"""Verify the delivered E1 encoder on a bounded Honduras sample, without training.

Uses the existing encoder architecture and preprocessing described by the owner.
Does not export full embeddings or substitute for E2 fine-tuning.
"""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn

from run_research_evaluation import read_csv, reference_table, require, sha256


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(.3),
                                     nn.Linear(hidden_dim, embedding_dim), nn.ReLU())

    def forward(self, x):
        return self.encoder(x)


def audit_delivered_embeddings(path, reference, selected, expected):
    """Scan the full delivery; compare sample vectors by ID, not row order."""
    dimensions = expected.shape[1]
    columns = [f"emb_{i:03d}" for i in range(dimensions)]
    header = pd.read_csv(path, nrows=0).columns.tolist()
    require("account_id" in header and all(c in header for c in columns), "Embedding CSV schema mismatch")
    require(len([c for c in header if c.startswith("emb_")]) == dimensions, "Unexpected embedding dimension")
    seen, unknown = set(), set()
    selected_set, sample_vectors = set(selected), {}
    fingerprints, fingerprint_labels = Counter(), {}
    rows = duplicate_ids = missing_ids = bad_values = label_mismatches = zero_rows = 0
    split_counts = Counter()
    reference_labels = reference.label.to_dict()
    reference_splits = reference.split.to_dict()
    for chunk in pd.read_csv(path, dtype={"account_id": str, "label": str, "split": str},
                             chunksize=10000, float_precision="round_trip", keep_default_na=False):
        values = chunk[columns].to_numpy(dtype=np.float32)
        finite = np.isfinite(values).all(axis=1)
        bad_values += int((~finite).sum())
        zero_rows += int((np.linalg.norm(values.astype(np.float64), axis=1) == 0).sum())
        ids = chunk.account_id.to_list()
        labels = chunk.label.to_list() if "label" in chunk else [None]*len(chunk)
        supplied_splits = chunk.split.to_list() if "split" in chunk else None
        for i, (account, vector, label) in enumerate(zip(ids, values, labels)):
            rows += 1
            if not account:
                missing_ids += 1
            if account in seen:
                duplicate_ids += 1
            seen.add(account)
            if account not in reference_labels:
                unknown.add(account)
            else:
                split_counts[reference_splits[account]] += 1
                if label is not None and label != reference_labels[account]:
                    label_mismatches += 1
                if supplied_splits is not None:
                    require(supplied_splits[i] == reference_splits[account], "Embedding split mismatch")
            if account in selected_set:
                sample_vectors[account] = vector.copy()
            if finite[i]:
                digest = hashlib.sha256(vector.tobytes()).digest()
                fingerprints[digest] += 1
                label_counts = fingerprint_labels.setdefault(digest, Counter())
                label_counts[reference_labels.get(account, "unknown")] += 1
    missing = set(reference.index) - seen
    missing_test = set(reference.index[reference.split == "test"]) - seen
    sample_missing = selected_set - sample_vectors.keys()
    sample_match, sample_max_difference = False, None
    mismatched_sample_rows = None
    if not sample_missing:
        actual = np.stack([sample_vectors[account] for account in selected])
        same = np.isclose(actual, expected, rtol=1e-6, atol=1e-7)
        sample_match = bool(same.all())
        sample_max_difference = float(np.max(np.abs(actual-expected)))
        mismatched_sample_rows = int((~same.all(axis=1)).sum())
    most_common = fingerprints.most_common(1)
    largest_group = most_common[0] if most_common else (None, 0)
    structural_pass = not (duplicate_ids or missing_ids or unknown or missing or bad_values or label_mismatches or zero_rows)
    return {
        "path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256(path),
        "n_rows": rows, "n_unique_account_ids": len(seen), "embedding_dimension": dimensions,
        "label_column_present": "label" in header, "duplicate_id_rows": duplicate_ids,
        "blank_id_rows": missing_ids, "unknown_account_ids": len(unknown),
        "missing_account_ids": len(missing), "missing_test_account_ids": len(missing_test),
        "label_mismatch_rows": label_mismatches, "nonfinite_embedding_rows": bad_values,
        "zero_norm_embedding_rows": zero_rows, "reference_split_counts": dict(split_counts),
        "structural_validation_passed": structural_pass,
        "sample_value_comparison": {"n_accounts": len(selected), "missing_sample_ids": len(sample_missing),
             "matches_recomputed_E1_encoder": sample_match, "mismatched_rows": mismatched_sample_rows,
             "max_absolute_difference": sample_max_difference, "rtol": 1e-6, "atol": 1e-7},
        "embedding_row_fingerprints": {"precision": "float32", "algorithm": "sha256",
             "n_unique_fingerprints": len(fingerprints), "largest_identical_vector_group": largest_group[1],
             "largest_group_labels": dict(fingerprint_labels.get(largest_group[0], {})),
             "groups_containing_both_labels": sum(bool(c["0"] and c["1"]) for c in fingerprint_labels.values())},
        "provenance_scope": "All rows checked structurally; only the fixed sample was re-encoded and compared. Not full regeneration or E2 fine-tuning."
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument("--processed", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--vectorizer", type=Path, required=True)
    p.add_argument("--per-label", type=int, default=64)
    p.add_argument("--embeddings", type=Path, help="Optional delivered full CSV to scan and compare by account ID")
    p.add_argument("--output", type=Path, default=Path(__file__).with_name("results") / "encoder_smoke_audit.json")
    args = p.parse_args()
    require(0 < args.per_label <= 512, "Smoke sample must be bounded: 1..512 accounts per label")
    tree = json.loads((args.snapshot / "repository_tree.json").read_text(encoding="utf-8"))
    blobs = {r["path"]: r["sha"] for r in tree["tree"] if r["type"] == "blob"}
    input_files = [args.checkpoint, args.vectorizer, args.processed / "honduras_account_splits.csv"]
    for path, name in [(args.checkpoint, "models/e1_behavioral_encoder.pt"),
                       (args.vectorizer, "preprocessing/tfidf_vectorizer.joblib")]:
        data = path.read_bytes()
        require(hashlib.sha1(f"blob {len(data)}\0".encode()+data).hexdigest() == blobs["Fox8 baseline bot detection/"+name],
                f"Artifact does not match GitHub snapshot: {name}")
    # Load only the verified, delivered artifacts. The checkpoint contains tensor state + metadata.
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    vectorizer = joblib.load(args.vectorizer)
    cfg = json.loads((args.snapshot / "Fox8 baseline bot detection/e1_config.json").read_text(encoding="utf-8"))
    prep = cfg["text_preprocessing"]
    dims = {key: int(checkpoint[key]) for key in ["input_dim", "hidden_dim", "embedding_dim"]}
    require(len(vectorizer.vocabulary_) == dims["input_dim"], "Vocabulary/encoder dimension mismatch")
    model = Encoder(**dims)
    model.encoder.load_state_dict(checkpoint["encoder_state_dict"], strict=True)
    model.eval()
    torch.manual_seed(42)
    torch.set_num_threads(2)
    reference = reference_table(read_csv(input_files[-1]), "user_id")
    test = reference[reference.split == "test"]
    selected = []
    for label in ["0", "1"]:
        selected.extend(sorted(test.index[test.label == label])[:args.per_label])
    texts = {account: [] for account in selected}
    records = 0
    for name in ["honduras_bad_normalized.jsonl", "honduras_good_normalized.jsonl"]:
        path = args.processed / name
        input_files.append(path)
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                records += 1
                row = json.loads(line)
                account = str(row["account_id"])
                if account not in texts or len(texts[account]) >= prep["max_tweets_per_account"]:
                    continue
                text = row.get("text")
                if isinstance(text, str) and text.strip():
                    texts[account].append(text.strip())
    missing = [account for account in selected if not texts[account]]
    require(not missing, f"Selected accounts have no usable text: {len(missing)}")
    account_text = [" ".join(texts[a])[:prep["max_characters_per_account"]] for a in selected]
    vectors = vectorizer.transform(account_text)
    tensor = torch.from_numpy(vectors.toarray().astype(np.float32))
    with torch.no_grad():
        first, second = model(tensor).numpy(), model(tensor).numpy()
    require(first.shape == (len(selected), dims["embedding_dim"]), "Unexpected embedding shape")
    require(np.isfinite(first).all(), "Embedding has NaN/Inf")
    stable = bool(np.array_equal(first, second))
    require(stable, "Encoder is not repeatable in eval mode")
    audit = {
        "status": "sample_compatibility_passed_not_full_embedding_validation",
        "snapshot_commit": tree["commit"], "architecture": dims,
        "sample_selection": "First sorted Test account IDs per label; stratified smoke sample, not a natural-distribution evaluation",
        "sample_per_label_requested": args.per_label, "n_accounts": len(selected),
        "sample_counts": {label: sum(test.loc[a, "label"] == label for a in selected) for label in ["0", "1"]},
        "account_ids_sha256": hashlib.sha256("\n".join(selected).encode()).hexdigest(),
        "normalized_records_scanned": records, "embedding_shape": list(first.shape),
        "all_finite": bool(np.isfinite(first).all()), "exact_repeatability_eval_mode": stable,
        "zero_tfidf_rows": int((vectors.getnnz(axis=1) == 0).sum()),
        "unique_embedding_rows": int(np.unique(first, axis=0).shape[0]),
        "preprocessing": {"max_tweets": prep["max_tweets_per_account"], "max_characters": prep["max_characters_per_account"],
                          "tweet_selection": "First valid texts in normalized input file order, as in delivered verification script; no refit of vectorizer"},
        "input_sha256": {str(path): sha256(path) for path in input_files},
        "environment": {name: importlib.metadata.version(name) for name in ["torch", "scikit-learn", "joblib", "pandas", "numpy"]},
        "limitations": ["No E2 fine-tuning performed; this is the Fox8 E1 encoder, not the Transferred E2 encoder.",
                        "Only the explicitly sampled Test accounts are validated, not all 224685 accounts or all Test IDs.",
                        "Finite, repeatable embeddings do not establish useful transfer or generalization.",
                        "Raw tweet count is not an input column of this TF-IDF encoder; tweet budget/activity can still affect representation."]
    }
    if args.embeddings:
        audit["delivered_embeddings"] = audit_delivered_embeddings(args.embeddings, reference, selected, first)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(json.dumps({k: audit[k] for k in ["status", "n_accounts", "embedding_shape", "zero_tfidf_rows", "unique_embedding_rows"]}))
    if args.embeddings:
        print(json.dumps(audit["delivered_embeddings"], ensure_ascii=False))
        require(audit["delivered_embeddings"]["structural_validation_passed"], "Delivered embedding structural audit failed; inspect saved JSON")
        require(audit["delivered_embeddings"]["sample_value_comparison"]["matches_recomputed_E1_encoder"], "Delivered sample differs from E1 encoder; inspect saved JSON")


if __name__ == "__main__":
    main()
