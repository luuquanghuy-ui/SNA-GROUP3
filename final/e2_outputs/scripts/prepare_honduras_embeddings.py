"""Validate the fixed Honduras splits and produce E1-compatible tweet embeddings."""

import argparse
import csv
import hashlib
import json
import platform
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory


E2_ROOT = Path(__file__).resolve().parents[1]
E1_HANDOFF = E2_ROOT.parent / "e1_outputs" / "e1_handoff"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LABELS = {"0": "Bad", "1": "Good"}
SPLITS = ("train", "validation", "test")
MAX_TWEETS = 5
MAX_LENGTH = 128
EMBEDDING_DIM = 384
TWEET_FILES = ("honduras_bad_normalized.jsonl", "honduras_good_normalized.jsonl")
AUDIT_FILES = ("group_a_exact_duplicate_candidates.csv", "group_b_metadata_conflicts.csv")
MANIFEST_FILE = "honduras_account_manifest.csv"
SPLIT_FILE = "honduras_account_splits.csv"
EMBEDDING_FILE = "honduras_tweet_embeddings.npy"
LABEL_FILE = "honduras_labels.npy"
SPLIT_NAMES_FILE = "honduras_split_names.npy"
METADATA_FILE = "honduras_account_metadata.csv"
CONFIG_FILE = "honduras_embedding_config.json"
VALIDATION_FILE = "honduras_embedding_validation.json"


def csv_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        yield from csv.DictReader(stream)


def exact_id(value):
    if type(value) is int:
        value = str(value)
    if not isinstance(value, str) or not value.isascii() or not value.isdecimal():
        raise ValueError(f"Expected a decimal ID without float conversion, got {value!r}")
    return value


def account_id(row):
    for name in ("user_id", "account_id", "userid"):
        if name in row:
            return exact_id(row[name])
    raise ValueError("Missing user_id, account_id or userid")


def read_accounts(paths):
    labels = {}
    for row in csv_rows(paths[MANIFEST_FILE]):
        user = account_id(row)
        if user in labels or str(row["label"]) not in LABELS:
            raise ValueError(f"Invalid or duplicate manifest entry: {user}")
        labels[user] = int(row["label"])

    split_by_user = {}
    expected_ids = {split: set() for split in SPLITS}
    for row in csv_rows(paths[SPLIT_FILE]):
        user = account_id(row)
        split = {"val": "validation"}.get(row["split"], row["split"])
        if user in split_by_user or user not in labels or split not in expected_ids:
            raise ValueError(f"Invalid or duplicate split assignment: {user}, {split}")
        if "label" in row and int(row["label"]) != labels[user]:
            raise ValueError(f"Split label mismatch: {user}")
        split_by_user[user] = split
        expected_ids[split].add(user)

    if not labels or labels.keys() != split_by_user.keys():
        raise ValueError("Manifest and split table must contain the same nonempty account set")
    for split in SPLITS:
        actual = set()
        for row in csv_rows(paths[f"honduras_{split}_ids.csv"]):
            user = account_id(row)
            if user in actual or user not in expected_ids[split]:
                raise ValueError(f"Unexpected or duplicate account in {split} IDs: {user}")
            if "label" in row and int(row["label"]) != labels[user]:
                raise ValueError(f"Label mismatch in {split} IDs: {user}")
            actual.add(user)
        if not actual or actual != expected_ids[split]:
            raise ValueError(f"{split} IDs do not match the split table or are empty")

    return labels, split_by_user


def audit_ids(path):
    counts = {}
    for row in csv_rows(path):
        tweet_id = int(exact_id(row["tweet_id"]))
        count = int(row["record_count"])
        if tweet_id in counts or count < 2:
            raise ValueError(f"Invalid duplicate audit row in {path.name}: {tweet_id}")
        counts[tweet_id] = count
    return counts


def read_tweets(paths, labels, split_by_user):
    audit_a, audit_b = (audit_ids(paths[name]) for name in AUDIT_FILES)
    if audit_a.keys() & audit_b.keys():
        raise ValueError("An audited tweet ID is present in both groups")
    source_rows = {}
    observed_accounts = set()
    by_user = {user: [] for user in labels}
    seen = {}
    conflicts = set()
    repeats = Counter()
    stats = Counter()

    for label, name in enumerate(TWEET_FILES):
        print(f"Reading {name}", flush=True)
        line_no = 0
        with paths[name].open(encoding="utf-8-sig") as stream:
            for line_no, line in enumerate(stream, 1):
                try:
                    row = json.loads(line)
                    user = account_id(row)
                    tweet_id = int(exact_id(row["tweet_id"]))
                    text = row["text"]
                    if text is not None and not isinstance(text, str):
                        raise ValueError("Tweet text must be a string or null")
                    text = (text or "").strip()
                    if labels.get(user) != label or int(row["label"]) != label:
                        raise ValueError("Account missing from manifest or label does not match")
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"{name}:{line_no}: {exc}") from exc
                observed_accounts.add(user)
                if tweet_id in seen:
                    repeats[tweet_id] += 1
                    prior_user, prior_text = seen[tweet_id]
                    if (prior_user, prior_text) != (user, text):
                        conflicts.add(tweet_id)
                    continue
                seen[tweet_id] = (user, text)
                by_user[user].append((tweet_id, text))
        source_rows[name] = line_no

    if observed_accounts != labels.keys():
        raise ValueError(
            f"{len(labels.keys() - observed_accounts)} manifest accounts absent from normalized JSONL"
        )
    del seen

    for group, audit in (("a", audit_a), ("b", audit_b)):
        stats[f"group_{group}_audited_ids"] = len(audit)
        stats[f"group_{group}_count_mismatches"] = sum(
            repeats[tweet_id] + 1 != count for tweet_id, count in audit.items()
        )
        stats[f"group_{group}_conflicting_ids_excluded"] = len(audit.keys() & conflicts)
        stats[f"group_{group}_safe_duplicate_ids"] = sum(
            tweet_id in repeats and tweet_id not in conflicts for tweet_id in audit
        )
    stats["duplicate_rows_collapsed"] = sum(
        count for tweet_id, count in repeats.items() if tweet_id not in conflicts
    )
    stats["conflicting_ids_excluded"] = len(conflicts)
    stats["conflicting_rows_excluded"] = sum(repeats[tweet_id] + 1 for tweet_id in conflicts)
    stats["conflicting_ids_outside_audit"] = len(conflicts - audit_a.keys() - audit_b.keys())

    # Detect exact repeated content independently of tweet ID, without moving accounts.
    split_bit = {split: 1 << index for index, split in enumerate(SPLITS)}
    text_splits = {}
    for user, rows in by_user.items():
        bit = split_bit[split_by_user[user]]
        eligible = []
        for tweet_id, text in rows:
            if tweet_id in conflicts:
                continue
            if not text:
                stats["empty_text_rows_excluded"] += 1
                continue
            digest = hashlib.sha256(" ".join(text.split()).encode("utf-8")).digest()
            text_splits[digest] = text_splits.get(digest, 0) | bit
            eligible.append((digest, text))
        by_user[user] = eligible

    cross_split = {key for key, mask in text_splits.items() if mask.bit_count() > 1}
    del text_splits
    stats["cross_split_text_groups_excluded"] = len(cross_split)
    for user, rows in by_user.items():
        eligible = []
        used_texts = set()
        for digest, text in rows:
            if digest in cross_split:
                stats["cross_split_tweet_rows_excluded"] += 1
            elif digest in used_texts:
                stats["repeated_text_within_account_excluded"] += 1
            else:
                used_texts.add(digest)
                eligible.append(text)
        count = len(eligible)
        if count > MAX_TWEETS:
            # Same inclusive, evenly spaced positions as E1's integer linspace.
            eligible = [eligible[i * (count - 1) // (MAX_TWEETS - 1)] for i in range(MAX_TWEETS)]
        by_user[user] = (count, eligible)

    return by_user, {"normalized_rows": source_rows, "duplicate_and_text_audit": dict(stats)}


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def before_encoding(paths, handoff):
    feature_path = handoff / "preprocessing" / "feature_config.json"
    checkpoint = handoff / "checkpoints" / "best_model.pt"
    features = json.loads(feature_path.read_text(encoding="utf-8"))
    if features["tweet_embedding_dim"] != EMBEDDING_DIM or not checkpoint.is_file():
        raise ValueError("Missing E1 checkpoint or incompatible tweet embedding dimension")

    labels, split_by_user = read_accounts(paths)
    by_user, validation = read_tweets(paths, labels, split_by_user)
    validation.update({
        "accounts": len(labels),
        "labels_by_split": {
            split: dict(Counter(LABELS[str(label)] for user, label in labels.items()
                                if split_by_user[user] == split))
            for split in SPLITS
        },
        "accounts_without_selected_tweets_by_split": {
            split: dict(Counter(LABELS[str(label)] for user, label in labels.items()
                                if split_by_user[user] == split and not by_user[user][1]))
            for split in SPLITS
        },
        "selected_tweet_count_distribution": dict(sorted(Counter(
            len(tweets) for _, tweets in by_user.values()
        ).items())),
        "manifest_split_and_id_files_match": True,
        "normalized_account_and_label_coverage_matches_manifest": True,
    })
    return labels, split_by_user, by_user, validation, feature_path, checkpoint


def encode(stage, by_user, args):
    import numpy as np
    import torch
    import transformers
    from transformers import AutoModel, AutoTokenizer

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.fp16 and device != "cuda":
        raise ValueError("--fp16 requires CUDA")
    model_source = args.model_dir or MODEL_NAME
    model_options = {"local_files_only": True} if args.model_dir else {"revision": args.revision}
    tokenizer = AutoTokenizer.from_pretrained(model_source, **model_options)
    model = AutoModel.from_pretrained(model_source, **model_options).to(device).eval()
    if args.fp16:
        model.half()

    vectors = np.lib.format.open_memmap(
        stage / EMBEDDING_FILE, mode="w+", dtype="float32",
        shape=(len(by_user), EMBEDDING_DIM),
    )
    vectors[:] = 0
    block = []
    block_start = 0
    block_size = 8192

    def encode_block():
        entries = [
            (offset, text) for offset, texts in enumerate(block) for text in texts
        ]
        entries.sort(key=lambda entry: len(entry[1]))
        sums = np.zeros((len(block), EMBEDDING_DIM), dtype=np.float32)
        for start in range(0, len(entries), args.batch_size):
            batch = entries[start:start + args.batch_size]
            inputs = tokenizer(
                [text for _, text in batch], padding=True, truncation=True,
                max_length=MAX_LENGTH, return_tensors="pt",
            ).to(device)
            with torch.inference_mode():
                hidden = model(**inputs).last_hidden_state.float()
                mask = inputs["attention_mask"].unsqueeze(-1)
                pooled = ((hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)).cpu().numpy()
            np.add.at(sums, [offset for offset, _ in batch], pooled)
        counts = np.array([max(1, len(texts)) for texts in block], dtype=np.float32)
        vectors[block_start:block_start + len(block)] = sums / counts[:, None]
        print(f"Encoded {block_start + len(block):,}/{len(by_user):,} accounts", flush=True)

    for _, texts in by_user.values():
        block.append(texts)
        if len(block) == block_size:
            encode_block()
            block_start += len(block)
            block.clear()
    if block:
        encode_block()
    vectors.flush()
    return {
        "model": MODEL_NAME,
        "model_revision": getattr(model.config, "_commit_hash", None) or args.revision,
        "device": torch.cuda.get_device_name() if device == "cuda" else "cpu",
        "batch_size_tweets": args.batch_size,
        "inference_dtype": "float16" if args.fp16 else "float32",
        "embedding_dtype": "float32",
        "versions": {
            "python": platform.python_version(), "numpy": np.__version__,
            "torch": torch.__version__, "transformers": transformers.__version__,
        },
    }


def export_and_verify(stage, labels, split_by_user, by_user):
    import numpy as np

    np.save(stage / LABEL_FILE, np.fromiter(labels.values(), dtype=np.int64), allow_pickle=False)
    np.save(stage / SPLIT_NAMES_FILE, np.array([split_by_user[user] for user in labels]),
            allow_pickle=False)
    fields = ("row_index", "account_id", "label", "split", "eligible_tweet_count",
              "selected_tweet_count")
    with (stage / METADATA_FILE).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        for index, (user, label) in enumerate(labels.items()):
            eligible, texts = by_user[user]
            writer.writerow((index, user, label, split_by_user[user], eligible, len(texts)))

    vectors = np.load(stage / EMBEDDING_FILE, mmap_mode="r", allow_pickle=False)
    label_array = np.load(stage / LABEL_FILE, allow_pickle=False)
    splits = np.load(stage / SPLIT_NAMES_FILE, allow_pickle=False)
    size = len(labels)
    if vectors.shape != (size, EMBEDDING_DIM) or vectors.dtype != np.float32:
        raise ValueError("Embedding shape or dtype is invalid")
    if label_array.shape != (size,) or splits.shape != (size,):
        raise ValueError("Label or split array is not aligned with embeddings")

    zero_rows = []
    for start in range(0, size, 8192):
        block = vectors[start:start + 8192]
        if not np.isfinite(block).all():
            raise ValueError(f"Non-finite embedding in block starting at {start}")
        zero_rows.extend((~np.any(block != 0, axis=1)).tolist())

    checked = 0
    for (index, row), (user, label) in zip(
        enumerate(csv_rows(stage / METADATA_FILE)), labels.items(), strict=True
    ):
        eligible, texts = by_user[user]
        split = split_by_user[user]
        expected = (index, user, label, split, eligible, len(texts))
        if tuple(row[field] for field in fields) != tuple(map(str, expected)):
            raise ValueError(f"Account metadata mismatch at row {index}")
        if label_array[index] != label or splits[index] != split:
            raise ValueError(f"Label or split disagrees with metadata at row {index}")
        if zero_rows[index] != (not texts):
            raise ValueError(f"Unexpected zero/nonzero vector at row {index}")
        checked += 1

    return {
        "status": "passed",
        "shape": list(vectors.shape),
        "metadata_rows_verified": checked,
        "all_finite": True,
        "zero_vector_rows": sum(zero_rows),
        "zero_vectors_match_accounts_without_tweets": True,
        "artifact_sha256": {
            name: file_hash(stage / name) for name in
            (EMBEDDING_FILE, LABEL_FILE, SPLIT_NAMES_FILE, METADATA_FILE)
        },
    }


def run(args):
    names = [MANIFEST_FILE, SPLIT_FILE, *(f"honduras_{split}_ids.csv" for split in SPLITS),
             *TWEET_FILES, *AUDIT_FILES]
    paths = {name: args.input_dir / name for name in names}
    for name in AUDIT_FILES:
        nested = args.input_dir / "quality_audit" / name
        if nested.is_file():
            paths[name] = nested
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    labels, split_by_user, by_user, validation, feature_path, checkpoint = before_encoding(
        paths, args.e1_handoff
    )
    print(json.dumps(validation, indent=2), flush=True)
    if args.validate_only:
        return

    destination = args.output_dir.resolve()
    if destination.exists():
        raise FileExistsError(f"Choose a new output directory: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".honduras-", dir=destination.parent) as temp:
        stage = Path(temp) / "data"
        stage.mkdir()
        config = encode(stage, by_user, args)
        validation.update(export_and_verify(stage, labels, split_by_user, by_user))
        config.update({
            "embedding_dim": EMBEDDING_DIM,
            "label_mapping": LABELS,
            "split_names": SPLITS,
            "account_order": "manifest CSV row order",
            "id_contract": "decimal strings, never parsed as floating-point",
            "max_tweets_per_account": MAX_TWEETS,
            "max_tokens_per_tweet": MAX_LENGTH,
            "selection": "evenly spaced inclusive indices in normalized JSONL order after cleaning",
            "pooling": "attention-masked token mean, then mean of tweet vectors without L2 normalization",
            "duplicate_policy": (
                "Keep first same-ID occurrence if account and stripped text agree, including "
                "metadata-only conflicts. Exclude all occurrences if account or text conflicts. "
                "Exclude whitespace-normalized exact text present in multiple splits and collapse "
                "repeated content within each account."
            ),
            "missing_text_policy": "keep account with an all-zero vector",
            "e1_handoff": {
                "feature_config_sha256": file_hash(feature_path),
                "checkpoint_sha256": file_hash(checkpoint),
                "checkpoint_use": "downstream transfer initialization only",
            },
            "input_sha256": {name: file_hash(path) for name, path in paths.items()},
        })
        write_json(stage / CONFIG_FILE, config)
        write_json(stage / VALIDATION_FILE, validation)
        stage.rename(destination)
    print(f"Verified and exported {destination}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--e1-handoff", type=Path, default=E1_HANDOFF)
    parser.add_argument("--output-dir", type=Path, default=E2_ROOT / "data")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--model-dir", type=Path, help="Local snapshot of the specified MiniLM revision")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.model_dir is not None and (not args.model_dir.is_dir() or args.revision == "main"):
        parser.error("--model-dir requires an existing directory and an explicit --revision")
    run(args)


if __name__ == "__main__":
    main()
