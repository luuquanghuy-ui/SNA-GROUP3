"""Read-only artifact evaluation for the new research assignment.

No training, graph construction, parameter selection, or Git operations.
Inputs are immutable snapshots; outputs go only to --output.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score
from sklearn.metrics import precision_score, recall_score


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def reference_table(frame, id_column, label_mapping=None):
    frame = frame.rename(columns={id_column: "account_id"}).copy()
    require({"account_id", "label", "split"}.issubset(frame), "Reference schema incomplete")
    require(not pd.api.types.is_float_dtype(frame.account_id), "Float account IDs may have lost precision")
    require(not frame[["account_id", "label", "split"]].isna().any().any(), "Missing reference metadata")
    for col in ["account_id", "label", "split"]:
        frame[col] = frame[col].astype(str)
    require(frame.account_id.str.len().gt(0).all(), "Blank reference IDs")
    require(frame.account_id.is_unique, "Reference has duplicate IDs or account overlap between splits")
    require(set(frame.split) == {"train", "validation", "test"}, "Unexpected or missing reference splits")
    if label_mapping:
        frame["label"] = frame.label.map(label_mapping)
    require(frame.label.notna().all() and set(frame.label) == {"0", "1"}, "Invalid reference labels")
    return frame.set_index("account_id", drop=False)


def check_accounts(data, reference):
    require(not data.account_id.isna().any(), "Missing prediction/node ID")
    require(data.account_id.ne("").all() and data.account_id.is_unique, "Blank or duplicate prediction/node ID")
    actual = data.set_index("account_id")
    require(set(actual.index) == set(reference.index),
            f"Account set mismatch: missing={len(set(reference.index)-set(actual.index))}, "
            f"extra={len(set(actual.index)-set(reference.index))}")
    require((actual.label.reindex(reference.index) == reference.label).all(), "Label mismatch with reference")
    if "split" in actual:
        require((actual.split.reindex(reference.index) == reference.split).all(), "Split mismatch with reference")


def classification_scores(truth, prediction, scores, positive, negative):
    truth, prediction = np.asarray(truth, dtype=str), np.asarray(prediction, dtype=str)
    require(len(truth) > 0 and len(truth) == len(prediction), "Empty/misaligned predictions")
    allowed = {positive, negative}
    require(set(truth) <= allowed and set(prediction) <= allowed, "Unexpected classification labels")
    y, yp = (truth == positive).astype(int), (prediction == positive).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yp, labels=[0, 1]).ravel()
    result = {
        "n_accounts": len(y), "n_positive": int(y.sum()), "n_negative": int((1-y).sum()),
        "positive_precision": precision_score(y, yp, zero_division=0),
        "positive_recall": recall_score(y, yp, zero_division=0),
        "positive_f1": f1_score(y, yp, zero_division=0),
        "macro_f1": f1_score(y, yp, labels=[0, 1], average="macro", zero_division=0),
        "accuracy": float((y == yp).mean()),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "average_precision": None,
    }
    if scores is not None:
        scores = np.asarray(scores, dtype=float)
        require(len(scores) == len(y) and np.isfinite(scores).all(), "Invalid or nonfinite positive scores")
        require(((scores >= 0) & (scores <= 1)).all(), "Positive probabilities outside [0, 1]")
        if y.sum():
            result["average_precision"] = average_precision_score(y, scores)
    return result


def probability_values(series):
    # Python float preserves the CSV decimal round trip; pd.to_numeric may merge
    # adjacent probabilities and change AP tie groups near 1.0.
    return series.map(float)


def close(actual, expected, name):
    require(actual is not None and math.isfinite(float(expected)), f"Missing/nonfinite reported {name}")
    require(math.isclose(float(actual), float(expected), rel_tol=1e-9, abs_tol=1e-10),
            f"Metric mismatch {name}: recomputed={actual}, reported={expected}")


def activity_bins(values):
    values = pd.to_numeric(values, errors="raise")
    require(np.isfinite(values).all() and (values >= 0).all(), "Invalid tweet_count")
    return pd.cut(values, [-1, 0, 1, 5, 30, np.inf], labels=["0", "1", "2-5", "6-30", "31+"])


def measure_graph(n_nodes, source, target, weights, membership, resolution=1):
    """Measure a simple undirected weighted graph, retaining all isolates."""
    source, target = np.asarray(source, dtype=np.int64), np.asarray(target, dtype=np.int64)
    weights, membership = np.asarray(weights, dtype=float), np.asarray(membership, dtype=str)
    require(len(membership) == n_nodes and len(source) == len(target) == len(weights), "Graph shape mismatch")
    require(n_nodes > 0 and len(source) < 2**63 // n_nodes, "Invalid graph size")
    require(np.isfinite(weights).all() and (weights > 0).all(), "Nonpositive/nonfinite edge weight")
    require(((source >= 0) & (source < n_nodes) & (target >= 0) & (target < n_nodes)).all(), "Unknown endpoint")
    require((source != target).all(), "Self-loop")
    keys = np.minimum(source, target) * n_nodes + np.maximum(source, target)
    require(len(np.unique(keys)) == len(keys), "Duplicate undirected edge, including reversed duplicates")
    degree = np.bincount(np.concatenate([source, target]), minlength=n_nodes)
    strength = np.bincount(np.concatenate([source, target]), weights=np.tile(weights, 2), minlength=n_nodes)
    active = degree > 0
    require((membership[active] != "-1").all(), "Active node lacks community assignment")
    require((membership[~active] == "-1").all(), "Expected isolate community=-1")
    community_ids = sorted(set(membership[active]))
    codes = pd.Categorical(membership, categories=community_ids).codes.astype(np.int64)
    nc = len(community_ids)
    community_size = np.bincount(codes[active], minlength=nc)
    community_strength = np.bincount(codes[active], weights=strength[active], minlength=nc)
    internal = codes[source] == codes[target]
    internal_weight = np.bincount(codes[source[internal]], weights=weights[internal], minlength=nc)
    internal_edges = np.bincount(codes[source[internal]], minlength=nc)
    matrix = coo_matrix((np.ones(len(source)), (source, target)), shape=(n_nodes, n_nodes)).tocsr()
    n_components, component_ids = connected_components(matrix, directed=False)
    total_weight = float(weights.sum())
    q = float(np.sum(internal_weight / total_weight - resolution * (community_strength / (2*total_weight))**2)) if total_weight else None
    metrics = {
        "n_nodes": n_nodes, "n_edges": len(source), "n_active_nodes": int(active.sum()),
        "n_isolates": int((~active).sum()), "isolated_node_rate": float((~active).mean()),
        "density": 2*len(source)/(n_nodes*(n_nodes-1)) if n_nodes > 1 else 0.,
        "n_components": int(n_components), "largest_component_nodes": int(np.bincount(component_ids).max()),
        "n_communities": nc, "largest_community_nodes": int(community_size.max()) if nc else 0,
        "modularity": q,
    }
    communities = pd.DataFrame({"community": community_ids, "n_nodes": community_size,
                                "n_edges": internal_edges, "internal_weight": internal_weight})
    return metrics, communities, degree, strength


def save_csv(path, records, columns=None):
    pd.DataFrame(records, columns=columns).to_csv(path, index=False, encoding="utf-8-sig")


def markdown_table(records, columns):
    def fmt(value):
        if value is None:
            return "Chưa có/không xác định"
        if isinstance(value, float):
            if value != 0 and abs(value) < 1e-5:
                return f"{value:.3e}"
            return f"{value:.6f}"
        return str(value).replace("|", "/").replace("\n", " ")
    return "\n".join(["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"]*len(columns)) + " |"] +
                     ["| " + " | ".join(fmt(r.get(c)) for c in columns) + " |" for r in records])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True, help="Immutable source snapshot, NOT output directory")
    parser.add_argument("--processed", type=Path, required=True, help="Canonical processed data with both account split CSVs")
    parser.add_argument("--features", type=Path, help="Optional override of snapshot/data/validation/honduras_features.parquet")
    parser.add_argument("--encoder-audit", type=Path, help="Optional sample audit from verify_encoder_handoff.py")
    parser.add_argument("--e1-inference-audit", type=Path, help="Optional full Test saved-checkpoint inference audit")
    parser.add_argument("--e2-handoff-audit", type=Path, help="Optional independently reviewed newer E2 delivery")
    parser.add_argument("--e2-snapshot", type=Path, help="Pinned snapshot used by the newer E2 review")
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("research_manifest.json"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("results"))
    args = parser.parse_args()
    require(args.output.resolve() != args.snapshot.resolve() and args.snapshot.resolve() not in args.output.resolve().parents,
            "Output must be outside the immutable input snapshot")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    tree = json.loads((args.snapshot / "repository_tree.json").read_text(encoding="utf-8"))
    require(tree["commit"] == manifest["snapshot_commit"], "Snapshot commit differs from manifest")
    require(not tree.get("truncated"), "Repository inventory is truncated")
    files = {item["path"]: item for item in tree["tree"] if item["type"] == "blob"}
    evidence = {}

    def source(relative):
        path = args.snapshot / relative
        require(relative in files and path.is_file(), f"Missing source artifact: {relative}")
        if relative not in evidence:
            data = path.read_bytes()
            blob_sha = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
            require(blob_sha == files[relative]["sha"], f"Source changed after snapshot: {relative}")
            evidence[relative] = {"sha256": hashlib.sha256(data).hexdigest(), "git_blob_sha": blob_sha, "bytes": len(data)}
        return path

    feature_path = args.features or source("data/validation/honduras_features.parquet")
    feature_raw = pd.read_parquet(feature_path) if feature_path.suffix == ".parquet" else read_csv(feature_path)
    features = reference_table(feature_raw, "account_id")
    honduras_path = args.processed / "honduras_account_splits.csv"
    fox8_path = args.processed / "fox8_account_splits.csv"
    honduras = reference_table(read_csv(honduras_path), "user_id")
    fox8 = reference_table(read_csv(fox8_path), "user_id", {"bot": "1", "human": "0"})
    check_accounts(features, honduras)
    test_honduras = honduras[honduras.split == "test"]
    classification, activities, checks, failures = [], [], [], []
    config_notes = []
    for run in manifest["classification_runs"]:
        try:
            raw = read_csv(source(run["predictions"]))
            if run.get("model_filter"):
                spec = run["model_filter"]
                raw = raw[raw[spec["column"]] == spec["value"]]
            cols = run["columns"]
            reference = fox8 if run["experiment"] == "E1" else honduras
            config = json.loads(source(run["config"]).read_text(encoding="utf-8"))
            if run.get("code"):
                source(run["code"])
            if run["experiment"] == "E1":
                require(config["random_seed"] == run["seed"], "E1 seed differs from config")
            elif "seed" not in config and "random_seed" not in config:
                config_notes.append(f"{run['run_id']}: seed, class weight, threshold and train-only scaler are not fully recorded in e2_config.json; notebook is separate evidence.")
            reported = read_csv(source(run["reported_metrics"]))
            require(set(raw.split) <= {"validation", "test"}, "Unexpected prediction split")
            require("test" in set(raw.split), "Test predictions missing")
            for split, subset in raw.groupby("split", sort=True):
                data = pd.DataFrame({"account_id": subset[cols["id"]], "label": subset[cols["truth"]], "split": subset.split})
                expected = reference[reference.split == split]
                check_accounts(data, expected)
                score = probability_values(subset[cols["score"]]) if cols.get("score") else None
                measured = classification_scores(data.label, subset[cols["prediction"]], score,
                                                  run["positive_label"], run["negative_label"])
                model_col = "model" if run["experiment"] == "E1" else "Model"
                claim = reported[reported[model_col] == run["method"]]
                if "Split" in claim:
                    claim = claim[claim.Split.str.lower() == split]
                require(len(claim) == 1, "Missing/ambiguous reported metric row")
                claim = claim.iloc[0]
                mapping = ({"positive_precision": "precision_bot", "positive_recall": "recall_bot", "positive_f1": "f1_bot",
                            "macro_f1": "macro_f1", "average_precision": "average_precision_bot", "accuracy": "accuracy",
                            "tp": "tp", "fp": "fp", "tn": "tn", "fn": "fn", "n_accounts": "n_test"}
                           if run["experiment"] == "E1" else
                           {"positive_precision": "Bad_Precision", "positive_recall": "Bad_Recall", "positive_f1": "Bad_F1",
                            "macro_f1": "Macro_F1", "average_precision": "Bad_PR_AUC"})
                for key, column in mapping.items():
                    close(measured[key], claim[column], f"{run['run_id']} {split} {key}")
                result = {"experiment": run["experiment"], "run_id": run["run_id"], "method": run["method"],
                          "split": split, "positive_label": run["positive_label"], "seed": run["seed"],
                          "feature_set": run["feature_set"], "status": "metrics_and_account_set_verified",
                          "training_rerun_this_audit": False, **measured, "source": run["predictions"]}
                classification.append(result)
                if run["experiment"] == "E2" and split == "test":
                    bins = activity_bins(features.loc[data.account_id, "tweet_count"]).astype(str).to_numpy()
                    for activity in ["0", "1", "2-5", "6-30", "31+"]:
                        selected = bins == activity
                        if not selected.any():
                            continue
                        group_scores = classification_scores(data.label.to_numpy()[selected], subset[cols["prediction"]].to_numpy()[selected],
                                                             score.to_numpy()[selected] if score is not None else None,
                                                             run["positive_label"], run["negative_label"])
                        activities.append({"experiment": "E2", "run_id": run["run_id"], "scope": "test", "activity_bin": activity,
                                           "analysis": "descriptive_existing_predictions_not_feature_ablation", **group_scores})
            checks.append({"item": run["run_id"], "status": "pass", "meaning": "Predictions/metrics/reference checked; not a fresh training rerun"})
        except (ValueError, KeyError, OSError) as exc:
            failures.append({"item": run["run_id"], "error": str(exc)})

    graphs, community_records = [], []
    for run in manifest["graph_runs"]:
        try:
            nodes = read_csv(source(run["nodes"]))
            reference = honduras if run["scope"] == "full_corpus" else test_honduras
            check_accounts(nodes, reference)
            source(run["config"])
            source(run["code"])
            source(run["source_report"])
            index = pd.Index(nodes.account_id)
            us, vs, ws = [], [], []
            for chunk in pd.read_csv(source(run["edges"]), dtype={"source": str, "target": str}, chunksize=200000):
                us.append(index.get_indexer(chunk.source))
                vs.append(index.get_indexer(chunk.target))
                ws.append(pd.to_numeric(chunk.weight, errors="raise").to_numpy())
            u = np.concatenate(us) if us else np.array([], dtype=np.int64)
            v = np.concatenate(vs) if vs else np.array([], dtype=np.int64)
            w = np.concatenate(ws) if ws else np.array([], dtype=float)
            measured, communities, degree, strength = measure_graph(len(nodes), u, v, w, nodes.community, run["resolution"])
            require(np.array_equal(degree, pd.to_numeric(nodes.degree).to_numpy()), "Node degrees differ from edges")
            require(np.allclose(strength, pd.to_numeric(nodes.strength).to_numpy(), rtol=1e-10, atol=1e-10), "Node strength mismatch")
            reported = read_csv(source(run["reported_metrics"]))
            require(len(reported) == 1, "Ambiguous graph metrics")
            for key in measured:
                if key in reported and measured[key] is not None:
                    close(measured[key], reported.iloc[0][key], f"{run['run_id']} {key}")
            official = run["scope"] == "test" and run["community_method"].lower() == "leiden" and manifest["protocol_status"] == "approved"
            bad = nodes.label.eq("0").to_numpy()
            n_bad, n_good = int(bad.sum()), int((~bad).sum())
            active = degree > 0
            graph_result = {"experiment": "E3", "run_id": run["run_id"], "method": run["method"], "scope": run["scope"],
                            "community_method": run["community_method"], "seed": run["seed"], "purpose": run["purpose"],
                            "official_comparison_eligible": official, "status": "artifact_metrics_verified_not_pipeline_rerun", **measured,
                            "n_bad": n_bad, "bad_active": int((bad & active).sum()), "good_active": int((~bad & active).sum()),
                            "bad_active_coverage": float((bad & active).sum()/n_bad) if n_bad else None,
                            "good_active_coverage": float((~bad & active).sum()/n_good) if n_good else None,
                            "global_bad_prevalence": n_bad/len(nodes), "source": run["reported_metrics"]}
            graphs.append(graph_result)
            node_data = nodes.assign(bad=bad.astype(int), tweet_count=pd.to_numeric(features.loc[nodes.account_id, "tweet_count"]).to_numpy())
            summaries = node_data[active].groupby("community").agg(bad_accounts=("bad", "sum"),
                    mean_tweet_count=("tweet_count", "mean"), median_tweet_count=("tweet_count", "median"))
            communities = communities.join(summaries, on="community")
            communities["good_accounts"] = communities.n_nodes - communities.bad_accounts
            communities["bad_ratio"] = communities.bad_accounts / communities.n_nodes
            communities["bad_coverage"] = communities.bad_accounts / n_bad if n_bad else np.nan
            communities["bad_enrichment"] = communities.bad_ratio / (n_bad/len(nodes)) if n_bad else np.nan
            communities["density"] = np.where(communities.n_nodes > 1,
                    2*communities.n_edges/(communities.n_nodes*(communities.n_nodes-1)), 0.)
            original = read_csv(source(run["communities"])).set_index("community")
            require(set(original.index) == set(communities.community), "Community IDs differ from node table")
            for row in communities.to_dict("records"):
                for key in ["n_nodes", "n_edges", "bad_accounts", "good_accounts", "bad_ratio", "density"]:
                    close(row[key], original.loc[row["community"], key], f"Community {row['community']} {key}")
                community_records.append({"run_id": run["run_id"], "scope": run["scope"], "purpose": run["purpose"], **row})
            bins = activity_bins(node_data.tweet_count).astype(str)
            for activity, subset in node_data.assign(activity_bin=bins, active=active).groupby("activity_bin"):
                nb = int(subset.bad.sum())
                activities.append({"experiment": "E3", "run_id": run["run_id"], "scope": run["scope"], "activity_bin": activity,
                    "analysis": "descriptive_prototype_not_official_E3", "n_accounts": len(subset), "n_positive": nb,
                    "n_negative": len(subset)-nb, "n_active": int(subset.active.sum()),
                    "bad_active_coverage": float(subset.loc[subset.bad.eq(1), "active"].mean()) if nb else None,
                    "good_active_coverage": float(subset.loc[subset.bad.eq(0), "active"].mean()) if len(subset)>nb else None})
            checks.append({"item": run["run_id"], "status": "pass", "meaning": "Edges/nodes/communities/structural metrics recomputed; prototype is not official protocol"})
        except (ValueError, KeyError, OSError) as exc:
            failures.append({"item": run["run_id"], "error": str(exc)})

    # A missing experiment is represented as pending, never a zero-valued result.
    ablations = [
        {"analysis": "E2 Scratch vs Transfer", "owner": "Le Phuc Thinh + evaluation", "priority": 1, "status": "pending_artifacts", "required_inputs": "Both models, same Test IDs, predictions, scores, checkpoint/config", "result": None},
        {"analysis": "E3 Combined vs +Scratch vs +Transferred", "owner": "Tran Tuyet Nhu / Nguyen Van Nam + evaluation", "priority": 1, "status": "pending_artifacts", "required_inputs": "Three graphs on fixed Test nodes with identical Leiden settings", "result": None},
        {"analysis": "E2 with/without tweet_count", "owner": "Le Phuc Thinh + evaluation", "priority": 2, "status": "pending_paired_runs", "required_inputs": "Paired models trained on same split; dropping a column only at inference is invalid", "result": None},
        {"analysis": "E3 edge-count control", "owner": "Graph owners + evaluation", "priority": 3, "status": "pending_artifacts", "required_inputs": "Keep top M edges in every graph, M=min(edge counts); rerun same Leiden, retain all Test nodes", "result": None},
        {"analysis": "E3 lambda/top-k/threshold/window sensitivity", "owner": "Protocol owner + graph owners + evaluation", "priority": 4, "status": "pending_predeclared_grid", "required_inputs": "Approved label-blind parameter grid; change one factor at a time; do not select by Test enrichment", "result": None},
        {"analysis": "Activity-level descriptive analysis", "owner": "Evaluation", "priority": 2, "status": "computed_on_available_artifacts" if activities else "pending", "required_inputs": "tweet_count bins fixed at 0, 1, 2-5, 6-30, 31+; no retraining", "result": "activity_analysis.csv" if activities else None},
        {"analysis": "ARI", "owner": "Protocol owner + evaluation", "priority": 4, "status": "pending_reference_definition", "required_inputs": "Specify whether comparing partitions or Bad/Good labels; node scope and isolate policy; not campaign accuracy", "result": None},
    ]
    e2_handoff = None
    if args.e2_handoff_audit:
        require(args.e2_snapshot is not None, "--e2-snapshot is required for the E2 handoff audit")
        e2_handoff = json.loads(args.e2_handoff_audit.read_text(encoding="utf-8"))
        e2_tree = json.loads((args.e2_snapshot/"repository_tree.json").read_text(encoding="utf-8"))
        require(e2_tree["commit"] == e2_handoff["snapshot_commit"], "E2 review snapshot mismatch")
        new_files = {r["path"]: r for r in e2_tree["tree"] if r["type"] == "blob"}
        for relative, claim in e2_handoff["artifact_evidence"].items():
            if relative.startswith("E2/"):
                path = args.e2_snapshot/relative
                data = path.read_bytes()
                require(hashlib.sha1(f"blob {len(data)}\0".encode()+data).hexdigest() == new_files[relative]["sha"], "E2 snapshot blob mismatch")
                require(sha256(path) == claim["sha256"], "E2 input changed after review")
            else:
                require(files[relative]["sha"] == new_files[relative]["sha"], "E1 input differs between snapshots")
        reference_claims = e2_handoff["local_input_sha256"]
        require(any(Path(p).name == "honduras_account_splits.csv" and h == sha256(honduras_path)
                    for p,h in reference_claims.items()), "E2 review reference differs")
        lr_path = args.e2_handoff_audit.parent/"lr_ablation_predictions.csv"
        require(sha256(lr_path) == e2_handoff["lr_ablation_predictions_sha256"], "LR reproduction predictions changed")
        neural_data = read_csv(args.e2_snapshot/"E2/artifacts/e2_predictions.csv")
        lr_data = read_csv(lr_path)
        for result in e2_handoff["classification_results"]:
            if result["method"] == "Train from Scratch":
                data, pred_col, score_col = neural_data, "pred_scratch", "prob_bad_scratch"
            elif result["method"] == "Fox8-pretrained + Fine-tuning":
                data, pred_col, score_col = neural_data, "pred_transfer", "prob_bad_transfer"
            else:
                data, pred_col, score_col = lr_data, "no_tweet_count_pred", "no_tweet_count_prob_bad"
            check_accounts(data.rename(columns={"true_label": "label"}).assign(split="test"), test_honduras)
            score = probability_values(data[score_col])
            measured = classification_scores(data.true_label, data[pred_col], score, "0", "1")
            for key, value in measured.items():
                close(value, result[key], "Reviewed E2 "+result["method"]+" "+key)
            classification.append({**result, "snapshot_commit": e2_handoff["snapshot_commit"]})
            bins = activity_bins(features.loc[data.account_id, "tweet_count"]).astype(str).to_numpy()
            for activity in ["0", "1", "2-5", "6-30", "31+"]:
                mask = bins == activity
                if mask.any():
                    group = classification_scores(data.true_label.to_numpy()[mask], data[pred_col].to_numpy()[mask],
                                                  score.to_numpy()[mask], "0", "1")
                    activities.append({"experiment": "E2", "run_id": result["run_id"], "scope": "test",
                                       "activity_bin": activity, "analysis": "descriptive_new_E2_delivery", **group})
        checks.append({"item": "new_E2_delivery", "status": "metrics_verified_protocol_issues",
                       "meaning": "Scratch/Transfer and LR ablation metrics rechecked against Test; see separate protocol issues and checkpoint inference audit"})
        for record in ablations:
            if record["analysis"] == "E2 Scratch vs Transfer":
                record.update(status="preliminary_metrics_verified_protocol_revision_needed", result="e2_handoff/scratch_transfer_deltas.csv")
            if record["analysis"] == "E2 with/without tweet_count":
                record.update(status="completed_for_LR_only" if all(r["matches_reported_metrics"] for r in e2_handoff["lr_ablation_reproduction"]) else "LR_reproduction_differs_from_claim",
                              result="e2_handoff/e2_verified_metrics.csv")
    encoder_smoke = None
    if args.encoder_audit:
        encoder_smoke = json.loads(args.encoder_audit.read_text(encoding="utf-8"))
        require(encoder_smoke["snapshot_commit"] == manifest["snapshot_commit"], "Encoder audit commit mismatch")
        require(encoder_smoke["status"] == "sample_compatibility_passed_not_full_embedding_validation"
                and encoder_smoke["all_finite"] and encoder_smoke["exact_repeatability_eval_mode"], "Encoder sample check failed")
        delivery = encoder_smoke.get("delivered_embeddings")
        if delivery:
            require(delivery["structural_validation_passed"] and delivery["sample_value_comparison"]["matches_recomputed_E1_encoder"],
                    "Delivered embedding audit failed")
            require(sha256(Path(delivery["path"])) == delivery["sha256"], "Embedding CSV changed since delivery audit")
    handoffs = [{**r, "status": "not_found_in_audited_snapshot"} for r in manifest["pending"]]
    if e2_handoff:
        for record in handoffs:
            if record["experiment"] == "E2" and record["method"] in {"Train from Scratch", "Fox8-pretrained + Fine-tuning"}:
                record.update(status="received_metrics_verified_protocol_issues", owner="Le Phuc Thinh / PThinhL (commit author)",
                    reason="New E2 notebook/checkpoints/predictions received. Metrics match; check loss weighting, validation use, neural seed and handoff documentation before official acceptance.")
    if encoder_smoke and encoder_smoke.get("delivered_embeddings"):
        for record in handoffs:
            if record["experiment"] == "E1":
                record["status"] = "received_local_csv_structure_and_sample_verified"
                record["reason"] = "CSV received outside GitHub; full ID/label/shape/finite checks passed, fixed sample matches E1 checkpoint within tolerance. This is E1-only, not the E2 fine-tuned representation; full regeneration provenance remains limited to the tested sample."
    e1_inference = None
    if args.e1_inference_audit:
        e1_inference = json.loads(args.e1_inference_audit.read_text(encoding="utf-8"))
        require(e1_inference["snapshot_commit"] == manifest["snapshot_commit"], "E1 inference audit commit mismatch")
        require(e1_inference["status"] == "inference_predictions_reproduced_not_retraining", "E1 inference audit failed")
        require(e1_inference["test_accounts"] == len(fox8[fox8.split == "test"]), "E1 inference Test size mismatch")
        require(e1_inference["reference_split_sha256"] == sha256(fox8_path), "E1 inference reference changed")
        require(e1_inference["encoder_weights_equal_full_model"], "Encoder/full-model weights differ")
        require(set(e1_inference["models"]) == {"LogisticRegression", "RandomForest", "MLP"}, "Missing E1 inference model")
        require(all(r["prediction_agreement_fraction"] == 1 and r["n_accounts"] == e1_inference["test_accounts"]
                    for r in e1_inference["models"].values()), "E1 inference predictions differ")
        expected_hash = sha256(source("Fox8 baseline bot detection/evaluation/e1_predictions.csv"))
        require(e1_inference["artifact_sha256"]["evaluation/e1_predictions.csv"] == expected_hash, "E1 inference used another prediction file")
        checks.append({"item": "e1_saved_checkpoint_inference", "status": "pass",
                       "meaning": "All three saved models reproduce labels on full Fox8 Test; encoder weights equal classifier encoder; no retraining"})
    args.output.mkdir(parents=True, exist_ok=True)
    save_csv(args.output / "classification_comparison.csv", classification)
    save_csv(args.output / "graph_comparison.csv", graphs)
    save_csv(args.output / "community_metrics.csv", community_records)
    save_csv(args.output / "activity_analysis.csv", activities)
    save_csv(args.output / "ablation_results.csv", ablations)
    save_csv(args.output / "handoff_status.csv", handoffs)
    counts = {name: frame.groupby(["split", "label"]).size().to_dict() for name, frame in [("Fox8", fox8), ("Honduras", honduras)]}
    audit = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(), "repository": manifest["repository"],
        "commit": manifest["snapshot_commit"], "status": "pass_with_protocol_gaps" if not failures else "failed",
        "checks": checks, "failures": failures, "config_notes": sorted(set(config_notes)),
        "source_evidence": evidence,
        "reference_evidence": {str(p): sha256(p) for p in [fox8_path, honduras_path, feature_path]},
        "split_counts": {name: [{"split": split, "label": label, "n_accounts": n} for (split, label), n in values.items()] for name, values in counts.items()},
        "environment": {"python": platform.python_version(), **{p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scipy", "scikit-learn", "pyarrow"]}},
        "scope": "Recomputed delivered predictions/graph edges; optional separately executed encoder and full-Test inference audits are identified explicitly. No fresh training, Leiden rerun or full tweet-to-graph pipeline rerun.",
        "missing_artifacts_scope": "Audited GitHub commit only; files on private Drive/messages may exist and have not been audited",
        "ari": "not_computed_pending_protocol_definition",
        "protocol_approved": False,
        "encoder_smoke": encoder_smoke,
        "encoder_smoke_audit_sha256": sha256(args.encoder_audit) if args.encoder_audit else None,
        "encoder_audit_file": args.encoder_audit.name if args.encoder_audit else None,
        "handoff_status": handoffs,
        "e1_checkpoint_inference": e1_inference,
        "e2_handoff": e2_handoff,
        "e2_handoff_audit_sha256": sha256(args.e2_handoff_audit) if args.e2_handoff_audit else None,
        "e1_inference_audit_sha256": sha256(args.e1_inference_audit) if args.e1_inference_audit else None,
        "fox8_source_label_counts": (fox8.groupby(["datasets", "label"]).size().rename("n_accounts").reset_index().to_dict("records")
                                     if "datasets" in fox8 else []),
        "score_parser": "CSV columns read as strings, then Python float; no probability rounding before AP",
    }
    (args.output / "verification_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    report = make_report(manifest, classification, graphs, activities, ablations, checks, failures, audit)
    (args.output / "E1_E2_E3_COMPARISON.md").write_text(report, encoding="utf-8")
    make_figure(classification, args.output)
    print(json.dumps({"status": audit["status"], "classification_rows": len(classification), "graphs": len(graphs),
                      "communities": len(community_records), "activity_rows": len(activities), "failures": failures,
                      "output": str(args.output.resolve())}, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


def make_report(manifest, classification, graphs, activities, ablations, checks, failures, audit):
    test = [r for r in classification if r["split"] == "test"]
    e1 = [r for r in test if r["experiment"] == "E1"]
    e2 = [r for r in test if r["experiment"] == "E2"]
    new_e2 = audit.get("e2_handoff")
    lines = ["# E1–E2–E3 — Kiểm định và so sánh kết quả hiện có", "",
        f"Nguồn đã kiểm tra: [{manifest['snapshot_commit'][:12]}]({manifest['repository']}/tree/{manifest['snapshot_commit']}).",
        "Đây là báo cáo của Người 6 theo phân công mới. Chưa phải kết luận cuối về hiệu quả transfer.", "",
        f"Trạng thái kiểm định artifact: **{audit['status']}**. Nếu có failures, chưa nghiệm thu run bị lỗi.", "",
        "## 1. Câu hỏi nghiên cứu và phạm vi kiểm định", "",
        "- E1: phân loại Bot/Human trên Fox8; kết quả bị giới hạn bởi source-label confounding.",
        ("- E2: đã có cặp Scratch/Transfer để so sánh định lượng trong run được bàn giao; còn vấn đề protocol nên chưa chốt kết luận nghiên cứu cuối."
         if new_e2 else "- E2: Scratch và Fox8-pretrained + Fine-tuning có khác nhau về Bad-F1/Macro-F1/AP không? Chưa đủ artifact để trả lời."),
        "- E3: representation có thay đổi graph/cộng đồng và phân bố Bad không? Chưa có bộ graph chính thức để trả lời.",
        "Lần này tính lại metrics từ predictions và cấu trúc graph từ edge list; không train lại model, xuất full embeddings hoặc dựng lại graph từ tweet. Kiểm tra encoder trên một mẫu nhỏ được ghi riêng ở mục 10 nếu có audit kèm theo.", "",
        "## 2. Dữ liệu và kiểm tra split", "",
        "ID được đọc như chuỗi. Đối chiếu toàn bộ ID/label/split với hai manifest local; kiểm tra feature table trên repo khớp manifest Honduras.", ""]
    for dataset, counts in audit["split_counts"].items():
        lines += [f"### {dataset}", "", markdown_table(counts, ["split", "label", "n_accounts"]), ""]
    if audit.get("fox8_source_label_counts"):
        lines += ["### Fox8 — đối chiếu nguồn thu thập và nhãn", "",
                  markdown_table(audit["fox8_source_label_counts"], ["datasets", "label", "n_accounts"]), "",
                  "Nhãn số trong bảng này là Bot=1/Human=0. Mỗi account xuất hiện đúng một lần; một Human có metadata ghi nhận hai nguồn. Tất cả Bot thuộc Fox8, Human thuộc các nguồn khác: đây là bằng chứng trực tiếp về source-label confounding ở cấp nguồn. Không dùng trường dataset làm feature vẫn chưa loại bỏ dấu vết nguồn trong nội dung.", ""]
    lines += ["Fox8: Bot=1, Human=0. Honduras: Bad=0, Good=1. AP dùng score của đúng lớp dương; Bad_PR_AUC trong file E2 thực chất được notebook tính bằng average_precision_score, nên ở đây ghi AP.", "",
        "## 3. E1 — kết quả tính lại trên Test", "",
        markdown_table(e1, ["method", "n_accounts", "positive_precision", "positive_recall", "positive_f1", "macro_f1", "average_precision", "fp", "fn"]), "",
        "LR và MLP cùng đạt F1=1 trên tập Test hiện có; không có căn cứ xếp MLP tốt hơn LR từ chỉ số này. RF có một false positive. Không suy rộng kết quả này thành phát hiện bot tổng quát.", "",
        "E1_VALIDATION.md đã bàn giao mô tả encoder 128 chiều và kiểm tra checkpoint. Repo có script verify_e1_honduras.py; full embedding CSV không được lưu trên GitHub. Nếu đã nhận CSV riêng, trạng thái và kết quả kiểm tra nằm ở mục 8/10 dưới đây. Script gốc đặt checkpoint/vectorizer trong e1_outputs/, khác bố cục models/ và preprocessing/ trên repo; cần chỉnh đường dẫn khi chạy. Không coi sự hiện diện của script là bằng chứng nó đã chạy thành công.", "",
        "## 4. E2 — baseline đã kiểm định", "",
        markdown_table(e2, ["method", "n_accounts", "positive_precision", "positive_recall", "positive_f1", "macro_f1", "average_precision", "tp", "fp", "fn", "tn"]), "",
        "Mọi baseline E2 dùng đúng cùng 44.938 tài khoản Test: 373 Bad, 44.565 Good. Majority không phát hiện Bad. LR phát hiện 360/373 Bad, bỏ sót 13 và gán nhầm 3.679 Good thành Bad. Recall cao đi kèm precision thấp; không dùng Accuracy làm tiêu chí chính.",
        "Notebook fit StandardScaler trên train, transform validation/test; LR dùng class_weight='balanced', random_state=42, max_iter=1000. e2_config.json chưa lưu đầy đủ các tham số này và đang đặt tên project là 'Honduras Bot Detection', dù nhiệm vụ là Bad/Good. Người 3 cần bổ sung config; không sửa artifact nguồn trong lần kiểm định này.",
        (f"Bổ sung bàn giao E2 từ [{new_e2['snapshot_commit'][:12]}]({new_e2['repository']}/tree/{new_e2['snapshot_commit']}), author PThinhL. E1/E3 và baseline cũ vẫn giữ nguồn snapshot ghi đầu báo cáo. Nhóm có đổi phân công nên xác định chủ sở hữu theo artifact/commit, không theo số folder."
         if new_e2 else "Scratch, Transfer và optional Frozen chưa xuất hiện trong snapshot được kiểm tra. Delta Transfer−Scratch chưa tính; không dùng 0 để thay kết quả chưa nhận."), "",
        "## 5. E3 — prototype hiện có, tách khỏi kết quả chính thức", "",
        markdown_table(graphs, ["method", "scope", "community_method", "n_nodes", "n_edges", "n_isolates", "n_components", "n_communities", "modularity", "official_comparison_eligible"]), "",
        "Bản E3 trên GitHub vẫn ghi hashtag–giờ UTC, weight>=2, Louvain, toàn corpus. Việc đã thống nhất chạy Leiden trong tin nhắn chưa được thể hiện bằng artifact Leiden ở commit này. Không so prototype này với graph tăng cường Test+Leiden như một đối chứng chính thức.",
        "Các số cộng đồng/modularity ở bảng trên được tính lại từ graph/assignment mới nhất đã merge, không lấy từ baseline_results.csv cũ. Tính lại đúng metrics của assignment không chứng minh tái lập thuật toán tìm assignment.", "",
        "### Định nghĩa phân tích cộng đồng đang dùng", "",
        "- Bad Ratio(C) = số Bad trong C / số node trong C.",
        "- Bad Coverage(C) = số Bad trong C / tổng Bad của phạm vi graph.",
        "- Bad Enrichment(C) = Bad Ratio(C) / tỷ lệ Bad toàn bộ phạm vi graph, bao gồm isolates trong mẫu số nền.",
        "- Bad active coverage = số Bad có cạnh / tổng Bad trong phạm vi graph.",
        "- Community=-1 là isolate; không gom tất cả isolates thành một cộng đồng.",
        "Các định nghĩa này được công khai để đối chiếu; cần Người 1 chốt cho bảng E3 chính thức. ARI chưa tính do chưa có định nghĩa cặp phân hoạch/nhãn tham chiếu và chính sách isolates được phê duyệt.",
        "Toàn bộ cộng đồng và số lượng Bad/Good, coverage/enrichment, kích thước và activity nằm trong community_metrics.csv. Không chọn community theo Bad Ratio để điều chỉnh graph.", "",
        "## 6. Phân tích activity đã làm", "",
        "Dùng các mức tweet_count cố định 0, 1, 2–5, 6–30, 31+, không chọn bin theo kết quả Test. Tính lại metrics E2 theo từng mức và coverage graph prototype theo activity; chi tiết activity_analysis.csv. Đây là phân tích mô tả predictions/graph đã có, KHÔNG phải thí nghiệm loại bỏ tweet_count và không chứng minh quan hệ nhân quả.", ""]
    activity_rows = [r for r in activities if r["run_id"] == "e2_logistic_regression"]
    lines += [markdown_table(activity_rows, ["activity_bin", "n_accounts", "n_positive", "positive_precision", "positive_recall", "positive_f1", "average_precision"]), ""]
    lr = next((r for r in e2 if r["method"] == "LogisticRegression"), None)
    one_tweet = next((r for r in activity_rows if r["activity_bin"] == "1"), None)
    if lr and one_tweet and lr["fp"]:
        fraction = one_tweet["fp"] / lr["fp"]
        lines += [f"Phát hiện từ phân tích bổ sung: nhóm một tweet/account có {one_tweet['n_accounts']:,} tài khoản, {one_tweet['n_positive']} Bad; LR tạo {one_tweet['fp']:,}/{lr['fp']:,} false positives tại nhóm này ({fraction:.2%} tổng FP). Nhóm này đồng thời lớn và có tỷ lệ Bad thấp. Sự khác biệt theo activity là lý do cần kiểm tra có/không tweet_count, chưa chứng minh tweet_count là nguyên nhân trực tiếp.", ""]
    if graphs:
        lines += ["Coverage E3 prototype theo nhãn, chỉ để mô tả toàn corpus:", "",
                  markdown_table(graphs, ["run_id", "bad_active", "good_active", "bad_active_coverage", "good_active_coverage"]), ""]
    lines += [
        "## 7. Ablation và đối chứng còn thiếu", "",
        markdown_table(ablations, ["analysis", "priority", "status", "required_inputs"]), "",
        "Đối chứng số cạnh theo proposal: M=min số cạnh của các graph được so; mỗi graph giữ M cạnh có weight cao nhất, hòa thì dùng thứ tự endpoint ID đã cố định. Giữ toàn bộ node Test và chạy lại cùng Leiden; không chỉ cắt CSV rồi tái sử dụng community cũ.",
        "Thay đổi tweet_count cần huấn luyện một cấu hình đối chứng hợp lệ; không xóa feature lúc inference của model đã train với feature đó. Nếu input của encoder không có cột tweet_count mà là văn bản, cần chốt ablation về giới hạn tweet/activity phù hợp với kiến trúc; không gọi việc xóa một cột thống kê không được model sử dụng là ablation của encoder.", "",
        "## 8. Trạng thái bàn giao và artifact còn thiếu", "",
        markdown_table(audit["handoff_status"], ["experiment", "method", "owner", "status", "required", "reason"]), "",
        "'Chưa nhận' chỉ có nghĩa là chưa thấy trong commit đã kiểm tra. File có thể đã được tạo hoặc gửi riêng ngoài GitHub; không kết luận thành viên chưa làm.", "",
        "## 9. Hạn chế và kết luận hiện tại", "",
        "- E1 có source-label confounding; chưa có external-source validation trong snapshot.",
        ("- E2: đã kiểm tra cặp Scratch/Transfer và tái lập LR ablation; cần xử lý class weighting, validation/checkpoint selection và neural seed trước khi nghiệm thu theo protocol."
         if new_e2 else "- E2 mất cân bằng; LR có nhiều false positives. Chưa có cặp Scratch/Transfer và ablation tweet_count."),
        "- E3 hiện là prototype toàn corpus/Louvain; chưa có bộ graph chính thức Test/Leiden, representation và edge-count control.",
        "- Encoder chạy được trên Honduras không đồng nghĩa representation chuyển giao có ích; cần kết quả E2/E3 đối chứng.",
        "- Không có campaign ground truth độc lập. Modularity, enrichment và ARI không tự xác nhận campaign thật.",
        "- Đây không phải chạy notebook nguyên trạng trong môi trường sạch; phiên bản môi trường kiểm định nằm trong verification_audit.json.",
        ("Kết luận: cặp E2 đã nhận có Transfer tăng Bad-F1/Recall nhưng giảm Precision/AP; chỉ là kết quả của run hiện tại với các vấn đề protocol chưa giải quyết. LR ablation đã được tái lập bằng hàm tác giả. Chưa có bộ ba graph chính thức để đánh giá E3; không tự dùng checkpoint E2 đang cần sửa làm kết quả đề xuất đã nghiệm thu."
         if new_e2 else "Kết luận: đã kiểm tra lại các baseline nhận được và tạo phân tích activity/cộng đồng từ artifact hiện có. Chưa đủ bằng chứng kết luận transfer cải thiện E2 hoặc E3. Ưu tiên nhận Scratch/Transfer và ba graph Combined/+Scratch/+Transferred đúng protocol trước các sensitivity mở rộng."), "",
        "## 10. Checklist kiểm định lần này", ""]
    lines += [f"- [x] {c['item']}: {c['meaning']}" for c in checks]
    lines += [f"- [ ] Lỗi {c['item']}: {c['error']}" for c in failures]
    smoke = audit.get("encoder_smoke")
    inference = audit.get("e1_checkpoint_inference")
    if inference:
        lines += ["", "### Chạy lại inference checkpoint E1 trên toàn bộ Test", "",
                  markdown_table([{"method": name, **row} for name, row in inference["models"].items()],
                                 ["method", "n_accounts", "prediction_agreement_fraction", "max_absolute_probability_difference"]), "",
                  "Đọc raw Fox8, tổng hợp văn bản và dùng vectorizer/checkpoint có hash khớp bản trên GitHub. Cả ba model tái tạo đúng nhãn dự đoán trên toàn Test; xác suất có sai số float như bảng. prediction_agreement_fraction=1 không có nghĩa Accuracy=100% (RF vẫn có một FP so với nhãn thật). Encoder riêng có tensor trọng số bằng chính xác phần encoder trong checkpoint MLP đầy đủ. Đây là inference bằng model đã lưu, không phải huấn luyện lại trong môi trường sạch. Chi tiết e1_checkpoint_inference_audit.json.", ""]
    if smoke:
        lines += ["", "### Kiểm tra tương thích encoder độc lập trên mẫu Honduras", "",
                  f"Đã nạp checkpoint/vectorizer có Git blob SHA khớp artifact GitHub. Chọn {smoke['n_accounts']} account Test theo ID cố định, cân bằng hai nhãn chỉ để smoke test. Embedding shape={smoke['embedding_shape']}; finite={smoke['all_finite']}; repeatability={smoke['exact_repeatability_eval_mode']}; zero TF-IDF rows={smoke['zero_tfidf_rows']}; unique embedding rows={smoke['unique_embedding_rows']}.",
                  f"Đây là encoder E1 gốc, không phải encoder E2 đã fine-tune. Kiểm tra nạp/chạy encoder chỉ áp dụng cho mẫu đã thử. Chi tiết {audit['encoder_audit_file']}. Có cảnh báo vectorizer lưu bằng scikit-learn 1.6.1 nhưng môi trường kiểm tra dùng 1.8.0; chưa phải môi trường gốc đã khóa phiên bản.", ""]
        delivery = smoke.get("delivered_embeddings")
        if delivery:
            sample = delivery["sample_value_comparison"]
            fingerprints = delivery["embedding_row_fingerprints"]
            lines += ["### CSV embeddings E1 nhận riêng ngoài GitHub", "",
                      f"File `{delivery['path']}` ({delivery['bytes']:,} bytes), SHA256 `{delivery['sha256']}`.", "",
                      markdown_table([delivery], ["n_rows", "n_unique_account_ids", "embedding_dimension", "duplicate_id_rows", "missing_account_ids", "missing_test_account_ids", "label_mismatch_rows", "nonfinite_embedding_rows", "zero_norm_embedding_rows"]), "",
                      f"Đối chiếu manifest: {delivery['reference_split_counts']}. Có đủ toàn bộ tài khoản Test. Kiểm tra {sample['n_accounts']} vector bằng cách tạo lại từ checkpoint E1 và ghép theo account_id: khớp trong rtol={sample['rtol']}, atol={sample['atol']}; max absolute difference={sample['max_absolute_difference']:.9g}. Không khẳng định byte-identical hoặc đã tạo lại toàn bộ CSV.",
                      f"Sau khi đọc float32 có {fingerprints['n_unique_fingerprints']:,} dấu vân tay vector khác nhau; nhóm vector giống nhau lớn nhất có {fingerprints['largest_identical_vector_group']:,} tài khoản, phân bố nhãn {fingerprints['largest_group_labels']}. Vector trùng không có nghĩa ID trùng hoặc dữ liệu lỗi. Nếu tạo similarity graph, cần tie-break top-k cố định khi cosine bằng nhau; không coi vector giống nhau là bằng chứng campaign.",
                      "CSV này gồm account_id, 128 cột emb_000…emb_127 và label. Chỉ 128 cột embedding được dùng làm vector; ID/label là metadata, không đưa vào feature hoặc cosine similarity.",
                      "Phân loại nguồn representation: Fox8 E1 encoder áp dụng lên Honduras, chưa có fine-tuning E2. Có thể dùng làm artifact kiểm tra/tái sử dụng encoder hoặc đối chứng frozen/Fox8-only nếu protocol cho phép. Không dùng thay Scratch embeddings hay Fox8-pretrained + fine-tuned embeddings trong hai graph đề xuất.", ""]
    if new_e2:
        lines += ["", "## 11. E2 mới — delta, inference và các điểm cần sửa", "",
                  markdown_table(new_e2["deltas"], ["metric", "scratch", "transfer", "transfer_minus_scratch", "difference_percentage_points"]), "",
                  "Đơn vị cột cuối là điểm phần trăm, không phải phần trăm cải thiện tương đối. Scratch: TP=254, FP=1, FN=119. Transfer: TP=315, FP=10, FN=58. Recall tăng đổi lại FP tăng; AP giảm nhẹ. Không nói Transfer tốt hơn ở mọi metric.", "",
                  markdown_table([{"method": key, **value} for key,value in new_e2["neural_inference"].items()],
                                 ["method", "n_accounts", "prediction_agreement_fraction", "max_absolute_probability_difference"]), "",
                  "Đã chạy inference từ hai checkpoint trên toàn Test bằng raw text và vectorizer E1, không retrain neural models. LR có/không tweet_count được tái lập bằng chính hàm trong notebook tác giả. Xem e2_handoff/E2_HANDOFF_REVIEW.md và e2_handoff/e2_handoff_audit.json.", "",
                  "Điểm cần xử lý: BCEWithLogitsLoss áp dụng pos_weight cho target=1. Notebook giữ Good=1 nhưng đặt pos_weight≈119,43, nên weight lớn đang áp dụng cho Good, không phải Bad. Cần tác giả giải trình hoặc sửa loss/remap có ghi nhận rồi chạy lại hai phương pháp. Việc chọn Bad=0 làm lớp dương khi tính metrics không thay đổi cách BCE hiểu target=1.", "",
                  markdown_table(new_e2["protocol_issues"], ["severity", "item", "finding", "action"]), "",
                  "Không có validation evaluation/chọn checkpoint trong neural training function đã bàn giao; không có torch/numpy seed trong notebook và config neural. Đây là thiếu bằng chứng protocol/tái lập, không tự động chứng minh tác giả đã tune trên Test. Báo cáo còn cần sửa cách gọi Bad=Bot/Good=Human và đường dẫn artifact. Không sửa artifact nguồn thay tác giả.", ""]
    lines += ["- [ ] Protocol chính thức đã chốt, có objective/resolution/seeds, lambda/top-k/threshold và định nghĩa ARI.",
              ("- [ ] Giải quyết các điểm protocol của E2 mới và nhận bộ graph E3 chính thức." if new_e2 else "- [ ] Nhận và kiểm định encoder verification evidence, E2 Scratch/Transfer và E3 chính thức."),
              ("- [x] LR có/không tweet_count đã tái lập; edge-count control và sensitivity E3 còn chờ graph chính thức."
               if new_e2 and all(r["matches_reported_metrics"] for r in new_e2["lr_ablation_reproduction"])
               else "- [ ] Hoàn tất ablation tweet_count và edge-count control; ghi rõ sensitivity nào đã chạy/chưa chạy."),
              "- [ ] Chạy tái lập trong môi trường mới với input đầy đủ và lưu bằng chứng.",
              "- [ ] Chốt bảng baseline–proposed và kết luận cuối sau khi đủ đầu vào.", ""]
    return "\n".join(lines)


def make_figure(classification, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [r for r in classification if r["experiment"] == "E2" and r["split"] == "test"]
    if not rows:
        return
    def plot(chosen, filename, title, caption):
        if not chosen:
            return
        fig, ax = plt.subplots(figsize=(8, 4.5))
        metrics = ["positive_precision", "positive_recall", "positive_f1", "macro_f1", "average_precision"]
        x = np.arange(len(metrics))
        width = min(.32, .8/len(chosen))
        for i, row in enumerate(chosen):
            bars = ax.bar(x+(i-(len(chosen)-1)/2)*width, [row[m] for m in metrics], width,
                          label=row["method"], color=["#B8B8B8", "#285477"][i % 2])
            ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=3)
        ax.set_xticks(x, ["Bad Precision", "Bad Recall", "Bad F1", "Macro F1", "AP"])
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Score")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.01), ncol=2, fontsize=9)
        fig.suptitle(title, fontsize=12, y=.99)
        fig.text(.5, .015, caption, ha="center", fontsize=8)
        fig.tight_layout(rect=[0, .05, 1, .94])
        fig.savefig(output/filename, dpi=180)
        plt.close(fig)
    plot([r for r in rows if r["method"] in {"DummyClassifier", "LogisticRegression"}],
         "e2_baseline_comparison.png", "E2 delivered baselines — fixed Test, 373 Bad / 44,565 Good",
         "Recomputed from delivered predictions; statistical baselines only.")
    neural = [r for r in rows if r["method"] in {"Train from Scratch", "Fox8-pretrained + Fine-tuning"}]
    if neural:
        plot(neural, "e2_scratch_transfer_comparison.png", "E2 Scratch vs Transfer — preliminary received run",
             "Same Test IDs; class weighting, validation and seed issues remain. See E2 handoff review.")
        plot([r for r in rows if r["method"] in {"LogisticRegression", "LR (No tweet_count)"}],
             "e2_tweet_count_ablation.png", "E2 LR ablation — with vs without tweet_count",
             "Owner's LR function reproduced on the same train/Test split; this is not a neural ablation.")


if __name__ == "__main__":
    main()
