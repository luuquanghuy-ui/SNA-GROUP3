import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.dont_write_bytecode = True
from evaluate_predictions import FIELDS, binary_scores


ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent
E1 = ROOT / "nguoi2" / "evaluation-20261018T031128Z-1-001" / "evaluation"
E2 = ROOT / "nguoi3" / "evaluation-20261018T031434Z-1-001" / "evaluation"
E3 = ROOT / "nguoi4" / "E3_outputs-20261018T031701Z-1-001" / "E3_outputs"
FEATURES = ROOT / "nguoi 1" / "honduras_features.csv"
PROCESSED = ROOT / "nguoi0" / "processed-20261018T032211Z-1-001" / "processed"


def rows(path):
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        yield from csv.DictReader(stream)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_close(actual, reported, name):
    if abs(float(actual) - float(reported)) > 1e-9:
        raise ValueError(f"Mismatch {name}: computed={actual}, reported={reported}")


def make_result(experiment, run_id, model, split, seed, positive_label, metrics, n_accounts, source, feature_set, notes):
    return [{
        "experiment": experiment,
        "run_id": run_id,
        "model": model,
        "split": split,
        "seed": seed,
        "positive_label": positive_label,
        "metric": name,
        "value": value,
        "n_accounts": n_accounts,
        "input": str(source.resolve()),
        "feature_set": feature_set,
        "notes": notes,
    } for name, value in metrics.items()]


def audit_e1():
    predictions_path = E1 / "e1_predictions.csv"
    metrics_path = E1 / "e1_metrics.csv"
    config = json.loads((ROOT / "nguoi2" / "e1_config.json").read_text(encoding="utf-8"))
    grouped = defaultdict(list)
    for row in rows(predictions_path):
        if row["split"] != "test":
            raise ValueError("E1 predictions contain non-test rows")
        if row["label"].lower() != ("bot" if row["y_true"] == "1" else "human"):
            raise ValueError("E1 label text disagrees with numeric label")
        grouped[row["model"]].append(row)
    reported = {row["model"]: row for row in rows(metrics_path)}
    if set(grouped) != set(reported):
        raise ValueError("E1 models differ between predictions and metrics")
    result_rows = []
    summaries = {}
    reference = None
    for model, data in sorted(grouped.items()):
        labels = {row["user_id"]: row["y_true"] for row in data}
        if len(labels) != len(data):
            raise ValueError(f"E1 duplicate user ID in {model}")
        if reference is None:
            reference = labels
        elif labels != reference:
            raise ValueError("E1 models use different test account sets or labels")
        truth = [int(row["y_true"] == "1") for row in data]
        pred = [int(row["y_pred"] == "1") for row in data]
        score = [float(row["bot_probability"]) for row in data]
        if any(not 0 <= value <= 1 for value in score):
            raise ValueError("E1 score outside [0, 1]")
        measured = binary_scores(truth, pred, score)
        claim = reported[model]
        comparisons = {
            "positive_precision": "precision_bot",
            "positive_recall": "recall_bot",
            "positive_f1": "f1_bot",
            "macro_f1": "macro_f1",
            "average_precision": "average_precision_bot",
            "tn": "tn", "fp": "fp", "fn": "fn", "tp": "tp",
        }
        require_close(len(data), claim["n_test"], f"E1 {model} n_test")
        for measured_name, claim_name in comparisons.items():
            require_close(measured[measured_name], claim[claim_name], f"E1 {model} {claim_name}")
        require_close((measured["tp"] + measured["tn"]) / len(data), claim["accuracy"], f"E1 {model} accuracy")
        summaries[model] = {"n_test": len(data), **measured}
        result_rows += make_result("E1", f"e1_{model.lower()}_seed42", model, "test", config["random_seed"], 1, measured, len(data), predictions_path, "fox8_account_text_tfidf_v1", "Bot=1; source-label confounding; no external-source validation")
    source_test = {row["user_id"]: row["label"] for row in rows(PROCESSED / "fox8_account_splits.csv") if row["split"] == "test"}
    if source_test != {account_id: ("bot" if value == "1" else "human") for account_id, value in reference.items()}:
        raise ValueError("E1 prediction test accounts or labels differ from source split")
    return {"models": summaries, "test_accounts_same_across_models": True, "test_matches_processed_fox8_split": True, "prediction_sha256": sha256(predictions_path), "metrics_sha256": sha256(metrics_path), "source_split_sha256": sha256(PROCESSED / "fox8_account_splits.csv"), "source_manifest_sha256": sha256(PROCESSED / "fox8_account_manifest.csv")}, result_rows


def audit_e2():
    predictions_path = E2 / "e2_predictions.csv"
    metrics_path = E2 / "e2_metrics.csv"
    reference = {row["account_id"]: row["label"] for row in rows(FEATURES) if row["split"] == "test"}
    data = [row for row in rows(predictions_path) if row["split"] == "test"]
    ids = {row["account_id"]: row["label"] for row in data}
    if len(ids) != len(data) or ids != reference:
        raise ValueError(f"E2 test account set or labels differ from Person 1: rows={len(data)}, unique={len(ids)}, expected={len(reference)}")
    reported = {(row["Model"], row["Split"].lower()): row for row in rows(metrics_path)}
    result_rows = []
    summaries = {}
    truth = [int(row["label"] == "0") for row in data]
    for model, pred_col, score_col in [("DummyClassifier", "dummy_pred", "dummy_prob_bad"), ("LogisticRegression", "lr_pred", "lr_prob_bad")]:
        pred = [int(row[pred_col] == "0") for row in data]
        score = [float(row[score_col]) for row in data]
        if any(not 0 <= value <= 1 for value in score):
            raise ValueError("E2 score outside [0, 1]")
        measured = binary_scores(truth, pred, score)
        claim = reported[(model, "test")]
        comparisons = {
            "positive_precision": "Bad_Precision",
            "positive_recall": "Bad_Recall",
            "positive_f1": "Bad_F1",
            "macro_f1": "Macro_F1",
            "average_precision": "Bad_PR_AUC",
        }
        for measured_name, claim_name in comparisons.items():
            require_close(measured[measured_name], claim[claim_name], f"E2 {model} {claim_name}")
        summaries[model] = {"n_test": len(data), **measured}
        result_rows += make_result("E2", f"e2_{model.lower()}_seed42", model, "test", 42, 0, measured, len(data), predictions_path, "honduras_six_account_statistics_v1", "Bad=0; full unbalanced Person 1 test; baseline only, no Fox8 transfer")
    return {"models": summaries, "test_matches_person1": True, "prediction_sha256": sha256(predictions_path), "metrics_sha256": sha256(metrics_path)}, result_rows


def audit_e3():
    metrics_path = E3 / "e3_graph_metrics.csv"
    node_path = E3 / "e3_graph_nodes.csv"
    edge_path = E3 / "e3_graph_edges.csv"
    claim = next(rows(metrics_path))
    config = json.loads((E3 / "e3_run_config.json").read_text(encoding="utf-8"))
    input_hashes_match = {name: sha256(PROCESSED / name) == expected for name, expected in config["input_sha256"].items()}
    if not all(input_hashes_match.values()):
        raise ValueError("E3 input hashes differ from processed data in nguoi0")
    reference = {row["account_id"]: (row["label"], row["split"]) for row in rows(FEATURES)}
    ids = set()
    degree_sum = 0
    isolate_count = 0
    activity_by_label = Counter()
    active_communities = set()
    for row in rows(node_path):
        account_id = row["account_id"]
        if account_id in ids:
            raise ValueError("E3 duplicate node ID")
        ids.add(account_id)
        if reference.get(account_id) != (row["label"], row["split"]):
            raise ValueError("E3 node label/split differs from Person 1")
        degree = int(row["degree"])
        activity_by_label[(row["label"], "active" if degree else "isolate")] += 1
        degree_sum += degree
        if degree == 0:
            isolate_count += 1
            if row["community"] != "-1":
                raise ValueError("E3 isolate assigned active community")
        else:
            if row["community"] == "-1":
                raise ValueError("E3 active node has no community")
            active_communities.add(row["community"])
    if ids != reference.keys():
        raise ValueError("E3 node set differs from Person 1")
    edge_count = 0
    for row in rows(edge_path):
        edge_count += 1
        if row["source"] not in ids or row["target"] not in ids:
            raise ValueError("E3 edge endpoint missing from node table")
        if row["source"] == row["target"] or int(row["weight"]) < 2:
            raise ValueError("E3 invalid edge")
    checks = {
        "n_nodes": len(ids),
        "n_active_nodes": len(ids) - isolate_count,
        "n_edges": edge_count,
        "n_isolates": isolate_count,
        "n_communities": len(active_communities),
    }
    for name, measured in checks.items():
        require_close(measured, claim[name], f"E3 {name}")
    require_close(degree_sum, 2 * edge_count, "E3 degree sum")
    community_rows = list(rows(E3 / "e3_communities.csv"))
    if len(community_rows) != len(active_communities) or sum(int(row["n_nodes"]) for row in community_rows) != len(ids) - isolate_count:
        raise ValueError("E3 community summary differs from node table")
    result_metrics = {name: claim[name] for name in [
        "n_nodes", "n_active_nodes", "n_edges", "n_isolates", "density", "n_components", "largest_component_nodes", "n_communities", "largest_community_nodes", "modularity", "pair_updates", "dropped_popular_events"
    ]}
    result_rows = make_result("E3", "e3_hashtag_hour_louvain_seed42", claim["method"], "full_corpus", claim["seed"], "", result_metrics, len(ids), metrics_path, "hashtag_hour_edges_v1", "Descriptive full-corpus graph; not E2 test-only; no campaign ground truth")
    by_label = {
        "bad_active": activity_by_label[("0", "active")],
        "bad_isolate": activity_by_label[("0", "isolate")],
        "good_active": activity_by_label[("1", "active")],
        "good_isolate": activity_by_label[("1", "isolate")],
    }
    return {"counts_checked": checks, "activity_by_label": by_label, "input_hashes_match_nguoi0": input_hashes_match, "degree_sum_matches_edges": True, "node_labels_splits_match_person1": True, "edge_endpoints_valid": True, "community_summary_matches_nodes": True, "metrics_sha256": sha256(metrics_path), "nodes_sha256": sha256(node_path), "edges_sha256": sha256(edge_path)}, result_rows


def main():
    e1, e1_rows = audit_e1()
    e2, e2_rows = audit_e2()
    e3, e3_rows = audit_e3()
    artifact_folders = [
        ROOT / "nguoi2" / "models-20261018T031152Z-1-001" / "models",
        ROOT / "nguoi2" / "preprocessing-20261018T031232Z-1-001" / "preprocessing",
        ROOT / "nguoi3" / "models-20261018T031439Z-1-001" / "models",
        ROOT / "nguoi3" / "preprocessing-20261018T031441Z-1-001" / "preprocessing",
    ]
    artifact_hashes = {str(path.relative_to(ROOT)): sha256(path) for folder in artifact_folders for path in folder.iterdir() if path.is_file()}
    all_rows = e1_rows + e2_rows + e3_rows
    with (OUT / "baseline_results.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)
    audit = {"status": "verified_against_delivered_files", "e1": e1, "e2": e2, "e3": e3, "model_artifact_sha256": artifact_hashes, "result_rows": len(all_rows), "limitations": [
        "Output files were checked against processed inputs in nguoi0; notebooks were not rerun unchanged in a fresh runtime.",
        "E1 and E2 use different input representations, so direct checkpoint transfer is not implemented.",
        "E3 uses full corpus and is descriptive, not the held-out E2 test protocol in the proposal.",
    ]}
    (OUT / "baseline_handoff_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": audit["status"], "result_rows": len(all_rows), "e1_models": list(e1["models"]), "e2_models": list(e2["models"]), "e3": e3["counts_checked"]}, ensure_ascii=True))


if __name__ == "__main__":
    main()
