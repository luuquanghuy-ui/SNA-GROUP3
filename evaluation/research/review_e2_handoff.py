"""Audit the received E2 run; reuse owner's inference/classes and LR function.

No neural training, label remapping, loss correction, or source-file changes.
"""
import argparse
import ast
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support, f1_score, average_precision_score
import torch
from torch import nn

from run_research_evaluation import check_accounts, classification_scores, close, probability_values
from run_research_evaluation import read_csv, reference_table, require, sha256, markdown_table


METRICS = {"positive_precision": "Bad-Precision", "positive_recall": "Bad-Recall", "positive_f1": "Bad-F1",
           "macro_f1": "Macro-F1", "average_precision": "AP"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument("--processed", type=Path, required=True)
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--e1-dir", type=Path, required=True)
    p.add_argument("--bad-jsonl", type=Path, required=True)
    p.add_argument("--good-jsonl", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path(__file__).with_name("results") / "e2_handoff")
    args = p.parse_args()
    tree = json.loads((args.snapshot / "repository_tree.json").read_text(encoding="utf-8"))
    items = {r["path"]: r for r in tree["tree"] if r["type"] == "blob"}
    evidence = {}

    def source(name, local=None):
        path = local or args.snapshot/name
        b = path.read_bytes()
        require(hashlib.sha1(f"blob {len(b)}\0".encode()+b).hexdigest() == items[name]["sha"], f"Artifact changed: {name}")
        evidence[name] = {"sha256": hashlib.sha256(b).hexdigest(), "bytes": len(b)}
        return path

    notebook = json.loads(source("E2/E2_Honduras.ipynb").read_text(encoding="utf-8"))
    cells = ["".join(c.get("source", [])) for c in notebook["cells"] if c["cell_type"] == "code"]
    config = json.loads(source("E2/artifacts/e2_config.json").read_text(encoding="utf-8"))
    source("E2/e2_results_report.md")
    predictions = read_csv(source("E2/artifacts/e2_predictions.csv"))
    reference_path = args.processed/"honduras_account_splits.csv"
    reference = reference_table(read_csv(reference_path), "user_id")
    test = reference[reference.split == "test"]
    check_accounts(predictions.rename(columns={"true_label": "label"}).assign(split="test"), test)
    claims = read_csv(source("E2/evaluation/E2_RESULTS_TABLE.csv")).set_index("Method")
    names = {"scratch": "Train from Scratch", "transfer": "Fox8-pretrained + Fine-tuning"}
    results, checks = [], []
    for key, name in names.items():
        good = probability_values(predictions["prob_good_"+key]).to_numpy()
        bad = probability_values(predictions["prob_bad_"+key]).to_numpy()
        require(np.allclose(good+bad, 1, rtol=0, atol=1e-7), "Good/Bad probabilities do not complement")
        require(np.array_equal((good > .5).astype(int).astype(str), predictions["pred_"+key].to_numpy()), "Prediction/threshold mismatch")
        measured = classification_scores(predictions.true_label, predictions["pred_"+key], bad, "0", "1")
        discrepancies = []
        for metric, column in METRICS.items():
            reported = float(claims.loc[name, column])
            if not np.isclose(measured[metric], reported, rtol=0, atol=1e-9):
                discrepancies.append({"metric": metric, "reported": reported, "recomputed": measured[metric]})
        results.append({"experiment": "E2", "run_id": "e2_"+key+"_received", "method": name, "split": "test",
                        "positive_label": "0", "seed": None, "feature_set": "saved_Fox8_TFIDF_10000",
                        "status": "metrics_verified_protocol_revision_needed", "training_rerun_this_audit": False,
                        "source": "E2/artifacts/e2_predictions.csv", **measured})
        checks.append({"item": key+"_metrics", "status": "report_mismatch" if discrepancies else "pass", "discrepancies": discrepancies})
    print("PASS: full Test IDs, labels, probabilities and threshold; reported metric discrepancies recorded", flush=True)

    # Execute only the owner's class definitions, never notebook training cells.
    class_nodes = []
    for cell in cells:
        class_nodes.extend(n for n in ast.parse(cell).body if isinstance(n, ast.ClassDef))
    namespace = {"nn": nn}
    exec(compile(ast.Module(body=class_nodes, type_ignores=[]), "E2_owner_class_definitions", "exec"), namespace)
    models = {}
    states = {}
    torch.set_num_threads(2)
    for key in names:
        state = torch.load(source(f"E2/artifacts/e2_{key}_model.pt"), map_location="cpu", weights_only=True)
        model = namespace["HondurasBotClassifier"](namespace["BehavioralEncoder"]())
        model.load_state_dict(state, strict=True)
        model.eval()
        models[key], states[key] = model, state
    vectorizer_path = source("Fox8 baseline bot detection/preprocessing/tfidf_vectorizer.joblib",
                             args.e1_dir/"preprocessing/tfidf_vectorizer.joblib")
    vectorizer = joblib.load(vectorizer_path)
    original_encoder = torch.load(source("Fox8 baseline bot detection/models/e1_behavioral_encoder.pt",
                                    args.e1_dir/"models/e1_behavioral_encoder.pt"), map_location="cpu", weights_only=True)
    changed_tensors = {}
    for key, state in states.items():
        encoder = {k.removeprefix("encoder.encoder."): v for k,v in state.items() if k.startswith("encoder.encoder.")}
        base = original_encoder["encoder_state_dict"]
        require(set(encoder) == set(base), "E1/E2 encoder architecture mismatch")
        changed_tensors[key] = {k: not torch.equal(encoder[k], base[k]) for k in base}

    # Exact raw-text recipe used in the delivered notebook, restricted to Test IDs.
    selected = set(test.index)
    texts = defaultdict(list)
    for path in [args.bad_jsonl, args.good_jsonl]:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                r = json.loads(line)
                account = str(r.get("userid") or r.get("account_id") or r.get("user_id") or r.get("id") or "").strip()
                if account not in selected or len(texts[account]) >= 30:
                    continue
                text = r.get("tweet_text") or r.get("text")
                if isinstance(text, str) and text.strip():
                    texts[account].append(text.strip())
    ids = predictions.account_id.to_list()
    documents = [" ".join(texts[a])[:5000] for a in ids]
    empty_text = sum(not s for s in documents)
    require(empty_text == 0, f"Test accounts without usable text: {empty_text}")
    x = vectorizer.transform(documents)
    replay = {key: [] for key in names}
    for start in range(0, len(ids), 128):
        batch = torch.from_numpy(x[start:start+128].toarray().astype(np.float32))
        with torch.no_grad():
            for key, model in models.items():
                replay[key].extend(torch.sigmoid(model(batch).squeeze(1)).numpy().tolist())
        if start % (128*100) == 0:
            print(f"Inference progress: {min(start+128,len(ids))}/{len(ids)}", flush=True)
    inference = {}
    for key in names:
        actual = np.asarray(replay[key])
        expected = probability_values(predictions["prob_good_"+key]).to_numpy()
        expected_labels = predictions["pred_"+key].astype(int).to_numpy()
        matches = (actual > .5).astype(int) == expected_labels
        inference[key] = {"n_accounts": len(ids), "prediction_agreement_fraction": float(matches.mean()),
                          "prediction_mismatches": int((~matches).sum()),
                          "max_absolute_probability_difference": float(np.abs(actual-expected).max()),
                          "scores_close_rtol_1e_4_atol_1e_5": bool(np.allclose(actual, expected, rtol=1e-4, atol=1e-5))}

    # Reproduce the owner's published LR function without changing the algorithm.
    features = pd.read_csv(args.features, dtype={"account_id": str})
    canonical_features = reference_table(features, "account_id")
    check_accounts(canonical_features, reference)
    train_df = features[features.split == "train"].copy()
    test_df = features[features.split == "test"].copy()
    captured_models, captured_scalers = [], []
    def model_factory(*a, **kw):
        model = LogisticRegression(*a, **kw); captured_models.append(model); return model
    def scaler_factory(*a, **kw):
        scaler = StandardScaler(*a, **kw); captured_scalers.append(scaler); return scaler
    ns = {"StandardScaler": scaler_factory, "LogisticRegression": model_factory,
          "precision_recall_fscore_support": precision_recall_fscore_support, "f1_score": f1_score,
          "average_precision_score": average_precision_score, "train_df": train_df, "test_df": test_df,
          "y_train": train_df.label, "y_test": test_df.label}
    function = next(n for cell in cells for n in ast.parse(cell).body if isinstance(n, ast.FunctionDef) and n.name == "train_and_eval_lr")
    exec(compile(ast.Module(body=[function], type_ignores=[]), "E2_owner_LR_function", "exec"), ns)
    full = ["tweet_count", "avg_text_length", "url_ratio", "hashtag_ratio", "mention_ratio", "retweet_ratio"]
    lr_reproduction, lr_predictions = [], pd.DataFrame({"account_id": test_df.account_id, "true_label": test_df.label})
    for columns, name, key in [(full, "LR (Full Features)", "full"), (full[1:], "LR (No tweet_count)", "no_tweet_count")]:
        measured_owner = ns["train_and_eval_lr"](columns, name)
        model, scaler = captured_models[-1], captured_scalers[-1]
        xtest = scaler.transform(test_df[columns])
        pred = model.predict(xtest); score = model.predict_proba(xtest)[:, list(model.classes_).index(0)]
        measured = classification_scores(test_df.label, pred, score, "0", "1")
        differences = {column: measured_owner[column]-float(claims.loc[name,column]) for column in METRICS.values()}
        matches = all(abs(v) < 1e-9 for v in differences.values())
        lr_reproduction.append({"method": name, "matches_reported_metrics": matches, "differences": differences, **measured})
        lr_predictions[key+"_pred"] = pred
        lr_predictions[key+"_prob_bad"] = score
        if key == "no_tweet_count":
            results.append({"experiment": "E2", "run_id": "e2_lr_no_tweet_count_reproduced", "method": name,
                "split": "test", "positive_label": "0", "seed": 42, "feature_set": "five_account_statistics_no_tweet_count",
                "status": "owner_function_reproduced" if matches else "owner_function_rerun_differs_from_claim",
                "training_rerun_this_audit": True, "source": "e2_handoff/lr_ablation_predictions.csv", **measured})

    scratch, transfer = results[0], results[1]
    deltas = [{"metric": metric, "scratch": scratch[metric], "transfer": transfer[metric],
               "transfer_minus_scratch": transfer[metric]-scratch[metric],
               "difference_percentage_points": 100*(transfer[metric]-scratch[metric])} for metric in METRICS]
    issues = [
        {"severity": "major", "item": "BCE class weighting", "finding": f"Targets keep Good=1, Bad=0; BCE pos_weight={config['pos_weight']} multiplies Good loss, not Bad loss.",
         "action": "Owner must justify this objective or correct class weighting/remap explicitly and rerun both paired models; evaluation does not modify the loss."},
        {"severity": "major", "item": "Validation/checkpoint selection", "finding": "val_df exists, but the delivered neural training function uses fixed 10 epochs without validation evaluation or validation-selected checkpoint.",
         "action": "Provide protocol-approved fixed-budget justification or validation checkpoint selection evidence; do not infer Test tuning solely from missing validation."},
        {"severity": "major", "item": "Neural seed metadata", "finding": "No torch.manual_seed/np.random.seed call in delivered notebook; LR random_state=42 is not a seed for neural initialization or DataLoader shuffle. Config omits neural seed.",
         "action": "Owner should provide the actual run seed/environment/log or rerun both models with recorded seeds."},
        {"severity": "report", "item": "Label semantics", "finding": "E2 report equates Bad with Bot and Good with Human; Honduras labels do not establish that equivalence.", "action": "Use Bad/Good consistently."},
        {"severity": "report", "item": "Overstated improvement", "finding": "Current Transfer-minus-Scratch deltas: " + "; ".join(f"{r['metric']} {r['difference_percentage_points']:+.6f} pp" for r in deltas) + ". One paired run is not general transfer evidence.", "action": "Report metric-specific trade-offs and percentage points; qualify conclusions by this run and protocol issues."},
        {"severity": "handoff", "item": "Artifact paths/export provenance", "finding": "Report lists evaluation/e2_predictions_e1_emb.csv, preprocessing/scaler_e1_emb.joblib and models/model_*_e2.pt, while actual Git files are artifacts/e2_predictions.csv and artifacts/e2_*_model.pt. Export cells are absent from the received notebook.",
         "action": "Point handoff documentation to actual files; provide export/config generation code and any additional Drive-only artifacts."}
    ]
    inference_pass = all(r["prediction_mismatches"] == 0 and r["scores_close_rtol_1e_4_atol_1e_5"] for r in inference.values())
    audit = {"status": "metrics_verified_inference_passed_protocol_issues" if inference_pass else "metrics_verified_inference_mismatch_protocol_issues",
             "checked_at_utc": datetime.now(timezone.utc).isoformat(), "snapshot_commit": tree["commit"],
             "repository": "https://github.com/luuquanghuy-ui/SNA-GROUP3", "handoff_author": "PThinhL (Le Phuc Thinh), verified from commit metadata; not inferred from folder number",
             "classification_results": results, "neural_metric_checks": checks, "neural_inference": inference,
             "neural_inference_passed": inference_pass, "lr_ablation_reproduction": lr_reproduction,
             "deltas": deltas, "protocol_issues": issues, "official_comparison_eligible": False,
             "encoder_tensors_changed_from_E1": changed_tensors, "embedding_CSV_used_as_neural_input": False,
             "input_recipe": "First 30 valid raw tweets per account, cap joined text at 5000 chars, transform saved E1 TF-IDF; no E1 CSV embeddings or feature scaler for neural input",
             "artifact_evidence": evidence,
             "local_input_sha256": {str(path): sha256(path) for path in [reference_path, args.features, args.bad_jsonl, args.good_jsonl]},
             "limitations": ["No neural retraining performed; only saved-checkpoint inference replayed on full Test.",
                             "LR ablation executed the owner's function unchanged, with input paths supplied locally.",
                             "Fine-tuning initialization is supported by notebook code; final changed weights alone cannot prove complete training history.",
                             "Role numbering may have changed. This audit assigns ownership by commit author and artifacts, not by local folder number."]}
    args.output.mkdir(parents=True, exist_ok=True)
    lr_predictions.to_csv(args.output/"lr_ablation_predictions.csv", index=False)
    pd.DataFrame(results).to_csv(args.output/"e2_verified_metrics.csv", index=False)
    pd.DataFrame(deltas).to_csv(args.output/"scratch_transfer_deltas.csv", index=False)
    pd.DataFrame(issues).to_csv(args.output/"e2_review_issues.csv", index=False)
    audit["lr_ablation_predictions_sha256"] = sha256(args.output/"lr_ablation_predictions.csv")
    (args.output/"e2_handoff_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    report = ["# Kiểm định bàn giao E2 mới của Thịnh", "",
              f"Nguồn: [{tree['commit'][:12]}]({audit['repository']}/tree/{tree['commit']}). Người bàn giao xác định theo author PThinhL và nội dung E2, không theo số folder.", "",
              "## Kết quả tính lại từ predictions", "",
              markdown_table(results, ["method", "n_accounts", "positive_precision", "positive_recall", "positive_f1", "macro_f1", "average_precision", "tp", "fp", "fn", "tn"]), "",
              "Các dòng dùng đúng 44.938 Test IDs và nhãn đối chiếu manifest (373 Bad, 44.565 Good). LR No tweet_count là kết quả tự tái lập bằng chính hàm của tác giả; xem đối chiếu trong JSON.", "",
              "## Delta Transfer − Scratch", "", markdown_table(deltas, ["metric", "scratch", "transfer", "transfer_minus_scratch", "difference_percentage_points"]), "",
              "Đọc chiều tăng/giảm từ delta của đúng checkpoint đang kiểm tra; không kế thừa kết luận của run cũ. Đây là điểm phần trăm, không phải phần trăm tương đối. Chưa nghiệm thu kết luận transfer cuối vì các vấn đề protocol dưới đây.", "",
              "## Đối chiếu bảng do tác giả bàn giao", "", "```json", json.dumps(checks, indent=2, ensure_ascii=False), "```", "",
              "## Kiểm tra checkpoint inference", "",
              markdown_table([{"method": name, **row} for name,row in inference.items()], ["method", "n_accounts", "prediction_agreement_fraction", "max_absolute_probability_difference", "scores_close_rtol_1e_4_atol_1e_5"]), "",
              "Chỉ chạy inference, không sửa loss hay huấn luyện lại neural models. Input thực tế là TF-IDF văn bản 10.000 chiều, không phải CSV embeddings E1 128 chiều mà Diễm gửi riêng.", "",
              "## Các điểm cần người phụ trách xử lý", "", markdown_table(issues, ["severity", "item", "finding", "action"]), "",
              "Bad/Good không đồng nghĩa Bot/Human. Không sửa artifact của Thịnh, không tự chọn cấu hình khác để làm đẹp kết quả. Cần đối chiếu với protocol nhóm sau khi có giải trình hoặc run sửa.", ""]
    (args.output/"E2_HANDOFF_REVIEW.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"status": audit["status"], "n_test": len(ids), "inference": inference,
                      "lr_reproduction_matches": [r["matches_reported_metrics"] for r in lr_reproduction],
                      "output": str(args.output.resolve())}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
