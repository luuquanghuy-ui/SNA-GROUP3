import contextlib
import json
import platform
import tempfile
from collections import defaultdict
from pathlib import Path

import sklearn


ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "nguoi2" / "E1_Fox8_Baseline.ipynb"
PROCESSED = ROOT / "nguoi0" / "processed-20261018T032211Z-1-001" / "processed"
RAW = ROOT / "nguoi0" / "fox8_23_dataset.ndjson"
DELIVERED = ROOT / "nguoi2" / "evaluation-20261018T031128Z-1-001" / "evaluation"
OUTPUT = Path(__file__).resolve().parent / "e1_training_rerun_audit.json"
LOG = Path(__file__).resolve().parent / "e1_training_rerun_log.txt"


def main():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="e1_retrain_") as temp_name:
        temp = Path(temp_name)
        namespace = {"__name__": "__main__", "display": lambda value: None}

        def run_cell(index):
            source = "".join(notebook["cells"][index]["source"])
            if index == 2:
                source = source.replace('OUTPUT_DIR = Path("e1_outputs")', f"OUTPUT_DIR = Path({str(temp)!r})")
            exec(compile(source, f"{NOTEBOOK.name}:cell_{index}", "exec"), namespace)

        with LOG.open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log):
            run_cell(2)
            namespace["MANIFEST_PATH"] = PROCESSED / "fox8_account_manifest.csv"
            namespace["SPLIT_PATH"] = PROCESSED / "fox8_account_splits.csv"
            namespace["OUTPUT_DIR"] = temp
            run_cell(3)
            run_cell(4)
            data = namespace["data"]
            selected = set(data["user_id"])
            texts = defaultdict(list)
            with RAW.open("r", encoding="utf-8") as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    uid = str(record.get("user_id", ""))
                    if uid not in selected:
                        continue
                    for tweet in (record.get("user_tweets") or [])[:30]:
                        if isinstance(tweet, dict):
                            value = tweet.get("text")
                            if isinstance(value, str) and value.strip():
                                texts[uid].append(value.strip())
            data["text"] = data["user_id"].map(lambda uid: " ".join(texts.get(uid, []))[:5000])
            data["tweet_count_used"] = data["user_id"].map(lambda uid: len(texts.get(uid, [])))
            namespace["data"] = data[data["text"].str.strip().ne("")].copy()
            for index in (7, 8, 9, 10, 11):
                run_cell(index)
        pd = namespace["pd"]
        rerun_metrics = pd.read_csv(temp / "e1_metrics.csv").set_index("model")
        delivered_metrics = pd.read_csv(DELIVERED / "e1_metrics.csv").set_index("model")
        rerun_preds = pd.read_csv(temp / "e1_predictions.csv", dtype={"user_id": str})
        delivered_preds = pd.read_csv(DELIVERED / "e1_predictions.csv", dtype={"user_id": str})
        models = {}
        for model in delivered_metrics.index:
            local = rerun_preds.loc[rerun_preds["model"] == model].set_index("user_id")
            original = delivered_preds.loc[delivered_preds["model"] == model].set_index("user_id")
            local = local.reindex(original.index)
            if local["y_true"].isna().any() or not (local["y_true"] == original["y_true"]).all():
                raise ValueError(f"E1 {model} test accounts or true labels differ")
            models[model] = {
                "reported_bot_f1": float(delivered_metrics.loc[model, "f1_bot"]),
                "rerun_bot_f1": float(rerun_metrics.loc[model, "f1_bot"]),
                "prediction_agreement_fraction": float((local["y_pred"] == original["y_pred"]).mean()),
                "max_absolute_probability_difference": float((local["bot_probability"] - original["bot_probability"]).abs().max()),
            }
        audit = {
            "scope": "E1 notebook training/evaluation cells rerun from Fox8 raw NDJSON and Person 0 manifest/split; Colab ZIP read replaced by equivalent direct NDJSON read; output written to a temporary folder.",
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "torch": namespace["torch"].__version__,
            "models": models,
            "all_bot_f1_match": all(abs(item["reported_bot_f1"] - item["rerun_bot_f1"]) < 1e-12 for item in models.values()),
            "all_predictions_match": all(item["prediction_agreement_fraction"] == 1.0 for item in models.values()),
        }
        OUTPUT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(audit, ensure_ascii=True, indent=2))
        if not audit["all_bot_f1_match"] or not audit["all_predictions_match"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
