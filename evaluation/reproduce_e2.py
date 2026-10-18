import csv
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent.parent
FEATURES_PATH = ROOT / "nguoi 1" / "honduras_features.csv"
PREDICTIONS_PATH = ROOT / "nguoi3" / "evaluation-20261018T031434Z-1-001" / "evaluation" / "e2_predictions.csv"
OUTPUT_PATH = Path(__file__).resolve().parent / "e2_local_rerun_audit.json"
FEATURE_COLUMNS = ["tweet_count", "avg_text_length", "url_ratio", "hashtag_ratio", "mention_ratio", "retweet_ratio"]


def main():
    frame = pd.read_csv(FEATURES_PATH, dtype={"account_id": str})
    delivered = pd.read_csv(PREDICTIONS_PATH, dtype={"account_id": str})
    train = frame.loc[frame["split"] == "train"]
    results = {}
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train[FEATURE_COLUMNS])
    y_train = train["label"]
    models = {
        "DummyClassifier": (DummyClassifier(strategy="prior"), "dummy_pred", "dummy_prob_bad"),
        "LogisticRegression": (LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000), "lr_pred", "lr_prob_bad"),
    }
    for name, (model, pred_column, score_column) in models.items():
        model.fit(x_train, y_train)
        by_split = {}
        for split in ("validation", "test"):
            subset = frame.loc[frame["split"] == split]
            x_subset = scaler.transform(subset[FEATURE_COLUMNS])
            observed = delivered.loc[delivered["split"] == split].set_index("account_id")
            observed = observed.reindex(subset["account_id"])
            if observed[pred_column].isna().any():
                raise ValueError(f"Missing delivered predictions for {name} {split}")
            predicted = model.predict(x_subset)
            scores = model.predict_proba(x_subset)[:, list(model.classes_).index(0)]
            label_match = np.asarray(predicted == observed[pred_column].to_numpy()).mean()
            max_score_difference = float(np.max(np.abs(scores - observed[score_column].to_numpy())))
            by_split[split] = {
                "n_accounts": len(subset),
                "prediction_agreement_fraction": float(label_match),
                "max_absolute_probability_difference": max_score_difference,
                "all_predictions_match": bool(label_match == 1),
            }
        results[name] = by_split
    audit = {
        "scope": "Independent local rerun of E2 notebook's StandardScaler, DummyClassifier and LogisticRegression on Person 1 feature table; not a fresh virtual environment.",
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "features": FEATURE_COLUMNS,
        "results": results,
    }
    OUTPUT_PATH.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
