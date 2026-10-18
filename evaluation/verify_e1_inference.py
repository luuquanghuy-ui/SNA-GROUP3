import csv
import json
import platform
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import sklearn
import torch
from torch import nn


ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "nguoi0" / "processed-20261018T032211Z-1-001" / "processed"
RAW = ROOT / "nguoi0" / "fox8_23_dataset.ndjson"
E1 = ROOT / "nguoi2"
MODELS = E1 / "models-20261018T031152Z-1-001" / "models"
VECTORIZER = E1 / "preprocessing-20261018T031232Z-1-001" / "preprocessing" / "tfidf_vectorizer.joblib"
PREDICTIONS = E1 / "evaluation-20261018T031128Z-1-001" / "evaluation" / "e1_predictions.csv"
OUTPUT = Path(__file__).resolve().parent / "e1_inference_audit.json"


class BehavioralMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, embedding_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(embedding_dim, 1)

    def forward(self, x):
        return self.classifier(self.encoder(x)).squeeze(1)


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        yield from csv.DictReader(stream)


def main():
    test = {row["user_id"]: row["label"] for row in read_csv(PROCESSED / "fox8_account_splits.csv") if row["split"] == "test"}
    texts = defaultdict(list)
    with RAW.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            uid = str(record.get("user_id", ""))
            if uid not in test:
                continue
            for tweet in (record.get("user_tweets") or [])[:30]:
                if isinstance(tweet, dict):
                    value = tweet.get("text")
                    if isinstance(value, str) and value.strip():
                        texts[uid].append(value.strip())
    account_ids = sorted(test)
    documents = [" ".join(texts[uid])[:5000] for uid in account_ids]
    if any(not document.strip() for document in documents):
        raise ValueError("At least one E1 test account has no text")
    vectorizer = joblib.load(VECTORIZER)
    x_test = vectorizer.transform(documents)
    expected = defaultdict(dict)
    for row in read_csv(PREDICTIONS):
        expected[row["model"]][row["user_id"]] = row
    models = {
        "LogisticRegression": joblib.load(MODELS / "logisticregression.joblib"),
        "RandomForest": joblib.load(MODELS / "randomforest.joblib"),
    }
    result = {}
    for name, model in models.items():
        predicted = model.predict(x_test)
        probabilities = model.predict_proba(x_test)[:, 1]
        label_matches = [int(predicted[index]) == int(expected[name][uid]["y_pred"]) for index, uid in enumerate(account_ids)]
        score_differences = [abs(float(probabilities[index]) - float(expected[name][uid]["bot_probability"])) for index, uid in enumerate(account_ids)]
        result[name] = {"n_accounts": len(account_ids), "prediction_agreement_fraction": sum(label_matches) / len(label_matches), "max_absolute_probability_difference": max(score_differences)}
    checkpoint = torch.load(MODELS / "e1_mlp_full_checkpoint.pt", map_location="cpu", weights_only=True)
    mlp = BehavioralMLP(checkpoint["input_dim"], checkpoint["hidden_dim"], checkpoint["embedding_dim"])
    mlp.load_state_dict(checkpoint["model_state_dict"])
    mlp.eval()
    with torch.no_grad():
        mlp_scores = torch.sigmoid(mlp(torch.tensor(x_test.toarray(), dtype=torch.float32))).numpy()
    mlp_pred = (mlp_scores >= 0.5).astype(int)
    label_matches = [int(mlp_pred[index]) == int(expected["MLP"][uid]["y_pred"]) for index, uid in enumerate(account_ids)]
    score_differences = [abs(float(mlp_scores[index]) - float(expected["MLP"][uid]["bot_probability"])) for index, uid in enumerate(account_ids)]
    result["MLP"] = {"n_accounts": len(account_ids), "prediction_agreement_fraction": sum(label_matches) / len(label_matches), "max_absolute_probability_difference": max(score_differences)}
    audit = {
        "scope": "Independent inference on Fox8 test accounts from raw NDJSON using delivered TF-IDF vectorizer and model checkpoints; training was not rerun.",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "torch": torch.__version__,
        "test_accounts": len(account_ids),
        "models": result,
    }
    OUTPUT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=True, indent=2))
    if any(item["prediction_agreement_fraction"] != 1.0 for item in result.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
