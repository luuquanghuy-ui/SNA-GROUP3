import sys
import json
import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

ROOT = Path("e1_outputs/e1_handoff")
MODEL_DIR = ROOT / "model"
OUTPUT_DIR = ROOT / "evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(MODEL_DIR))

from e1_multibranch import E1MultiBranch
from prepare_dataloader import test_loader

device = torch.device("cpu")
model = E1MultiBranch().to(device)

checkpoint = torch.load(
    ROOT / "checkpoints" / "best_model.pt",
    map_location=device
)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

all_preds = []
all_labels = []
all_probs = []

with torch.no_grad():
    for tweet, profile, struct, neighbor, labels in test_loader:
        logits, _ = model(tweet, profile, struct, neighbor)
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)

        all_preds.extend(preds.numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.numpy())

metrics = {
    "best_checkpoint_epoch": checkpoint["epoch"],
    "best_dev_macro_f1": checkpoint["best_dev_macro_f1"],
    "test_accuracy": accuracy_score(all_labels, all_preds),
    "test_precision_macro": precision_score(
        all_labels, all_preds, average="macro", zero_division=0
    ),
    "test_recall_macro": recall_score(
        all_labels, all_preds, average="macro", zero_division=0
    ),
    "test_macro_f1": f1_score(
        all_labels, all_preds, average="macro", zero_division=0
    ),
    "confusion_matrix": confusion_matrix(
        all_labels, all_preds
    ).tolist(),
    "classification_report": classification_report(
        all_labels, all_preds,
        target_names=["Human", "Bot"],
        output_dict=True,
        zero_division=0
    )
}

account_ids = np.load(ROOT / "cache" / "account_ids.npy")
split_names = np.load(ROOT / "cache" / "split_names.npy").astype(str)
test_ids = account_ids[split_names == "test"]

np.savez(
    OUTPUT_DIR / "test_predictions.npz",
    account_ids=test_ids,
    labels=np.array(all_labels),
    predictions=np.array(all_preds),
    probabilities=np.array(all_probs)
)

with open(OUTPUT_DIR / "test_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("===== TEST RESULTS =====")
print("Checkpoint epoch:", checkpoint["epoch"])
print("Accuracy:", round(metrics["test_accuracy"], 4))
print("Precision:", round(metrics["test_precision_macro"], 4))
print("Recall:", round(metrics["test_recall_macro"], 4))
print("Macro-F1:", round(metrics["test_macro_f1"], 4))
print("Confusion Matrix:")
print(np.array(metrics["confusion_matrix"]))
print("\nClassification Report:")
print(classification_report(
    all_labels, all_preds,
    target_names=["Human", "Bot"],
    zero_division=0
))
print("Saved test metrics and predictions:", OUTPUT_DIR)
