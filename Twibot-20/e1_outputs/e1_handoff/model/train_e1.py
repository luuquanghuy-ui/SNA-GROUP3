import sys
import json
import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)

ROOT = Path("e1_outputs/e1_handoff")
MODEL_DIR = ROOT / "model"
OUTPUT_DIR = ROOT / "checkpoints"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(MODEL_DIR))

from e1_multibranch import E1MultiBranch
from e1_losses import combined_loss
from prepare_dataloader import train_loader, dev_loader

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cpu")
model = E1MultiBranch().to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=2e-4,
    weight_decay=1e-4
)

EPOCHS = 10
PATIENCE = 2
best_f1 = -1
patience_count = 0


def evaluate(loader):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for tweet, profile, struct, neighbor, labels in loader:
            logits, _ = model(
                tweet.to(device),
                profile.to(device),
                struct.to(device),
                neighbor.to(device)
            )
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    return {
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(
            all_labels, all_preds, zero_division=0
        ),
        "recall": recall_score(
            all_labels, all_preds, zero_division=0
        ),
        "macro_f1": f1_score(
            all_labels, all_preds,
            average="macro", zero_division=0
        ),
        "confusion_matrix": confusion_matrix(
            all_labels, all_preds
        ).tolist()
    }


history = []

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0
    total_cls = 0
    total_supcon = 0

    for step, batch in enumerate(train_loader, 1):
        tweet, profile, struct, neighbor, labels = batch

        optimizer.zero_grad()

        logits, z = model(
            tweet.to(device),
            profile.to(device),
            struct.to(device),
            neighbor.to(device)
        )

        loss, cls_loss, supcon_loss = combined_loss(
            logits, z, labels.to(device), alpha=0.2
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1.0
        )
        optimizer.step()

        total_loss += loss.item()
        total_cls += cls_loss.item()
        total_supcon += supcon_loss.item()

        if step % 300 == 0:
            print(
                f"Epoch {epoch} | Step {step}/{len(train_loader)} "
                f"| Loss {loss.item():.4f}",
                flush=True
            )

    metrics = evaluate(dev_loader)

    result = {
        "epoch": epoch,
        "train_loss": total_loss / len(train_loader),
        "classification_loss": total_cls / len(train_loader),
        "supcon_loss": total_supcon / len(train_loader),
        **metrics
    }
    history.append(result)

    print(
        f"\nEpoch {epoch}/{EPOCHS} "
        f"| Train loss: {result['train_loss']:.4f} "
        f"| Dev Acc: {metrics['accuracy']:.4f} "
        f"| Dev Macro-F1: {metrics['macro_f1']:.4f}",
        flush=True
    )

    if metrics["macro_f1"] > best_f1:
        best_f1 = metrics["macro_f1"]
        patience_count = 0

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_dev_macro_f1": best_f1,
            "config": {
                "alpha_supcon": 0.2,
                "learning_rate": 2e-4,
                "batch_size": 4,
                "seed": 42
            }
        }, OUTPUT_DIR / "best_model.pt")

        print("Saved best checkpoint.", flush=True)
    else:
        patience_count += 1

    with open(OUTPUT_DIR / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    if patience_count >= PATIENCE:
        print("Early stopping.", flush=True)
        break

print("\n===== TRAINING FINISHED =====")
print("Best Dev Macro-F1:", round(best_f1, 4))
print("Checkpoint:", OUTPUT_DIR / "best_model.pt")
