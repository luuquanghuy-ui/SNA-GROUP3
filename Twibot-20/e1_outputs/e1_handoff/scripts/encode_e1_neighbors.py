import json
import numpy as np
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModel

BASE = Path(__file__).resolve().parents[3]
DATA = BASE / "e1_outputs/e1_handoff/data"
CACHE = BASE / "e1_outputs/e1_handoff/cache"
CACHE.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
BATCH_SIZE = 8
MAX_LENGTH = 96
DEVICE = "cpu"

EDGE_FILE = DATA / "neighbor_edges.jsonl"
SUPPORT_FILE = DATA / "support_neighbors.jsonl"

IDS_FILE = CACHE / "neighbor_profile_ids.json"
EMB_FILE = CACHE / "neighbor_profile_embeddings.npy"
PROGRESS_FILE = CACHE / "neighbor_encode_progress.json"

TEXT_FIELDS = ["name", "screen_name", "description", "location"]

print("1. Collecting selected neighbor IDs...")

selected = set()

with EDGE_FILE.open(encoding="utf-8") as f:
    for line in f:
        x = json.loads(line)
        selected.update(list(dict.fromkeys(x["following"]))[:5])
        selected.update(list(dict.fromkeys(x["follower"]))[:5])

ids = sorted(selected)

if IDS_FILE.exists():
    saved_ids = json.loads(IDS_FILE.read_text())
    if saved_ids != ids:
        raise RuntimeError("Neighbor ID list changed. Stop to avoid cache mismatch.")
else:
    IDS_FILE.write_text(json.dumps(ids))

id_to_idx = {account_id: i for i, account_id in enumerate(ids)}
print("Unique IDs:", len(ids))

print("2. Loading MiniLM...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

if EMB_FILE.exists():
    embeddings = np.load(EMB_FILE, mmap_mode="r+")
    if embeddings.shape != (len(ids), 384):
        raise RuntimeError("Embedding cache has wrong shape.")
else:
    embeddings = np.lib.format.open_memmap(
        EMB_FILE, mode="w+", dtype="float32",
        shape=(len(ids), 384)
    )
    embeddings[:] = 0

if PROGRESS_FILE.exists():
    progress = json.loads(PROGRESS_FILE.read_text())
    last_line = progress["last_line"]
else:
    last_line = -1

print("Resume after source line:", last_line)

def profile_text(profile):
    return " ".join(
        str(profile.get(k) or "").strip()
        for k in TEXT_FIELDS
    ).strip() or "[empty profile]"

def encode_batch(batch):
    texts = [item[1] for item in batch]
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

    with torch.no_grad():
        output = model(**inputs)
        hidden = output.last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

    return pooled.cpu().numpy().astype("float32")

print("3. Encoding selected neighbor profiles...")

batch = []
processed = 0
found = 0

with SUPPORT_FILE.open(encoding="utf-8") as f:
    for line_no, line in enumerate(f):
        if line_no <= last_line:
            continue

        record = json.loads(line)
        account_id = str(record["ID"])

        if account_id not in id_to_idx:
            continue

        idx = id_to_idx[account_id]
        batch.append((idx, profile_text(record.get("profile") or ""), line_no))
        found += 1

        if len(batch) >= BATCH_SIZE:
            vectors = encode_batch(batch)

            for item, vector in zip(batch, vectors):
                embeddings[item[0]] = vector

            embeddings.flush()
            last_line = batch[-1][2]
            processed += len(batch)

            PROGRESS_FILE.write_text(json.dumps({
                "last_line": last_line,
                "processed": processed
            }))

            if processed % 800 == 0:
                print(f"Encoded: {processed} | Last line: {last_line}")

            batch.clear()

if batch:
    vectors = encode_batch(batch)
    for item, vector in zip(batch, vectors):
        embeddings[item[0]] = vector

    embeddings.flush()
    last_line = batch[-1][2]
    processed += len(batch)

    PROGRESS_FILE.write_text(json.dumps({
        "last_line": last_line,
        "processed": processed
    }))

print("4. Checking cache...")

missing = [
    account_id for i, account_id in enumerate(ids)
    if not np.any(embeddings[i])
]

print("Encoded this run:", processed)
print("Missing/zero embeddings:", len(missing))
print("Embedding shape:", embeddings.shape)
print("Saved:", EMB_FILE)
print("DONE")
