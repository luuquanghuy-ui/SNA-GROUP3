import json
import ijson
import numpy as np
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModel

BASE = Path(__file__).resolve().parents[3]
CACHE = BASE / "e1_outputs/e1_handoff/cache"
CACHE.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
BATCH_SIZE = 8
MAX_LENGTH = 128
MAX_TWEETS = 5

PROFILE_FILE = CACHE / "account_profile_embeddings.npy"
TWEET_FILE = CACHE / "account_tweet_embeddings.npy"
META_FILE = CACHE / "account_metadata.json"
PROGRESS_FILE = CACHE / "account_encode_progress.json"

TEXT_FIELDS = ["name", "screen_name", "description", "location"]
SPLITS = ["train", "dev", "test"]


def collect_tweets(obj, result):
    if isinstance(obj, str):
        if obj.strip():
            result.append(obj.strip())

    elif isinstance(obj, dict):
        found = False
        for key in ("full_text", "text"):
            value = obj.get(key)
            if isinstance(value, str) and value.strip():
                result.append(value.strip())
                found = True

        if not found:
            for value in obj.values():
                if isinstance(value, (dict, list)):
                    collect_tweets(value, result)

    elif isinstance(obj, list):
        for item in obj:
            collect_tweets(item, result)


def get_tweets(raw):
    texts = []
    collect_tweets(raw, texts)

    if len(texts) > MAX_TWEETS:
        indices = np.linspace(
            0, len(texts) - 1, MAX_TWEETS, dtype=int
        )
        texts = [texts[i] for i in indices]

    return texts


def get_profile_text(profile):
    return " ".join(
        str(profile.get(k) or "").strip()
        for k in TEXT_FIELDS
    ).strip() or "[empty profile]"


print("1. Preparing account metadata...")

metadata = []

for split in SPLITS:
    with (BASE / f"{split}.json").open("rb") as f:
        for account in ijson.items(f, "item"):
            metadata.append({
                "id": str(account["ID"]),
                "split": split,
                "label": account.get("label")
            })

if len({x["id"] for x in metadata}) != len(metadata):
    raise ValueError("Duplicate account IDs found across splits.")

if META_FILE.exists():
    old = json.loads(META_FILE.read_text())
    if old != metadata:
        raise RuntimeError("Metadata changed; refusing to reuse cache.")
else:
    META_FILE.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2)
    )

N = len(metadata)
print("Total accounts:", N)

print("2. Loading MiniLM...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).cpu()
model.eval()

def open_cache(path):
    if path.exists():
        arr = np.load(path, mmap_mode="r+")
        if arr.shape != (N, 384):
            raise RuntimeError(f"Wrong cache shape: {path}")
        return arr

    arr = np.lib.format.open_memmap(
        path, mode="w+", dtype="float32", shape=(N, 384)
    )
    arr[:] = 0
    arr.flush()
    return arr

profile_emb = open_cache(PROFILE_FILE)
tweet_emb = open_cache(TWEET_FILE)

if PROGRESS_FILE.exists():
    last_index = json.loads(PROGRESS_FILE.read_text())["last_index"]
else:
    last_index = -1

print("Resume after account index:", last_index)

def encode_batch(batch):
    texts = []
    profile_positions = []
    tweet_positions = []

    for item in batch:
        profile_positions.append(len(texts))
        texts.append(item["profile"])

        positions = []
        for tweet in item["tweets"]:
            positions.append(len(texts))
            texts.append(tweet)

        tweet_positions.append(positions)

    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

    with torch.inference_mode():
        output = model(**inputs).last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1)
        vectors = (
            (output * mask).sum(dim=1)
            / mask.sum(dim=1).clamp(min=1)
        ).cpu().numpy().astype("float32")

    for j, item in enumerate(batch):
        idx = item["index"]
        profile_emb[idx] = vectors[profile_positions[j]]

        positions = tweet_positions[j]
        if positions:
            tweet_emb[idx] = vectors[positions].mean(axis=0)
        else:
            tweet_emb[idx] = 0


print("3. Encoding Tweet + Profile...")

batch = []
processed = 0
global_index = 0
no_tweets = 0

for split in SPLITS:
    print("Reading:", split)

    with (BASE / f"{split}.json").open("rb") as f:
        for account in ijson.items(f, "item"):

            if global_index <= last_index:
                global_index += 1
                continue

            expected = metadata[global_index]["id"]
            if str(account["ID"]) != expected:
                raise RuntimeError("Account order mismatch.")

            tweets = get_tweets(account.get("tweet"))
            if not tweets:
                no_tweets += 1

            batch.append({
                "index": global_index,
                "profile": get_profile_text(
                    account.get("profile") or {}
                ),
                "tweets": tweets
            })

            global_index += 1

            if len(batch) >= BATCH_SIZE:
                encode_batch(batch)
                profile_emb.flush()
                tweet_emb.flush()

                processed += len(batch)
                last_index = batch[-1]["index"]

                PROGRESS_FILE.write_text(json.dumps({
                    "last_index": last_index,
                    "processed_this_run": processed
                }))

                if processed % 400 == 0:
                    print("Encoded this run:", processed)

                batch.clear()

if batch:
    encode_batch(batch)
    profile_emb.flush()
    tweet_emb.flush()

    processed += len(batch)
    last_index = batch[-1]["index"]

    PROGRESS_FILE.write_text(json.dumps({
        "last_index": last_index,
        "processed_this_run": processed
    }))

print("4. Final verification...")

profile_emb.flush()
tweet_emb.flush()

print("Total accounts:", N)
print("Encoded this run:", processed)
print("Profile shape:", profile_emb.shape)
print("Tweet shape:", tweet_emb.shape)
print("Accounts without tweets in this run:", no_tweets)
print("Profile cache:", PROFILE_FILE)
print("Tweet cache:", TWEET_FILE)
print("DONE")
