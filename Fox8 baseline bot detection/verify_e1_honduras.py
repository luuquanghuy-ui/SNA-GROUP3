import json
import sqlite3
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib


# =========================================================
# 1. CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
E1_DIR = BASE_DIR / "e1_outputs"
OUTPUT_DIR = BASE_DIR / "e1_honduras_verification"

BAD_FILE = BASE_DIR / "honduras_bad_normalized.jsonl"
GOOD_FILE = BASE_DIR / "honduras_good_normalized.jsonl"

CHECKPOINT_PATH = E1_DIR / "e1_behavioral_encoder.pt"
VECTORIZER_PATH = E1_DIR / "tfidf_vectorizer.joblib"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATABASE_PATH = OUTPUT_DIR / "temporary_tweets.sqlite"
EMBEDDINGS_PATH = OUTPUT_DIR / "honduras_e1_embeddings.csv"
REPORT_PATH = OUTPUT_DIR / "verification_report.json"

MAX_TWEETS = 30
MAX_CHARS = 5000
BATCH_SIZE = 512
DEVICE = torch.device("cpu")

torch.manual_seed(42)
np.random.seed(42)


# =========================================================
# 2. CHECK REQUIRED FILES
# =========================================================

for path in [
    BAD_FILE,
    GOOD_FILE,
    CHECKPOINT_PATH,
    VECTORIZER_PATH,
]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")

print("All required files found.")


# =========================================================
# 3. LOAD E1 VECTORIZER AND CHECKPOINT
# =========================================================

vectorizer = joblib.load(VECTORIZER_PATH)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=False,
)

input_dim = int(checkpoint["input_dim"])
hidden_dim = int(checkpoint["hidden_dim"])
embedding_dim = int(checkpoint["embedding_dim"])

print("\nE1 checkpoint configuration")
print("Input dimension:", input_dim)
print("Hidden dimension:", hidden_dim)
print("Embedding dimension:", embedding_dim)
print("Feature type:", checkpoint.get("feature_type"))

print("\nTF-IDF configuration")
print("Vocabulary size:", len(vectorizer.vocabulary_))
print("N-gram range:", vectorizer.ngram_range)
print("Max features:", vectorizer.max_features)

if len(vectorizer.vocabulary_) != input_dim:
    raise ValueError(
        "INCOMPATIBLE E1 ARTIFACTS: "
        f"Vectorizer vocabulary={len(vectorizer.vocabulary_)}, "
        f"but checkpoint input_dim={input_dim}. "
        "Do not generate embeddings until the matching vectorizer "
        "and checkpoint are identified."
    )


# =========================================================
# 4. REBUILD THE EXACT CHECKPOINT ARCHITECTURE
# =========================================================

class BehavioralEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, embedding_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.encoder(x)


model = BehavioralEncoder(
    input_dim=input_dim,
    hidden_dim=hidden_dim,
    embedding_dim=embedding_dim,
)

model.encoder.load_state_dict(checkpoint["encoder_state_dict"])
model.to(DEVICE)
model.eval()

print("\nE1 encoder checkpoint loaded successfully.")


# =========================================================
# 5. STREAM HONDURAS JSONL INTO SQLITE
#    Keep at most 30 valid tweets per account.
# =========================================================

if DATABASE_PATH.exists():
    DATABASE_PATH.unlink()

conn = sqlite3.connect(DATABASE_PATH)

conn.execute("""
             CREATE TABLE tweets (
                                     account_id TEXT NOT NULL,
                                     tweet_idx INTEGER NOT NULL,
                                     text TEXT NOT NULL,
                                     PRIMARY KEY (account_id, tweet_idx)
             )
             """)

tweet_counts = defaultdict(int)
seen_accounts = set()
account_labels = {}
source_stats = {}
invalid_records = 0
valid_tweets = 0

files = [
    ("bad", BAD_FILE),
    ("good", GOOD_FILE),
]

for source, path in files:
    records = 0
    source_tweets = 0

    print(f"\nReading {path.name} ...")

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                invalid_records += 1
                continue

            account_id = record.get("account_id")

            if account_id is None:
                account_id = record.get("user_id")

            if account_id is None:
                invalid_records += 1
                continue

            account_id = str(account_id).strip()

            if not account_id:
                invalid_records += 1
                continue

            records += 1
            seen_accounts.add(account_id)

            if "label" in record:
                account_labels.setdefault(
                    account_id, record["label"]
                )

            text = record.get("text")

            if not isinstance(text, str) or not text.strip():
                continue

            text = text.strip()

            # Same account-level limit as E1:
            # use at most the first 30 valid tweets.
            if tweet_counts[account_id] >= MAX_TWEETS:
                continue

            tweet_idx = tweet_counts[account_id]

            conn.execute(
                """
                INSERT INTO tweets (account_id, tweet_idx, text)
                VALUES (?, ?, ?)
                """,
                (account_id, tweet_idx, text),
            )

            tweet_counts[account_id] += 1
            source_tweets += 1
            valid_tweets += 1

            if valid_tweets % 10000 == 0:
                conn.commit()
                print(f"  Valid tweets stored: {valid_tweets:,}")

    conn.commit()

    source_stats[source] = {
        "file": path.name,
        "records_read": records,
        "valid_tweets_stored": source_tweets,
    }

    print(
        f"  Records: {records:,} | "
        f"valid tweets stored: {source_tweets:,}"
    )

print("\nJSONL ingestion completed.")
print("Unique account IDs:", f"{len(seen_accounts):,}")
print("Valid tweets stored:", f"{valid_tweets:,}")
print("Invalid records:", f"{invalid_records:,}")


# =========================================================
# 6. AGGREGATE TEXT PER ACCOUNT
#    Then apply the saved E1 TF-IDF vectorizer.
# =========================================================

cursor = conn.execute("""
                      SELECT account_id, text
                      FROM tweets
                      ORDER BY account_id, tweet_idx
                      """)

def account_text_batches(cursor, batch_size= BATCH_SIZE):
    batch_ids = []
    batch_texts = []
    current_id = None
    current_texts = []

    def emit(account_id, texts):
        text = " ".join(texts)[:MAX_CHARS]
        return account_id, text

    for account_id, text in cursor:
        if current_id is None:
            current_id = account_id

        if account_id != current_id:
            account_id_out, text_out = emit(
                current_id, current_texts
            )

            if text_out.strip():
                batch_ids.append(account_id_out)
                batch_texts.append(text_out)

            if len(batch_ids) >= batch_size:
                yield batch_ids, batch_texts
                batch_ids, batch_texts = [], []

            current_id = account_id
            current_texts = []

        current_texts.append(text)

    if current_id is not None:
        account_id_out, text_out = emit(current_id, current_texts)

        if text_out.strip():
            batch_ids.append(account_id_out)
            batch_texts.append(text_out)

    if batch_ids:
        yield batch_ids, batch_texts


# =========================================================
# 7. SMOKE TEST: SHAPE, FINITE VALUES, STABILITY
# =========================================================

smoke_texts = [
    "hello world",
    "this is a test tweet",
    " ".join(["sample"] * 20),
]

smoke_x = vectorizer.transform(smoke_texts)

if smoke_x.shape[1] != input_dim:
    raise ValueError(
        f"Unexpected TF-IDF shape: {smoke_x.shape}"
    )

smoke_tensor = torch.from_numpy(
    smoke_x.toarray().astype(np.float32)
).to(DEVICE)

with torch.no_grad():
    smoke_embedding_1 = model(smoke_tensor).cpu().numpy()
    smoke_embedding_2 = model(smoke_tensor).cpu().numpy()

smoke_finite = bool(np.isfinite(smoke_embedding_1).all())

smoke_stable = bool(
    np.allclose(
        smoke_embedding_1,
        smoke_embedding_2,
        rtol=1e-6,
        atol=1e-7,
    )
)

print("\nSmoke test")
print("TF-IDF shape:", tuple(smoke_x.shape))
print("Embedding shape:", tuple(smoke_embedding_1.shape))
print("All values finite:", smoke_finite)
print("Repeated output stable:", smoke_stable)

if not smoke_finite or not smoke_stable:
    raise RuntimeError("E1 encoder smoke test failed.")


# =========================================================
# 8. ENCODE ALL HONDURAS ACCOUNTS
# =========================================================

if EMBEDDINGS_PATH.exists():
    EMBEDDINGS_PATH.unlink()

embedding_columns = [
    f"emb_{i:03d}" for i in range(embedding_dim)
]

success_count = 0
failed_count = 0
nonfinite_count = 0
first_write = True
error_samples = []

print("\nGenerating Honduras embeddings ...")

for account_ids, texts in account_text_batches(cursor):

    try:
        X = vectorizer.transform(texts)

        if X.shape != (len(texts), input_dim):
            raise ValueError(f"Unexpected TF-IDF shape: {X.shape}")

        X_dense = X.toarray().astype(np.float32)

        if not np.isfinite(X_dense).all():
            raise ValueError("TF-IDF contains NaN/Inf")

        tensor_x = torch.from_numpy(X_dense).to(DEVICE)

        with torch.no_grad():
            embeddings = model(tensor_x).cpu().numpy()

        if embeddings.shape != (
                len(account_ids), embedding_dim
        ):
            raise ValueError(
                f"Unexpected embedding shape: {embeddings.shape}"
            )

        finite_rows = np.isfinite(embeddings).all(axis=1)
        nonfinite_count += int((~finite_rows).sum())

        valid_ids = [
            account_ids[i]
            for i in range(len(account_ids))
            if finite_rows[i]
        ]

        valid_embeddings = embeddings[finite_rows]

        if len(valid_ids) > 0:
            output_df = pd.DataFrame(
                valid_embeddings,
                columns=embedding_columns,
            )

            output_df.insert(0, "account_id", valid_ids)
            output_df["label"] = [
                account_labels.get(account_id)
                for account_id in valid_ids
            ]

            output_df.to_csv(
                EMBEDDINGS_PATH,
                mode="w" if first_write else "a",
                header=first_write,
                index=False,
            )

            first_write = False
            success_count += len(valid_ids)

        failed_count += len(account_ids) - len(valid_ids)

    except Exception as exc:
        failed_count += len(account_ids)

        if len(error_samples) < 20:
            error_samples.append({
                "account_ids": account_ids[:5],
                "error": str(exc),
            })

    processed = success_count + failed_count

    if processed and processed % 10000 < BATCH_SIZE:
        print(
            f"Processed: {processed:,} | "
            f"success: {success_count:,} | "
            f"failed: {failed_count:,}"
        )


# =========================================================
# 9. FINAL REPORT
# =========================================================

accounts_with_text = int(
    conn.execute(
        "SELECT COUNT(DISTINCT account_id) FROM tweets"
    ).fetchone()[0]
)

conn.close()

report = {
    "experiment": "Verify E1 Fox8 encoder on Honduras",
    "checkpoint": str(CHECKPOINT_PATH),
    "vectorizer": str(VECTORIZER_PATH),
    "feature_type": checkpoint.get("feature_type"),
    "input_dim": input_dim,
    "hidden_dim": hidden_dim,
    "embedding_dim": embedding_dim,
    "max_tweets_per_account": MAX_TWEETS,
    "max_characters_per_account": MAX_CHARS,
    "lowercase": getattr(vectorizer, "lowercase", None),
    "ngram_range": list(vectorizer.ngram_range),
    "vocabulary_size": len(vectorizer.vocabulary_),
    "source_stats": source_stats,
    "unique_accounts_seen": len(seen_accounts),
    "accounts_with_valid_text": accounts_with_text,
    "valid_tweets_stored": valid_tweets,
    "invalid_records": invalid_records,
    "smoke_test": {
        "embedding_shape": list(smoke_embedding_1.shape),
        "finite": smoke_finite,
        "stable": smoke_stable,
    },
    "full_run": {
        "success_count": success_count,
        "failed_count": failed_count,
        "nonfinite_rows": nonfinite_count,
        "output_file": str(EMBEDDINGS_PATH),
    },
    "errors": error_samples,
}

with REPORT_PATH.open("w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print("\n========== VERIFICATION REPORT ==========")
print(json.dumps(report, indent=2, ensure_ascii=False))
print("\nReport saved to:", REPORT_PATH)
print("Embeddings saved to:", EMBEDDINGS_PATH)