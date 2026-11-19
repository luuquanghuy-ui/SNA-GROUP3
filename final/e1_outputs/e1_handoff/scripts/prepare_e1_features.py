import json
import ijson
import numpy as np
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parents[3]
DATA = BASE / "e1_outputs/e1_handoff/data"
CACHE = BASE / "e1_outputs/e1_handoff/cache"
PREP = BASE / "e1_outputs/e1_handoff/preprocessing"
PREP.mkdir(parents=True, exist_ok=True)

NUMERIC = [
    "followers_count", "friends_count", "statuses_count",
    "favourites_count", "listed_count"
]

BOOLEAN = [
    "verified", "protected", "geo_enabled",
    "default_profile", "default_profile_image",
    "has_extended_profile", "profile_use_background_image",
    "contributors_enabled", "is_translator",
    "is_translation_enabled"
]

def number(value):
    try:
        return max(0.0, float(str(value).strip()))
    except (ValueError, TypeError):
        return 0.0

def boolean(value):
    return float(str(value).strip().lower() == "true")

print("1. Loading cached embeddings...")

metadata = json.loads(
    (CACHE / "account_metadata.json").read_text()
)
profile_text = np.load(
    CACHE / "account_profile_embeddings.npy", mmap_mode="r"
)
tweet = np.load(
    CACHE / "account_tweet_embeddings.npy", mmap_mode="r"
)

neighbor_ids = json.loads(
    (CACHE / "neighbor_profile_ids.json").read_text()
)
neighbor_index = {x: i for i, x in enumerate(neighbor_ids)}
neighbor_vectors = np.load(
    CACHE / "neighbor_profile_embeddings.npy", mmap_mode="r"
)

N = len(metadata)
assert profile_text.shape == (N, 384)
assert tweet.shape == (N, 384)

print("2. Extracting Profile numeric and boolean features...")

raw_features = np.zeros((N, 15), dtype=np.float32)
labels = np.zeros(N, dtype=np.int64)
split_names = []
account_ids = []

global_idx = 0

for split in ["train", "dev", "test"]:
    with (BASE / f"{split}.json").open("rb") as f:
        for account in ijson.items(f, "item"):
            if str(account["ID"]) != metadata[global_idx]["id"]:
                raise RuntimeError("Account order mismatch")

            p = account.get("profile") or {}

            for j, field in enumerate(NUMERIC):
                raw_features[global_idx, j] = np.log1p(
                    number(p.get(field))
                )

            for j, field in enumerate(BOOLEAN):
                raw_features[global_idx, 5 + j] = boolean(
                    p.get(field)
                )

            labels[global_idx] = int(account["label"])
            split_names.append(split)
            account_ids.append(str(account["ID"]))
            global_idx += 1

print("Accounts processed:", global_idx)

print("3. Fit scaler on TRAIN only...")

train_mask = np.array(split_names) == "train"
scaler = StandardScaler()
scaler.fit(raw_features[train_mask, :5])

profile_features = raw_features.copy()
profile_features[:, :5] = scaler.transform(
    raw_features[:, :5]
)

joblib.dump(scaler, PREP / "profile_numeric_scaler.joblib")

print("4. Aggregating Following and Follower embeddings...")

neighbor_features = np.zeros((N, 768), dtype=np.float32)
global_idx = 0

for split in ["train", "dev", "test"]:
    with (DATA / "neighbor_edges.jsonl").open(encoding="utf-8") as f:
        for line in f:
            edge = json.loads(line)
            if edge["split"] != split:
                continue

            if edge["account_id"] != account_ids[global_idx]:
                raise RuntimeError("Neighbor edge order mismatch")

            following = list(dict.fromkeys(edge["following"]))[:5]
            follower = list(dict.fromkeys(edge["follower"]))[:5]

            def average(ids):
                indexes = [
                    neighbor_index[x] for x in ids
                    if x in neighbor_index
                ]
                if not indexes:
                    return np.zeros(384, dtype=np.float32)
                return np.asarray(
                    neighbor_vectors[indexes]
                ).mean(axis=0)

            neighbor_features[global_idx, :384] = average(following)
            neighbor_features[global_idx, 384:] = average(follower)
            global_idx += 1

print("5. Saving feature arrays...")

np.save(CACHE / "profile_numeric_boolean.npy", profile_features)
np.save(CACHE / "neighbor_aggregated_embeddings.npy", neighbor_features)
np.save(CACHE / "labels.npy", labels)
np.save(CACHE / "split_names.npy", np.array(split_names))
np.save(CACHE / "account_ids.npy", np.array(account_ids))

config = {
    "numeric_fields": NUMERIC,
    "boolean_fields": BOOLEAN,
    "numeric_transform": "log1p then StandardScaler fitted on train only",
    "neighbor_limit": 5,
    "neighbor_aggregation": "mean pooling, following and follower separately",
    "tweet_embedding_dim": 384,
    "profile_text_embedding_dim": 384,
    "profile_numeric_boolean_dim": 15,
    "neighbor_embedding_dim": 768,
    "label_mapping": {"Human": 0, "Bot": 1}
}

(PREP / "feature_config.json").write_text(
    json.dumps(config, indent=2)
)

print("DONE")
print("Tweet:", tweet.shape)
print("Profile text:", profile_text.shape)
print("Profile numeric/boolean:", profile_features.shape)
print("Neighbor:", neighbor_features.shape)
print("Labels:", labels.shape)
print("Train/Dev/Test:", [
    int(np.sum(np.array(split_names) == s))
    for s in ["train", "dev", "test"]
])
