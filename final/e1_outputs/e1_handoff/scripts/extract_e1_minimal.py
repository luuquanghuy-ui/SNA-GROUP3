import json
import ijson
from pathlib import Path

BASE = Path(__file__).resolve().parents[3]
OUT = BASE / "e1_outputs" / "e1_handoff" / "data"
OUT.mkdir(parents=True, exist_ok=True)

TEXT_FIELDS = ["name", "screen_name", "description", "location"]
NUMERIC_FIELDS = [
    "followers_count", "friends_count", "statuses_count",
    "favourites_count", "listed_count"
]
BOOL_FIELDS = [
    "verified", "protected", "geo_enabled",
    "default_profile", "default_profile_image",
    "has_extended_profile", "profile_use_background_image",
    "contributors_enabled", "is_translator",
    "is_translation_enabled"
]

def get_id(value):
    if isinstance(value, dict):
        value = value.get("ID", value.get("id", ""))
    return str(value) if value is not None else ""

def extract_tweets(raw):
    result = []

    def collect(obj):
        if len(result) >= 5:
            return
        if isinstance(obj, str):
            if obj.strip():
                result.append(obj.strip())
        elif isinstance(obj, dict):
            for key in ("full_text", "text"):
                value = obj.get(key)
                if isinstance(value, str) and value.strip():
                    result.append(value.strip())
                    return
            for value in obj.values():
                if len(result) >= 5:
                    break
                if isinstance(value, (dict, list)):
                    collect(value)
        elif isinstance(obj, list):
            for item in obj:
                if len(result) >= 5:
                    break
                collect(item)

    collect(raw)
    return result[:5]

print("1. Reading split files...")
split_ids = set()
neighbor_ids = set()

edge_path = OUT / "neighbor_edges.jsonl"

with edge_path.open("w", encoding="utf-8") as edge_out:
    for split in ("train", "dev", "test"):
        with (BASE / f"{split}.json").open(encoding="utf-8") as f:
            accounts = json.load(f)

        for account in accounts:
            account_id = get_id(account.get("ID"))
            split_ids.add(account_id)

            neighbor = account.get("neighbor") or {}
            following = [
                get_id(x) for x in (neighbor.get("following") or [])
            ]
            follower = [
                get_id(x) for x in (neighbor.get("follower") or [])
            ]

            following = [x for x in following if x]
            follower = [x for x in follower if x]

            neighbor_ids.update(following)
            neighbor_ids.update(follower)

            edge_out.write(json.dumps({
                "account_id": account_id,
                "split": split,
                "following": following,
                "follower": follower
            }, ensure_ascii=False) + "\n")

        print(f"{split}: {len(accounts)} accounts")

external_ids = neighbor_ids - split_ids
print("External neighbor IDs:", len(external_ids))

print("2. Streaming support.json...")
support_path = OUT / "support_neighbors.jsonl"
found = set()

with (BASE / "support.json").open("rb") as source, \
     support_path.open("w", encoding="utf-8") as target:

    for record in ijson.items(source, "item"):
        account_id = get_id(record.get("ID"))

        if account_id not in external_ids:
            continue

        profile = record.get("profile") or {}

        compact = {
            "ID": account_id,
            "profile": {
                key: profile.get(key)
                for key in TEXT_FIELDS + NUMERIC_FIELDS + BOOL_FIELDS
            },
            "tweets": extract_tweets(record.get("tweet"))
        }

        target.write(json.dumps(compact, ensure_ascii=False) + "\n")
        found.add(account_id)

summary = {
    "external_neighbor_ids": len(external_ids),
    "support_accounts_found": len(found),
    "support_accounts_missing": len(external_ids - found),
    "neighbor_edges_file": edge_path.name,
    "support_features_file": support_path.name,
    "tweet_limit_per_account": 5,
    "profile_fields": TEXT_FIELDS + NUMERIC_FIELDS + BOOL_FIELDS
}

with (OUT / "extraction_summary.json").open("w") as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
print("DONE")
