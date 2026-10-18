import argparse
import csv
import hashlib
import json
import platform
from collections import Counter
from pathlib import Path


EXPECTED_COLUMNS = [
    "account_id",
    "label",
    "split",
    "tweet_count",
    "avg_text_length",
    "url_ratio",
    "hashtag_ratio",
    "mention_ratio",
    "retweet_ratio",
]
RATIO_COLUMNS = ["url_ratio", "hashtag_ratio", "mention_ratio", "retweet_ratio"]


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_summary(path):
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return {row["metric"]: int(row["value"]) for row in csv.DictReader(stream)}


def audit(features_path, summary_path, splits_path):
    counts = Counter()
    seen = set()
    feature_assignments = {}
    duplicate_ids = 0
    missing_values = Counter()
    invalid_rows = Counter()
    with features_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != EXPECTED_COLUMNS:
            raise ValueError(f"Sai schema: {reader.fieldnames}")
        for row in reader:
            account_id = row["account_id"]
            if account_id in seen:
                duplicate_ids += 1
            seen.add(account_id)
            feature_assignments[account_id] = (row["label"], row["split"])
            for column in EXPECTED_COLUMNS:
                if row[column] is None or not row[column].strip():
                    missing_values[column] += 1
            label = row["label"]
            split = row["split"]
            if label not in {"0", "1"} or split not in {"train", "validation", "test"}:
                invalid_rows["label_or_split"] += 1
                continue
            counts[(split, label)] += 1
            try:
                if int(row["tweet_count"]) <= 0 or float(row["avg_text_length"]) < 0:
                    invalid_rows["count_or_length"] += 1
                if any(not 0 <= float(row[column]) <= 1 for column in RATIO_COLUMNS):
                    invalid_rows["ratio"] += 1
            except ValueError:
                invalid_rows["numeric_parse"] += 1
    summary = load_summary(summary_path)
    split_assignments = {}
    split_duplicate_ids = 0
    with splits_path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            account_id = row["user_id"]
            if account_id in split_assignments:
                split_duplicate_ids += 1
            split_assignments[account_id] = (row["label"], row["split"])
    total = sum(counts.values())
    actual_bad = sum(value for (split, label), value in counts.items() if label == "0")
    actual_good = sum(value for (split, label), value in counts.items() if label == "1")
    checks = {
        "row_count_matches_summary": total == summary["feature_table_accounts"],
        "bad_count_matches_summary": actual_bad == summary["bad_accounts_label_0"],
        "good_count_matches_summary": actual_good == summary["good_accounts_label_1"],
        "account_ids_unique": duplicate_ids == 0,
        "no_missing_values": not missing_values,
        "valid_feature_values": not invalid_rows,
        "matches_processed_account_splits": feature_assignments == split_assignments and split_duplicate_ids == 0,
    }
    return {
        "audit_runtime": {"python": platform.python_version(), "platform": platform.platform()},
        "source_files": {
            "features": str(features_path.resolve()),
            "features_sha256": file_hash(features_path),
            "quality_summary": str(summary_path.resolve()),
            "quality_summary_sha256": file_hash(summary_path),
            "processed_account_splits": str(splits_path.resolve()),
            "processed_account_splits_sha256": file_hash(splits_path),
        },
        "rows": total,
        "unique_account_ids": len(seen),
        "bad_accounts_label_0": actual_bad,
        "good_accounts_label_1": actual_good,
        "split_by_label": {f"{split}_{label}": counts[(split, label)] for split in ("train", "validation", "test") for label in ("0", "1")},
        "duplicate_account_ids": duplicate_ids,
        "duplicate_split_account_ids": split_duplicate_ids,
        "missing_values": dict(missing_values),
        "invalid_rows": dict(invalid_rows),
        "checks": checks,
        "all_checks_pass": all(checks.values()),
        "scope": "Kiểm tra bảng feature bàn giao với manifest split trong nguoi0; không chạy lại phần trích xuất feature từ JSONL gốc.",
    }


def main():
    folder = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=folder.parent / "nguoi 1" / "honduras_features.csv")
    parser.add_argument("--summary", type=Path, default=folder.parent / "nguoi 1" / "data_quality_summary.csv")
    parser.add_argument("--splits", type=Path, default=folder.parent / "nguoi0" / "processed-20261018T032211Z-1-001" / "processed" / "honduras_account_splits.csv")
    parser.add_argument("--output", type=Path, default=folder / "honduras_features_audit.json")
    args = parser.parse_args()
    result = audit(args.features, args.summary, args.splits)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True, indent=2))
    if not result["all_checks_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
