import argparse
import csv
from collections import Counter
from pathlib import Path


FIELDS = ["experiment", "run_id", "model", "split", "seed", "positive_label", "metric", "value", "n_accounts", "input", "feature_set", "notes"]


def average_precision(labels, scores):
    pairs = sorted(zip(scores, labels), key=lambda pair: pair[0], reverse=True)
    positives = sum(labels)
    if positives == 0:
        raise ValueError("Không có mẫu dương để tính AP")
    retrieved = 0
    true_retrieved = 0
    area = 0.0
    index = 0
    while index < len(pairs):
        score = pairs[index][0]
        group_positives = 0
        group_size = 0
        while index < len(pairs) and pairs[index][0] == score:
            group_positives += pairs[index][1]
            group_size += 1
            index += 1
        retrieved += group_size
        true_retrieved += group_positives
        area += (group_positives / positives) * (true_retrieved / retrieved)
    if 1 < area < 1 + 1e-12:
        return 1.0
    return area


def binary_scores(truth, prediction, score):
    counts = Counter(zip(truth, prediction))
    tp = counts[(1, 1)]
    fp = counts[(0, 1)]
    tn = counts[(0, 0)]
    fn = counts[(1, 0)]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    negative_f1 = 2 * tn / (2 * tn + fp + fn) if 2 * tn + fp + fn else 0.0
    result = {
        "positive_precision": precision,
        "positive_recall": recall,
        "positive_f1": f1,
        "macro_f1": (f1 + negative_f1) / 2,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "n_positive": tp + fn,
        "n_negative": tn + fp,
    }
    if score is not None:
        result["average_precision"] = average_precision(truth, score)
    return result


def read_predictions(path, args):
    truth = []
    prediction = []
    score = []
    ids = set()
    labels_by_id = {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {args.id_column, args.truth_column, args.pred_column, args.split_column}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Thiếu cột: {sorted(required - set(reader.fieldnames or []))}")
        has_score = args.score_column in (reader.fieldnames or [])
        for row in reader:
            if row[args.split_column] != args.split:
                continue
            account_id = row[args.id_column].strip()
            if not account_id or account_id in ids:
                raise ValueError("Thiếu hoặc trùng account ID trong split được chọn")
            ids.add(account_id)
            if row[args.truth_column] not in {args.positive_label, args.negative_label}:
                raise ValueError("Nhãn thật nằm ngoài hai nhãn đã khai báo")
            if row[args.pred_column] not in {args.positive_label, args.negative_label}:
                raise ValueError("Nhãn dự đoán nằm ngoài hai nhãn đã khai báo")
            truth.append(int(row[args.truth_column] == args.positive_label))
            labels_by_id[account_id] = row[args.truth_column]
            prediction.append(int(row[args.pred_column] == args.positive_label))
            if has_score:
                value = float(row[args.score_column])
                if not 0 <= value <= 1:
                    raise ValueError("Điểm lớp dương phải nằm trong [0, 1]")
                score.append(value)
    if not truth:
        raise ValueError("Không có dự đoán trong split được chọn")
    return truth, prediction, score if has_score else None, labels_by_id


def verify_reference(prediction_labels, reference_path, split):
    expected = {}
    with reference_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"account_id", "label", "split"}.issubset(reader.fieldnames or []):
            raise ValueError("Bảng reference thiếu account_id, label hoặc split")
        for row in reader:
            if row["split"] == split:
                account_id = row["account_id"]
                if account_id in expected:
                    raise ValueError("Reference chứa account ID trùng")
                expected[account_id] = row["label"]
    if prediction_labels.keys() != expected.keys():
        missing = len(expected.keys() - prediction_labels.keys())
        extra = len(prediction_labels.keys() - expected.keys())
        raise ValueError(f"Tập account không khớp reference: thiếu {missing}, thừa {extra}")
    mismatches = sum(prediction_labels[account_id] != label for account_id, label in expected.items())
    if mismatches:
        raise ValueError(f"Nhãn thật không khớp reference ở {mismatches} tài khoản")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "baseline_results.csv")
    parser.add_argument("--experiment", choices=["E1", "E2"], required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--feature-set", required=True)
    parser.add_argument("--positive-label", required=True)
    parser.add_argument("--negative-label", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--id-column", default="account_id")
    parser.add_argument("--truth-column", default="y_true")
    parser.add_argument("--pred-column", default="y_pred")
    parser.add_argument("--score-column", default="y_score_positive")
    parser.add_argument("--split-column", default="split")
    parser.add_argument("--notes", default="")
    parser.add_argument("--reference-features", type=Path)
    args = parser.parse_args()
    if args.positive_label == args.negative_label:
        raise ValueError("Nhãn dương và âm phải khác nhau")
    truth, prediction, score, prediction_labels = read_predictions(args.input, args)
    if args.experiment == "E2" and args.reference_features is None:
        args.reference_features = Path(__file__).resolve().parent.parent / "nguoi 1" / "honduras_features.csv"
    if args.reference_features is not None:
        verify_reference(prediction_labels, args.reference_features, args.split)
    metrics = binary_scores(truth, prediction, score)
    output = args.output
    existing = []
    if output.exists():
        with output.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != FIELDS:
                raise ValueError("Schema baseline_results.csv không khớp")
            existing = list(reader)
    existing = [row for row in existing if not (row["experiment"] == args.experiment and row["run_id"] == args.run_id)]
    for metric, value in metrics.items():
        existing.append({
            "experiment": args.experiment,
            "run_id": args.run_id,
            "model": args.model,
            "split": args.split,
            "seed": args.seed,
            "positive_label": args.positive_label,
            "metric": metric,
            "value": value,
            "n_accounts": len(truth),
            "input": str(args.input.resolve()),
            "feature_set": args.feature_set,
            "notes": args.notes,
        })
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(existing)
    print(f"Wrote {len(metrics)} metrics for {args.run_id}; n={len(truth)}; output={output}")


if __name__ == "__main__":
    main()
