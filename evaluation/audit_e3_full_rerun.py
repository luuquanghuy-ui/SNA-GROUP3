import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
ORIGINAL = ROOT / "nguoi4" / "E3_outputs-20261018T031701Z-1-001" / "E3_outputs"
RERUN = Path(__file__).resolve().parent / "e3_rerun_from_tweets"


def normalized_bytes_equal(name):
    original = (ORIGINAL / name).read_bytes().replace(b"\r\n", b"\n")
    rerun = (RERUN / name).read_bytes().replace(b"\r\n", b"\n")
    return original == rerun


def first_row(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return next(csv.DictReader(stream))


def node_comparison():
    fields = ["account_id", "label", "split", "degree", "strength"]
    mismatches = {field: 0 for field in fields}
    community_id_mismatches = 0
    count = 0
    with (ORIGINAL / "e3_graph_nodes.csv").open(encoding="utf-8-sig", newline="") as left_stream:
        with (RERUN / "e3_graph_nodes.csv").open(encoding="utf-8-sig", newline="") as right_stream:
            left = csv.DictReader(left_stream)
            right = csv.DictReader(right_stream)
            for a, b in zip(left, right):
                count += 1
                for field in fields:
                    mismatches[field] += a[field] != b[field]
                community_id_mismatches += a["community"] != b["community"]
    return {"rows": count, "structural_field_mismatches": mismatches, "community_id_mismatches": community_id_mismatches}


def main():
    original_metrics = first_row(ORIGINAL / "e3_graph_metrics.csv")
    rerun_metrics = first_row(RERUN / "e3_graph_metrics.csv")
    result = {
        "scope": "Notebook E3 code cells 2-4 rerun from original normalized tweets, with local paths and display changed to print; no edits to graph logic.",
        "runtime": {"python": "3.11.9", "networkx": "3.6.1", "original_notebook_networkx": "3.7"},
        "normalized_csv_equal": {
            name: normalized_bytes_equal(name)
            for name in ["e3_data_quality.csv", "e3_graph_edges.csv", "e3_event_audit.csv", "e3_components.csv"]
        },
        "node_comparison": node_comparison(),
        "metrics": {
            key: {"original": value, "rerun": rerun_metrics[key], "equal": value == rerun_metrics[key]}
            for key, value in original_metrics.items()
        },
        "memory_status": json.loads((RERUN / "rerun_status.json").read_text(encoding="utf-8")),
    }
    (RERUN / "comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"normalized_csv_equal": result["normalized_csv_equal"], "node_comparison": result["node_comparison"], "metric_differences": {k: v for k, v in result["metrics"].items() if not v["equal"]}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
