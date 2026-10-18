import contextlib
import hashlib
import json
import platform
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "nguoi 1" / "honduras_data_validation.ipynb"
PROCESSED = ROOT / "nguoi0" / "processed-20261018T032211Z-1-001" / "processed"
DELIVERED = ROOT / "nguoi 1"
OUTPUT = Path(__file__).resolve().parent / "honduras_preprocessing_rerun_audit.json"
SKIP_CELLS = {0, 1, 2, 5, 89}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="p1_rerun_") as temp_name:
        temp = Path(temp_name)
        namespace = {
            "__name__": "__main__",
            "display": lambda value: None,
            "BASE_DIR": str(PROCESSED.parent),
            "PROCESSED_DIR": str(PROCESSED),
            "OUTPUT_DIR": str(temp),
        }
        executed = []
        with (temp / "notebook_stdout.txt").open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log):
            for index, cell in enumerate(notebook["cells"]):
                if cell["cell_type"] != "code" or index in SKIP_CELLS:
                    continue
                source = "".join(cell.get("source", []))
                if not source.strip():
                    continue
                if index == 90:
                    source = source.replace('OUTPUT_DIR = "/content/drive/MyDrive/MXH_output"', f"OUTPUT_DIR = {str(temp)!r}")
                try:
                    exec(compile(source, f"{NOTEBOOK.name}:cell_{index}", "exec"), namespace)
                except Exception as error:
                    raise RuntimeError(f"Notebook cell {index} failed: {error}") from error
                executed.append(index)
        files = ["honduras_features.csv", "data_quality_summary.csv", "honduras_feature_profile.csv"]
        comparisons = {
            name: {
                "rerun_sha256": sha256(temp / name),
                "delivered_sha256": sha256(DELIVERED / name),
            }
            for name in files
        }
        for item in comparisons.values():
            item["sha256_matches"] = item["rerun_sha256"] == item["delivered_sha256"]
        rerun_features = pd.read_csv(temp / "honduras_features.csv", dtype={"account_id": str})
        delivered_features = pd.read_csv(DELIVERED / "honduras_features.csv", dtype={"account_id": str})
        rerun_features = rerun_features.sort_values("account_id").reset_index(drop=True)
        delivered_features = delivered_features.sort_values("account_id").reset_index(drop=True)
        feature_columns = list(delivered_features.columns)
        feature_comparison = {
            "same_shape": rerun_features.shape == delivered_features.shape,
            "same_columns": list(rerun_features.columns) == feature_columns,
            "same_account_ids": rerun_features["account_id"].equals(delivered_features["account_id"]),
        }
        if all(feature_comparison.values()):
            feature_comparison["categorical_or_count_mismatches"] = {
                name: int((rerun_features[name] != delivered_features[name]).sum())
                for name in ["label", "split", "tweet_count"]
            }
            feature_comparison["numeric_max_abs_difference"] = {
                name: float(np.max(np.abs(rerun_features[name].to_numpy() - delivered_features[name].to_numpy())))
                for name in ["avg_text_length", "url_ratio", "hashtag_ratio", "mention_ratio", "retweet_ratio"]
            }
        rerun_quality = dict(zip(pd.read_csv(temp / "data_quality_summary.csv")["metric"], pd.read_csv(temp / "data_quality_summary.csv")["value"]))
        delivered_quality = dict(zip(pd.read_csv(DELIVERED / "data_quality_summary.csv")["metric"], pd.read_csv(DELIVERED / "data_quality_summary.csv")["value"]))
        quality_mismatches = {key: [rerun_quality.get(key), delivered_quality.get(key)] for key in set(rerun_quality) | set(delivered_quality) if rerun_quality.get(key) != delivered_quality.get(key)}
        rerun_profile = pd.read_csv(temp / "honduras_feature_profile.csv").sort_values(["split", "label"]).reset_index(drop=True)
        delivered_profile = pd.read_csv(DELIVERED / "honduras_feature_profile.csv").sort_values(["split", "label"]).reset_index(drop=True)
        profile_comparison = {
            "same_shape": rerun_profile.shape == delivered_profile.shape,
            "same_columns": list(rerun_profile.columns) == list(delivered_profile.columns),
            "same_split_and_label": rerun_profile[["split", "label"]].equals(delivered_profile[["split", "label"]]),
            "numeric_values_match": bool(np.allclose(rerun_profile.select_dtypes(include="number").to_numpy(), delivered_profile.select_dtypes(include="number").to_numpy(), rtol=0, atol=1e-12)),
        }
        semantic_match = (
            all(feature_comparison[key] for key in ["same_shape", "same_columns", "same_account_ids"])
            and not any(feature_comparison["categorical_or_count_mismatches"].values())
            and not any(feature_comparison["numeric_max_abs_difference"].values())
            and not quality_mismatches
            and all(profile_comparison.values())
        )
        audit = {
            "scope": "Original P1 notebook code cells executed in order with local processed input/output paths; Colab mount, pip install and processed-folder write-test cells skipped.",
            "python": platform.python_version(),
            "executed_code_cells": executed,
            "skipped_code_cells": sorted(SKIP_CELLS),
            "comparisons": comparisons,
            "feature_semantic_comparison": feature_comparison,
            "quality_summary_mismatches": quality_mismatches,
            "feature_profile_semantic_comparison": profile_comparison,
            "all_csv_outputs_match": all(item["sha256_matches"] for item in comparisons.values()),
            "all_semantic_outputs_match": semantic_match,
        }
        OUTPUT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(audit, ensure_ascii=True, indent=2))
        if not audit["all_semantic_outputs_match"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
