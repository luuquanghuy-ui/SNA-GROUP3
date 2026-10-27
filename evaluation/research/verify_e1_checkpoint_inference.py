"""Run the existing E1 inference verifier with explicit paths and pinned artifacts."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path

from run_research_evaluation import require, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--e1-dir", type=Path, required=True)
    parser.add_argument("--processed", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("results") / "e1_checkpoint_inference_audit.json")
    args = parser.parse_args()
    require(args.raw.is_file(), "Raw Fox8 NDJSON missing")
    tree = json.loads((args.snapshot / "repository_tree.json").read_text(encoding="utf-8"))
    blobs = {item["path"]: item for item in tree["tree"] if item["type"] == "blob"}
    paths = ["models/logisticregression.joblib", "models/randomforest.joblib", "models/e1_mlp_full_checkpoint.pt",
             "models/e1_behavioral_encoder.pt", "preprocessing/tfidf_vectorizer.joblib", "evaluation/e1_predictions.csv"]
    hashes = {}
    for relative in paths:
        path = args.e1_dir / relative
        data = path.read_bytes()
        require(hashlib.sha1(f"blob {len(data)}\0".encode()+data).hexdigest() == blobs["Fox8 baseline bot detection/"+relative]["sha"],
                f"Artifact differs from snapshot: {relative}")
        hashes[relative] = hashlib.sha256(data).hexdigest()
    verifier_path = Path(__file__).resolve().parents[1] / "verify_e1_inference.py"
    spec = importlib.util.spec_from_file_location("existing_e1_inference_verifier", verifier_path)
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    verifier.PROCESSED = args.processed
    verifier.RAW = args.raw
    verifier.MODELS = args.e1_dir / "models"
    verifier.VECTORIZER = args.e1_dir / "preprocessing/tfidf_vectorizer.joblib"
    verifier.PREDICTIONS = args.e1_dir / "evaluation/e1_predictions.csv"
    verifier.OUTPUT = args.output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    verifier.torch.set_num_threads(2)
    verifier.main()
    audit = json.loads(args.output.read_text(encoding="utf-8"))
    encoder = verifier.torch.load(args.e1_dir / "models/e1_behavioral_encoder.pt", map_location="cpu", weights_only=True)
    full = verifier.torch.load(args.e1_dir / "models/e1_mlp_full_checkpoint.pt", map_location="cpu", weights_only=True)
    full_encoder = {key.removeprefix("encoder."): value for key, value in full["model_state_dict"].items() if key.startswith("encoder.")}
    saved_encoder = encoder["encoder_state_dict"]
    require(set(full_encoder) == set(saved_encoder), "Encoder/full-model state keys differ")
    identical = all(verifier.torch.equal(full_encoder[key], value) for key, value in saved_encoder.items())
    require(identical, "Encoder weights differ from classifier checkpoint")
    audit.update({"status": "inference_predictions_reproduced_not_retraining",
                  "checked_at_utc": datetime.now(timezone.utc).isoformat(), "snapshot_commit": tree["commit"],
                  "encoder_weights_equal_full_model": identical, "artifact_sha256": hashes,
                  "raw_fox8_sha256": sha256(args.raw),
                  "reference_split_sha256": sha256(args.processed / "fox8_account_splits.csv"),
                  "verifier_code_sha256": sha256(verifier_path),
                  "limitations": ["Saved model inference only, not training from scratch or a clean environment rerun.",
                                  "Prediction agreement does not remove source-label confounding.",
                                  "Float probability differences are recorded separately from exact class agreement."]})
    args.output.write_text(json.dumps(audit, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"status": audit["status"], "n_test": audit["test_accounts"],
                      "encoder_weights_equal_full_model": identical, "output": str(args.output.resolve())}))


if __name__ == "__main__":
    main()
