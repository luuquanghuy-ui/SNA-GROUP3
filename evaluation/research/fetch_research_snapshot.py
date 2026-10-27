"""Download only registered evaluation inputs from a pinned public GitHub commit."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import urllib.parse
import urllib.request


def read(url):
    request = urllib.request.Request(url, headers={"User-Agent": "SNA-research-evaluation"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("research_manifest.json"))
    parser.add_argument("--include-encoder", action="store_true")
    parser.add_argument("--include-e1-models", action="store_true", help="Also fetch LR/RF/full MLP for saved-model inference")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    parsed = urllib.parse.urlparse(manifest["repository"])
    if parsed.scheme != "https" or parsed.netloc != "github.com" or not re.fullmatch(r"/[\w.-]+/[\w.-]+", parsed.path):
        raise ValueError("Expected a standard public GitHub repository URL")
    commit = manifest["snapshot_commit"]
    if not re.fullmatch("[0-9a-f]{40}", commit):
        raise ValueError("Use a pinned 40-character commit SHA")
    repository = parsed.path.lstrip("/")
    root = args.destination.resolve()
    root.mkdir(parents=True, exist_ok=True)
    tree = json.loads(read(f"https://api.github.com/repos/{repository}/git/trees/{commit}?recursive=1"))
    if tree.get("truncated"):
        raise ValueError("Truncated repository inventory")
    tree.update(repository=repository, commit=commit)
    entries = {r["path"]: r for r in tree["tree"] if r["type"] == "blob"}
    paths = {"data/validation/honduras_features.parquet", "Fox8 baseline bot detection/E1_VALIDATION.md"}
    for run in manifest["classification_runs"] + manifest["graph_runs"]:
        paths.update(run[k] for k in ["predictions", "reported_metrics", "config", "code", "nodes", "edges", "communities", "source_report"] if k in run)
    if args.include_encoder or args.include_e1_models:
        paths.update("Fox8 baseline bot detection/"+p for p in ["models/e1_behavioral_encoder.pt", "preprocessing/tfidf_vectorizer.joblib"])
    if args.include_e1_models:
        paths.update("Fox8 baseline bot detection/models/"+name for name in
                     ["logisticregression.joblib", "randomforest.joblib", "e1_mlp_full_checkpoint.pt"])
    downloaded = []
    for relative in sorted(paths):
        entry = entries[relative]
        target = (root / relative).resolve()
        if root not in target.parents:
            raise ValueError("Source path escapes snapshot directory")
        data = None
        if target.exists():
            candidate = target.read_bytes()
            if hashlib.sha1(f"blob {len(candidate)}\0".encode()+candidate).hexdigest() == entry["sha"]:
                data = candidate
        url = f"https://raw.githubusercontent.com/{repository}/{commit}/"+urllib.parse.quote(relative, safe="/")
        if data is None:
            for attempt in range(3):
                try:
                    data = read(url)
                    if len(data) != entry["size"] or hashlib.sha1(f"blob {len(data)}\0".encode()+data).hexdigest() != entry["sha"]:
                        raise ValueError(f"Size/blob hash mismatch: {relative}")
                    break
                except Exception:
                    if attempt == 2:
                        raise
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        downloaded.append({"path": relative, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "url": url})
        print(f"Verified {relative}", flush=True)
    (root / "repository_tree.json").write_text(json.dumps(tree, indent=2), encoding="utf-8")
    (root / "download_manifest.json").write_text(json.dumps({"repository": repository, "commit": commit, "files": downloaded}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
