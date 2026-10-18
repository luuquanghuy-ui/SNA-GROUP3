import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil


ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "nguoi4" / "E3_Coordination_Graph.ipynb"
DATA = ROOT / "nguoi0" / "processed-20261018T032211Z-1-001" / "processed"
OUT = Path(__file__).resolve().parent / "e3_rerun_from_tweets"


def worker():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    scope = {"__name__": "__main__"}
    for index in (2, 3, 4):
        source = "".join(notebook["cells"][index]["source"])
        if index == 2:
            source = source.replace(
                "BASE = Path('/content/drive/MyDrive/Colab Notebooks/MXH_4_output')",
                f"BASE = Path({str(ROOT)!r})",
            )
            source = source.replace("DATA = BASE / 'MXH/processed'", f"DATA = Path({str(DATA)!r})")
            source = source.replace("OUT = BASE / 'E3_outputs'", f"OUT = Path({str(OUT)!r})")
            source = source.replace("from IPython.display import display", "display = print")
        print(f"START_CELL_{index}", flush=True)
        exec(compile(source, f"E3_notebook_cell_{index}", "exec"), scope)
        print(f"DONE_CELL_{index}", flush=True)


def supervise():
    OUT.mkdir(parents=True, exist_ok=True)
    log_path = OUT / "rerun_log.txt"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [sys.executable, "-X", "utf8", "-u", str(Path(__file__).resolve()), "--worker"],
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        child = psutil.Process(process.pid)
        peak_rss = 0
        lowest_available = psutil.virtual_memory().available
        stopped_for_memory = False
        while process.poll() is None:
            try:
                peak_rss = max(peak_rss, child.memory_info().rss)
                lowest_available = min(lowest_available, psutil.virtual_memory().available)
            except psutil.NoSuchProcess:
                break
            if lowest_available < 900_000_000:
                process.kill()
                stopped_for_memory = True
                break
            time.sleep(2)
        process.wait()
    status = {
        "exit_code": process.returncode,
        "stopped_for_memory": stopped_for_memory,
        "peak_rss_gb": peak_rss / 1e9,
        "lowest_available_gb": lowest_available / 1e9,
        "log": str(log_path),
        "output": str(OUT),
    }
    (OUT / "rerun_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False))
    if process.returncode:
        raise SystemExit(process.returncode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    worker() if args.worker else supervise()
