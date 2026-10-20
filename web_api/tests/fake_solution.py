"""Stand-in for solution.py: writes the same artifact shapes without any ML or LLM."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("task", type=Path)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--search-mode")
    for name in (
        "seed",
        "threads",
        "time-budget",
        "worker-timeout",
        "memory-limit-mb",
        "llm-token-limit",
    ):
        parser.add_argument("--" + name)
    args = parser.parse_args()
    harness, inputs = args.run_dir, args.task.parent
    pause = float(os.environ.get("FAKE_PAUSE", "0.2"))
    submission = inputs / "submission.csv"
    dump(
        harness / "prepared.json",
        {"spec": {"kind": "binary", "metric": "roc_auc", "submission_path": str(submission)}},
    )
    submission.write_text("id,target\n1,0.5\n", encoding="utf-8")
    time.sleep(pause)
    dump(harness / "baseline.json", {"score": {"value": 0.8, "metric": "roc_auc"}})
    events = harness / "research" / "memory" / "events.jsonl"
    events.parent.mkdir(parents=True)
    if os.environ.get("FAKE_MODE") == "hang":
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
        (harness / "child.pid").write_text(str(child.pid))
        time.sleep(600)
    finished = {
        "event_id": "experiment:e1:finished",
        "kind": "experiment_finished",
        "payload": {
            "id": "e1",
            "parent_id": "baseline",
            "hypothesis": "Частотное кодирование region",
            "status": "succeeded",
            "backend": "catboost",
            "parameters": {"depth": 6},
            "full_score": {"value": 0.83, "metric": "roc_auc"},
            "level": "full",
            "elapsed_seconds": 12.5,
        },
    }
    with events.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(finished) + "\n")
        stream.write('{"event_id": "torn", "kind": "experiment_fin')  # unfinished last line
    time.sleep(pause)
    submission.write_text("id,target\n1,0.9\n", encoding="utf-8")
    if os.environ.get("FAKE_MODE") == "fail":
        return 2
    dump(harness / "summary.json", {"research": {"status": "completed", "tokens_used": 100}})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
