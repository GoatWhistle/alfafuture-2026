"""Competition entrypoint and explicit composition of the baseline pipeline."""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv

from ml_harness.adapters.data.task import CsvTaskData
from ml_harness.adapters.execution.runner import SubprocessRunner
from ml_harness.adapters.storage.submission import AtomicSubmissionWriter, IndependentEvaluator
from ml_harness.application.use_cases.baseline import run_baseline
from ml_harness.application.use_cases.prepare_task import prepare_task
from ml_harness.domain.models import ContractError
from ml_harness.entrypoints.research import research_after_baseline


def main() -> int:
    parser = argparse.ArgumentParser(description="Autonomous tabular ML research harness")
    parser.add_argument("task", nargs="?", default=Path("task.md"), type=Path)
    parser.add_argument("--backend", choices=("catboost", "lightgbm"), default="catboost")
    parser.add_argument("--threads", type=int, choices=range(1, 9), default=8)
    parser.add_argument("--worker-timeout", type=float, default=1800)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--research-steps", type=int, default=12)
    parser.add_argument("--llm-token-limit", type=int, default=32000)
    args = parser.parse_args()
    if args.research_steps < 1 or args.llm_token_limit < 1:
        parser.error("Research steps and token limit must be positive")
    load_dotenv(Path.cwd() / ".env", override=False)
    work_dir = (args.run_dir or Path("runs") / uuid4().hex).resolve()
    try:
        task = prepare_task(str(args.task), CsvTaskData())
        work_dir.mkdir(parents=True, exist_ok=True)
        (work_dir / "prepared.json").write_text(
            json.dumps(asdict(task), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        result = run_baseline(
            task,
            SubprocessRunner(),
            IndependentEvaluator(),
            AtomicSubmissionWriter(),
            work_dir=str(work_dir),
            backend=args.backend,
            threads=args.threads,
            timeout_seconds=args.worker_timeout,
        )
        details = asdict(result)
        if not args.baseline_only:
            details["research"] = research_after_baseline(
                task,
                result,
                work_dir / "research",
                threads=args.threads,
                timeout_seconds=args.worker_timeout,
                max_steps=args.research_steps,
                token_limit=args.llm_token_limit,
            )
        summary = json.dumps(details, ensure_ascii=False, indent=2)
        (work_dir / "summary.json").write_text(summary, encoding="utf-8")
        print(summary)
    except (ContractError, OSError, ValueError) as exc:
        print(f"Task failed: {exc}", file=sys.stderr)
        return 2
    return 0
