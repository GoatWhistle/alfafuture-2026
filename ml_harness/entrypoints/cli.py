"""Competition entrypoint and explicit composition of the baseline pipeline."""

import argparse
import json
import sys
from dataclasses import asdict
from math import isfinite
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv

from ml_harness.adapters.data.task import CsvTaskData, read_csv
from ml_harness.adapters.execution.runner import SubprocessRunner
from ml_harness.adapters.ml.evaluation import validate_predictions
from ml_harness.adapters.research.workspace import write_json
from ml_harness.adapters.storage.lease import RunLease
from ml_harness.adapters.storage.submission import AtomicSubmissionWriter, IndependentEvaluator
from ml_harness.application.use_cases.baseline import BaselineResult, run_baseline
from ml_harness.application.use_cases.prepare_task import prepare_task
from ml_harness.domain.models import (
    ContractError,
    Metric,
    PredictionKind,
    PreparedTask,
    Score,
    TaskKind,
)
from ml_harness.entrypoints.research import research_after_baseline


def _resume_baseline(work_dir: Path, task: PreparedTask) -> BaselineResult:
    """Restore measured baseline and validate the currently saved answer independently."""
    try:
        manifest = json.loads((work_dir / "prepared.json").read_text(encoding="utf-8"))
        if manifest["protocol"]["fingerprint"] != task.protocol.fingerprint:
            raise ContractError("Resume validation protocol fingerprint changed")
        baseline = json.loads((work_dir / "baseline.json").read_text(encoding="utf-8"))
        measured = baseline["score"]
        score = Score(
            measured["value"],
            Metric(measured["metric"]),
            measured["protocol_id"],
            measured["level"],
        )
        if (score.metric, score.protocol_id, score.level) != (
            task.spec.metric,
            task.protocol.fingerprint,
            task.protocol.level,
        ):
            raise ContractError("Resume baseline score disagrees with prepared task")
        submission = Path(baseline["submission_path"]).resolve()
        if submission != Path(task.spec.submission_path).resolve():
            raise ContractError("Resume baseline submission path disagrees with task")
        columns, rows = read_csv(submission)
        if columns != task.spec.submission_columns:
            raise ContractError("Resume submission schema disagrees with task")
        _, test = read_csv(Path(task.spec.test_path))
        spec = task.spec
        predictions: list[object] = []
        for row in rows:
            if spec.prediction_kind == PredictionKind.PROBABILITIES:
                values = [float(row[column]) for column in spec.submission_columns[1:]]
                predictions.append(
                    [1 - values[0], values[0]] if spec.kind == TaskKind.BINARY else values
                )
            elif spec.prediction_kind == PredictionKind.VALUES:
                predictions.append(float(row[spec.target]))
            else:
                predictions.append(row[spec.target])
        validate_predictions(
            spec,
            [row[spec.id_column] for row in test],
            {
                "ids": [row[spec.id_column] for row in rows],
                "predictions": predictions,
                "classes": list(spec.classes),
            },
        )
        durations = [float(baseline[key]) for key in ("validation_seconds", "refit_seconds")]
        if not all(isfinite(value) and value >= 0 for value in durations):
            raise ContractError("Resume baseline durations must be finite and nonnegative")
        return BaselineResult(score, str(submission), *durations)
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        raise ContractError(f"Cannot resume saved baseline: {error}") from error


def main() -> int:
    parser = argparse.ArgumentParser(description="Autonomous tabular ML research harness")
    parser.add_argument("task", nargs="?", default=Path("task.md"), type=Path)
    parser.add_argument(
        "--backend", choices=("catboost", "lightgbm", "autogluon"), default="catboost"
    )
    parser.add_argument("--threads", type=int, choices=range(1, 9), default=8)
    parser.add_argument("--worker-timeout", type=float, default=1800)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--research-sessions", type=int, default=3)
    parser.add_argument("--research-steps", type=int, default=12)
    parser.add_argument("--llm-token-limit", type=int, default=32000)
    args = parser.parse_args()
    if args.research_steps < 1 or args.llm_token_limit < 1 or args.research_sessions < 1:
        parser.error("Research steps, sessions and token limit must be positive")
    if args.resume and args.run_dir is None:
        parser.error("--resume requires --run-dir")
    load_dotenv(Path.cwd() / ".env", override=False)
    work_dir = (args.run_dir or Path("runs") / uuid4().hex).resolve()
    try:
        with RunLease(work_dir / ".run.lock"):
            if not args.resume and any(path.name != ".run.lock" for path in work_dir.iterdir()):
                raise ContractError("Run directory already contains artifacts; use --resume")
            task = prepare_task(str(args.task), CsvTaskData())
            if args.resume:
                result = _resume_baseline(work_dir, task)
            else:
                write_json(work_dir / "prepared.json", asdict(task))
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
                write_json(work_dir / "baseline.json", asdict(result))
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
                    max_sessions=args.research_sessions,
                )
            summary = json.dumps(details, ensure_ascii=False, indent=2)
            write_json(work_dir / "summary.json", details)
            print(summary)
    except (ContractError, OSError, ValueError) as exc:
        print(f"Task failed: {exc}", file=sys.stderr)
        return 2
    return 0
