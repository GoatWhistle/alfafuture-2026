"""Competition entrypoint and explicit composition of the baseline pipeline."""

import argparse
import json
import sys
import time
from dataclasses import asdict
from math import isfinite
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv

from ml_harness.adapters.data.clarification import ClarifyingTaskData
from ml_harness.adapters.data.task import CsvTaskData, read_csv
from ml_harness.adapters.execution.budget import BudgetedRunner, BudgetedSubmissionWriter
from ml_harness.adapters.execution.clock import SystemClock
from ml_harness.adapters.execution.deadline import DeadlineGuard
from ml_harness.adapters.execution.runner import SubprocessRunner
from ml_harness.adapters.ml.evaluation import validate_predictions
from ml_harness.adapters.research.workspace import write_json
from ml_harness.adapters.storage.lease import RunLease
from ml_harness.adapters.storage.submission import (
    AtomicSubmissionWriter,
    IndependentEvaluator,
    ensure_ready_submission,
    load_submission,
)
from ml_harness.application.use_cases.baseline import BaselineResult, run_baseline, run_portfolio
from ml_harness.application.use_cases.prepare_task import prepare_task
from ml_harness.domain.budget import BudgetExpired, WallClockBudget
from ml_harness.domain.models import (
    ContractError,
    Metric,
    PredictionKind,
    PreparedTask,
    RunBudget,
    Score,
    TaskKind,
)
from ml_harness.entrypoints.preparation import ConfiguredTaskModel
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
        return BaselineResult(
            score,
            str(submission),
            durations[0],
            durations[1],
            backend=baseline.get("backend", "catboost"),
            validation_dir=baseline.get("validation_dir"),
            candidates=tuple(baseline.get("candidates", [])),
        )
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        raise ContractError(f"Cannot resume saved baseline: {error}") from error


def main(*, started_at: float | None = None) -> int:
    clock = SystemClock()
    started = started_at if started_at is not None else clock.now()
    parser = argparse.ArgumentParser(description="Autonomous tabular ML research harness")
    parser.add_argument("task", nargs="?", default=Path("task.md"), type=Path)
    parser.add_argument(
        "--backend", choices=("portfolio", "catboost", "lightgbm", "autogluon"), default="portfolio"
    )
    parser.add_argument("--threads", type=int, choices=range(1, 9), default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--worker-timeout", type=float, default=1800)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--time-budget", type=float, default=10800)
    parser.add_argument("--memory-limit-mb", type=int, default=28000)
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--search-mode", choices=("qwen", "optuna", "none"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--research-sessions", type=int, default=12)
    parser.add_argument("--research-steps", type=int, default=12)
    parser.add_argument("--llm-token-limit", type=int, default=32000)
    args = parser.parse_args()
    if args.baseline_only and args.search_mode not in (None, "none"):
        parser.error("--baseline-only conflicts with --search-mode")
    args.search_mode = "none" if args.baseline_only else (args.search_mode or "qwen")
    if args.research_steps < 1 or args.llm_token_limit < 1 or args.research_sessions < 1:
        parser.error("Research steps, sessions and token limit must be positive")
    if (
        not isfinite(args.time_budget)
        or not 0 < args.time_budget <= 10800
        or not 1 <= args.memory_limit_mb <= 28000
    ):
        parser.error("Time budget must be in (0,10800] and memory in 1..28000 MiB")
    if args.resume and args.run_dir is None:
        parser.error("--resume requires --run-dir")
    if not 0 <= args.seed < 2**32:
        parser.error("Seed must be in 0..2**32-1")
    if not isfinite(args.worker_timeout) or args.worker_timeout <= 0:
        parser.error("Worker timeout must be finite and positive")
    load_dotenv(Path.cwd() / ".env", override=False)
    work_dir = (args.run_dir or Path("runs") / uuid4().hex).resolve()
    ready = False
    task: PreparedTask | None = None
    try:
        with RunLease(work_dir / ".run.lock"):
            if not args.resume and any(path.name != ".run.lock" for path in work_dir.iterdir()):
                raise ContractError("Run directory already contains artifacts; use --resume")
            budget_path = work_dir / "budget.json"
            saved = (
                json.loads(budget_path.read_text()) if args.resume and budget_path.exists() else {}
            )
            if not isinstance(saved, dict):
                raise ContractError("Saved budget must be an object")
            for key in (
                "started_unix",
                "time_limit",
                "token_limit",
                "validation_seconds",
                "refit_seconds",
                "predict_seconds",
                "write_seconds",
                "preparation_tokens",
                "preparation_attempts",
            ):
                if key in saved and (
                    isinstance(saved[key], bool)
                    or not isinstance(saved[key], int | float)
                    or not isfinite(saved[key])
                    or saved[key] < 0
                ):
                    raise ContractError(f"Invalid saved budget field: {key}")
            if "token_limit" in saved and (
                not isinstance(saved["token_limit"], int) or saved["token_limit"] < 1
            ):
                raise ContractError("Saved token limit must be a positive integer")
            for key in ("preparation_tokens", "preparation_attempts"):
                if key in saved and type(saved[key]) is not int:
                    raise ContractError(f"Saved budget counter must be an integer: {key}")
            origin = saved.get("started_unix", time.time() - max(0, clock.now() - started))
            limit = min(args.time_budget, saved.get("time_limit", args.time_budget))
            args.llm_token_limit = min(
                args.llm_token_limit, saved.get("token_limit", args.llm_token_limit)
            )
            remaining = origin + limit - time.time()
            if remaining <= 0:
                raise BudgetExpired("Original run deadline is exhausted")
            budget = WallClockBudget(clock.now() + remaining)
            preparation_budget = RunBudget(
                budget.deadline,
                args.llm_token_limit,
                2,
                int(saved.get("preparation_tokens", 0)),
                int(saved.get("preparation_attempts", 0)),
            )
            data = ClarifyingTaskData(
                CsvTaskData(),
                ConfiguredTaskModel(),
                work_dir / "task-clarification",
                preparation_budget,
                clock,
                reserve_seconds=budget.reserve_seconds,
            )
            data.usage_known = saved.get("preparation_usage_known", True) is True
            preparation_inflight = False
            for key in ("validation_seconds", "refit_seconds", "predict_seconds", "write_seconds"):
                if key in saved:
                    budget.observe(key.removesuffix("_seconds"), saved[key])

            def save_budget() -> None:
                write_json(
                    budget_path,
                    {
                        "started_unix": origin,
                        "time_limit": limit,
                        "token_limit": args.llm_token_limit,
                        "preparation_tokens": preparation_budget.tokens_used,
                        "preparation_attempts": preparation_budget.attempts_used,
                        "preparation_usage_known": data.usage_known and not preparation_inflight,
                        **{
                            key: value for key, value in asdict(budget).items() if key != "deadline"
                        },
                    },
                )

            save_budget()
            with DeadlineGuard(remaining):
                preparation_inflight = True
                save_budget()
                try:
                    task = prepare_task(str(args.task), data, seed=args.seed)
                except BaseException as error:
                    # An interrupted request can have provider-side cost without returned usage.
                    if not isinstance(error, Exception):
                        data.usage_known = False
                    raise
                finally:
                    preparation_inflight = False
                    save_budget()
                underlying = SubprocessRunner(memory_limit_mb=args.memory_limit_mb)
                runner = BudgetedRunner(underlying, clock, budget, finalizing=True)
                writer = BudgetedSubmissionWriter(AtomicSubmissionWriter(), clock, budget)
                if args.resume:
                    result = _resume_baseline(work_dir, task)
                    ready = True
                else:
                    write_json(work_dir / "prepared.json", asdict(task))
                    fallback = ensure_ready_submission(task, writer)
                    write_json(work_dir / "ready.json", fallback)
                    ready = True
                    baseline_runner = run_portfolio if args.backend == "portfolio" else run_baseline
                    backend_arguments = (
                        {} if args.backend == "portfolio" else {"backend": args.backend}
                    )
                    result = baseline_runner(
                        task,
                        runner,
                        IndependentEvaluator(),
                        writer,
                        work_dir=str(work_dir),
                        **backend_arguments,
                        threads=args.threads,
                        timeout_seconds=args.worker_timeout,
                    )
                    write_json(work_dir / "baseline.json", asdict(result))
                budget.observe("validation", result.validation_seconds)
                budget.observe("refit", result.refit_seconds)
                save_budget()
                details = asdict(result)
                if args.search_mode != "none":
                    details["research"] = research_after_baseline(
                        task,
                        result,
                        work_dir / "research",
                        threads=args.threads,
                        timeout_seconds=args.worker_timeout,
                        max_steps=args.research_steps,
                        search_mode=args.search_mode,
                        token_limit=args.llm_token_limit,
                        max_sessions=args.research_sessions,
                        runtime_budget=budget,
                        base_runner=underlying,
                        initial_usage={
                            "tokens": preparation_budget.tokens_used,
                            "attempts": preparation_budget.attempts_used,
                            "known": data.usage_known,
                        },
                    )
                else:
                    from ml_harness.entrypoints.research import finalize_baseline

                    details["finalization"] = finalize_baseline(
                        task,
                        result,
                        work_dir / "research",
                        threads=args.threads,
                        timeout_seconds=args.worker_timeout,
                        runtime_budget=budget,
                        base_runner=underlying,
                    )
                save_budget()
                write_json(work_dir / "summary.json", details)
                print(json.dumps(details, ensure_ascii=False, indent=2))
    except (BudgetExpired, ContractError, OSError, ValueError) as exc:
        if ready and task is not None:
            try:
                load_submission(task)
            except (ContractError, OSError, ValueError):
                ready = False
        if ready and task is not None:
            details = {
                "status": "fallback_preserved",
                "submission_path": task.spec.submission_path,
                "reason": str(exc),
            }
            write_json(work_dir / "summary.json", details)
            print(json.dumps(details, ensure_ascii=False, indent=2))
            return 0
        print(f"Task failed: {exc}", file=sys.stderr)
        return 124 if isinstance(exc, BudgetExpired) else 2
    return 0
