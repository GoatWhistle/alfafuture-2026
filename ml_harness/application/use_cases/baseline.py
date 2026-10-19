"""One independently evaluated candidate and full-data prediction."""

from dataclasses import asdict, dataclass, replace
from typing import Any

from ml_harness.application.ports.execution import (
    Evaluator,
    ExperimentRunner,
    RunRequest,
    SubmissionWriter,
)
from ml_harness.domain.models import ContractError, ExperimentStatus, PreparedTask, Score


@dataclass(frozen=True)
class BaselineResult:
    score: Score
    submission_path: str
    validation_seconds: float
    refit_seconds: float
    backend: str = "catboost"
    validation_dir: str | None = None
    candidates: tuple[dict[str, Any], ...] = ()


def run_baseline(
    task: PreparedTask,
    runner: ExperimentRunner,
    evaluator: Evaluator,
    writer: SubmissionWriter,
    *,
    work_dir: str,
    backend: str = "catboost",
    threads: int = 8,
    timeout_seconds: float = 1800,
) -> BaselineResult:
    validation = runner.run(
        RunRequest(task, work_dir + "/validation", backend=backend, threads=threads),
        timeout_seconds,
    )
    if validation.status != ExperimentStatus.SUCCEEDED or validation.output is None:
        raise ContractError(f"Validation {validation.status}: {validation.detail}")
    score = evaluator.evaluate(task, validation.output)
    refit = runner.run(
        RunRequest(
            task,
            work_dir + "/refit",
            mode="fallback" if task.protocol.control_indices else "refit",
            backend=backend,
            threads=threads,
        ),
        timeout_seconds,
    )
    if refit.status != ExperimentStatus.SUCCEEDED or refit.output is None:
        raise ContractError(f"Refit {refit.status}: {refit.detail}")
    submission = writer.write(task, refit.output)
    return BaselineResult(
        score,
        submission,
        validation.elapsed_seconds,
        refit.elapsed_seconds,
        backend,
        work_dir + "/validation",
    )


def run_portfolio(
    task: PreparedTask,
    runner: ExperimentRunner,
    evaluator: Evaluator,
    writer: SubmissionWriter,
    *,
    work_dir: str,
    threads: int,
    timeout_seconds: float,
) -> BaselineResult:
    """Publish an early learned answer, then retain both comparable starting points."""
    candidates: list[dict[str, Any]] = []
    winner: BaselineResult | None = None
    failure = "No portfolio model completed"
    for backend in ("catboost", "lightgbm"):
        directory = work_dir if backend == "catboost" else work_dir + "/portfolio/lightgbm"
        try:
            if winner is None:
                result = run_baseline(
                    task,
                    runner,
                    evaluator,
                    writer,
                    work_dir=directory,
                    backend=backend,
                    threads=threads,
                    timeout_seconds=timeout_seconds,
                )
                candidates.append(
                    {
                        "backend": backend,
                        "score": asdict(result.score),
                        "validation_dir": directory + "/validation",
                    }
                )
                winner = result
                continue
            outcome = runner.run(
                RunRequest(task, directory + "/validation", backend=backend, threads=threads),
                timeout_seconds,
            )
            if outcome.status != ExperimentStatus.SUCCEEDED or outcome.output is None:
                continue
            score = evaluator.evaluate(task, outcome.output)
            candidates.append(
                {
                    "backend": backend,
                    "score": asdict(score),
                    "validation_dir": directory + "/validation",
                }
            )
            if score.better_than(winner.score):
                refit = runner.run(
                    RunRequest(
                        task,
                        directory + "/refit",
                        mode="fallback" if task.protocol.control_indices else "refit",
                        backend=backend,
                        threads=threads,
                    ),
                    timeout_seconds,
                )
                if refit.status == ExperimentStatus.SUCCEEDED and refit.output is not None:
                    path = writer.write(task, refit.output)
                    winner = BaselineResult(
                        score,
                        path,
                        max(winner.validation_seconds, outcome.elapsed_seconds),
                        max(winner.refit_seconds, refit.elapsed_seconds),
                        backend,
                        directory + "/validation",
                    )
        except (ContractError, OSError, ValueError) as exc:
            failure = str(exc)
    if winner is None:
        raise ContractError(failure)
    return replace(winner, candidates=tuple(candidates))
