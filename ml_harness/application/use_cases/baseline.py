"""One independently evaluated candidate and full-data prediction."""

from dataclasses import dataclass

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
        RunRequest(task, work_dir + "/refit", mode="refit", backend=backend, threads=threads),
        timeout_seconds,
    )
    if refit.status != ExperimentStatus.SUCCEEDED or refit.output is None:
        raise ContractError(f"Refit {refit.status}: {refit.detail}")
    submission = writer.write(task, refit.output)
    return BaselineResult(score, submission, validation.elapsed_seconds, refit.elapsed_seconds)
