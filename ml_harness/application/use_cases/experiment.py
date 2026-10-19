"""Smoke then full execution; only the evaluator can produce successful measured scores."""

from dataclasses import asdict
from typing import Any

from ml_harness.application.ports.execution import Evaluator, ExperimentRunner, RunRequest
from ml_harness.domain.models import (
    ContractError,
    Experiment,
    ExperimentStatus,
    PreparedTask,
)


def execute_experiment(
    experiment: Experiment,
    task: PreparedTask,
    smoke: PreparedTask,
    runner: ExperimentRunner,
    evaluator: Evaluator,
    *,
    candidate_path: str,
    output_dir: str,
    threads: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    experiment.start()
    measured: dict[str, Any] = {}
    elapsed = 0.0
    try:
        for level, prepared in (("smoke", smoke), ("full", task)):
            result = runner.run(
                RunRequest(
                    prepared,
                    output_dir + "/" + level,
                    candidate_path=candidate_path,
                    threads=threads,
                ),
                min(60, timeout_seconds) if level == "smoke" else timeout_seconds,
            )
            elapsed += result.elapsed_seconds
            if result.status != ExperimentStatus.SUCCEEDED or result.output is None:
                status = (
                    ExperimentStatus.TIMED_OUT
                    if result.status == ExperimentStatus.TIMED_OUT
                    else ExperimentStatus.FAILED
                )
                experiment.finish(status)
                measured.update(failure_phase=level, error=result.detail)
                break
            score = evaluator.evaluate(prepared, result.output)
            measured[level + "_score"] = asdict(score)
        else:
            experiment.finish(ExperimentStatus.SUCCEEDED, score)
    except (ContractError, OSError, ValueError) as exc:
        experiment.finish(ExperimentStatus.FAILED)
        measured["error"] = str(exc)[:2000]
    experiment.elapsed_seconds = elapsed
    measured.update(status=experiment.status.value, elapsed_seconds=elapsed)
    return measured
