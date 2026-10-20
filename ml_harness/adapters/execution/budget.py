"""Clamp each worker to remaining time while retaining measured finalization costs."""

from math import isfinite
from typing import Any

from ml_harness.application.ports.clock import Clock
from ml_harness.application.ports.execution import (
    ExecutionResult,
    ExperimentRunner,
    RunRequest,
    SubmissionWriter,
)
from ml_harness.domain.budget import WallClockBudget
from ml_harness.domain.models import ContractError, ExperimentStatus, PreparedTask


class BudgetedSubmissionWriter:
    def __init__(self, delegate: SubmissionWriter, clock: Clock, budget: WallClockBudget) -> None:
        self.delegate, self.clock, self.budget = delegate, clock, budget

    def write(self, task: PreparedTask, output: dict[str, Any]) -> str:
        started = self.clock.now()
        try:
            return self.delegate.write(task, output)
        finally:
            self.budget.observe("write", max(0.0, self.clock.now() - started))


class BudgetedRunner:
    def __init__(
        self,
        delegate: ExperimentRunner,
        clock: Clock,
        budget: WallClockBudget,
        finalizing: bool = False,
    ) -> None:
        self.delegate, self.clock, self.budget = delegate, clock, budget
        self.finalizing = finalizing

    def run(self, request: RunRequest, timeout_seconds: float) -> ExecutionResult:
        if not isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ContractError("Worker timeout must be finite and positive")
        started = self.clock.now()
        allowed = min(timeout_seconds, self.budget.available(started, self.finalizing))
        if allowed <= 0:
            return ExecutionResult(
                ExperimentStatus.TIMED_OUT, 0, detail="Wall-clock budget exhausted"
            )
        result = self.delegate.run(request, allowed)
        elapsed = max(0.0, self.clock.now() - started, result.elapsed_seconds)
        # Unsuccessful candidates are not finalists. Their elapsed time still
        # consumes the absolute deadline, but must not inflate the cost forecast.
        if result.status != ExperimentStatus.SUCCEEDED:
            return result
        stage = {"fallback": "refit", "control": "control", "refit": "refit"}.get(
            request.mode, "validation"
        )
        self.budget.observe(stage, elapsed)
        if result.output is not None:
            prediction = result.output.get("predict_seconds")
            if (
                not isinstance(prediction, bool)
                and isinstance(prediction, int | float)
                and isfinite(prediction)
                and prediction >= 0
            ):
                self.budget.observe("predict", prediction)
        return result
