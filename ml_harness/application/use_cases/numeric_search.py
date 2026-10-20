"""Deterministic alternating numeric search under the same measured wall-clock reserve."""

from collections.abc import Callable
from typing import Protocol

from ml_harness.application.ports.clock import Clock
from ml_harness.domain.budget import WallClockBudget


class NumericWorkspace(Protocol):
    def numeric_batch(self, backend: str, index: int) -> bool: ...
    def numeric_completed(self) -> int: ...


def run_numeric_search(
    workspace: NumericWorkspace,
    budget: WallClockBudget,
    clock: Clock,
    *,
    checkpoint: Callable[[], None] = lambda: None,
) -> str:
    index = workspace.numeric_completed()
    failures = 0
    while budget.available(clock.now()) > 1:
        backend = ("catboost", "lightgbm")[index % 2]
        if workspace.numeric_batch(backend, index):
            failures = 0
        else:
            failures += 1
        checkpoint()
        index += 1
        if failures >= 3:
            return "numeric_error_limit"
    return "budget_exhausted"
