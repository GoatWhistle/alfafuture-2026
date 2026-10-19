"""Dynamic finalization reserves measured only from observed execution times."""

from dataclasses import dataclass
from math import isfinite

from ml_harness.domain.models import ContractError


class BudgetExpired(BaseException):
    """The process reached its allocated wall-clock deadline."""


@dataclass
class WallClockBudget:
    deadline: float
    validation_seconds: float = 0
    refit_seconds: float = 0
    predict_seconds: float = 0
    write_seconds: float = 5
    finalists: int = 3

    def __post_init__(self) -> None:
        values = (
            self.validation_seconds,
            self.refit_seconds,
            self.predict_seconds,
            self.write_seconds,
        )
        if (
            not isfinite(self.deadline)
            or any(not isfinite(value) or value < 0 for value in values)
            or isinstance(self.finalists, bool)
            or not isinstance(self.finalists, int)
            or self.finalists < 0
        ):
            raise ContractError("Invalid wall-clock budget")

    def reserve_seconds(self) -> float:
        return max(
            15.0,
            1.5
            # A frozen ensemble has at most two members, each requiring its own refit.
            * (
                2 * self.refit_seconds
                + self.finalists * self.validation_seconds
                + 2 * self.predict_seconds
            )
            + self.write_seconds,
        )

    def available(self, now: float, finalizing: bool = False) -> float:
        if not isfinite(now):
            raise ContractError("Clock must be finite")
        return max(0.0, self.deadline - now - (0 if finalizing else self.reserve_seconds()))

    def observe(self, stage: str, seconds: float) -> None:
        columns = {
            "validation": "validation_seconds",
            "control": "validation_seconds",
            "refit": "refit_seconds",
            "predict": "predict_seconds",
            "write": "write_seconds",
        }
        if stage not in columns or not isfinite(seconds) or seconds < 0:
            raise ContractError("Invalid budget stage or duration")
        column = columns[stage]
        setattr(self, column, max(getattr(self, column), seconds))
