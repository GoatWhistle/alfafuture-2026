"""Monotonic wall clock boundary."""

from typing import Protocol


class Clock(Protocol):
    def now(self) -> float: ...
