"""POSIX process-wide deadline that restores an enclosing alarm on exit."""

import math
import signal
import time
from types import FrameType, TracebackType
from typing import Any

from ml_harness.domain.budget import BudgetExpired


class DeadlineGuard:
    def __init__(self, seconds: float) -> None:
        if isinstance(seconds, bool) or not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("Deadline duration must be finite and positive")
        self.seconds = seconds
        self._active = False
        self._handler: Any = None
        self._timer = (0.0, 0.0)
        self._started = 0.0

    @staticmethod
    def _expired(signum: int, frame: FrameType | None) -> None:
        raise BudgetExpired("Wall-clock deadline reached")

    def __enter__(self) -> "DeadlineGuard":
        if self._active:
            raise ValueError("Deadline guard is already active")
        self._handler = signal.getsignal(signal.SIGALRM)
        self._timer = signal.getitimer(signal.ITIMER_REAL)
        self._started = time.monotonic()
        signal.signal(signal.SIGALRM, self._expired)
        delay = min(self.seconds, self._timer[0]) if self._timer[0] > 0 else self.seconds
        signal.setitimer(signal.ITIMER_REAL, delay)
        self._active = True
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._active:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, self._handler)
            remaining = self._timer[0]
            if remaining > 0:
                remaining = max(0.000001, remaining - (time.monotonic() - self._started))
            signal.setitimer(signal.ITIMER_REAL, remaining, self._timer[1])
            self._active = False
