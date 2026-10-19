"""Actual POSIX alarms interrupt blocked execution and restore enclosing timers."""

import signal
import time

import pytest

from ml_harness.adapters.execution.deadline import BudgetExpired, DeadlineGuard


@pytest.mark.parametrize("seconds", [0, -1, float("inf"), float("nan"), True])
def test_invalid_deadline(seconds):
    with pytest.raises(ValueError):
        DeadlineGuard(seconds)


def test_actual_alarm_interrupts_wait_and_restores_handler():
    original = signal.getsignal(signal.SIGALRM)
    started = time.monotonic()
    with pytest.raises(BudgetExpired):
        with DeadlineGuard(0.03):
            time.sleep(1)
    assert time.monotonic() - started < 0.5
    assert signal.getsignal(signal.SIGALRM) == original
    assert signal.getitimer(signal.ITIMER_REAL) == (0, 0)


def test_previous_timer_is_preserved_with_elapsed_time_and_original_interval():
    original = signal.getsignal(signal.SIGALRM)
    old_timer = signal.getitimer(signal.ITIMER_REAL)

    def previous(signum, frame):
        pass

    try:
        signal.signal(signal.SIGALRM, previous)
        signal.setitimer(signal.ITIMER_REAL, 0.5, 0.1)
        with DeadlineGuard(1):
            assert signal.getitimer(signal.ITIMER_REAL)[0] <= 0.5
            time.sleep(0.02)
        remaining, interval = signal.getitimer(signal.ITIMER_REAL)
        assert 0.1 < remaining < 0.49
        assert interval == 0.1
        assert signal.getsignal(signal.SIGALRM) == previous
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, original)
        signal.setitimer(signal.ITIMER_REAL, *old_timer)


def test_nested_guard_restores_outer_deadline():
    with DeadlineGuard(0.5):
        outer = signal.getsignal(signal.SIGALRM)
        with pytest.raises(BudgetExpired):
            with DeadlineGuard(0.02):
                time.sleep(1)
        assert signal.getsignal(signal.SIGALRM) == outer
        assert 0 < signal.getitimer(signal.ITIMER_REAL)[0] < 0.5
