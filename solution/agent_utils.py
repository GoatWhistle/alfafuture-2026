"""Общие утилиты агента: логирование и бюджет времени."""

import os
import time

START = time.time()

# Полный лимит платформы — 3 часа. Резерв на предикт/запись — отдельно.
TOTAL_BUDGET_SEC = float(os.environ.get("AGENT_TIME_LIMIT_SEC", 3 * 3600))
# Жёсткий запас: к этому моменту submission.csv уже должен лежать финальный.
SAFETY_SEC = float(os.environ.get("AGENT_SAFETY_SEC", 600))


def log(msg: str) -> None:
    print(f"[agent +{time.time() - START:8.1f}s] {msg}", flush=True)


def elapsed() -> float:
    return time.time() - START


def time_left() -> float:
    """Сколько секунд осталось до дедлайна с учётом запаса."""
    return TOTAL_BUDGET_SEC - SAFETY_SEC - elapsed()
