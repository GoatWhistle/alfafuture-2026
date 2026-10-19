"""Validate task materials and freeze the protocol before any training."""

from ml_harness.application.ports.task_data import TaskData
from ml_harness.domain.models import PreparedTask


def prepare_task(task_path: str, data: TaskData, *, seed: int = 42) -> PreparedTask:
    return data.prepare(task_path, seed)
