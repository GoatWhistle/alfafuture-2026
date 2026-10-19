"""Task preparation boundary: file formats stay in the adapter."""

from typing import Protocol

from ml_harness.domain.models import PreparedTask


class TaskData(Protocol):
    def prepare(self, task_path: str, seed: int) -> PreparedTask: ...
