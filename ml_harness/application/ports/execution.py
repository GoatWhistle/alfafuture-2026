"""Training execution and independent output boundaries."""

from dataclasses import dataclass
from typing import Any, Protocol

from ml_harness.domain.models import ExperimentStatus, PreparedTask, Score


@dataclass(frozen=True)
class RunRequest:
    task: PreparedTask
    output_dir: str
    mode: str = "validation"
    backend: str = "catboost"
    candidate_path: str | None = None
    parameters: dict[str, Any] | None = None
    threads: int = 8


@dataclass(frozen=True)
class ExecutionResult:
    status: ExperimentStatus
    elapsed_seconds: float
    output: dict[str, Any] | None = None
    detail: str = ""


class ExperimentRunner(Protocol):
    def run(self, request: RunRequest, timeout_seconds: float) -> ExecutionResult: ...


class Evaluator(Protocol):
    def evaluate(self, task: PreparedTask, output: dict[str, Any]) -> Score: ...


class SubmissionWriter(Protocol):
    def write(self, task: PreparedTask, output: dict[str, Any]) -> str: ...
