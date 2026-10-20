"""Task and experiment invariants, independent of data/ML frameworks."""

from dataclasses import dataclass, field
from enum import StrEnum
from math import isfinite

Scalar = str | int | float


class ContractError(ValueError):
    """The task or result contradicts an explicit contract."""


class TaskKind(StrEnum):
    REGRESSION = "regression"
    BINARY = "binary"
    MULTICLASS = "multiclass"


class Metric(StrEnum):
    RMSE = "rmse"
    MAE = "mae"
    MACRO_F1 = "macro_f1"
    ACCURACY = "accuracy"
    ROC_AUC = "roc_auc"
    LOG_LOSS = "log_loss"

    @property
    def minimize(self) -> bool:
        return self in (Metric.RMSE, Metric.MAE, Metric.LOG_LOSS)


class PredictionKind(StrEnum):
    VALUES = "values"
    LABELS = "labels"
    PROBABILITIES = "probabilities"


@dataclass(frozen=True)
class TaskSpec:
    task_path: str
    train_path: str
    test_path: str
    dictionary_path: str
    sample_path: str
    submission_path: str
    target: str
    id_column: str
    features: tuple[str, ...]
    kind: TaskKind
    metric: Metric
    prediction_kind: PredictionKind
    submission_columns: tuple[str, ...]
    classes: tuple[Scalar, ...] = ()
    time_column: str | None = None
    group_column: str | None = None

    def __post_init__(self) -> None:
        if not self.features or self.target in self.features or self.id_column in self.features:
            raise ContractError("Features must be nonempty and exclude target/ID")
        if len(set(self.features)) != len(self.features):
            raise ContractError("Duplicate feature names")
        regression = self.kind == TaskKind.REGRESSION
        if regression != (self.metric in (Metric.RMSE, Metric.MAE)):
            raise ContractError("Task kind and metric disagree")
        if regression:
            if self.classes or self.prediction_kind != PredictionKind.VALUES:
                raise ContractError("Regression requires values and no classes")
        else:
            if len(set(self.classes)) != len(self.classes):
                raise ContractError("Duplicate classes")
            if len(self.classes) < 2 or (self.kind == TaskKind.BINARY) != (len(self.classes) == 2):
                raise ContractError("Classification kind and class count disagree")
            expected = (
                PredictionKind.PROBABILITIES
                if self.metric in (Metric.ROC_AUC, Metric.LOG_LOSS)
                else PredictionKind.LABELS
            )
            if self.prediction_kind != expected:
                raise ContractError("Metric and prediction format disagree")
        expected_columns = 2
        if self.prediction_kind == PredictionKind.PROBABILITIES:
            expected_columns = 2 if self.kind == TaskKind.BINARY else 1 + len(self.classes)
        if (
            len(self.submission_columns) != expected_columns
            or self.submission_columns[0] != self.id_column
            or len(set(self.submission_columns)) != len(self.submission_columns)
        ):
            raise ContractError("Invalid submission columns")
        if self.prediction_kind != PredictionKind.PROBABILITIES:
            if self.submission_columns[1] != self.target:
                raise ContractError("Submission target disagrees with training target")
        for column in (self.group_column, self.time_column):
            if column is not None and column not in self.features:
                raise ContractError(f"Validation column {column!r} is not a feature")
        if self.group_column and self.time_column:
            raise ContractError("Combined group/time validation requires an explicit protocol")


class SplitKind(StrEnum):
    RANDOM = "random"
    STRATIFIED = "stratified"
    GROUP = "group"
    TIME = "time"


@dataclass(frozen=True)
class ValidationProtocol:
    fingerprint: str
    data_fingerprint: str
    kind: SplitKind
    seed: int
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    control_indices: tuple[int, ...] = ()
    level: str = "full"

    def __post_init__(self) -> None:
        partitions = (self.train_indices, self.validation_indices, self.control_indices)
        flat = [index for part in partitions for index in part]
        if not self.train_indices or not self.validation_indices:
            raise ContractError("Training and validation must be nonempty")
        if len(flat) != len(set(flat)) or any(index < 0 for index in flat):
            raise ContractError("Validation indices overlap, repeat or are negative")
        if self.level not in ("smoke", "full"):
            raise ContractError("Unknown validation level")


@dataclass(frozen=True)
class PreparedTask:
    spec: TaskSpec
    protocol: ValidationProtocol
    train_rows: int
    test_rows: int
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class Score:
    value: float
    metric: Metric
    protocol_id: str
    level: str = "full"

    def __post_init__(self) -> None:
        if not isfinite(self.value):
            raise ContractError("Score must be finite")
        if self.level not in ("smoke", "full") or not self.protocol_id:
            raise ContractError("Score requires protocol and validation level")

    def better_than(self, other: "Score") -> bool:
        if (self.metric, self.protocol_id, self.level) != (
            other.metric,
            other.protocol_id,
            other.level,
        ):
            raise ContractError("Scores from different metrics/protocols/levels are incomparable")
        return self.value < other.value if self.metric.minimize else self.value > other.value


class ExperimentStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMED_OUT = "timed_out"


@dataclass(frozen=True)
class Hypothesis:
    description: str
    expected_effect: str
    parent_id: str | None = None


@dataclass(frozen=True)
class ErrorReport:
    observations: tuple[str, ...] = ()
    # Measured diagnostics are separate from the model's interpretation.
    interpretation: str | None = None


@dataclass
class Experiment:
    id: str
    hypothesis: Hypothesis
    code_hash: str
    parameters_hash: str
    data_fingerprint: str
    protocol_id: str
    level: str = "full"
    status: ExperimentStatus = ExperimentStatus.PENDING
    score: Score | None = None
    error_report: ErrorReport = field(default_factory=ErrorReport)
    elapsed_seconds: float = 0.0
    artifacts: tuple[str, ...] = ()

    def start(self) -> None:
        if self.status != ExperimentStatus.PENDING:
            raise ContractError("Only a pending experiment can start")
        self.status = ExperimentStatus.RUNNING

    def finish(self, status: ExperimentStatus, score: Score | None = None) -> None:
        if self.status != ExperimentStatus.RUNNING or status not in (
            ExperimentStatus.SUCCEEDED,
            ExperimentStatus.FAILED,
            ExperimentStatus.TIMED_OUT,
        ):
            raise ContractError("Invalid terminal transition")
        if (status == ExperimentStatus.SUCCEEDED) != (score is not None):
            raise ContractError("Only successful experiments must have a score")
        if score is not None and (score.protocol_id, score.level) != (self.protocol_id, self.level):
            raise ContractError("Experiment score protocol/level mismatch")
        self.status, self.score = status, score


# Retain the established mutable state machine as the canonical measured result contract.
ExperimentResult = Experiment


@dataclass
class RunBudget:
    deadline: float
    token_limit: int
    attempt_limit: int
    tokens_used: int = 0
    attempts_used: int = 0

    def __post_init__(self) -> None:
        if (
            not isfinite(self.deadline)
            or min(self.token_limit, self.attempt_limit, self.tokens_used, self.attempts_used) < 0
        ):
            raise ContractError("Invalid budget")

    def can_search(self, now: float, reserve_seconds: float) -> bool:
        if not isfinite(now) or not isfinite(reserve_seconds) or reserve_seconds < 0:
            raise ContractError("Invalid time/reserve")
        return (
            now + reserve_seconds < self.deadline
            and self.tokens_used < self.token_limit
            and self.attempts_used < self.attempt_limit
        )

    def record_attempt(self, tokens: int) -> None:
        if tokens < 0:
            raise ContractError("Token usage cannot be negative")
        self.attempts_used += 1
        self.tokens_used += tokens
