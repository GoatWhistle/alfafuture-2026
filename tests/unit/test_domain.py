"""Behavioral contracts for the framework-independent domain models."""

from dataclasses import replace

import pytest

from ml_harness.domain.models import (
    ContractError,
    Experiment,
    ExperimentStatus,
    Hypothesis,
    Metric,
    PredictionKind,
    RunBudget,
    Score,
    SplitKind,
    TaskKind,
    TaskSpec,
    ValidationProtocol,
)


@pytest.mark.parametrize("metric", list(Metric))
def test_score_direction_and_ties(metric: Metric) -> None:
    low, high = Score(0.2, metric, "split"), Score(0.8, metric, "split")
    minimize = metric in (Metric.RMSE, Metric.MAE, Metric.LOG_LOSS)
    assert low.better_than(high) == minimize
    assert high.better_than(low) != minimize
    assert not low.better_than(low)


@pytest.mark.parametrize(
    "other",
    [
        Score(0.1, Metric.MAE, "split"),
        Score(0.1, Metric.RMSE, "other"),
        Score(0.1, Metric.RMSE, "split", "smoke"),
    ],
)
def test_score_rejects_incomparable_results(other: Score) -> None:
    with pytest.raises(ContractError, match="incomparable"):
        Score(0.2, Metric.RMSE, "split").better_than(other)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_score_requires_finite_value(value: float) -> None:
    with pytest.raises(ContractError, match="finite"):
        Score(value, Metric.RMSE, "split")


@pytest.mark.parametrize("protocol,level", [("", "full"), ("split", "unknown")])
def test_score_requires_protocol_and_level(protocol: str, level: str) -> None:
    with pytest.raises(ContractError):
        Score(0.2, Metric.RMSE, protocol, level)


@pytest.fixture
def experiment() -> Experiment:
    return Experiment("candidate", Hypothesis("test", "improve"), "code", "params", "data", "split")


@pytest.mark.parametrize(
    "status",
    [
        ExperimentStatus.SUCCEEDED,
        ExperimentStatus.FAILED,
        ExperimentStatus.TIMED_OUT,
    ],
)
def test_experiment_terminal_transitions(experiment: Experiment, status: ExperimentStatus) -> None:
    score = Score(0.2, Metric.RMSE, "split") if status == ExperimentStatus.SUCCEEDED else None
    experiment.start()
    assert experiment.status == ExperimentStatus.RUNNING
    experiment.finish(status, score)
    assert experiment.status == status
    assert experiment.score == score
    with pytest.raises(ContractError):
        experiment.start()
    with pytest.raises(ContractError):
        experiment.finish(status, score)


def test_experiment_requires_running_before_finish(experiment: Experiment) -> None:
    with pytest.raises(ContractError):
        experiment.finish(ExperimentStatus.FAILED)
    assert experiment.status == ExperimentStatus.PENDING
    experiment.start()
    with pytest.raises(ContractError):
        experiment.start()


@pytest.mark.parametrize(
    "status,score",
    [
        (ExperimentStatus.SUCCEEDED, None),
        (ExperimentStatus.FAILED, Score(0.2, Metric.RMSE, "split")),
        (ExperimentStatus.TIMED_OUT, Score(0.2, Metric.RMSE, "split")),
        (ExperimentStatus.PENDING, None),
        (ExperimentStatus.RUNNING, None),
        (ExperimentStatus.SUCCEEDED, Score(0.2, Metric.RMSE, "other")),
        (ExperimentStatus.SUCCEEDED, Score(0.2, Metric.RMSE, "split", "smoke")),
    ],
)
def test_invalid_finish_preserves_running_state(
    experiment: Experiment,
    status: ExperimentStatus,
    score: Score | None,
) -> None:
    experiment.start()
    with pytest.raises(ContractError):
        experiment.finish(status, score)
    assert experiment.status == ExperimentStatus.RUNNING
    assert experiment.score is None


def test_budget_reserves_finalization_time() -> None:
    budget = RunBudget(100, 10, 2)
    assert budget.can_search(89, 10)
    assert not budget.can_search(90, 10)
    assert not budget.can_search(101, 0)


@pytest.mark.parametrize("tokens,attempts", [(10, 0), (11, 0), (0, 2), (0, 3)])
def test_budget_stops_at_either_usage_limit(tokens: int, attempts: int) -> None:
    assert not RunBudget(100, 10, 2, tokens, attempts).can_search(0, 0)


def test_budget_records_zero_token_attempt_and_rejects_negative_usage() -> None:
    budget = RunBudget(100, 10, 2)
    budget.record_attempt(0)
    assert (budget.tokens_used, budget.attempts_used) == (0, 1)
    budget.record_attempt(10)
    assert (budget.tokens_used, budget.attempts_used) == (10, 2)
    with pytest.raises(ContractError):
        budget.record_attempt(-1)
    assert (budget.tokens_used, budget.attempts_used) == (10, 2)


@pytest.mark.parametrize(
    "values",
    [
        (float("nan"), 1, 1, 0, 0),
        (float("inf"), 1, 1, 0, 0),
        (-float("inf"), 1, 1, 0, 0),
        (100, -1, 1, 0, 0),
        (100, 1, -1, 0, 0),
        (100, 1, 1, -1, 0),
        (100, 1, 1, 0, -1),
    ],
)
def test_invalid_budget(values: tuple[float, int, int, int, int]) -> None:
    with pytest.raises(ContractError):
        RunBudget(*values)


@pytest.mark.parametrize(
    "now,reserve",
    [
        (float("nan"), 0),
        (float("inf"), 0),
        (-float("inf"), 0),
        (0, float("nan")),
        (0, float("inf")),
        (0, -float("inf")),
        (0, -1),
    ],
)
def test_invalid_budget_time(now: float, reserve: float) -> None:
    with pytest.raises(ContractError):
        RunBudget(100, 10, 2).can_search(now, reserve)


def test_zero_budget_cannot_search() -> None:
    assert not RunBudget(100, 0, 1).can_search(0, 0)
    assert not RunBudget(100, 1, 0).can_search(0, 0)


@pytest.fixture
def spec() -> TaskSpec:
    return TaskSpec(
        task_path="task.md",
        train_path="train.csv",
        test_path="test.csv",
        dictionary_path="dict.csv",
        sample_path="sample.csv",
        submission_path="submission.csv",
        target="outcome",
        id_column="row_id",
        features=("value", "group", "time"),
        kind=TaskKind.REGRESSION,
        metric=Metric.RMSE,
        prediction_kind=PredictionKind.VALUES,
        submission_columns=("row_id", "outcome"),
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"features": ()},
        {"features": ("outcome",)},
        {"features": ("row_id",)},
        {"features": ("value", "value")},
        {"metric": Metric.ACCURACY},
        {"classes": (0, 1)},
        {"prediction_kind": PredictionKind.LABELS},
        {"submission_columns": ()},
        {"submission_columns": ("outcome", "row_id")},
        {"submission_columns": ("row_id", "row_id")},
        {"submission_columns": ("row_id", "wrong")},
        {"submission_columns": ("row_id", "outcome", "extra")},
        {"group_column": "missing"},
        {"time_column": "missing"},
        {"group_column": "group", "time_column": "time"},
    ],
)
def test_task_spec_rejects_contradictions(spec: TaskSpec, changes: dict[str, object]) -> None:
    with pytest.raises(ContractError):
        replace(spec, **changes)


@pytest.fixture
def binary_spec(spec: TaskSpec) -> TaskSpec:
    return replace(
        spec,
        kind=TaskKind.BINARY,
        metric=Metric.ACCURACY,
        prediction_kind=PredictionKind.LABELS,
        classes=(1, 2),
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"classes": (0, 0)},
        {"classes": ()},
        {"classes": (0,)},
        {"classes": (0, 1, 2)},
        {"kind": TaskKind.MULTICLASS},
        {"metric": Metric.RMSE},
        {"prediction_kind": PredictionKind.PROBABILITIES},
        {"metric": Metric.LOG_LOSS},
    ],
)
def test_classification_rejects_contradictions(
    binary_spec: TaskSpec,
    changes: dict[str, object],
) -> None:
    with pytest.raises(ContractError):
        replace(binary_spec, **changes)


def test_numeric_labels_and_probability_submission_contracts(binary_spec: TaskSpec) -> None:
    assert binary_spec.classes == (1, 2)
    binary = replace(
        binary_spec,
        metric=Metric.ROC_AUC,
        prediction_kind=PredictionKind.PROBABILITIES,
        submission_columns=("row_id", "probability"),
    )
    multi = replace(
        binary,
        kind=TaskKind.MULTICLASS,
        metric=Metric.LOG_LOSS,
        classes=("a", "b", "c"),
        submission_columns=("row_id", "a", "b", "c"),
    )
    assert multi.classes == ("a", "b", "c")
    with pytest.raises(ContractError):
        replace(multi, submission_columns=("row_id", "p"))
    assert replace(
        multi,
        metric=Metric.MACRO_F1,
        prediction_kind=PredictionKind.LABELS,
        submission_columns=("row_id", "outcome"),
    ).classes == ("a", "b", "c")


@pytest.mark.parametrize("column", ["group_column", "time_column"])
def test_validation_column_can_reference_feature(spec: TaskSpec, column: str) -> None:
    assert replace(spec, **{column: "value"}).features == spec.features


@pytest.mark.parametrize(
    "train,validation,control",
    [
        ((), (1,), ()),
        ((0,), (), ()),
        ((0, 0), (1,), ()),
        ((0,), (1, 1), ()),
        ((0,), (1,), (2, 2)),
        ((0,), (0,), ()),
        ((0,), (1,), (0,)),
        ((0,), (1,), (1,)),
        ((-1,), (1,), ()),
        ((0,), (-1,), ()),
        ((0,), (1,), (-1,)),
    ],
)
def test_validation_partitions_are_nonempty_disjoint_and_unique(
    train: tuple[int, ...],
    validation: tuple[int, ...],
    control: tuple[int, ...],
) -> None:
    with pytest.raises(ContractError):
        ValidationProtocol("split", "data", SplitKind.RANDOM, 42, train, validation, control)


@pytest.mark.parametrize("level", ["smoke", "full"])
def test_validation_accepts_separate_control_partition(level: str) -> None:
    protocol = ValidationProtocol("split", "data", SplitKind.RANDOM, 42, (0, 2), (1,), (3,), level)
    assert protocol.control_indices == (3,)
    assert protocol.level == level


def test_validation_rejects_unknown_level() -> None:
    with pytest.raises(ContractError):
        ValidationProtocol("split", "data", SplitKind.RANDOM, 42, (0,), (1,), level="unknown")
