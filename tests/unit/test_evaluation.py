"""Independent metric calculation and worker-result validation."""

from dataclasses import replace
from math import log, sqrt
from typing import Any

import pytest

from ml_harness.adapters.ml.evaluation import evaluate, validate_predictions
from ml_harness.domain.models import (
    ContractError,
    Metric,
    PredictionKind,
    SplitKind,
    TaskKind,
    TaskSpec,
    ValidationProtocol,
)


@pytest.fixture
def spec() -> TaskSpec:
    return TaskSpec(
        "task",
        "train",
        "test",
        "dict",
        "sample",
        "out",
        "y",
        "id",
        ("x",),
        TaskKind.REGRESSION,
        Metric.RMSE,
        PredictionKind.VALUES,
        ("id", "y"),
    )


@pytest.fixture
def binary(spec: TaskSpec) -> TaskSpec:
    return replace(
        spec,
        kind=TaskKind.BINARY,
        metric=Metric.ROC_AUC,
        prediction_kind=PredictionKind.PROBABILITIES,
        classes=("z", "a"),
    )


@pytest.fixture
def protocol() -> ValidationProtocol:
    return ValidationProtocol("split", "data", SplitKind.RANDOM, 42, (0,), (2, 1), level="smoke")


def result(predictions: list[Any], classes: list[Any] | None = None) -> dict[str, Any]:
    return {"ids": ["002", "001"], "predictions": predictions, "classes": classes or []}


@pytest.mark.parametrize("metric,expected", [(Metric.RMSE, sqrt(2.5)), (Metric.MAE, 1.5)])
def test_regression_uses_protocol_order_and_ignores_worker_score(
    spec: TaskSpec,
    protocol: ValidationProtocol,
    metric: Metric,
    expected: float,
) -> None:
    rows = [{"id": "train", "y": "999"}, {"id": "001", "y": "3"}, {"id": "002", "y": "1"}]
    worker = result([2, 5]) | {"score": -100000}
    score = evaluate(replace(spec, metric=metric), protocol, rows, worker)
    assert score.value == pytest.approx(expected)
    assert (score.protocol_id, score.level, score.metric) == ("split", "smoke", metric)


@pytest.mark.parametrize("metric,expected", [(Metric.ROC_AUC, 1), (Metric.LOG_LOSS, -log(0.8))])
def test_binary_probability_order_and_positive_class(
    binary: TaskSpec,
    protocol: ValidationProtocol,
    metric: Metric,
    expected: float,
) -> None:
    rows = [{"id": "train"}, {"id": "001", "y": "z"}, {"id": "002", "y": "a"}]
    worker = result([[0.8, 0.2], [0.2, 0.8]], ["a", "z"])
    assert validate_predictions(binary, ["002", "001"], worker) == [[0.2, 0.8], [0.8, 0.2]]
    assert evaluate(replace(binary, metric=metric), protocol, rows, worker).value == pytest.approx(
        expected,
    )


@pytest.mark.parametrize("metric,expected", [(Metric.ACCURACY, 0.5), (Metric.MACRO_F1, 1 / 3)])
def test_numeric_labels_are_evaluated_as_classes(
    binary: TaskSpec,
    protocol: ValidationProtocol,
    metric: Metric,
    expected: float,
) -> None:
    spec = replace(binary, classes=(2, 1), metric=metric, prediction_kind=PredictionKind.LABELS)
    rows = [{"id": "train"}, {"id": "001", "y": "2"}, {"id": "002", "y": "1"}]
    assert evaluate(spec, protocol, rows, result([1, 1], [1, 2])).value == pytest.approx(expected)


@pytest.mark.parametrize("metric,expected", [(Metric.ROC_AUC, 1), (Metric.LOG_LOSS, -log(0.8))])
def test_multiclass_probabilities_in_nonlexical_class_order(
    binary: TaskSpec,
    metric: Metric,
    expected: float,
) -> None:
    spec = replace(
        binary,
        kind=TaskKind.MULTICLASS,
        classes=("z", "a", "m"),
        metric=metric,
        submission_columns=("id", "z", "a", "m"),
    )
    protocol = ValidationProtocol("split", "data", SplitKind.STRATIFIED, 42, (3,), (0, 1, 2))
    rows = [{"id": str(i), "y": label} for i, label in enumerate(["z", "a", "m", "z"])]
    worker = {
        "ids": ["0", "1", "2"],
        "classes": ["m", "z", "a"],
        "predictions": [[0.1, 0.8, 0.1], [0.1, 0.1, 0.8], [0.8, 0.1, 0.1]],
    }
    assert evaluate(spec, protocol, rows, worker).value == pytest.approx(expected)


@pytest.mark.parametrize(
    "changes",
    [
        {"ids": ["001", "002"]},
        {"ids": [2, 1]},
        {"ids": ["002", "002"]},
        {"ids": ["002"]},
        {"predictions": [1]},
        {"predictions": [[1], [2]]},
        {"predictions": [float("nan"), 1]},
        {"predictions": [1, float("inf")]},
        {"predictions": ["1", 2]},
        {"predictions": [True, 2]},
        {"classes": ["a"]},
        {"classes": None},
        {"predictions": None},
    ],
)
def test_invalid_regression_results(spec: TaskSpec, changes: dict[str, Any]) -> None:
    with pytest.raises(ContractError):
        validate_predictions(spec, ["002", "001"], result([1, 2]) | changes)


@pytest.mark.parametrize(
    "predictions,classes",
    [
        ([0.5, 0.5], ["z", "a"]),
        ([[0.5], [0.5]], ["z", "a"]),
        ([[0.5, 0.5], [0.5, 0.5]], ["z", "z"]),
        ([[0.5, 0.5], [0.5, 0.5]], ["z", "unknown"]),
        ([[0.5, 0.5], [0.5, 0.5]], []),
        ([[0.5, 0.5, 0], [0.5, 0.5, 0]], ["z", "a"]),
        ([[1.1, -0.1], [0.5, 0.5]], ["z", "a"]),
        ([[0.2, 0.2], [0.5, 0.5]], ["z", "a"]),
        ([[float("nan"), 0.5], [0.5, 0.5]], ["z", "a"]),
        ([[float("inf"), 0.5], [0.5, 0.5]], ["z", "a"]),
    ],
)
def test_invalid_probabilities(
    binary: TaskSpec,
    predictions: list[Any],
    classes: list[Any],
) -> None:
    with pytest.raises(ContractError):
        validate_predictions(binary, ["002", "001"], result(predictions, classes))


def test_unknown_and_nested_labels_rejected(binary: TaskSpec) -> None:
    spec = replace(binary, metric=Metric.ACCURACY, prediction_kind=PredictionKind.LABELS)
    for predictions in [["z", "other"], [["z"], ["a"]], [float("nan"), "a"]]:
        with pytest.raises(ContractError):
            validate_predictions(spec, ["002", "001"], result(predictions, ["z", "a"]))


def test_unmeasurable_auc_and_missing_targets_are_contract_errors(
    binary: TaskSpec,
    protocol: ValidationProtocol,
) -> None:
    worker = result([[0.5, 0.5], [0.5, 0.5]], ["z", "a"])
    rows = [{"id": "train"}, {"id": "001", "y": "z"}, {"id": "002", "y": "z"}]
    with pytest.raises(ContractError):
        evaluate(binary, protocol, rows, worker)
    del rows[1]["y"]
    with pytest.raises(ContractError):
        evaluate(binary, protocol, rows, worker)


def test_regression_nonfinite_targets_and_invalid_indices(
    spec: TaskSpec,
    protocol: ValidationProtocol,
) -> None:
    rows = [{"id": "train"}, {"id": "001", "y": "inf"}, {"id": "002", "y": "1"}]
    with pytest.raises(ContractError):
        evaluate(spec, protocol, rows, result([1, 2]))
    with pytest.raises(ContractError):
        evaluate(spec, replace(protocol, validation_indices=(5,)), rows, result([1]))


def test_probability_tolerance_preserves_finite_rows(binary: TaskSpec) -> None:
    worker = result([[0.5, 0.5000001], [1, 0]], ["z", "a"])
    assert validate_predictions(binary, ["002", "001"], worker) == worker["predictions"]


@pytest.mark.parametrize(
    "classes,csv_targets",
    [
        ((0, 1), ("0.0", "1.0")),
        ((-1.5, 2.5), ("-1.50", "2.50")),
    ],
)
def test_numeric_csv_target_representation(
    binary: TaskSpec,
    protocol: ValidationProtocol,
    classes: tuple[int | float, ...],
    csv_targets: tuple[str, str],
) -> None:
    spec = replace(binary, classes=classes)
    rows = [{"id": "train"}, {"id": "001", "y": csv_targets[0]}, {"id": "002", "y": csv_targets[1]}]
    worker = result([[0.2, 0.8], [0.8, 0.2]], list(classes))
    assert evaluate(spec, protocol, rows, worker).value == 1.0
    labels = replace(spec, metric=Metric.ACCURACY, prediction_kind=PredictionKind.LABELS)
    worker_labels = result([csv_targets[1], csv_targets[0]], list(classes))
    assert evaluate(labels, protocol, rows, worker_labels).value == 1.0


def test_text_labels_preserve_leading_zeroes_and_na(binary: TaskSpec) -> None:
    spec = replace(
        binary, classes=("001", "NA"), metric=Metric.ACCURACY, prediction_kind=PredictionKind.LABELS
    )
    worker = result(["001", "NA"], ["001", "NA"])
    assert validate_predictions(spec, ["002", "001"], worker) == ["001", "NA"]
    for label in ["1", "1.0", 1, "nan", "na"]:
        with pytest.raises(ContractError):
            validate_predictions(spec, ["002", "001"], result([label, "NA"], ["001", "NA"]))


@pytest.mark.parametrize("label", ["nan", "inf", "-inf", "unknown"])
def test_numeric_class_rejects_nonfinite_and_invalid_strings(binary: TaskSpec, label: str) -> None:
    spec = replace(
        binary, classes=(0, 1), metric=Metric.ACCURACY, prediction_kind=PredictionKind.LABELS
    )
    with pytest.raises(ContractError):
        validate_predictions(spec, ["002", "001"], result([label, 1], [0, 1]))
