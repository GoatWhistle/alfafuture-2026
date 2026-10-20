"""Validate worker predictions and measure them independently of worker output."""

from math import isclose, isfinite
from typing import Any

from sklearn import metrics  # type: ignore[import-untyped]

from ml_harness.domain.models import (
    ContractError,
    Metric,
    PredictionKind,
    Scalar,
    Score,
    TaskKind,
    TaskSpec,
    ValidationProtocol,
)

Prediction = Scalar | list[float]


def _number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not isfinite(value):
        raise ContractError("Predictions must contain finite numbers")
    return float(value)


def _class_index(value: Any, classes: tuple[Scalar, ...]) -> int:
    # CSV values are strings; preserve textual labels while accepting numeric CSV targets.
    matches = [i for i, label in enumerate(classes) if value == label]
    if (
        not matches
        and isinstance(value, str)
        and all(isinstance(label, int | float) and not isinstance(label, bool) for label in classes)
    ):
        try:
            numeric = float(value)
        except ValueError:
            numeric = float("nan")
        if isfinite(numeric):
            matches = [i for i, label in enumerate(classes) if numeric == label]
    if len(matches) != 1:
        raise ContractError(f"Unknown or ambiguous class {value!r}")
    return matches[0]


def validate_predictions(
    spec: TaskSpec,
    expected_ids: list[str],
    result: dict[str, Any],
) -> list[Prediction]:
    """Return scalar predictions or probability rows in the task's class order."""
    ids = result.get("ids")
    if (
        not isinstance(ids, list)
        or any(not isinstance(value, str) for value in ids)
        or ids != expected_ids
        or len(set(ids)) != len(ids)
        or not ids
    ):
        raise ContractError("Prediction IDs must match expected IDs in order without duplicates")
    predictions = result.get("predictions")
    if not isinstance(predictions, list) or len(predictions) != len(ids):
        raise ContractError("Prediction row count disagrees with IDs")
    classes = result.get("classes")
    if not isinstance(classes, list):
        raise ContractError("Worker classes must be a list")
    if spec.prediction_kind == PredictionKind.VALUES:
        if classes:
            raise ContractError("Regression predictions cannot contain classes")
        return [_number(value) for value in predictions]
    order = [_class_index(label, spec.classes) for label in classes]
    if sorted(order) != list(range(len(spec.classes))):
        raise ContractError("Worker classes disagree with task classes")
    if spec.prediction_kind == PredictionKind.LABELS:
        if "probabilities" in result:
            validate_probabilities(spec, expected_ids, result)
        return [spec.classes[_class_index(value, spec.classes)] for value in predictions]
    return _probability_rows(predictions, order)


def validate_probabilities(
    spec: TaskSpec, expected_ids: list[str], result: dict[str, Any]
) -> list[Prediction]:
    """Validate auxiliary class probabilities independently of submission format."""
    if result.get("ids") != expected_ids:
        raise ContractError("Probability IDs must match expected IDs")
    classes = result.get("classes")
    if not isinstance(classes, list):
        raise ContractError("Worker classes must be a list")
    order = [_class_index(label, spec.classes) for label in classes]
    if sorted(order) != list(range(len(spec.classes))):
        raise ContractError("Worker classes disagree with task classes")
    rows = result.get("probabilities")
    if not isinstance(rows, list) or len(rows) != len(expected_ids):
        raise ContractError("Probability row count disagrees with IDs")
    return _probability_rows(rows, order)


def _probability_rows(predictions: list[Any], order: list[int]) -> list[Prediction]:
    output: list[Prediction] = []
    for row in predictions:
        if not isinstance(row, list) or len(row) != len(order):
            raise ContractError("Probability predictions must have shape rows by classes")
        values = [_number(value) for value in row]
        if any(value < 0 or value > 1 for value in values) or not isclose(
            sum(values),
            1.0,
            rel_tol=1e-6,
            abs_tol=1e-6,
        ):
            raise ContractError("Probabilities must lie in [0, 1] and sum to one")
        output.append([values[order.index(index)] for index in range(len(order))])
    return output


def evaluate(
    spec: TaskSpec,
    protocol: ValidationProtocol,
    rows: list[dict[str, str]],
    result: dict[str, Any],
) -> Score:
    """Calculate the declared metric using only held-out targets and predictions."""
    try:
        validation = [rows[index] for index in protocol.validation_indices]
        predictions = validate_predictions(
            spec,
            [row[spec.id_column] for row in validation],
            result,
        )
        raw_targets = [row[spec.target] for row in validation]
        if spec.kind == TaskKind.REGRESSION:
            targets: list[Any] = [float(value) for value in raw_targets]
            if not all(isfinite(value) for value in targets):
                raise ContractError("Targets must be finite")
        else:
            targets = [_class_index(value, spec.classes) for value in raw_targets]
            if spec.prediction_kind == PredictionKind.LABELS:
                predictions = [_class_index(value, spec.classes) for value in predictions]
        if spec.metric == Metric.RMSE:
            value = metrics.root_mean_squared_error(targets, predictions)
        elif spec.metric == Metric.MAE:
            value = metrics.mean_absolute_error(targets, predictions)
        elif spec.metric == Metric.ACCURACY:
            value = metrics.accuracy_score(targets, predictions)
        elif spec.metric == Metric.MACRO_F1:
            value = metrics.f1_score(
                targets,
                predictions,
                average="macro",
                zero_division=0,
                labels=list(range(len(spec.classes))),
            )
        elif spec.metric == Metric.LOG_LOSS:
            value = metrics.log_loss(targets, predictions, labels=list(range(len(spec.classes))))
        elif spec.kind == TaskKind.BINARY:
            positive = [row[1] for row in predictions if isinstance(row, list)]
            value = metrics.roc_auc_score(targets, positive)
        else:
            value = metrics.roc_auc_score(
                targets,
                predictions,
                multi_class="ovr",
                average="macro",
                labels=list(range(len(spec.classes))),
            )
        return Score(float(value), spec.metric, protocol.fingerprint, protocol.level)
    except (IndexError, KeyError, TypeError, ValueError, OverflowError) as error:
        if isinstance(error, ContractError):
            raise
        raise ContractError(f"Cannot evaluate predictions: {error}") from error
