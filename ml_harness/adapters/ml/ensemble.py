"""Fixed averaging/voting with explicit ID and class alignment."""

from math import isfinite
from typing import Any

from ml_harness.adapters.ml.evaluation import validate_predictions
from ml_harness.domain.models import ContractError, PredictionKind, TaskSpec


def combine_predictions(
    spec: TaskSpec,
    expected_ids: list[str],
    outputs: list[dict[str, Any]],
    weights: tuple[float, ...] | None = None,
) -> dict[str, Any]:
    if not outputs:
        raise ContractError("An ensemble requires members")
    if not expected_ids or any(not isinstance(item, str) for item in expected_ids):
        raise ContractError("Ensemble requires string IDs")
    if len(set(expected_ids)) != len(expected_ids):
        raise ContractError("Ensemble expected IDs repeat")
    coefficients = weights if weights is not None else tuple(1 / len(outputs) for _ in outputs)
    if len(coefficients) != len(outputs) or any(
        isinstance(w, bool) or not isinstance(w, int | float) or not isfinite(w) or w <= 0
        for w in coefficients
    ):
        raise ContractError("Ensemble weights must be positive and finite")
    total = sum(coefficients)
    if not isfinite(total):
        raise ContractError("Ensemble weight sum must be finite")
    coefficients = tuple(w / total for w in coefficients)
    members = []
    for output in outputs:
        if not isinstance(output, dict):
            raise ContractError("Ensemble member must be a prediction object")
        ids, values = output.get("ids"), output.get("predictions")
        if (
            not isinstance(ids, list)
            or not isinstance(values, list)
            or any(not isinstance(identifier, str) for identifier in ids)
            or len(ids) != len(values)
            or len(set(ids)) != len(ids)
            or set(ids) != set(expected_ids)
        ):
            raise ContractError("Ensemble IDs disagree or repeat")
        positions = {identifier: index for index, identifier in enumerate(ids)}
        aligned = {
            **output,
            "ids": expected_ids,
            "predictions": [values[positions[identifier]] for identifier in expected_ids],
        }
        members.append(validate_predictions(spec, expected_ids, aligned))
    predictions: list[Any] = []
    for row in zip(*members, strict=True):
        if spec.prediction_kind == PredictionKind.PROBABILITIES:
            if not all(isinstance(item, list) for item in row):
                raise ContractError("Expected probability matrices")
            predictions.append(
                [
                    sum(
                        weight * item[column]
                        for weight, item in zip(coefficients, row, strict=True)
                    )
                    for column in range(len(spec.classes))
                ]
            )
        elif spec.prediction_kind == PredictionKind.LABELS:
            votes: dict[Any, float] = {}
            for weight, label in zip(coefficients, row, strict=True):
                votes[label] = votes.get(label, 0.0) + weight
            predictions.append(max(spec.classes, key=lambda label: votes.get(label, 0.0)))
        else:
            predictions.append(
                sum(weight * value for weight, value in zip(coefficients, row, strict=True))
            )
    result = {"ids": expected_ids, "predictions": predictions, "classes": list(spec.classes)}
    validate_predictions(spec, expected_ids, result)
    return result
