"""Paired held-out diagnostics with segment definitions learned on training rows."""

import hashlib
import json
from collections import Counter
from dataclasses import asdict
from math import isfinite, sqrt
from pathlib import Path
from typing import Any, cast

from ml_harness.adapters.data.task import read_csv
from ml_harness.adapters.ml.evaluation import _class_index, evaluate, validate_predictions
from ml_harness.domain.models import (
    ContractError,
    PredictionKind,
    PreparedTask,
    TaskKind,
    TaskSpec,
    ValidationProtocol,
)

MAX_FEATURES = 20
MAX_CATEGORIES = 8


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _missing(value: str) -> bool:
    return value == ""


def _numeric(value: str) -> float | None:
    try:
        number = float(value)
        return number if isfinite(number) else None
    except ValueError:
        return None


def _losses(errors: list[float]) -> dict[str, float]:
    size = len(errors)
    return {
        "rmse": sqrt(sum(e * e for e in errors) / size),
        "mae": sum(abs(e) for e in errors) / size,
        "bias": sum(errors) / size,
    }


def _classification(targets: list[int], predictions: list[int], count: int) -> dict[str, Any]:
    matrix = [[0] * count for _ in range(count)]
    for target, prediction in zip(targets, predictions, strict=True):
        matrix[target][prediction] += 1
    perclass = []
    for index in range(count):
        support = sum(matrix[index])
        predicted = sum(row[index] for row in matrix)
        correct = matrix[index][index]
        precision = correct / predicted if predicted else 0.0
        recall = correct / support if support else 0.0
        perclass.append(
            {
                "support": support,
                "precision": precision,
                "recall": recall,
                "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            }
        )
    return {"confusion_matrix": matrix, "class_report": perclass}


def _definitions(
    feature: str,
    train: list[dict[str, str]],
) -> list[dict[str, Any]]:
    present = [row[feature] for row in train if not _missing(row[feature])]
    numbers = [_numeric(value) for value in present]
    definitions: list[dict[str, Any]] = [{"feature": feature, "kind": "missing"}]
    if numbers and all(value is not None for value in numbers):
        ordered = sorted(cast(list[float], numbers))
        cuts = sorted({ordered[min(len(ordered) - 1, len(ordered) * q // 4)] for q in (1, 2, 3)})
        bounds: list[float | None] = [None, *cuts, None]
        for lower, upper in zip(bounds[:-1], bounds[1:], strict=True):
            definitions.append(
                {
                    "feature": feature,
                    "kind": "numeric",
                    "lower": lower,
                    "upper": upper,
                    "interval": "(lower,upper]",
                }
            )
    else:
        counts = Counter(present)
        for value, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[
            :MAX_CATEGORIES
        ]:
            definitions.append({"feature": feature, "kind": "category", "value": value})
    return definitions


def _matches(row: dict[str, str], definition: dict[str, Any]) -> bool:
    value = row[definition["feature"]]
    if definition["kind"] == "missing":
        return _missing(value)
    if _missing(value):
        return False
    if definition["kind"] == "category":
        return bool(value == definition["value"])
    numeric = _numeric(value)
    return (
        numeric is not None
        and (definition["lower"] is None or numeric > definition["lower"])
        and (definition["upper"] is None or numeric <= definition["upper"])
    )


def analyze_errors(
    spec: TaskSpec,
    protocol: ValidationProtocol,
    rows: list[dict[str, str]],
    result: dict[str, Any],
    parent_result: dict[str, Any] | None = None,
    *,
    min_support: int = 30,
) -> dict[str, Any]:
    """Compare errors on identical ordered validation rows; do not infer causality."""
    if protocol.level != "full":
        raise ContractError("Error analysis requires full validation")
    if isinstance(min_support, bool) or not isinstance(min_support, int) or min_support < 1:
        raise ContractError("Minimum diagnostic support must be a positive integer")
    score = evaluate(spec, protocol, rows, result)
    parent_score = (
        evaluate(spec, protocol, rows, parent_result) if parent_result is not None else None
    )
    try:
        validation = [rows[index] for index in protocol.validation_indices]
        train = [rows[index] for index in protocol.train_indices]
        ids = [row[spec.id_column] for row in validation]
        predictions = validate_predictions(spec, ids, result)
        parent_predictions = (
            validate_predictions(spec, ids, parent_result) if parent_result is not None else None
        )
        output: dict[str, Any] = {
            "measured_by": "harness",
            "protocol_id": protocol.fingerprint,
            "level": "full",
            "validation_ids_hash": _hash(ids),
            "overall_score": asdict(score),
            "parent_score": asdict(parent_score) if parent_score is not None else None,
            "delta": (
                {
                    "value": score.value - parent_score.value,
                    "improved": score.better_than(parent_score),
                }
                if parent_score is not None
                else None
            ),
            "observations": ["Measured on the same ordered validation rows."],
            "interpretation": None,
            "limits": {
                "min_support": min_support,
                "max_features": MAX_FEATURES,
                "max_categories_per_feature": MAX_CATEGORIES,
                "numeric_bins": 4,
                "segment_fit_partition": "train",
            },
        }
        if spec.kind != TaskKind.REGRESSION:
            targets = [_class_index(row[spec.target], spec.classes) for row in validation]

            def labels(values: list[Any]) -> list[int]:
                if spec.prediction_kind == PredictionKind.PROBABILITIES:
                    return [max(range(len(spec.classes)), key=lambda i: row[i]) for row in values]
                return [_class_index(value, spec.classes) for value in values]

            diagnostic = _classification(targets, labels(predictions), len(spec.classes))
            output.update(classes=list(spec.classes), **diagnostic)
            output["diagnostic_prediction_rule"] = (
                "argmax in spec class order"
                if spec.prediction_kind == PredictionKind.PROBABILITIES
                else "labels"
            )
            for index, item in enumerate(output["class_report"]):
                item["class"] = spec.classes[index]
                item["sufficient_support"] = item["support"] >= min_support
            if parent_predictions is not None:
                parent_labels = labels(parent_predictions)
                child_labels = labels(predictions)
                parent = _classification(targets, parent_labels, len(spec.classes))
                output["parent_confusion_matrix"] = parent["confusion_matrix"]
                paired = {"corrected": 0, "worsened": 0, "unchanged": 0}
                for target, previous_label, child_label in zip(
                    targets,
                    parent_labels,
                    child_labels,
                    strict=True,
                ):
                    previous_correct = previous_label == target
                    child_correct = child_label == target
                    if child_correct and not previous_correct:
                        paired["corrected"] += 1
                    elif previous_correct and not child_correct:
                        paired["worsened"] += 1
                    else:
                        paired["unchanged"] += 1
                output["paired_changes"] = paired
                output["confusion_matrix_delta"] = [
                    [
                        child - previous
                        for child, previous in zip(child_row, parent_row, strict=True)
                    ]
                    for child_row, parent_row in zip(
                        output["confusion_matrix"],
                        parent["confusion_matrix"],
                        strict=True,
                    )
                ]
                for item, previous in zip(
                    output["class_report"], parent["class_report"], strict=True
                ):
                    item["parent"] = previous
                    item["delta"] = {
                        key: item[key] - previous[key] for key in ("precision", "recall", "f1")
                    }
        else:
            targets_float = [float(row[spec.target]) for row in validation]
            errors = [
                float(value) - target
                for value, target in zip(cast(list[float], predictions), targets_float, strict=True)
            ]
            parent_errors = (
                [
                    float(value) - target
                    for value, target in zip(
                        cast(list[float], parent_predictions), targets_float, strict=True
                    )
                ]
                if parent_predictions is not None
                else None
            )
            output["regression"] = _losses(errors)
            if parent_errors is not None:
                output["parent_regression"] = _losses(parent_errors)
                output["regression_delta"] = {
                    key: value - output["parent_regression"][key]
                    for key, value in output["regression"].items()
                }
            features = [f for f in spec.features if f not in (spec.target, spec.id_column)]
            selected = features[:MAX_FEATURES]
            missing = [f for f in selected if any(f not in row for row in train + validation)]
            definitions = [
                definition
                for feature in selected
                if feature not in missing
                for definition in _definitions(feature, train)
            ]
            output["segment_definitions"] = definitions
            output["segment_fingerprint"] = _hash(definitions)
            output["limits"].update(
                features_omitted=max(0, len(features) - MAX_FEATURES), missing_features=missing
            )
            segments = []
            suppressed = 0
            for definition in definitions:
                train_support = sum(_matches(row, definition) for row in train)
                positions = [i for i, row in enumerate(validation) if _matches(row, definition)]
                if train_support < min_support or len(positions) < min_support:
                    suppressed += 1
                    continue
                losses = _losses([errors[i] for i in positions])
                segment = {
                    "definition": definition,
                    "train_support": train_support,
                    "support": len(positions),
                    "metrics": losses,
                }
                if parent_errors is not None:
                    previous = _losses([parent_errors[i] for i in positions])
                    segment["parent_metrics"] = previous
                    segment["delta"] = {key: value - previous[key] for key, value in losses.items()}
                segments.append(segment)
            output["segments"] = segments
            output["limits"]["segments_suppressed"] = suppressed
        json.dumps(output, allow_nan=False)
        return output
    except (IndexError, KeyError, TypeError, ValueError, OverflowError) as error:
        raise ContractError(f"Cannot analyze errors: {type(error).__name__}") from error


class IndependentErrorAnalyzer:
    def __init__(self, min_support: int = 30) -> None:
        self.min_support = min_support

    def analyze(
        self,
        task: PreparedTask,
        output: dict[str, Any],
        parent_output: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        _, rows = read_csv(Path(task.spec.train_path))
        return analyze_errors(
            task.spec, task.protocol, rows, output, parent_output, min_support=self.min_support
        )
