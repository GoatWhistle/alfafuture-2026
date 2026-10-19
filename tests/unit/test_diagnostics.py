"""Diagnostic measurements use paired rows and train-only segment definitions."""

import csv
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from ml_harness.adapters.ml.diagnostics import IndependentErrorAnalyzer, analyze_errors
from ml_harness.domain.models import (
    ContractError,
    Metric,
    PredictionKind,
    PreparedTask,
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
        ("x", "cat"),
        TaskKind.REGRESSION,
        Metric.RMSE,
        PredictionKind.VALUES,
        ("id", "y"),
    )


@pytest.fixture
def protocol() -> ValidationProtocol:
    return ValidationProtocol("split", "data", SplitKind.RANDOM, 42, (0, 1, 2, 3), (4, 5, 6, 7))


@pytest.fixture
def rows() -> list[dict[str, str]]:
    return [
        {"id": str(i), "y": "1", "x": str(i % 4), "cat": "a" if i % 2 else "b"} for i in range(8)
    ]


def worker(predictions: list[Any], classes: list[Any] | None = None) -> dict[str, Any]:
    return {"ids": ["4", "5", "6", "7"], "predictions": predictions, "classes": classes or []}


def test_regression_global_segment_and_paired_deltas(
    spec: TaskSpec,
    protocol: ValidationProtocol,
    rows: list[dict[str, str]],
) -> None:
    result = analyze_errors(
        spec, protocol, rows, worker([1, 2, 1, 2]), worker([3] * 4), min_support=1
    )
    assert result["regression"] == {"rmse": pytest.approx(2**-0.5), "mae": 0.5, "bias": 0.5}
    assert result["parent_regression"]["mae"] == 2
    assert result["regression_delta"]["bias"] == -1.5
    assert result["delta"]["improved"] is True
    assert result["delta"]["value"] < 0
    assert result["segments"]
    for segment in result["segments"]:
        assert segment["train_support"] >= 1
        assert segment["support"] >= 1
        assert segment["delta"]["mae"] <= 0
    assert result["interpretation"] is None
    assert result["measured_by"] == "harness"
    assert "caus" not in " ".join(result["observations"]).lower()
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize(
    "classes,targets",
    [
        (("z", "a"), ["z", "a", "z", "a"]),
        ((2, -1.5), ["2.0", "-1.50", "2", "-1.5"]),
    ],
)
def test_class_order_numeric_strings_and_label_deltas(
    spec: TaskSpec,
    protocol: ValidationProtocol,
    rows: list[dict[str, str]],
    classes: tuple[str | int | float, ...],
    targets: list[str],
) -> None:
    spec = replace(
        spec,
        kind=TaskKind.BINARY,
        metric=Metric.ACCURACY,
        prediction_kind=PredictionKind.LABELS,
        classes=classes,
    )
    for row, target in zip(rows[4:], targets, strict=True):
        row["y"] = target
    result = analyze_errors(
        spec,
        protocol,
        rows,
        worker([classes[0], classes[1], classes[0], classes[1]], list(classes)),
        worker([classes[1]] * 4, list(classes)),
        min_support=1,
    )
    assert result["classes"] == list(classes)
    assert result["confusion_matrix"] == [[2, 0], [0, 2]]
    assert result["parent_confusion_matrix"] == [[0, 2], [0, 2]]
    assert result["class_report"][0]["support"] == 2
    assert result["class_report"][0]["f1"] == 1
    assert result["class_report"][0]["delta"]["recall"] == 1
    assert result["delta"] == {"value": 0.5, "improved": True}


def test_probability_argmax_is_only_diagnostic(
    spec: TaskSpec,
    protocol: ValidationProtocol,
    rows: list[dict[str, str]],
) -> None:
    spec = replace(
        spec,
        kind=TaskKind.BINARY,
        metric=Metric.LOG_LOSS,
        prediction_kind=PredictionKind.PROBABILITIES,
        classes=("z", "a"),
    )
    for row, target in zip(rows[4:], ["z", "a", "z", "a"], strict=True):
        row["y"] = target
    result = analyze_errors(
        spec,
        protocol,
        rows,
        worker([[0.2, 0.8], [0.8, 0.2], [0.1, 0.9], [0.9, 0.1]], ["a", "z"]),
        min_support=1,
    )
    assert result["confusion_matrix"] == [[2, 0], [0, 2]]
    assert result["overall_score"]["value"] > 0
    assert result["diagnostic_prediction_rule"] == "argmax in spec class order"


def test_multiclass_confusion_has_all_classes_in_spec_order(
    spec: TaskSpec,
    protocol: ValidationProtocol,
    rows: list[dict[str, str]],
) -> None:
    spec = replace(
        spec,
        kind=TaskKind.MULTICLASS,
        metric=Metric.ACCURACY,
        prediction_kind=PredictionKind.LABELS,
        classes=("z", "a", "m"),
    )
    for row, target in zip(rows[4:], ["z", "a", "m", "z"], strict=True):
        row["y"] = target
    result = analyze_errors(spec, protocol, rows, worker(["a", "a", "m", "z"], ["m", "z", "a"]))
    assert result["confusion_matrix"] == [[1, 1, 0], [0, 1, 0], [0, 0, 1]]
    assert not result["class_report"][0]["sufficient_support"]


def test_rare_segments_suppressed_on_both_partitions(
    spec: TaskSpec,
    protocol: ValidationProtocol,
    rows: list[dict[str, str]],
) -> None:
    rows[0]["cat"] = "rare_train"
    rows[4]["cat"] = "rare_validation"
    result = analyze_errors(spec, protocol, rows, worker([1] * 4), min_support=2)
    assert all(
        segment["support"] >= 2 and segment["train_support"] >= 2 for segment in result["segments"]
    )
    assert not any(
        segment["definition"].get("value", "").startswith("rare") for segment in result["segments"]
    )
    assert result["limits"]["segments_suppressed"] > 0


def test_validation_extremes_do_not_change_train_bins(
    spec: TaskSpec,
    protocol: ValidationProtocol,
    rows: list[dict[str, str]],
) -> None:
    first = analyze_errors(spec, protocol, rows, worker([1] * 4), min_support=1)
    for row in rows[4:]:
        row["x"] = "999999999"
        row["cat"] = "validation_only"
    second = analyze_errors(spec, protocol, rows, worker([1] * 4), min_support=1)
    assert first["segment_definitions"] == second["segment_definitions"]
    assert first["segment_fingerprint"] == second["segment_fingerprint"]
    assert "validation_only" not in json.dumps(second["segment_definitions"])


def test_missing_values_and_missing_feature_are_explicit(
    spec: TaskSpec,
    protocol: ValidationProtocol,
    rows: list[dict[str, str]],
) -> None:
    rows[0]["x"] = ""
    rows[4]["x"] = ""
    del rows[1]["cat"]
    result = analyze_errors(spec, protocol, rows, worker([1] * 4), min_support=1)
    assert result["limits"]["missing_features"] == ["cat"]
    assert any(segment["definition"]["kind"] == "missing" for segment in result["segments"])


def test_feature_and_category_limits_are_explicit(
    spec: TaskSpec,
    protocol: ValidationProtocol,
    rows: list[dict[str, str]],
) -> None:
    spec = replace(spec, features=tuple(f"x{i}" for i in range(30)))
    for row in rows:
        for feature in spec.features:
            row[feature] = "a"
    result = analyze_errors(spec, protocol, rows, worker([1] * 4), min_support=1)
    assert result["limits"]["features_omitted"] == 10
    assert len({d["feature"] for d in result["segment_definitions"]}) == 20
    assert result["limits"]["max_categories_per_feature"] == 8


@pytest.mark.parametrize("parent", [False, True])
def test_mismatched_ids_rejected(
    spec: TaskSpec,
    protocol: ValidationProtocol,
    rows: list[dict[str, str]],
    parent: bool,
) -> None:
    invalid = worker([1] * 4) | {"ids": ["5", "4", "6", "7"]}
    with pytest.raises(ContractError):
        analyze_errors(
            spec,
            protocol,
            rows,
            worker([1] * 4) if parent else invalid,
            invalid if parent else None,
        )


@pytest.mark.parametrize("min_support", [0, -1, True])
def test_invalid_support_rejected(
    spec: TaskSpec,
    protocol: ValidationProtocol,
    rows: list[dict[str, str]],
    min_support: int,
) -> None:
    with pytest.raises(ContractError):
        analyze_errors(spec, protocol, rows, worker([1] * 4), min_support=min_support)


def test_smoke_cannot_produce_full_diagnostics(
    spec: TaskSpec,
    protocol: ValidationProtocol,
    rows: list[dict[str, str]],
) -> None:
    with pytest.raises(ContractError):
        analyze_errors(spec, replace(protocol, level="smoke"), rows, worker([1] * 4))


def test_file_adapter_reads_train(
    spec: TaskSpec,
    protocol: ValidationProtocol,
    rows: list[dict[str, str]],
    tmp_path: Path,
) -> None:
    path = tmp_path / "train.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id", "y", "x", "cat"])
        writer.writeheader()
        writer.writerows(rows)
    task = PreparedTask(replace(spec, train_path=str(path)), protocol, 8, 1)
    result = IndependentErrorAnalyzer(1).analyze(task, worker([1] * 4), worker([2] * 4))
    assert result["regression"]["mae"] == 0
    assert len(result["validation_ids_hash"]) == 64


def test_categorical_top_values_are_capped_and_come_from_training(spec: TaskSpec) -> None:
    protocol = ValidationProtocol(
        "split", "data", SplitKind.RANDOM, 42, tuple(range(12)), tuple(range(12, 24))
    )
    rows = [
        {"id": str(i), "y": "1", "x": str(i % 12), "cat": f"category{i % 12:02d}"}
        for i in range(24)
    ]
    result = analyze_errors(
        spec,
        protocol,
        rows,
        {"ids": [str(i) for i in range(12, 24)], "predictions": [1] * 12, "classes": []},
        min_support=1,
    )
    categories = [d["value"] for d in result["segment_definitions"] if d["kind"] == "category"]
    assert categories == [f"category{i:02d}" for i in range(8)]


def test_unchanged_parent_score_is_not_an_improvement(
    spec: TaskSpec,
    protocol: ValidationProtocol,
    rows: list[dict[str, str]],
) -> None:
    result = analyze_errors(spec, protocol, rows, worker([1] * 4), worker([1] * 4))
    assert result["delta"] == {"value": 0, "improved": False}


def test_literal_na_tokens_are_categories_not_missing(spec: TaskSpec) -> None:
    categories = ["NA", "null", "nan", "none", " ", ""]
    rows = [{"id": str(i), "y": "1", "x": str(i), "cat": categories[i % 6]} for i in range(12)]
    protocol = ValidationProtocol(
        "split", "data", SplitKind.RANDOM, 42, tuple(range(6)), tuple(range(6, 12))
    )
    output = {"ids": [str(i) for i in range(6, 12)], "predictions": [1] * 6, "classes": []}
    result = analyze_errors(spec, protocol, rows, output, min_support=1)
    segments = [
        segment for segment in result["segments"] if segment["definition"]["feature"] == "cat"
    ]
    category_values = {
        segment["definition"]["value"]
        for segment in segments
        if segment["definition"]["kind"] == "category"
    }
    assert category_values == {"NA", "null", "nan", "none", " "}
    missing = next(segment for segment in segments if segment["definition"]["kind"] == "missing")
    assert missing["support"] == missing["train_support"] == 1


def test_paired_classification_changes_count_correctness_transitions(
    spec: TaskSpec,
    protocol: ValidationProtocol,
    rows: list[dict[str, str]],
) -> None:
    spec = replace(
        spec,
        kind=TaskKind.BINARY,
        metric=Metric.ACCURACY,
        prediction_kind=PredictionKind.LABELS,
        classes=("z", "a"),
    )
    for row, target in zip(rows[4:], ["z", "a", "z", "a"], strict=True):
        row["y"] = target
    result = analyze_errors(
        spec,
        protocol,
        rows,
        worker(["z", "z", "z", "z"], ["z", "a"]),
        worker(["a", "a", "z", "z"], ["z", "a"]),
    )
    assert result["paired_changes"] == {"corrected": 1, "worsened": 1, "unchanged": 2}
    assert result["confusion_matrix_delta"] == [[1, -1], [1, -1]]
