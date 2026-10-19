"""Fixed ensembles align IDs and classes before numerical combination."""

from dataclasses import replace

import pytest

from ml_harness.adapters.ml.ensemble import combine_predictions
from ml_harness.domain.models import ContractError, Metric, PredictionKind, TaskKind, TaskSpec


def spec():
    return TaskSpec(
        "task",
        "train",
        "test",
        "dict",
        "sample",
        "submission",
        "y",
        "id",
        ("x",),
        TaskKind.REGRESSION,
        Metric.RMSE,
        PredictionKind.VALUES,
        ("id", "y"),
    )


def test_average_aligns_ids():
    result = combine_predictions(
        spec(),
        ["001", "NA"],
        [
            {"ids": ["001", "NA"], "predictions": [1, 4], "classes": []},
            {"ids": ["NA", "001"], "predictions": [8, 3], "classes": []},
        ],
    )
    assert result == {"ids": ["001", "NA"], "predictions": [2, 6], "classes": []}


def test_probability_average_aligns_classes_and_ids():
    task = replace(
        spec(),
        kind=TaskKind.BINARY,
        metric=Metric.LOG_LOSS,
        prediction_kind=PredictionKind.PROBABILITIES,
        classes=("z", "a"),
    )
    result = combine_predictions(
        task,
        ["1", "2"],
        [
            {"ids": ["1", "2"], "predictions": [[0.2, 0.8], [0.4, 0.6]], "classes": ["z", "a"]},
            {"ids": ["2", "1"], "predictions": [[0.2, 0.8], [0.6, 0.4]], "classes": ["a", "z"]},
        ],
    )
    assert result["predictions"][0] == pytest.approx([0.3, 0.7])
    assert result["predictions"][1] == pytest.approx([0.6, 0.4])


def test_label_vote_tie_uses_explicit_class_order():
    task = replace(
        spec(),
        kind=TaskKind.BINARY,
        metric=Metric.ACCURACY,
        prediction_kind=PredictionKind.LABELS,
        classes=(3.5, 1.5),
    )
    result = combine_predictions(
        task,
        ["1"],
        [
            {"ids": ["1"], "predictions": [1.5], "classes": [1.5, 3.5]},
            {"ids": ["1"], "predictions": [3.5], "classes": [3.5, 1.5]},
        ],
    )
    assert result["predictions"] == [3.5]


@pytest.mark.parametrize(
    "output",
    [
        {"ids": ["1", "1"], "predictions": [1, 2], "classes": []},
        {"ids": ["1", "wrong"], "predictions": [1, 2], "classes": []},
        {"ids": [["1"], "2"], "predictions": [1, 2], "classes": []},
        {"ids": ["1", "2"], "predictions": [1, float("inf")], "classes": []},
    ],
)
def test_invalid_members_have_structured_errors(output):
    with pytest.raises(ContractError):
        combine_predictions(spec(), ["1", "2"], [output])


@pytest.mark.parametrize("weights", [(), (-1.0,), (float("nan"),), (True,), ("invalid",)])
def test_invalid_weights_have_structured_errors(weights):
    with pytest.raises(ContractError):
        combine_predictions(
            spec(), ["1"], [{"ids": ["1"], "predictions": [1], "classes": []}], weights
        )
