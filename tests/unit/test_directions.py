"""Branch selection is based on comparable measurements, never LLM claims."""

from dataclasses import replace

import pytest

from ml_harness.application.use_cases.directions import choose_direction
from ml_harness.domain.models import (
    ContractError,
    Metric,
    PredictionKind,
    PreparedTask,
    Score,
    SplitKind,
    TaskKind,
    TaskSpec,
    ValidationProtocol,
)


def task():
    spec = TaskSpec(
        "task",
        "train",
        "test",
        "dict",
        "sample",
        "output",
        "y",
        "id",
        ("x",),
        TaskKind.REGRESSION,
        Metric.RMSE,
        PredictionKind.VALUES,
        ("id", "y"),
    )
    return PreparedTask(
        spec, ValidationProtocol("protocol", "data", SplitKind.RANDOM, 42, (0, 1), (2, 3)), 4, 2
    )


def report(identifier, score, segment_loss, **overrides):
    return {
        "id": identifier,
        "status": "succeeded",
        "level": "full",
        "protocol_id": "protocol",
        "measured_by": "harness",
        "full_score": {
            "value": score,
            "metric": "rmse",
            "protocol_id": "protocol",
            "level": "full",
        },
        "error_report": {
            "measured_by": "harness",
            "level": "full",
            "protocol_id": "protocol",
            "segments": [
                {
                    "definition": {"feature": "x", "kind": "category", "value": "large"},
                    "support": 40,
                    "metrics": {"mae": segment_loss},
                }
            ],
        },
        **overrides,
    }


def test_best_parent_keeps_segment_specific_alternative_and_branches():
    prepared = task()
    evidence = [
        report("best", 1, 5),
        report("segment_specialist", 2, 1),
        {"id": "failure", "status": "failed"},
    ]
    direction = choose_direction(prepared, Score(3, Metric.RMSE, "protocol"), evidence)
    assert direction.parent_id == "best"
    assert direction.alternatives == ("segment_specialist",)
    assert direction.recent_failures == ("failure",)
    assert direction.target_errors[0]["loss"] == 5
    branch = choose_direction(
        prepared, Score(3, Metric.RMSE, "protocol"), evidence, session_index=2
    )
    assert branch.parent_id == "segment_specialist" and branch.best_id == "best"


@pytest.mark.parametrize(
    "invalid",
    [
        {"level": "smoke"},
        {"measured_by": "llm"},
        {"status": "failed"},
        {"full_score": {"value": 0, "metric": "rmse", "protocol_id": "other", "level": "full"}},
    ],
)
def test_incomparable_or_unmeasured_claims_cannot_win(invalid):
    prepared = task()
    candidate = report("claim", 0, 0, **invalid)
    direction = choose_direction(prepared, Score(3, Metric.RMSE, "protocol"), [candidate])
    assert direction.parent_id == "baseline" and direction.alternatives == ()


def test_maximize_metric_direction():
    prepared = task()
    spec = replace(
        prepared.spec,
        kind=TaskKind.BINARY,
        metric=Metric.MACRO_F1,
        prediction_kind=PredictionKind.LABELS,
        classes=(0, 1),
    )
    prepared = replace(prepared, spec=spec)
    a = report("high", 0.8, 2)
    b = report("low", 0.6, 2)
    for item in (a, b):
        item["full_score"]["metric"] = "macro_f1"
    assert (
        choose_direction(prepared, Score(0.5, Metric.MACRO_F1, "protocol"), [a, b]).parent_id
        == "high"
    )


def test_baseline_comparability_is_required():
    with pytest.raises(ContractError):
        choose_direction(task(), Score(1, Metric.RMSE, "different"), [])
