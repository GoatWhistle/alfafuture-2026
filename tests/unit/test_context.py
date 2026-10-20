"""Bounded context selects comparable evidence without copying raw artifacts."""

import json
from dataclasses import replace
from typing import Any

import pytest

from ml_harness.application.use_cases.context import build_context
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


@pytest.fixture
def task() -> PreparedTask:
    spec = TaskSpec(
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
    protocol = ValidationProtocol("split", "data", SplitKind.RANDOM, 1, (0,), (1,))
    return PreparedTask(spec, protocol, 2, 1)


def report(identifier: str, value: float = 1, **changes: Any) -> dict[str, Any]:
    return {
        "id": identifier,
        "protocol_id": "split",
        "level": "full",
        "status": "succeeded",
        "measured_by": "harness",
        "hypothesis": "try another model",
        "expected_effect": "reduce error",
        "llm_commentary": "interpretation only",
        "full_score": {"value": value, "metric": "rmse", "protocol_id": "split", "level": "full"},
    } | changes


def baseline(task: PreparedTask) -> Score:
    return Score(1, task.spec.metric, "split")


def test_parent_best_failure_and_alternative_are_selected(task: PreparedTask) -> None:
    reports = [
        report("best", 0.1),
        report("parent", 0.4),
        report("alternative", 0.5),
        report("failure", status="failed", error="training crashed", full_score=None),
    ]
    context = json.loads(build_context(task, baseline(task), reports, parent_id="parent"))
    summaries = {r["id"]: r for r in context["reports"]}
    assert set(summaries) == {"best", "parent", "alternative", "failure"}
    assert summaries["parent"]["role"] == "parent"
    assert summaries["best"]["role"] == "best"
    assert summaries["failure"]["error"] == "training crashed"
    assert summaries["best"]["measured"]["value"] == 0.1
    assert summaries["best"]["interpretation"] == "interpretation only"


@pytest.mark.parametrize(
    "metric,kind,prediction,classes,expected",
    [
        (Metric.RMSE, TaskKind.REGRESSION, PredictionKind.VALUES, (), "low"),
        (Metric.ACCURACY, TaskKind.BINARY, PredictionKind.LABELS, ("a", "b"), "high"),
    ],
)
def test_metric_direction(
    task: PreparedTask,
    metric: Metric,
    kind: TaskKind,
    prediction: PredictionKind,
    classes: tuple[str, ...],
    expected: str,
) -> None:
    task = replace(
        task,
        spec=replace(
            task.spec, metric=metric, kind=kind, prediction_kind=prediction, classes=classes
        ),
    )
    reports = [report("low", 0.1), report("high", 0.9)]
    for item in reports:
        item["full_score"]["metric"] = metric.value
    summaries = json.loads(build_context(task, baseline(task), reports))["reports"]
    assert next(r["id"] for r in summaries if r["role"] == "best") == expected


@pytest.mark.parametrize(
    "change",
    [
        {"protocol_id": "other"},
        {"level": "smoke"},
        {"measured_by": "llm"},
        {"full_score": {"value": -1000, "metric": "mae", "protocol_id": "split", "level": "full"}},
        {"full_score": {"value": -1000, "metric": "rmse", "protocol_id": "other", "level": "full"}},
        {
            "full_score": {
                "value": -1000,
                "metric": "rmse",
                "protocol_id": "split",
                "level": "smoke",
            }
        },
        {
            "full_score": {
                "value": float("nan"),
                "metric": "rmse",
                "protocol_id": "split",
                "level": "full",
            }
        },
        {"full_score": None},
    ],
)
def test_incomparable_or_unmeasured_results_never_rank(
    task: PreparedTask, change: dict[str, Any]
) -> None:
    text = build_context(task, baseline(task), [report("good"), report("bad", -1000, **change)])
    summaries = json.loads(text)["reports"]
    assert next(r["id"] for r in summaries if r["role"] == "best") == "good"
    assert "bad" not in [r["id"] for r in summaries]


def test_smoke_score_cannot_replace_missing_full_score(task: PreparedTask) -> None:
    smoke = report("smoke", full_score=None, smoke_score={"value": -10000})
    assert json.loads(build_context(task, baseline(task), [smoke]))["reports"] == []


@pytest.mark.parametrize("limit", [512, 700, 2000, 12000])
def test_huge_history_and_task_remain_strictly_bounded(task: PreparedTask, limit: int) -> None:
    spec = replace(
        task.spec,
        target="target" * 1000,
        submission_columns=("id", "target" * 1000),
        features=tuple("feature" + str(i) + "x" * 1000 for i in range(1000)),
    )
    task = replace(task, spec=spec)
    reports = [
        report(
            str(i),
            float(i),
            hypothesis="h" * 20000,
            expected_effect="e" * 20000,
            error="error" * 20000,
            llm_commentary="comment" * 20000,
        )
        for i in range(1000)
    ]
    text = build_context(
        task, baseline(task), reports, max_chars=limit, budget={"remaining": "x" * 10000}
    )
    assert len(text) <= limit
    parsed = json.loads(text)
    assert parsed["task"]["features_omitted"] > 0
    assert parsed["task"]["metric"] == "rmse"


def test_many_classes_are_explicitly_omitted(task: PreparedTask) -> None:
    spec = replace(
        task.spec,
        kind=TaskKind.MULTICLASS,
        metric=Metric.ACCURACY,
        prediction_kind=PredictionKind.LABELS,
        classes=tuple(range(1000)),
    )
    task = replace(task, spec=spec)
    parsed = json.loads(build_context(task, baseline(task), [], max_chars=512))
    assert parsed["task"]["classes_omitted"] > 0
    assert len(parsed["task"]["classes"]) + parsed["task"]["classes_omitted"] == 1000


def test_logs_code_and_artifacts_are_not_copied(task: PreparedTask) -> None:
    item = report("best", code="SECRET_CODE", stdout="SECRET_LOG", artifacts=["SECRET_ARTIFACT"])
    text = build_context(task, baseline(task), [item], budget={"tokens_left": 100, "bad": []})
    assert "SECRET_" not in text
    assert json.loads(text)["budget"] == {"tokens_left": 100}
    assert item["code"] == "SECRET_CODE"


@pytest.mark.parametrize("limit", [0, 511, -1, True])
def test_invalid_context_limit(task: PreparedTask, limit: int) -> None:
    with pytest.raises(ContractError):
        build_context(task, baseline(task), [], max_chars=limit)


def test_recent_failure_and_timeout_are_kept(task: PreparedTask) -> None:
    reports = [
        report("old", status="failed"),
        report("parent"),
        report("failure", status="failed"),
        report("timeout", status="timed_out"),
    ]
    summaries = json.loads(build_context(task, baseline(task), reports, parent_id="parent"))[
        "reports"
    ]
    assert {"parent", "failure", "timeout"} <= {r["id"] for r in summaries}


def test_compact_report_distinguishes_selected_classes_from_full_metric_changes():
    from ml_harness.application.use_cases.observations import experiment_observation

    rows = [
        {
            "class": str(i),
            "support": 100,
            "sufficient_support": True,
            "f1": 0.1 + i * 0.07,
            "recall": 0.2,
            "delta": {"precision": -0.1 if i < 7 else 0.1, "recall": 0.1, "f1": 0.01},
        }
        for i in range(10)
    ]
    compact = experiment_observation(
        {
            "id": "measured",
            "status": "succeeded",
            "measured_by": "harness",
            "parent_backend": "catboost",
            "hypothesis_source": "tool_call",
            "error_report": {"class_report": rows, "measured_by": "harness"},
        }
    )
    measured = compact["measured_errors"]
    assert len(measured["target_errors"]) == 3
    assert measured["focus_scope"] == {"selected": 3, "classes_total": 10, "exhaustive": False}
    assert measured["class_metric_changes"]["precision"] == {
        "increased": 3,
        "decreased": 7,
        "unchanged": 0,
        "unavailable": 0,
    }
    assert compact["parent_backend"] == "catboost"
    assert compact["hypothesis_source"] == "tool_call"


def test_final_bound_preserves_contract_and_late_feature_description():
    from ml_harness.application.use_cases.context import bound_context

    evidence = [{"key": "segment:late", "definition": {"feature": "late"}, "support": 50}]
    state = {
        "task": {"metric": "rmse"},
        "budget": {"seconds_remaining": 100},
        "parent_recipe": {"backend": "catboost", "parameters": {"depth": 3}},
        "selected_evidence": evidence,
        "dataset_profile": {
            "fit_partition": "train",
            "columns": {
                **{f"x{i}": {"dictionary": {"description": "unused"}} for i in range(200)},
                "late": {"dictionary": {"description": "important late feature"}},
            },
        },
        "last_measurements": ["x" * 50000],
    }
    result = bound_context(state, max_chars=2000)
    assert len(result) <= 2000
    parsed = json.loads(result)
    for key in ("task", "budget", "parent_recipe", "selected_evidence"):
        assert parsed[key] == state[key]
    assert (
        parsed["dataset_profile"]["details"]["late"]["dictionary"]["description"]
        == "important late feature"
    )
