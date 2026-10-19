"""Final quality comparisons must be paired and must never open holdout early."""

import importlib
from pathlib import Path

import pytest


@pytest.fixture
def comparison(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]))
    return importlib.import_module("verification.run_agent_comparison")


def pair(name, metric, qwen, numeric):
    return [
        {
            "task": name,
            "seed": 42,
            "mode": mode,
            "protocol": "frozen",
            "quality": {"metric": metric, "value": value},
        }
        for mode, value in [("qwen", qwen), ("optuna", numeric)]
    ]


def test_pair_direction_and_benefit_gate(comparison):
    report = comparison.paired_results(
        pair("regression", "rmse", 9, 10)
        + pair("classification", "macro_f1", 0.6, 0.5)
        + pair("diabetes", "rmse", 10.5, 10)
    )
    assert report["task_mean_relative_gains"]["regression"] == pytest.approx(0.1)
    assert report["task_mean_relative_gains"]["classification"] == pytest.approx(0.2)
    assert report["benefit_gate_passed"]


def test_incomplete_pair_and_mismatched_protocol_rejected(comparison):
    rows = pair("regression", "rmse", 9, 10)
    with pytest.raises(ValueError, match="complete"):
        comparison.paired_results(rows[:1])
    rows[1]["protocol"] = "other"
    with pytest.raises(ValueError, match="protocol"):
        comparison.paired_results(rows)


def test_incomplete_cohort_never_opens_holdout(tmp_path, comparison):
    source = tmp_path / "diabetes/task.md"
    with pytest.raises(ValueError, match="entire cohort"):
        comparison.complete_holdout([], [source], 18)
