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


def test_verify_run_counts_generated_baseline_prefix_by_provenance(
    tmp_path, monkeypatch, comparison
):
    import json

    from test_prepare_task import task_fixture
    from test_research_workspace import ScriptedModel

    import ml_harness.entrypoints.research as entrypoint
    from ml_harness.adapters.data.task import CsvTaskData
    from ml_harness.adapters.research.workspace import STARTER
    from ml_harness.application.use_cases.baseline import BaselineResult
    from ml_harness.domain.models import Metric, Score

    task_path = task_fixture(tmp_path)
    task = CsvTaskData().prepare(str(task_path), 42)
    baseline = BaselineResult(
        Score(1000, Metric.RMSE, task.protocol.fingerprint), task.spec.submission_path, 1, 1
    )
    model = ScriptedModel(
        STARTER.replace('"iterations": 200', '"iterations": 3'),
        experiment_id="baseline-agent-change",
    )
    monkeypatch.setattr(entrypoint, "QwenLanguageModel", lambda *args: model)
    for name in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL"):
        monkeypatch.setenv(name, "synthetic-qwen")
    harness = tmp_path / "harness"
    report = entrypoint.research_after_baseline(
        task,
        baseline,
        harness / "research",
        threads=1,
        timeout_seconds=10,
        max_steps=6,
        token_limit=32000,
        max_sessions=1,
    )
    (harness / "summary.json").write_text(json.dumps({"research": report}))
    verified = comparison.verify_run(tmp_path, task_path, 42)
    assert verified["refit_verified"]
    assert verified["experiments"] == 1
    assert verified["changed_successes"] == 1
    assert verified["interpreted_successes"] == 1
