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


def test_verify_run_verifies_structured_proposal_without_llm_summary(
    tmp_path, monkeypatch, comparison
):
    import json

    from test_prepare_task import task_fixture
    from test_research_workspace import ScriptedProposalModel

    import ml_harness.entrypoints.research as entrypoint
    from ml_harness.adapters.data.task import CsvTaskData
    from ml_harness.application.use_cases.baseline import BaselineResult
    from ml_harness.domain.models import Metric, Score

    task_path = task_fixture(tmp_path)
    task = CsvTaskData().prepare(str(task_path), 42)
    baseline = BaselineResult(
        Score(1000, Metric.RMSE, task.protocol.fingerprint), task.spec.submission_path, 1, 1
    )
    model = ScriptedProposalModel()
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
    from ml_harness.adapters.storage.run_result import save_run_result

    canonical = save_run_result(harness, task, {"research": report}, {})
    (harness / "summary.json").write_text(json.dumps({"research": report, "run_result": canonical}))
    verified = comparison.verify_run(tmp_path, task_path, 42)
    assert verified["refit_verified"]
    assert verified["experiments"] == 1
    assert verified["changed_successes"] == 1
    assert verified["proposal_successes"] == 1

    from copy import deepcopy

    from verification.pipeline_gate import verify_recipe

    original = report["reports"][-1]
    for key, value in [
        ("proposal_hash", "forged"),
        ("resolved_hash", "forged"),
        ("effective_code_hash", "forged"),
        ("executed_recipe_hash", "forged"),
        ("parent_code_hash", "forged"),
        ("status", "running"),
        ("pipeline_version", 1),
        ("full_score", {**original["full_score"], "value": -1}),
    ]:
        changed = deepcopy(original)
        changed[key] = value
        with pytest.raises(ValueError):
            verify_recipe(changed, task, report["reports"])
    changed_canonical = {**canonical, "selected": "forged"}
    (harness / "run-result.json").write_text(json.dumps(changed_canonical))
    with pytest.raises(ValueError):
        comparison.verify_run(tmp_path, task_path, 42)
    (harness / "run-result.json").write_text(json.dumps(canonical))
    Path(task.spec.submission_path).write_text("key,response\nwrong,1\n")
    with pytest.raises(ValueError):
        comparison.verify_run(tmp_path, task_path, 42)


def test_gate_rejects_legacy_empty_missing_and_changed_configuration(comparison):
    from copy import deepcopy

    from verification.pipeline_gate import validate_gate_shape

    expected = {"regression": {"train": "a"}, "classification": {"train": "b"}}
    config = {
        "gate_version": 3,
        "gate": True,
        "seeds": [42],
        "modes": ["qwen"],
        "tasks": [{"path": k, "inputs": v} for k, v in expected.items()],
        "sessions": 12,
        "threads": 4,
        "time_budget": 1800,
        "worker_timeout": 300,
        "token_limit": 32000,
        "memory_limit_mb": 6000,
    }
    runs = [{"task": k, "seed": 42, "mode": "qwen"} for k in expected]
    validate_gate_shape(config, runs, expected)
    for key, value in [
        ("gate_version", 1),
        ("sessions", 3),
        ("threads", 8),
        ("tasks", config["tasks"][:1]),
        ("seeds", [7]),
    ]:
        changed = deepcopy(config)
        changed[key] = value
        with pytest.raises(ValueError):
            validate_gate_shape(changed, runs, expected)
    for incomplete in ([], runs[:1], [runs[0], runs[0]]):
        with pytest.raises(ValueError):
            validate_gate_shape(config, incomplete, expected)


def test_saved_gate_rechecks_runtime_configuration_and_artifacts(tmp_path, monkeypatch, comparison):
    import json

    tasks = {"regression": {"data": "abc"}, "classification": {"data": "abc"}}
    config = {
        "gate_version": 3,
        "gate": True,
        "seeds": [42],
        "modes": ["qwen"],
        "tasks": [{"path": k, "inputs": v} for k, v in tasks.items()],
        "sessions": 12,
        "threads": 4,
        "time_budget": 1800,
        "worker_timeout": 300,
        "token_limit": 32000,
        "memory_limit_mb": 6000,
        "code": {"runtime": "abc"},
        "model": "qwen",
        "endpoint_fingerprint": "hashed",
        "runtime": {"python": "3.12"},
    }
    configuration = tmp_path / "configuration.json"
    configuration.write_text(json.dumps(config))
    report = {
        "gate_version": 3,
        "gate_passed": False,
        "configuration": str(configuration),
        "configuration_hash": comparison._hash(configuration),
        "code": config["code"],
        "runs": [
            {
                "task": name,
                "seed": 42,
                "mode": "qwen",
                "returncode": 0,
                "csv_sha256": "csv",
                "api_trace_hashes": {},
                "command": ["--run-dir", str(tmp_path / name / "harness")],
            }
            for name in tasks
        ],
    }
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report))
    monkeypatch.setattr(comparison, "public_tasks", lambda: tasks)
    monkeypatch.setattr(comparison, "_runtime_versions", lambda: config["runtime"])
    monkeypatch.setattr(comparison, "_task_inputs", lambda path: {"data": "abc"})
    monkeypatch.setattr(
        comparison,
        "verify_run",
        lambda *args: {"live_proposal_successes": 1, "api_trace_hashes": {}, "csv_sha256": "csv"},
    )
    # Boolean is deliberately false: evidence, not the flag, decides acceptance.
    comparison.validate_saved_gate(path, config["code"], "qwen", "hashed")
    for code, model, endpoint in [
        ({"runtime": "other"}, "qwen", "hashed"),
        (config["code"], "other", "hashed"),
        (config["code"], "qwen", "other"),
    ]:
        with pytest.raises(ValueError, match="runtime"):
            comparison.validate_saved_gate(path, code, model, endpoint)
    monkeypatch.setattr(
        comparison,
        "verify_run",
        lambda *args: {"live_proposal_successes": 0, "api_trace_hashes": {}, "csv_sha256": "csv"},
    )
    with pytest.raises(ValueError, match="artifact"):
        comparison.validate_saved_gate(path, config["code"], "qwen", "hashed")
    configuration.write_text("{}")
    with pytest.raises(ValueError, match="hash"):
        comparison.validate_saved_gate(path, config["code"], "qwen", "hashed")


@pytest.fixture(autouse=True)
def focused_semantic_search(monkeypatch):
    """These tests isolate proposal/refit behavior; numeric continuation has its own test."""
    from ml_harness.adapters.research.workspace import ResearchWorkspace

    monkeypatch.setattr(ResearchWorkspace, "numeric_batch", lambda *args: False)
