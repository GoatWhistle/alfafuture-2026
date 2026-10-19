"""Failures observed in the real Qwen audit exercised through production interfaces."""

import json
from pathlib import Path

import pytest
from test_prepare_task import task_fixture

from ml_harness.adapters.data.task import CsvTaskData
from ml_harness.adapters.execution.runner import SubprocessRunner
from ml_harness.adapters.llm.tools import ResearchTools
from ml_harness.adapters.research.finalization import FinalizationWorkspace
from ml_harness.adapters.research.workspace import ResearchWorkspace
from ml_harness.adapters.storage.submission import AtomicSubmissionWriter, IndependentEvaluator
from ml_harness.application.ports.language_model import ToolCall
from ml_harness.domain.models import ContractError, Score
from ml_harness.entrypoints.verify_research import candidate_code


def workspace(tmp_path):
    task = CsvTaskData().prepare(str(task_fixture(tmp_path / "inputs")), 42)
    runner = SubprocessRunner()
    evaluator = IndependentEvaluator()
    return (
        task,
        ResearchWorkspace(task, tmp_path / "research", runner, evaluator, threads=1),
        runner,
        evaluator,
    )


def test_original_task_and_api_are_readable_but_not_editable(tmp_path):
    task, work, _, _ = workspace(tmp_path)
    tools = ResearchTools(work.path, work.inspect_data, work.run_experiment, work.read_experiments)
    for name in ("task.md", "data_dictionary.csv", "candidate_api.md"):
        result = tools.execute(ToolCall(name, "read_file", json.dumps({"path": name})))
        assert result["ok"] and result["result"]["text"]
        assert not tools.execute(
            ToolCall("edit-" + name, "edit_file", json.dumps({"path": name, "content": "bad"}))
        )["ok"]
    assert (work.path / "task.md").read_bytes() == Path(task.spec.task_path).read_bytes()


def test_unchanged_parent_never_starts_training(tmp_path):
    task, work, _, _ = workspace(tmp_path)
    work.register_baseline(Score(100, task.spec.metric, task.protocol.fingerprint), None)
    (work.path / "candidate.py").write_bytes(Path(work.baseline_report["code_path"]).read_bytes())
    with pytest.raises(ContractError, match="unchanged"):
        work.run_experiment(
            {
                "experiment_id": "fake-change",
                "parent_id": "baseline",
                "hypothesis": "log target",
                "expected_effect": "lower RMSE",
            }
        )
    assert not (work.path / "experiments/fake-change").exists()


def test_real_tuned_candidate_freezes_parameters_and_refits_without_search(tmp_path):
    task, work, runner, evaluator = workspace(tmp_path)
    (work.path / "candidate.py").write_text(
        candidate_code(
            {"backend": "catboost", "iterations": 3, "tuning": {"trials": 2, "time_limit": 5}}
        )
    )
    report = work.run_experiment(
        {
            "experiment_id": "tuned",
            "hypothesis": "train-only search",
            "expected_effect": "lower RMSE",
        }
    )
    assert report["status"] == "succeeded"
    assert report["parameters"] == {}
    assert report["refit_parameters"]["tuning"] == {"enabled": False}
    result = FinalizationWorkspace(
        task, work, runner, evaluator, AtomicSubmissionWriter(), threads=1
    ).finish()
    assert result["refit_status"] == "succeeded" and result["selected"] == "tuned"
    requests = list((work.path / "finalization/refit").glob("*/request.json"))
    assert len(requests) == 1
    frozen = json.loads(requests[0].read_text())
    assert frozen["parameters"] == report["refit_parameters"]
    output = json.loads(requests[0].with_name("result.json").read_text())
    assert "tuning_report" not in output


def test_numeric_studies_continue_and_finalists_deduplicate(tmp_path):
    import optuna

    task, work, runner, evaluator = workspace(tmp_path)
    assert work.numeric_batch("catboost", 0)
    assert work.numeric_batch("catboost", 1)
    storage = "sqlite:///" + str(work.path / "numeric-studies.sqlite3")
    summaries = optuna.get_all_study_summaries(storage)
    assert len(summaries) == 2  # smoke and full have independent inner partitions.
    assert all(summary.n_trials == 10 for summary in summaries)
    result = FinalizationWorkspace(
        task, work, runner, evaluator, AtomicSubmissionWriter(), threads=1
    ).finish()
    assert result["refit_status"] == "succeeded"


def test_default_portfolio_cli_and_resume_keep_measured_artifacts(tmp_path, monkeypatch):
    import sys

    from ml_harness.entrypoints import cli

    monkeypatch.chdir(tmp_path)
    path = task_fixture(tmp_path / "inputs")
    directory = tmp_path / "run"
    arguments = [
        "solution.py",
        str(path),
        "--run-dir",
        str(directory),
        "--search-mode",
        "none",
        "--threads",
        "1",
        "--time-budget",
        "60",
    ]
    monkeypatch.setattr(sys, "argv", arguments)
    assert cli.main() == 0
    baseline = json.loads((directory / "baseline.json").read_text())
    assert {c["backend"] for c in baseline["candidates"]} == {"catboost", "lightgbm"}
    assert baseline["backend"] in ("catboost", "lightgbm")
    before = {str(p): p.stat().st_mtime_ns for p in directory.rglob("request.json")}
    monkeypatch.setattr(sys, "argv", arguments + ["--resume"])
    assert cli.main() == 0
    assert {str(p): p.stat().st_mtime_ns for p in directory.rglob("request.json")} == before
