"""A scripted LLM changes real code, trains it and sees independently measured results."""

import json
from pathlib import Path

import pytest
from test_prepare_task import task_fixture

from ml_harness.adapters.data.task import CsvTaskData
from ml_harness.adapters.execution.clock import SystemClock
from ml_harness.adapters.execution.runner import SubprocessRunner
from ml_harness.adapters.llm.tools import ResearchTools
from ml_harness.adapters.research.workspace import STARTER, ResearchWorkspace
from ml_harness.adapters.storage.submission import IndependentEvaluator
from ml_harness.application.ports.language_model import ModelReply, ToolCall
from ml_harness.application.use_cases.research import run_session
from ml_harness.domain.models import RunBudget


class ScriptedModel:
    def __init__(self, code, experiment_id="regularization"):
        self.replies = iter(
            [
                ModelReply(
                    "Reduce depth to test regularization.",
                    (
                        ToolCall(
                            "edit",
                            "edit_file",
                            json.dumps({"path": "candidate.py", "content": code}),
                        ),
                        ToolCall(
                            "run",
                            "run_experiment",
                            json.dumps(
                                {
                                    "experiment_id": experiment_id,
                                    "hypothesis": "Shallower trees generalize",
                                    "expected_effect": "Lower RMSE",
                                    "parent_id": "baseline",
                                }
                            ),
                        ),
                    ),
                ),
                ModelReply("Measured result received; this observation does not prove causality."),
                ModelReply(
                    "Reviewed the measured result; a single split does not prove causality."
                ),
            ]
        )
        self.observed = []

    def complete(self, messages, tools):
        self.observed = list(messages)
        return next(self.replies)


class ScriptedProposalModel:
    def __init__(self):
        self.starts = []
        self.observed = []

    def complete(self, messages, tools):
        self.observed = list(messages)
        state = json.loads(messages[1]["content"])
        self.starts.append(state)
        index = len(self.starts)
        return ModelReply(
            "",
            (
                ToolCall(
                    str(index),
                    "propose_experiment",
                    json.dumps(
                        {
                            "parent_id": state["parent_id"],
                            "task_kind": state["task_kind"],
                            "backend": "catboost",
                            "target_error": "exploratory: simpler model",
                            "rationale": "Reduce overfit with shallow trees",
                            "expected_effect": "lower RMSE",
                            "parameters": {"iterations": 10 + index, "depth": 3},
                        }
                    ),
                ),
            ),
            5,
            2,
        )


def test_interrupted_model_session_never_resets_unknown_provider_usage(tmp_path, monkeypatch):
    import ml_harness.entrypoints.research as entrypoint
    from ml_harness.application.use_cases.baseline import BaselineResult
    from ml_harness.domain.models import Metric, Score

    task = CsvTaskData().prepare(str(task_fixture(tmp_path)), 42)
    baseline = BaselineResult(
        Score(100, Metric.RMSE, task.protocol.fingerprint), task.spec.submission_path, 1, 1
    )
    Path(task.spec.submission_path).write_text("saved baseline bytes")

    class InterruptedModel:
        def complete(self, messages, tools):
            raise KeyboardInterrupt("Process interrupted while provider usage is unknown")

    monkeypatch.setattr(entrypoint, "QwenLanguageModel", lambda *args: InterruptedModel())
    for name in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL"):
        monkeypatch.setenv(name, "synthetic")
    arguments = dict(threads=1, timeout_seconds=10, max_steps=3, token_limit=1000)
    with pytest.raises(KeyboardInterrupt):
        entrypoint.research_after_baseline(task, baseline, tmp_path / "research", **arguments)

    def forbidden_model(*args):
        pytest.fail("Resume must stop further calls when prior provider usage is unknown")

    monkeypatch.setattr(entrypoint, "QwenLanguageModel", forbidden_model)
    report = entrypoint.research_after_baseline(task, baseline, tmp_path / "research", **arguments)
    assert report["status"] == "unknown_usage"
    assert report["sessions"] == []
    assert Path(task.spec.submission_path).read_text() == "saved baseline bytes"


def test_edit_smoke_full_evaluation_and_idempotence(tmp_path):
    task = CsvTaskData().prepare(str(task_fixture(tmp_path)), 42)
    workspace = ResearchWorkspace(
        task,
        tmp_path / "research",
        SubprocessRunner(),
        IndependentEvaluator(),
        threads=1,
        timeout_seconds=10,
    )
    tools = ResearchTools(
        workspace.path, workspace.inspect_data, workspace.run_experiment, workspace.read_experiments
    )
    code = STARTER.replace('"iterations": 200', '"iterations": 3').replace(
        '"depth": 6', '"depth": 3'
    )
    model = ScriptedModel(code)
    clock = SystemClock()
    result = run_session(
        model,
        tools,
        "regression task",
        RunBudget(clock.now() + 60, 10000, 10),
        clock,
        reserve_seconds=0,
    )
    assert result.reason == "model_finished"
    assert result.reports[0]["status"] == "succeeded"
    report = workspace.reports[0]
    assert report["smoke_score"]["level"] == "smoke"
    assert report["full_score"]["level"] == "full"
    assert report["full_score"]["protocol_id"] == task.protocol.fingerprint
    assert report["smoke_score"]["protocol_id"] != task.protocol.fingerprint
    assert any("full_score" in m.get("content", "") for m in model.observed if m["role"] == "tool")
    report_file = workspace.path / "experiments/regularization/experiment.json"
    before = report_file.stat().st_mtime_ns
    repeated = workspace.run_experiment(
        {
            "experiment_id": "regularization",
            "hypothesis": "Shallower trees generalize",
            "expected_effect": "Lower RMSE",
            "parent_id": "baseline",
        }
    )
    assert repeated == report and report_file.stat().st_mtime_ns == before
    assert len(workspace.read_experiments()) == 1
    assert (workspace.path / "candidate.py").read_text() == code
    assert (workspace.path / "experiments/regularization/candidate.py").read_text() == code


def test_bad_candidate_saved_then_independent_attempt_succeeds(tmp_path):
    task = CsvTaskData().prepare(str(task_fixture(tmp_path)), 42)
    workspace = ResearchWorkspace(
        task,
        tmp_path / "research",
        SubprocessRunner(),
        IndependentEvaluator(),
        threads=1,
        timeout_seconds=10,
    )
    candidate = workspace.path / "candidate.py"
    candidate.write_text("raise RuntimeError('broken candidate')")
    arguments = {
        "experiment_id": "broken",
        "hypothesis": "test",
        "expected_effect": "test",
        "parent_id": None,
    }
    failed = workspace.run_experiment(arguments)
    assert failed["status"] == "failed" and failed["failure_phase"] == "smoke"
    assert "full_score" not in failed
    candidate.write_text(STARTER.replace('"iterations": 200', '"iterations": 3'))
    good = workspace.run_experiment({**arguments, "experiment_id": "fixed"})
    assert good["status"] == "succeeded"
    assert len(workspace.reports) == 2


def test_research_promotes_measured_improvement_after_valid_refit(tmp_path, monkeypatch):
    import ml_harness.entrypoints.research as entrypoint
    from ml_harness.application.use_cases.baseline import BaselineResult
    from ml_harness.domain.models import Metric, Score

    task = CsvTaskData().prepare(str(task_fixture(tmp_path)), 42)
    baseline = BaselineResult(
        Score(1000, Metric.RMSE, task.protocol.fingerprint), task.spec.submission_path, 2.5, 4.0
    )
    Path(task.spec.submission_path).write_text("baseline placeholder")
    model = ScriptedProposalModel()
    monkeypatch.setattr(entrypoint, "QwenLanguageModel", lambda *args: model)
    for name, value in (
        ("OPENAI_API_KEY", "synthetic"),
        ("OPENAI_BASE_URL", "https://invalid.test"),
        ("OPENAI_MODEL", "qwen-issued"),
    ):
        monkeypatch.setenv(name, value)
    report = entrypoint.research_after_baseline(
        task,
        baseline,
        tmp_path / "research",
        threads=1,
        timeout_seconds=10,
        max_steps=3,
        token_limit=1000,
        max_sessions=1,
    )
    initial_context = json.loads(model.observed[1]["content"])
    assert initial_context["parent_recipe"]["backend"] == "catboost"
    assert initial_context["task_kind"] == "regression"
    context_budget = initial_context["budget"]
    assert {
        key: context_budget[key]
        for key in (
            "worker_timeout_seconds",
            "smoke_timeout_seconds",
            "threads",
            "baseline_validation_seconds",
            "baseline_refit_seconds",
        )
    } == {
        "worker_timeout_seconds": 10,
        "smoke_timeout_seconds": 10,
        "threads": 1,
        "baseline_validation_seconds": 2.5,
        "baseline_refit_seconds": 4.0,
    }
    assert report["reports"][-1]["parent_backend"] == "catboost"
    assert report["selected"] != "baseline"
    assert report["sessions"][0]["reason"] == "experiment_completed"
    assert report["refit_status"] == "succeeded"
    assert Path(task.spec.submission_path).read_text().startswith("key,response")
    published = Path(task.spec.submission_path).read_bytes()

    def forbidden_model(*args):
        raise AssertionError("Frozen control/finalization must never restart model research")

    monkeypatch.setattr(entrypoint, "QwenLanguageModel", forbidden_model)
    resumed = entrypoint.research_after_baseline(
        task,
        baseline,
        tmp_path / "research",
        threads=1,
        timeout_seconds=10,
        max_steps=3,
        token_limit=1000,
        max_sessions=10,
    )
    assert resumed["status"] == "finalization_resumed"
    assert resumed["sessions"] == []
    assert Path(task.spec.submission_path).read_bytes() == published


def test_api_failure_retains_baseline_csv(tmp_path, monkeypatch):
    import ml_harness.entrypoints.research as entrypoint
    from ml_harness.application.ports.language_model import ModelError
    from ml_harness.application.use_cases.baseline import BaselineResult
    from ml_harness.domain.models import Metric, Score

    task = CsvTaskData().prepare(str(task_fixture(tmp_path)), 42)
    Path(task.spec.submission_path).write_text("previous valid file")
    baseline = BaselineResult(
        Score(1, Metric.RMSE, task.protocol.fingerprint), task.spec.submission_path, 1, 1
    )

    class UnavailableModel:
        def complete(self, messages, tools):
            raise ModelError("unavailable", reason="api_error")

    monkeypatch.setattr(entrypoint, "QwenLanguageModel", lambda *args: UnavailableModel())
    for name in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL"):
        monkeypatch.setenv(name, "synthetic")
    report = entrypoint.research_after_baseline(
        task,
        baseline,
        tmp_path / "research",
        threads=1,
        timeout_seconds=10,
        max_steps=3,
        token_limit=1000,
        max_sessions=1,
    )
    assert report["status"] == "api_error" and report["selected"] == "baseline"
    assert len(report["sessions"]) == 1
    assert report["sessions"][0]["error_detail"] == "unavailable"
    assert Path(task.spec.submission_path).read_text() == "previous valid file"


def test_new_hypotheses_get_short_sessions_with_prior_evidence(tmp_path, monkeypatch):
    import ml_harness.entrypoints.research as entrypoint
    from ml_harness.application.use_cases.baseline import BaselineResult
    from ml_harness.domain.models import Metric, Score

    task = CsvTaskData().prepare(str(task_fixture(tmp_path)), 42)
    model = ScriptedProposalModel()
    monkeypatch.setattr(entrypoint, "QwenLanguageModel", lambda *args: model)
    for key in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL"):
        monkeypatch.setenv(key, "synthetic")
    baseline = BaselineResult(
        Score(1000, Metric.RMSE, task.protocol.fingerprint), task.spec.submission_path, 1, 1
    )
    result = entrypoint.research_after_baseline(
        task,
        baseline,
        tmp_path / "research",
        threads=1,
        timeout_seconds=10,
        max_steps=3,
        token_limit=1000,
        max_sessions=2,
    )
    assert len(model.starts) == 2 and len(result["sessions"]) == 2
    assert model.starts[0]["reports"] == []
    previous = model.starts[1]["reports"][0]
    assert previous["id"] == result["reports"][1]["id"]
    assert not previous["interpretation"]
    assert model.starts[1]["last_measurements"][-1]["full_score"]
    assert model.starts[1]["budget"]["tokens_remaining"] == 993
    assert previous["measured"]["protocol_id"] == task.protocol.fingerprint
    assert result["tokens_used"] == 14
    restored = ResearchWorkspace(
        task, tmp_path / "research", SubprocessRunner(), IndependentEvaluator()
    )
    assert restored.reports[0]["usage"]["scope"] == "shared_session"
    assert restored.reports[0]["usage"]["input_tokens"] == 5
    assert restored.reports[0]["usage"]["api_attempts"] == 1


def test_resume_preserves_conservative_charge_after_successful_retry(tmp_path, monkeypatch):
    import ml_harness.entrypoints.research as entrypoint
    from ml_harness.application.use_cases.baseline import BaselineResult
    from ml_harness.domain.models import Metric, Score

    task = CsvTaskData().prepare(str(task_fixture(tmp_path)), 42)
    baseline = BaselineResult(
        Score(100, Metric.RMSE, task.protocol.fingerprint), task.spec.submission_path, 1, 1
    )
    Path(task.spec.submission_path).write_text("saved baseline bytes")
    directory = tmp_path / "research"
    workspace = ResearchWorkspace(
        task, directory, SubprocessRunner(), IndependentEvaluator(), threads=1, timeout_seconds=10
    )
    workspace.register_session(
        "recovered",
        {
            "summary": "recovered request",
            "reports": [],
            "reason": "model_finished",
            "input_tokens": 9,
            "output_tokens": 1,
            "usage_known": False,
            "api_attempts": 2,
            "tokens_charged": 1400,
            "budget_accounted": True,
        },
    )

    (directory / "pipeline-version.json").write_text(json.dumps({"version": 2}))

    class ForbiddenModel:
        def complete(self, messages, tools):
            pytest.fail("The reserved retry cost must not reset to ten reported tokens")

    monkeypatch.setattr(entrypoint, "QwenLanguageModel", lambda *args: ForbiddenModel())
    for name in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL"):
        monkeypatch.setenv(name, "synthetic")
    report = entrypoint.research_after_baseline(
        task,
        baseline,
        directory,
        threads=1,
        timeout_seconds=10,
        max_steps=3,
        token_limit=1000,
        max_sessions=2,
    )
    assert report["tokens_used"] == 1400
    assert report["budget_accounted"] and not report["usage_known"]
    assert report["status"] == "budget_exhausted"


def test_real_worker_failure_reaches_model_and_repairs_in_same_session(tmp_path):
    task = CsvTaskData().prepare(str(task_fixture(tmp_path)), 42)
    workspace = ResearchWorkspace(
        task,
        tmp_path / "research",
        SubprocessRunner(),
        IndependentEvaluator(),
        threads=1,
        timeout_seconds=10,
    )
    tools = ResearchTools(
        workspace.path, workspace.inspect_data, workspace.run_experiment, workspace.read_experiments
    )
    bad_options = {"backend": "lightgbm", "n_estimators": 3, "early_stopping_rounds": 2}
    good_options = {"backend": "lightgbm", "n_estimators": 3}

    def source(options):
        return (
            "from ml_harness.adapters.ml.candidates import build_model as base_model\n"
            "def build_model(kind, parameters, threads, seed):\n"
            f"    options = {options!r}\n"
            "    options.update(parameters)\n"
            "    return base_model(kind, options, threads, seed)\n"
        )

    def call(name, arguments, identifier):
        return ModelReply("", (ToolCall(identifier, name, json.dumps(arguments)),))

    class RepairModel:
        step = 0

        def set_limits(self, *limits):
            pass

        def complete(self, messages, schemas):
            step = self.step
            self.step += 1
            if step in (0, 2):
                if step == 2:
                    context = json.dumps(messages)
                    assert "ValueError" in context and "at least one dataset" in context
                    assert '"failure_phase":"smoke"' in messages[-1]["content"]
                    assert [s["function"]["name"] for s in schemas] == ["edit_file"]
                return call(
                    "edit_file",
                    {
                        "path": "candidate.py",
                        "content": source(bad_options if step == 0 else good_options),
                    },
                    f"edit-{step}",
                )
            if step in (1, 3):
                return call(
                    "run_experiment",
                    {
                        "experiment_id": "broken" if step == 1 else "fixed",
                        "hypothesis": "Test the written LightGBM configuration",
                        "expected_effect": "A measured RMSE",
                    },
                    f"run-{step}",
                )
            assert step in (4, 5)
            return ModelReply("The repaired written candidate completed full validation.")

    model = RepairModel()
    clock = SystemClock()
    result = run_session(
        model,
        tools,
        json.dumps({"candidate_code": STARTER}),
        RunBudget(clock.now() + 60, 32000, 10),
        clock,
        reserve_seconds=0,
    )
    assert result.reason == "model_finished" and result.tool_errors == 1
    assert [report["status"] for report in result.reports] == ["failed", "succeeded"]
    assert result.reports[1]["full_score"]["level"] == "full"
    assert model.step == 6
