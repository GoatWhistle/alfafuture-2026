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
    def __init__(self, code):
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
                                    "experiment_id": "regularization",
                                    "hypothesis": "Shallower trees generalize",
                                    "expected_effect": "Lower RMSE",
                                    "parent_id": "baseline",
                                }
                            ),
                        ),
                    ),
                ),
                ModelReply("Measured result received; this observation does not prove causality."),
            ]
        )
        self.observed = []

    def complete(self, messages, tools):
        self.observed = list(messages)
        return next(self.replies)


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
    assert "full_score" in model.observed[-2]["content"]
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
        Score(1000, Metric.RMSE, task.protocol.fingerprint), task.spec.submission_path, 1, 1
    )
    Path(task.spec.submission_path).write_text("baseline placeholder")
    code = STARTER.replace('"iterations": 200', '"iterations": 3')
    monkeypatch.setattr(entrypoint, "QwenLanguageModel", lambda *args: ScriptedModel(code))
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
    assert report["selected"] == "regularization"
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
    code = STARTER.replace('"iterations": 200', '"iterations": 3')
    replies = []
    for index in range(2):
        attempt_code = code.replace('"iterations": 3', f'"iterations": {3 + index}')
        replies.extend(
            [
                ModelReply(
                    "hypothesis",
                    (
                        ToolCall(
                            f"edit-{index}",
                            "edit_file",
                            json.dumps({"path": "candidate.py", "content": attempt_code}),
                        ),
                        ToolCall(
                            f"run-{index}",
                            "run_experiment",
                            json.dumps(
                                {
                                    "experiment_id": f"attempt-{index}",
                                    "hypothesis": f"hypothesis {index}",
                                    "expected_effect": "lower RMSE",
                                    "parent_id": None,
                                }
                            ),
                        ),
                    ),
                    5,
                    2,
                ),
                ModelReply(f"interpretation {index}", (), 5, 2),
            ]
        )

    class MultipleSessions:
        def __init__(self):
            self.replies = iter(replies)
            self.starts = []

        def complete(self, messages, tools):
            if len(messages) == 2:
                self.starts.append(json.loads(messages[1]["content"]))
            return next(self.replies)

    model = MultipleSessions()
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
    assert previous["id"] == "attempt-0"
    assert previous["interpretation"] == "interpretation 0"
    assert previous["measured"]["protocol_id"] == task.protocol.fingerprint
    assert result["tokens_used"] == 28
    restored = ResearchWorkspace(
        task, tmp_path / "research", SubprocessRunner(), IndependentEvaluator()
    )
    assert restored.reports[0]["usage"]["scope"] == "shared_session"
    assert restored.reports[0]["usage"]["input_tokens"] == 10
    assert restored.reports[0]["usage"]["api_attempts"] == 2


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
