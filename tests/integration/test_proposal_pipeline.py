"""Real workers test feature refit, bounded admission and persistence across an API outage."""

import json
from pathlib import Path

from test_prepare_task import task_fixture
from test_research_workspace import ScriptedProposalModel

from ml_harness.adapters.data.task import CsvTaskData
from ml_harness.application.ports.language_model import ModelError, ModelReply, ToolCall
from ml_harness.application.use_cases.baseline import BaselineResult
from ml_harness.domain.models import Metric, Score


def setup(tmp_path, monkeypatch, model, baseline_seconds=1):
    import ml_harness.entrypoints.research as entrypoint

    task = CsvTaskData().prepare(str(task_fixture(tmp_path)), 42)
    baseline = BaselineResult(
        Score(1000, Metric.RMSE, task.protocol.fingerprint),
        task.spec.submission_path,
        baseline_seconds,
        1,
    )
    Path(task.spec.submission_path).write_text("saved baseline")
    monkeypatch.setattr(entrypoint, "QwenLanguageModel", lambda *args: model)
    for key in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL"):
        monkeypatch.setenv(key, "synthetic")
    result = entrypoint.research_after_baseline(
        task,
        baseline,
        tmp_path / "research",
        threads=1,
        timeout_seconds=10,
        max_steps=3,
        token_limit=10000,
        max_sessions=2,
    )
    return task, result


def test_real_feature_transform_refit_survives_next_provider_failure(tmp_path, monkeypatch):
    class Model(ScriptedProposalModel):
        def complete(self, messages, tools):
            if self.starts:
                raise ModelError("synthetic outage", attempts=1, reason="api_error")
            reply = super().complete(messages, tools)
            raw = json.loads(reply.tool_calls[0].arguments)
            raw["python_transform"] = """
import pandas as pd
class Transform:
    def fit(self, X):
        self.means = X.select_dtypes(include="number").mean()
    def transform(self, X):
        result = X.copy()
        for name in self.means.index:
            values = pd.to_numeric(result[name], errors="coerce")
            result[name + "_centered"] = values - self.means[name]
        return result
def build_transformer():
    return Transform()
"""
            return ModelReply(
                "", (ToolCall("feature", "propose_experiment", json.dumps(raw)),), 5, 2
            )

    task, result = setup(tmp_path, monkeypatch, Model())
    assert result["status"] == "api_error"
    assert result["reports"][-1]["status"] == "succeeded"
    assert result["reports"][-1]["resolved_spec"]["python_transform"]
    assert result["finalization"]["refit_status"] == "succeeded"
    assert Path(task.spec.submission_path).read_text().startswith("key,response")
    assert result["sessions"][0]["reason"] == "experiment_completed"


def test_expensive_recipe_smoke_never_starts_full_and_repeat_deduplicates(tmp_path, monkeypatch):
    class Model(ScriptedProposalModel):
        def complete(self, messages, tools):
            reply = super().complete(messages, tools)
            raw = json.loads(reply.tool_calls[0].arguments)
            raw["parameters"] = {"iterations": 600, "depth": 8}
            return ModelReply("", (ToolCall("expensive", "propose_experiment", json.dumps(raw)),))

    _, result = setup(tmp_path, monkeypatch, Model(), baseline_seconds=100)
    experiments = [r for r in result["reports"] if r.get("hypothesis_source")]
    assert len(experiments) == 1
    assert experiments[0]["failure_phase"] == "admission"
    assert experiments[0]["admission"]["estimated_full_seconds"] > 10
    assert not list((tmp_path / "research/experiments").glob("*/full/request.json"))
    assert Path(tmp_path / "submission.csv").exists() or result["selected"] == "baseline"


def test_next_hypothesis_sees_actual_optuna_parameters(tmp_path, monkeypatch):
    class Model(ScriptedProposalModel):
        def complete(self, messages, tools):
            reply = super().complete(messages, tools)
            if len(self.starts) != 1:
                return reply
            raw = json.loads(reply.tool_calls[0].arguments)
            raw["tune"] = True
            return ModelReply("", (ToolCall("tune", "propose_experiment", json.dumps(raw)),), 5, 2)

    model = Model()
    _, result = setup(tmp_path, monkeypatch, model)
    first = next(r for r in result["reports"] if r.get("tuning_report"))
    fitted = {k: v for k, v in first["refit_parameters"].items() if k not in ("backend", "tuning")}
    assert first["executed_recipe"]["parameters"] == fitted
    assert model.starts[1]["parent_recipe"]["parameters"] == fitted
    assert model.starts[1]["parent_recipe"]["tuning"] == {}
