"""A provider outage leaves time for a complete real numeric pipeline and final CSV."""

import json
from pathlib import Path

from test_prepare_task import task_fixture

from ml_harness.adapters.data.task import CsvTaskData
from ml_harness.adapters.research.workspace import ResearchWorkspace
from ml_harness.application.ports.language_model import ModelError
from ml_harness.application.use_cases.baseline import BaselineResult
from ml_harness.domain.budget import WallClockBudget
from ml_harness.domain.models import Metric, Score


def test_provider_outage_continues_real_numeric_worker_until_budget(tmp_path, monkeypatch):
    import ml_harness.entrypoints.research as entrypoint

    class Clock:
        value = 0.0

        def now(self):
            return self.value

    class Unavailable:
        calls = 0

        def complete(self, messages, tools):
            self.calls += 1
            raise ModelError("issued provider unavailable", attempts=1, reason="api_error")

    clock, model = Clock(), Unavailable()
    wall = WallClockBudget(1000)
    monkeypatch.setattr(entrypoint, "SystemClock", lambda: clock)
    monkeypatch.setattr(entrypoint, "QwenLanguageModel", lambda *args: model)
    for key in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL"):
        monkeypatch.setenv(key, "synthetic")
    task = CsvTaskData().prepare(str(task_fixture(tmp_path)), 42)
    validation = tmp_path / "validation"
    validation.mkdir()
    (validation / "request.json").write_text(
        json.dumps({"backend": "catboost", "parameters": {"iterations": 10, "depth": 3}})
    )
    baseline = BaselineResult(
        Score(1000, Metric.RMSE, task.protocol.fingerprint),
        task.spec.submission_path,
        1,
        1,
        validation_dir=str(validation),
    )
    original = ResearchWorkspace.numeric_batch
    batches = []

    def numeric(workspace, backend, index):
        result = original(workspace, backend, index)
        batches.append(result)
        # End search after one real batch while retaining the final refit reserve.
        clock.value = wall.deadline - wall.reserve_seconds() + 0.5
        return result

    monkeypatch.setattr(ResearchWorkspace, "numeric_batch", numeric)
    result = entrypoint.research_after_baseline(
        task,
        baseline,
        tmp_path / "research",
        threads=1,
        timeout_seconds=20,
        max_steps=3,
        token_limit=1000,
        runtime_budget=wall,
    )
    assert model.calls == 1
    assert batches == [True]
    assert result["status"] == "budget_exhausted"
    numeric_report = next(
        r for r in result["reports"] if r.get("hypothesis_source") == "deterministic_optuna"
    )
    assert numeric_report["status"] == "succeeded"
    assert numeric_report["executed_recipe"]["tuning"] == {}
    assert result["finalization"]["refit_status"] == "succeeded"
    assert Path(task.spec.submission_path).read_text().startswith("key,response")
    events = (tmp_path / "research/memory/events.jsonl").read_text()
    assert "local_only" in events and "api_error" in events
