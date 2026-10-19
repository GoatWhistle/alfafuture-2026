"""Scripted declarations clarify evidence; source files always remain authoritative."""

import csv
import json
import sys
from pathlib import Path

import pytest

from ml_harness.adapters.data.clarification import ClarifyingTaskData
from ml_harness.adapters.data.task import CsvTaskData
from ml_harness.application.ports.language_model import ModelError, ModelReply
from ml_harness.domain.models import ContractError, Metric, RunBudget


class Clock:
    def now(self):
        return 0


class Model:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.calls, self.limits = [], []

    def set_limits(self, *args):
        self.limits.append(args)

    def complete(self, messages, tools):
        self.calls.append((list(messages), tools))
        result = next(self.replies)
        if isinstance(result, Exception):
            raise result
        return result


def fixture(tmp_path, *, missing_role=False, text="Критерий — средняя абсолютная ошибка."):
    task = tmp_path / "task.md"
    task.write_text(text + "\nOutput: `answer.csv`\n", encoding="utf-8")
    with (tmp_path / "train.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(["key", "x", "response"])
        writer.writerows([[i, i, i / 3] for i in range(30)])
    (tmp_path / "test.csv").write_text("key,x\n001,40\nNA,41\n")
    (tmp_path / "sample_submission.csv").write_text("key,response\n001,0\nNA,0\n")
    (tmp_path / "data_dictionary.csv").write_text(
        "feature,role,dtype\nkey,id,string\nx,feature,numeric\nresponse,"
        + ("feature" if missing_role else "target")
        + ",numeric\n"
    )
    return task


def declaration(metric="mae", target="response", quote="Критерий — средняя абсолютная ошибка."):
    return ModelReply(
        json.dumps(
            {"metric": metric, "target": target, "evidence": {"metric": quote}}, ensure_ascii=False
        ),
        input_tokens=10,
        output_tokens=5,
    )


def adapter(tmp_path, model, **kwargs):
    budget = kwargs.pop("budget", RunBudget(100, 10000, 10))
    return ClarifyingTaskData(
        CsvTaskData(), model, tmp_path / "artifacts", budget, Clock(), **kwargs
    )


def snapshots(tmp_path):
    return {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}


def test_real_cli_accounts_clarification_and_resumes_without_another_request(tmp_path, monkeypatch):
    from ml_harness.entrypoints import cli

    path = fixture(tmp_path)
    original = snapshots(tmp_path)
    model = Model([declaration()])
    monkeypatch.setattr(cli, "ConfiguredTaskModel", lambda: model)
    for name in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL"):
        monkeypatch.delenv(name, raising=False)
    run = tmp_path / "run"
    arguments = [
        "solution.py",
        str(path),
        "--run-dir",
        str(run),
        "--time-budget",
        "120",
        "--seed",
        "7",
        "--backend",
        "lightgbm",
    ]
    monkeypatch.setattr(sys, "argv", arguments)
    assert cli.main() == 0
    assert json.loads((run / "prepared.json").read_text())["protocol"]["seed"] == 7
    summary = json.loads((run / "summary.json").read_text())
    assert summary["research"]["tokens_used"] == 15
    assert summary["research"]["status"] == "not_configured"
    assert (
        summary["research"]["finalization"]["finalists"][0]["members"][0]["backend"] == "lightgbm"
    )
    usage = json.loads((run / "budget.json").read_text())
    assert usage["preparation_tokens"] == 15 and usage["preparation_attempts"] == 1
    saved = (tmp_path / "answer.csv").read_bytes()
    monkeypatch.setattr(sys, "argv", arguments + ["--resume"])
    assert cli.main() == 0
    assert len(model.calls) == 1
    assert (tmp_path / "answer.csv").read_bytes() == saved
    assert {name: (tmp_path / name).read_bytes() for name in original} == original


def test_unknown_metric_evidence_is_verified_and_original_materials_unchanged(tmp_path):
    task = fixture(tmp_path)
    before = snapshots(tmp_path)
    model = Model([declaration()])
    data = adapter(tmp_path, model)
    result = data.prepare(str(task), 42)
    assert result.spec.metric == Metric.MAE
    assert result.spec.target == "response" and result.spec.task_path == str(task)
    assert result.spec.dictionary_path == str(tmp_path / "data_dictionary.csv")
    assert result.spec.submission_path == str(tmp_path / "answer.csv")
    assert snapshots(tmp_path) == before
    assert len(model.calls) == 1 and model.calls[0][1] == []
    assert model.limits == [(70, 2, 10000)]
    assert data.budget.tokens_used == 15 and data.budget.attempts_used == 1
    saved = list((tmp_path / "artifacts").glob("*/clarification.json"))
    assert len(saved) == 1
    # A deterministic independently checked declaration resumes without another API call.
    restored = adapter(tmp_path, Model([])).prepare(str(task), 42)
    assert restored == result


def test_missing_target_role_requires_unique_train_only_field_and_revalidates_overlay(tmp_path):
    task = fixture(tmp_path, missing_role=True, text="Main metric: RMSE")
    original = snapshots(tmp_path)
    model = Model([declaration(metric="rmse", quote="")])
    result = adapter(tmp_path, model).prepare(str(task), 42)
    assert result.spec.target == "response" and result.spec.metric == Metric.RMSE
    assert Path(result.spec.dictionary_path) != tmp_path / "data_dictionary.csv"
    assert "response,target,numeric" in Path(result.spec.dictionary_path).read_text()
    assert snapshots(tmp_path) == original


def test_two_attempts_allow_malformed_then_verified_declaration(tmp_path):
    task = fixture(tmp_path)
    model = Model([ModelReply("not-json"), declaration()])
    data = adapter(tmp_path, model)
    assert data.prepare(str(task), 42).spec.metric == Metric.MAE
    assert len(model.calls) == 2 and data.budget.attempts_used == 2
    assert model.limits[-1][1] == 1


@pytest.mark.parametrize(
    "bad",
    [
        declaration(metric="rmse"),
        declaration(target="x"),
        declaration(quote="invented supported MAE quotation"),
        ModelReply(
            json.dumps({"metric": "mae", "target": "response", "train_path": "/etc/passwd"})
        ),
    ],
)
def test_unverifiable_or_overriding_declarations_never_pass(tmp_path, bad):
    task = fixture(tmp_path)
    model = Model([bad, bad])
    with pytest.raises(ContractError, match="bounded attempts"):
        adapter(tmp_path, model).prepare(str(task), 42)
    assert len(model.calls) == 2
    assert not list((tmp_path / "artifacts").glob("*/clarification.json"))


@pytest.mark.parametrize(
    "damage",
    [
        "target_role",
        "duplicate_ids",
        "sample_ids",
        "dictionary_schema",
        "explicit_metric",
        "explicit_target",
        "no_metric_evidence",
    ],
)
def test_file_or_explicit_description_contradictions_never_call_model(tmp_path, damage):
    task = fixture(tmp_path)
    if damage == "target_role":
        dictionary = tmp_path / "data_dictionary.csv"
        dictionary.write_text(dictionary.read_text().replace("x,feature", "x,target"))
    elif damage == "duplicate_ids":
        test = tmp_path / "test.csv"
        test.write_text("key,x\n001,40\n001,41\n")
    elif damage == "sample_ids":
        (tmp_path / "sample_submission.csv").write_text("key,response\nother,0\nNA,0\n")
    elif damage == "dictionary_schema":
        dictionary = tmp_path / "data_dictionary.csv"
        dictionary.write_text(dictionary.read_text().replace("response,target", "wrong,target"))
    elif damage == "explicit_metric":
        task.write_text(task.read_text() + '\n```task-json\n{"metric":"unsupported"}\n```\n')
    elif damage == "explicit_target":
        task.write_text(task.read_text() + '\n```task-json\n{"target":"x"}\n```\n')
    else:
        task.write_text("Predict the best possible numbers.\n")
    model = Model([])
    with pytest.raises(ContractError):
        adapter(tmp_path, model).prepare(str(task), 42)
    assert not model.calls


def test_ambiguous_metrics_need_actual_primary_evidence(tmp_path):
    quote = "Главный критерий — средняя абсолютная ошибка."
    task = fixture(tmp_path, text=quote + "\nDiagnostics: RMSE")
    result = adapter(tmp_path, Model([declaration(quote=quote)])).prepare(str(task), 42)
    assert result.spec.metric == Metric.MAE


def test_unknown_usage_and_model_error_stop_without_leaking_or_retrying(tmp_path):
    task = fixture(tmp_path)
    model = Model([ModelReply("{}", usage_known=False)])
    data = adapter(tmp_path, model)
    with pytest.raises(ContractError, match="usage is unknown"):
        data.prepare(str(task), 42)
    with pytest.raises(ContractError, match="usage is unknown"):
        data.prepare(str(task), 42)
    assert len(model.calls) == 1
    rejected = Model([ModelError("private-api-secret", attempts=0)])
    data = adapter(tmp_path, rejected)
    with pytest.raises(ContractError) as error:
        data.prepare(str(task), 42)
    assert "private-api-secret" not in str(error.value)
    assert data.budget.attempts_used == 0 and len(rejected.calls) == 1
    assert data.usage_known  # No provider call was made.


def test_ordinary_supported_task_needs_no_clarification(tmp_path):
    task = fixture(tmp_path, text="RMSE")
    model = Model([])
    result = adapter(tmp_path, model).prepare(str(task), 42)
    assert result.spec.metric == Metric.RMSE and not model.calls


def test_sdk_retry_attempts_count_against_two_attempt_cap_and_global_budget(tmp_path):
    task = fixture(tmp_path)
    model = Model([ModelReply("invalid-json", attempts=2)])
    data = adapter(tmp_path, model)
    with pytest.raises(ContractError, match="bounded attempts"):
        data.prepare(str(task), 42)
    assert len(model.calls) == 1 and data.budget.attempts_used == 2
    exhausted = Model([])
    with pytest.raises(ContractError, match="bounded attempts"):
        adapter(tmp_path, exhausted, budget=RunBudget(0, 0, 0)).prepare(str(task), 42)
    assert not exhausted.calls


def test_callable_reserve_and_explicit_primary_conflict(tmp_path):
    task = fixture(tmp_path)
    model = Model([declaration()])
    adapter(tmp_path, model, reserve_seconds=lambda: 40).prepare(str(task), 42)
    assert model.limits == [(60, 2, 10000)]
    task.write_text(
        'Главный критерий — средняя абсолютная ошибка.\n```task-json\n{"metric":"rmse"}\n```\n'
    )
    no_calls = Model([])
    with pytest.raises(ContractError, match="contradicts"):
        adapter(tmp_path, no_calls).prepare(str(task), 42)
    assert not no_calls.calls
