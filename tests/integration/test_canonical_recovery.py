"""Canonical result describes the actual checkpoint CSV after interrupted research."""

import hashlib
import json
from pathlib import Path

from test_prepare_task import task_fixture

from ml_harness.adapters.data.task import CsvTaskData
from ml_harness.adapters.storage.run_result import save_run_result


def test_interrupted_checkpoint_selection_and_unknown_inflight_usage(tmp_path):
    task = CsvTaskData().prepare(str(task_fixture(tmp_path)), 42)
    path = Path(task.spec.submission_path)
    path.write_text("key,response\n102,2\n101,1\n")
    research = tmp_path / "research"
    (research / "memory").mkdir(parents=True)
    score = {
        "value": 0.5,
        "metric": "rmse",
        "protocol_id": task.protocol.fingerprint,
        "level": "full",
    }
    recipe = {
        "backend": "catboost",
        "parameters": {"depth": 3},
        "transforms": [{"operation": "frequency", "columns": ["x"]}],
    }
    (research / "checkpoint.json").write_text(
        json.dumps(
            {
                "id": "improved",
                "score": score,
                "protocol_id": task.protocol.fingerprint,
                "submission_hash": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    )
    events = [
        {
            "event_id": "experiment",
            "kind": "experiment_finished",
            "payload": {
                "id": "improved",
                "status": "succeeded",
                "executed_recipe": recipe,
                "hypothesis_source": "qwen_proposal",
            },
        },
        {"event_id": "start", "kind": "session_started", "payload": {"id": "inflight"}},
    ]
    (research / "memory/events.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events)
    )
    result = save_run_result(tmp_path, task, {"score": {"value": 100}, "reason": "deadline"}, {})
    assert result["selected"] == "improved"
    assert result["score"] == score and result["recipe"] == recipe
    assert result["result_status"] == "preserved"
    assert result["usage"]["usage_known"] is False
    assert result["usage"]["budget_accounted"] is False
    path.write_text("key,response\n102,4\n101,3\n")
    assert save_run_result(tmp_path, task, {}, {})["selected"] == "baseline"


def test_no_llm_usage_is_known_and_malformed_checkpoint_does_not_destroy_csv(tmp_path):
    task = CsvTaskData().prepare(str(task_fixture(tmp_path)), 42)
    path = Path(task.spec.submission_path)
    path.write_text("key,response\n102,2\n101,1\n")
    (tmp_path / "research").mkdir()
    (tmp_path / "research/checkpoint.json").write_text("{torn")
    result = save_run_result(tmp_path, task, {}, {})
    assert result["result_status"] == "preserved"
    assert result["usage"]["tokens_used"] == 0
    assert result["usage"]["usage_known"] is True


def test_canonical_ensemble_contains_actual_members_weights_and_prediction_rule(tmp_path):
    task = CsvTaskData().prepare(str(task_fixture(tmp_path)), 42)
    Path(task.spec.submission_path).write_text("key,response\n102,2\n101,1\n")
    recipe = {
        "backend": "catboost",
        "parameters": {"depth": 3},
        "transforms": [{"operation": "frequency", "columns": ["x"]}],
    }
    finalist = {
        "id": "ensemble:a+b",
        "score": {"value": 1},
        "weights": [0.25, 0.75],
        "members": [
            {"id": "a", "backend": "catboost", "parameters": {"depth": 3}},
            {"id": "b", "backend": "lightgbm", "parameters": {"num_leaves": 7}},
        ],
    }
    details = {
        "research": {
            "finalization": {
                "selected": finalist["id"],
                "refit_status": "succeeded",
                "finalists": [finalist],
            },
            "reports": [{"id": "a", "status": "succeeded", "executed_recipe": recipe}],
        }
    }
    result = save_run_result(tmp_path, task, details, {})
    assert result["origin"] == "ensemble"
    assert result["recipe"]["weights"] == [0.25, 0.75]
    assert result["recipe"]["members"][0]["recipe"] == recipe
    assert result["recipe"]["members"][1]["parameters"] == {"num_leaves": 7}
    assert result["recipe"]["prediction_rule"] == "weighted mean of values"
