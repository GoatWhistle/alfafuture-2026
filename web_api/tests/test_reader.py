import json
from pathlib import Path

from app import reader


def _event(payload):
    return json.dumps(
        {"event_id": "x:" + payload["id"], "kind": "experiment_finished", "payload": payload}
    )


def test_v2_payload_separates_requested_and_executed(tmp_path: Path):
    payload = {
        "id": "a-1",
        "parent_id": "baseline",
        "hypothesis": "H",
        "status": "succeeded",
        "backend": "catboost",
        "parameters": {},
        "level": "full",
        "full_score": {"value": 0.91, "metric": "roc_auc"},
        "elapsed_seconds": 3.0,
        "requested_proposal": {
            "target_error": "ошибки в классе 1",
            "rationale": "R",
            "parameters": {"depth": 9},
            "transforms": [],
            "tune": False,
        },
        "executed_recipe": {
            "parameters": {"depth": 8},
            "transforms": [{"operation": "frequency", "columns": ["f2"]}],
            "python_transform": "",
            "tuning": {},
        },
        "admission": {"allowed": True},
    }
    [row] = reader.experiments([json.loads(_event(payload))], "roc_auc", 0.90, tmp_path)
    assert row["proposedParams"] == {"depth": 9} and row["executedParams"] == {"depth": 8}
    assert row["observed"] == "ошибки в классе 1" and row["proposed"] == "R"
    assert row["featureChanges"] == ["frequency: f2"]
    assert row["delta"] == 0.91 - 0.90 and row["status"] == "ok"


def test_rejected_by_admission_reports_numbers(tmp_path: Path):
    payload = {
        "id": "a-2",
        "parent_id": "baseline",
        "status": "rejected",
        "error": "admission",
        "level": "full",
        "requested_proposal": {"parameters": {"iterations": 600}, "rationale": "R"},
        "admission": {"allowed": False, "estimated_full_seconds": 1140, "available_seconds": 610},
    }
    [row] = reader.experiments([json.loads(_event(payload))], "roc_auc", 0.9, tmp_path)
    assert row["status"] == "rejected" and "1140" in row["measured"] and "610" in row["measured"]
    assert row["executedParams"] == {} and row["proposedParams"] == {"iterations": 600}


def test_optuna_shows_refit_parameters(tmp_path: Path):
    payload = {
        "id": "numeric-00001",
        "parent_id": "baseline",
        "hypothesis": "Continue train-only lightgbm study, batch 1",
        "status": "succeeded",
        "backend": "lightgbm",
        "parameters": {},
        "level": "full",
        "full_score": {"value": 0.9, "metric": "roc_auc"},
        "refit_parameters": {"backend": "lightgbm", "num_leaves": 28, "tuning": {"enabled": False}},
    }
    [row] = reader.experiments([json.loads(_event(payload))], "roc_auc", 0.9, tmp_path)
    assert row["executedParams"] == {"num_leaves": 28}
    assert row["executed"] == "num_leaves 28"


def test_best_is_the_selected_finalist(tmp_path: Path):
    harness = tmp_path / "harness"
    harness.mkdir()
    (harness / "baseline.json").write_text(
        json.dumps({"score": {"value": 0.90, "metric": "roc_auc"}})
    )
    summary = {
        "research": {
            "status": "budget_exhausted",
            "finalization": {
                "selected": "ensemble:baseline+n1",
                "finalists": [
                    {
                        "id": "ensemble:baseline+n1",
                        "score": {"value": 0.911, "metric": "roc_auc", "level": "full"},
                    }
                ],
            },
        }
    }
    (harness / "summary.json").write_text(json.dumps(summary))
    facts = reader.run_facts(tmp_path)
    assert facts["best"] == 0.911
    assert facts["selected"] == {"id": "ensemble:baseline+n1", "score": 0.911}


def test_failed_qwen_session_is_shown_as_attempt(tmp_path: Path):
    event = {
        "event_id": "session:abc",
        "kind": "session_finished",
        "payload": {
            "id": "abcdef0123",
            "experiment_ids": [],
            "reason": "api_error",
            "error_detail": "Qwen APIConnectionError after 3 attempt(s)",
        },
    }
    [row] = reader.experiments([event], "roc_auc", 0.9, tmp_path)
    assert row["kind"] == "llm_request" and row["status"] == "failed"
    assert row["error"] == "Qwen APIConnectionError after 3 attempt(s)"
    assert row["score"] is None
