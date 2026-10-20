import json
import os
import time

import psutil
import pytest
from fastapi.testclient import TestClient

from .conftest import ORIGIN, create


def wait_status(client: TestClient, run_id: str, wanted: str, timeout: float = 20) -> dict:
    end = time.time() + timeout
    while time.time() < end:
        run = client.get(f"/api/runs/{run_id}").json()
        if run["status"] == wanted:
            return run
        time.sleep(0.1)
    raise AssertionError(f"status {run['status']} != {wanted}")


def test_full_cycle_reads_artifacts_and_serves_own_csv(client, settings):
    run_id = create(client)
    assert client.post(f"/api/runs/{run_id}/start").status_code == 200
    run = wait_status(client, run_id, "completed")
    assert run["baseline"] == 0.8 and run["best"] == 0.83
    assert run["metric"] == "roc_auc" and run["stage"] == "result"
    [experiment] = run["experiments"]
    assert experiment["id"] == "e1" and experiment["delta"] == pytest.approx(0.03)
    assert experiment["executedParams"] == {"depth": 6}
    assert run["finalCsv"] is True
    final = client.get(f"/api/runs/{run_id}/download/final")
    assert final.text == "id,target\n1,0.9\n"
    assert client.get(f"/api/runs/{run_id}/download/report").status_code == 200
    assert "token" not in json.dumps(run).lower().replace("tokens", "")


def test_duplicate_start_creates_no_second_process(client):
    run_id = create(client)
    assert client.post(f"/api/runs/{run_id}/start").status_code == 200
    assert client.post(f"/api/runs/{run_id}/start").status_code == 409
    other = create(client)
    assert client.post(f"/api/runs/{other}/start").status_code == 409  # single active run
    wait_status(client, run_id, "completed")


def test_viewing_does_not_modify_journal(client, settings):
    run_id = create(client)
    client.post(f"/api/runs/{run_id}/start")
    wait_status(client, run_id, "completed")
    journal = settings.runs_root / run_id / "harness/research/memory/events.jsonl"
    before = journal.read_bytes()
    assert before.endswith(b"fin")  # torn tail present
    for _ in range(3):
        client.get(f"/api/runs/{run_id}")
        client.get(f"/api/runs/{run_id}/experiments/e1")
    assert journal.read_bytes() == before
    assert not (journal.parent / "recovery").exists()


def test_abort_kills_tree_and_keeps_csv(client, settings, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "hang")
    run_id = create(client)
    client.post(f"/api/runs/{run_id}/start")
    child_file = settings.runs_root / run_id / "harness/child.pid"
    end = time.time() + 10
    while not child_file.exists() and time.time() < end:
        time.sleep(0.1)
    child = int(child_file.read_text())
    assert psutil.pid_exists(child)
    response = client.post(f"/api/runs/{run_id}/abort")
    assert response.status_code == 200 and response.json()["status"] == "stopped"
    time.sleep(0.3)
    assert not psutil.pid_exists(child) or psutil.Process(child).status() == psutil.STATUS_ZOMBIE
    assert client.get(f"/api/runs/{run_id}/download/current").status_code == 200
    assert client.get(f"/api/runs/{run_id}/download/final").status_code == 409


def test_failure_keeps_current_csv(client, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "fail")
    run_id = create(client)
    client.post(f"/api/runs/{run_id}/start")
    run = wait_status(client, run_id, "failed")
    assert run["csvAvailable"] and not run["finalCsv"]
    assert client.get(f"/api/runs/{run_id}/download/current").text == "id,target\n1,0.9\n"


def test_status_survives_restart(client, settings):
    from app.main import create_app
    from fastapi.testclient import TestClient as Fresh

    run_id = create(client)
    client.post(f"/api/runs/{run_id}/start")
    wait_status(client, run_id, "completed")
    again = Fresh(
        create_app(settings), base_url="http://127.0.0.1:8765", headers={"Origin": ORIGIN}
    )
    assert again.get(f"/api/runs/{run_id}").json()["status"] == "completed"
    assert [r["id"] for r in again.get("/api/runs").json()] == [run_id]


def test_downloads_are_per_run(client):
    a = create(client)
    client.post(f"/api/runs/{a}/start")
    wait_status(client, a, "completed")
    b = create(client)
    assert client.get(f"/api/runs/{b}/download/current").status_code == 404
    assert client.get(f"/api/runs/{a}/download/current").status_code == 200


def test_origin_and_host_are_checked(client, settings):
    bare = TestClient(client.app, base_url="http://127.0.0.1:8765")
    assert bare.post("/api/runs/abc/abort").status_code == 403
    evil = TestClient(
        client.app, base_url="http://127.0.0.1:8765", headers={"Origin": "http://evil.test"}
    )
    assert evil.post("/api/runs/abc/abort").status_code == 403
    rebinding = TestClient(client.app, base_url="http://evil.test:8765")
    assert rebinding.get("/api/runs").status_code == 403
    assert client.get("/api/runs").status_code == 200


@pytest.mark.parametrize("run_id", ["../etc", "..%2f..", "0" * 31, "x" * 32])
def test_arbitrary_ids_never_name_paths(client, run_id):
    assert client.get(f"/api/runs/{run_id}").status_code == 404
    assert client.get(f"/api/runs/{run_id}/download/current").status_code == 404


def test_description_paths_are_confined(client):
    bad = {"title": "t", "description": "см. ../../secret/train.csv"}
    from .conftest import files

    assert client.post("/api/runs", data=bad, files=files()).status_code == 422
    bad["description"] = "данные в /etc/passwd.csv"
    assert client.post("/api/runs", data=bad, files=files()).status_code == 422


def test_description_prefix_is_rewritten_to_run_files(client, settings):
    run_id = create(client, description="Файл multiclass/participants/train.csv — обучение")
    task = (settings.runs_root / run_id / "inputs/task.md").read_text(encoding="utf-8")
    assert "`train.csv`" in task and "multiclass/" not in task


def test_no_secrets_or_paths_in_responses(client, settings, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-value")
    run_id = create(client)
    client.post(f"/api/runs/{run_id}/start")
    run = wait_status(client, run_id, "completed")
    text = json.dumps(run) + client.get("/api/qwen").text
    assert "sk-secret-value" not in text
    assert str(settings.runs_root) not in text
    assert os.environ["OPENAI_API_KEY"] not in client.get("/api/runs").text
