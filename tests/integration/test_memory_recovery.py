"""Recover authoritative facts in fresh processes and never replay completed training."""

import json
import os
import subprocess
import sys

import psutil
import pytest
from test_prepare_task import task_fixture

from ml_harness.adapters.data.task import CsvTaskData
from ml_harness.adapters.execution.runner import SubprocessRunner
from ml_harness.adapters.research.workspace import STARTER, ResearchWorkspace
from ml_harness.adapters.storage.submission import IndependentEvaluator
from ml_harness.domain.models import ContractError

ARGUMENTS = {
    "experiment_id": "once",
    "hypothesis": "regularization",
    "expected_effect": "lower RMSE",
    "parent_id": None,
}


def workspace(tmp_path):
    task = CsvTaskData().prepare(str(task_fixture(tmp_path)), 42)
    work = ResearchWorkspace(
        task,
        tmp_path / "research",
        SubprocessRunner(),
        IndependentEvaluator(),
        threads=1,
        timeout_seconds=10,
    )
    (work.path / "candidate.py").write_text(STARTER.replace('"iterations": 200', '"iterations": 3'))
    return work


def test_completed_training_is_not_reexecuted_in_fresh_process(tmp_path):
    work = workspace(tmp_path)
    first = work.run_experiment(ARGUMENTS)
    assert first["status"] == "succeeded"
    script = """
import json,sys
from pathlib import Path
from ml_harness.adapters.data.task import CsvTaskData
from ml_harness.adapters.research.workspace import ResearchWorkspace
from ml_harness.adapters.storage.submission import IndependentEvaluator
class NeverRun:
    def run(self,*args):
        raise AssertionError('Completed training was replayed')
task=CsvTaskData().prepare(sys.argv[1],42)
w=ResearchWorkspace(task,Path(sys.argv[2]),NeverRun(),IndependentEvaluator())
result=w.run_experiment(json.loads(sys.argv[3]))
assert result['status']=='succeeded' and len(w.reports)==1
print(result['code_hash'])
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            work.task.spec.task_path,
            str(work.path),
            json.dumps(ARGUMENTS),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.stdout.strip() == first["code_hash"]
    assert len(work.store.read_events()) == 2


def test_result_is_recovered_if_parent_died_before_terminal_record(tmp_path):
    work = workspace(tmp_path)
    first = work.run_experiment(ARGUMENTS)
    journal = work.store.path
    journal.write_bytes(journal.read_bytes().splitlines(keepends=True)[0])

    class NeverRun:
        def run(self, *args):
            raise AssertionError("Should evaluate existing full result")

    restored = ResearchWorkspace(work.task, work.path, NeverRun(), IndependentEvaluator())
    report = restored.read_experiments()[0]
    assert report["status"] == "succeeded" and report["recovered"]
    assert report["full_score"] == first["full_score"]
    assert len(restored.store.read_events()) == 2


def test_live_recorded_worker_is_not_mistaken_for_interrupted(tmp_path):
    work = workspace(tmp_path)
    record = {"id": "active", "status": "running"}
    work.store.append(
        {"event_id": "experiment:active:started", "kind": "experiment_started", "payload": record}
    )
    directory = work.path / "experiments/active/smoke"
    directory.mkdir(parents=True)
    process = psutil.Process(os.getpid())
    (directory / "process.json").write_text(
        json.dumps({"pid": process.pid, "created_at": process.create_time()})
    )
    with pytest.raises(ContractError, match="still running"):
        ResearchWorkspace(work.task, work.path, SubprocessRunner(), IndependentEvaluator())
    assert len(work.store.read_events()) == 1


def test_failed_interrupted_attempt_remains_in_next_context(tmp_path):
    work = workspace(tmp_path)
    record = {
        "id": "interrupted",
        "status": "running",
        "hypothesis": "bad idea",
        "protocol_id": work.task.protocol.fingerprint,
        "level": "full",
    }
    work.store.append(
        {
            "event_id": "experiment:interrupted:started",
            "kind": "experiment_started",
            "payload": record,
        }
    )
    restored = ResearchWorkspace(work.task, work.path, SubprocessRunner(), IndependentEvaluator())
    assert restored.reports[0]["status"] == "failed"
    assert "Interrupted" in restored.reports[0]["error"]
    events = restored.store.read_events()
    assert events[-1]["kind"] == "experiment_finished"
