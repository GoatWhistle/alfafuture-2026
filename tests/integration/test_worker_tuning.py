"""Changing outer held-out targets cannot change inner tuning or its predictions."""

import json
from pathlib import Path

from test_prepare_task import task_fixture, write_csv

from ml_harness.adapters.data.task import CsvTaskData, read_csv
from ml_harness.adapters.execution.runner import SubprocessRunner
from ml_harness.application.ports.execution import RunRequest
from ml_harness.domain.models import ExperimentStatus


def test_worker_passes_time_context_without_outer_target_leakage(tmp_path):
    path = task_fixture(tmp_path, role="time")
    task = CsvTaskData().prepare(str(path), 42)
    params = {"iterations": 3, "tuning": {"trials": 2, "time_limit": 5}}
    runner = SubprocessRunner()
    first = runner.run(RunRequest(task, str(tmp_path / "first"), parameters=params, threads=1), 10)
    assert first.status == ExperimentStatus.SUCCEEDED
    columns, rows = read_csv(Path(task.spec.train_path))
    for index in task.protocol.validation_indices:
        rows[index][task.spec.target] = str(float(rows[index][task.spec.target]) + 1000)
    write_csv(
        Path(task.spec.train_path), columns, [[row[column] for column in columns] for row in rows]
    )
    changed = CsvTaskData().prepare(str(path), 42)
    assert changed.protocol.validation_indices == task.protocol.validation_indices
    assert changed.protocol.fingerprint != task.protocol.fingerprint
    second = runner.run(
        RunRequest(changed, str(tmp_path / "second"), parameters=params, threads=1), 10
    )
    assert second.status == ExperimentStatus.SUCCEEDED
    for result in (first, second):
        assert result.output["tuning_report"]["inner_split"]["kind"] == "time"
        split = result.output["tuning_report"]["inner_split"]
        assert set(split["train_indices"]).isdisjoint(split["validation_indices"])
        assert max(split["train_indices"] + split["validation_indices"]) < len(
            task.protocol.train_indices
        )
    assert first.output["predictions"] == second.output["predictions"]
    assert first.output["tuning_report"] == second.output["tuning_report"]
    assert (tmp_path / "first/tuning.json").exists()


def test_all_failed_trials_keep_structured_tuning_artifact(tmp_path):
    task = CsvTaskData().prepare(str(task_fixture(tmp_path)), 42)
    output = tmp_path / "failed"
    result = SubprocessRunner().run(
        RunRequest(
            task,
            str(output),
            threads=1,
            parameters={"iterations": -1, "tuning": {"trials": 1, "time_limit": 5}},
        ),
        10,
    )
    assert result.status == ExperimentStatus.FAILED
    report = json.loads((output / "tuning.json").read_text())
    assert report["best"] is None and report["trials"][0]["state"] == "FAIL"
    assert not (output / "result.json").exists()
