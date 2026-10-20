"""Frozen plans survive restart without rerunning a started control worker."""

import csv
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

import psutil
import pytest

from ml_harness.adapters.data.task import read_csv
from ml_harness.adapters.execution.runner import SubprocessRunner
from ml_harness.adapters.research.finalization import FinalizationWorkspace
from ml_harness.adapters.research.workspace import ResearchWorkspace, write_json
from ml_harness.adapters.storage.submission import AtomicSubmissionWriter, IndependentEvaluator
from ml_harness.application.ports.execution import ExecutionResult, RunRequest
from ml_harness.domain.models import (
    ContractError,
    ExperimentStatus,
    Metric,
    PredictionKind,
    PreparedTask,
    SplitKind,
    TaskKind,
    TaskSpec,
    ValidationProtocol,
)


def setup(tmp_path, control=True):
    train, test, submission = [tmp_path / name for name in ("train.csv", "test.csv", "answer.csv")]
    with train.open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "x", "y"])
        writer.writerows([[f"row{i}", i, float(i)] for i in range(30)])
    test.write_text("id,x\n002,31\n001,32\n")
    submission.write_text("id,y\n002,10\n001,11\n")
    spec = TaskSpec(
        "task",
        str(train),
        str(test),
        "dict",
        "sample",
        str(submission),
        "y",
        "id",
        ("x",),
        TaskKind.REGRESSION,
        Metric.RMSE,
        PredictionKind.VALUES,
        ("id", "y"),
    )
    protocol = ValidationProtocol(
        "search",
        "data",
        SplitKind.RANDOM,
        42,
        tuple(range(18)),
        tuple(range(18, 24)),
        tuple(range(24, 30)) if control else (),
    )
    task = PreparedTask(spec, protocol, 30, 2)
    evaluator = IndependentEvaluator()
    runner = FakeRunner()
    workspace = ResearchWorkspace(task, tmp_path / "research", runner, evaluator, threads=1)
    runner.workspace = workspace
    return task, workspace, runner, evaluator


class FakeRunner:
    def __init__(self):
        self.calls = []
        self.fail_refit = False
        self.crash_control = False
        self.workspace = None

    def run(self, request, timeout):
        assert (self.workspace.path / "finalists.json").exists()
        offset = request.parameters.get("offset", 0)
        self.calls.append((request.mode, offset))
        if request.mode == "refit" and self.fail_refit:
            return ExecutionResult(ExperimentStatus.FAILED, 0.01, detail="synthetic refit failure")
        output = predictions(request.task, offset, request.mode == "refit")
        if request.mode == "control" and self.crash_control:
            write_json(Path(request.output_dir) / "result.json", output)
            raise KeyboardInterrupt("simulate parent death after worker result")
        return ExecutionResult(ExperimentStatus.SUCCEEDED, 0.01, output)


def predictions(task, offset, refit=False):
    _, rows = read_csv(Path(task.spec.test_path if refit else task.spec.train_path))
    selected = rows if refit else [rows[i] for i in task.protocol.validation_indices]
    values = [float(row["x"]) + offset for row in selected]
    return {"ids": [row["id"] for row in selected], "predictions": values, "classes": []}


def register(task, workspace, evaluator, identifier, offset):
    code = f"# immutable candidate {identifier}\n".encode()
    digest, path = workspace.store.save_code(code)
    output = predictions(task, offset)
    artifact = workspace.path / f"{identifier}-validation.json"
    write_json(artifact, output)
    measured = evaluator.evaluate(task, output)
    report = {
        "id": identifier,
        "status": "succeeded",
        "measured_by": "harness",
        "level": "full",
        "protocol_id": task.protocol.fingerprint,
        "full_score": asdict(measured),
        "code_path": path,
        "code_hash": digest,
        "parameters": {"offset": offset},
        "prediction_path": str(artifact),
        "prediction_hash": hashlib.sha256(artifact.read_bytes()).hexdigest(),
    }
    if identifier == "baseline":
        workspace.baseline_report = report
    else:
        workspace.reports.append(report)
    return report


def finish(task, workspace, runner, evaluator):
    return FinalizationWorkspace(
        task, workspace, runner, evaluator, AtomicSubmissionWriter(), threads=1, timeout_seconds=10
    ).finish()


def test_control_shared_once_freeze_ignores_new_reports_and_reuses_selection(tmp_path):
    task, workspace, runner, evaluator = setup(tmp_path)
    register(task, workspace, evaluator, "baseline", -2)
    register(task, workspace, evaluator, "regularization", 1)
    result = finish(task, workspace, runner, evaluator)
    assert result["selected"].startswith("ensemble:")
    assert runner.calls.count(("control", -2)) == runner.calls.count(("control", 1)) == 1
    assert len(result["control_scores"]) == 3
    manifest = (workspace.path / "finalists.json").read_bytes()
    calls = list(runner.calls)
    register(task, workspace, evaluator, "later-must-be-ignored", 0)
    assert finish(task, workspace, runner, evaluator) == result
    assert runner.calls == calls
    assert (workspace.path / "finalists.json").read_bytes() == manifest
    assert not any(item["id"] == "later-must-be-ignored" for item in result["finalists"])


def test_recovered_result_after_parent_crash_never_reexecutes_control(tmp_path):
    task, workspace, runner, evaluator = setup(tmp_path)
    register(task, workspace, evaluator, "regularization", 1)
    runner.crash_control = True
    with pytest.raises(KeyboardInterrupt):
        finish(task, workspace, runner, evaluator)
    runner.crash_control = False
    result = finish(task, workspace, runner, evaluator)
    assert result["selected"] == "regularization"
    assert runner.calls.count(("control", 1)) == 1


def test_live_process_identity_prevents_resume_and_preserves_csv(tmp_path):
    task, workspace, runner, evaluator = setup(tmp_path)
    register(task, workspace, evaluator, "regularization", 1)
    runner.crash_control = True
    with pytest.raises(KeyboardInterrupt):
        finish(task, workspace, runner, evaluator)
    directory = next((workspace.path / "finalization/control").iterdir())
    write_json(
        directory / "process.json",
        {"pid": os.getpid(), "created_at": psutil.Process().create_time()},
    )
    previous = Path(task.spec.submission_path).read_bytes()
    with pytest.raises(ContractError, match="still running"):
        finish(task, workspace, runner, evaluator)
    assert Path(task.spec.submission_path).read_bytes() == previous
    assert len(runner.calls) == 1


def test_started_without_result_never_reexecutes_and_failed_refit_preserves_csv(tmp_path):
    task, workspace, runner, evaluator = setup(tmp_path)
    register(task, workspace, evaluator, "regularization", 1)
    runner.fail_refit = True
    previous = Path(task.spec.submission_path).read_bytes()
    result = finish(task, workspace, runner, evaluator)
    assert result["refit_status"] == "failed_baseline_preserved"
    assert Path(task.spec.submission_path).read_bytes() == previous
    assert finish(task, workspace, runner, evaluator) == result
    assert runner.calls.count(("refit", 1)) == 1


def test_no_control_selects_best_search_and_empty_baseline_has_no_invented_refit(tmp_path):
    task, workspace, runner, evaluator = setup(tmp_path, control=False)
    workspace.baseline_report = {
        "id": "baseline",
        "status": "succeeded",
        "measured_by": "harness",
        "level": "full",
        "protocol_id": task.protocol.fingerprint,
    }
    previous = Path(task.spec.submission_path).read_bytes()
    result = finish(task, workspace, runner, evaluator)
    assert result["selected"] == "baseline" and result["refit_status"] == "not_needed"
    assert not runner.calls and Path(task.spec.submission_path).read_bytes() == previous


def test_checksum_tampering_rejected_before_control(tmp_path):
    task, workspace, runner, evaluator = setup(tmp_path)
    register(task, workspace, evaluator, "regularization", 1)
    manager = FinalizationWorkspace(task, workspace, runner, evaluator, AtomicSubmissionWriter())
    manager._freeze()
    manifest = workspace.path / "finalists.json"
    contents = json.loads(manifest.read_text())
    contents["plan"]["finalists"][0]["weights"] = [0.4]
    manifest.chmod(0o644)
    write_json(manifest, contents)
    with pytest.raises(ContractError, match="checksum"):
        finish(task, workspace, runner, evaluator)
    assert not runner.calls


def test_actual_tiny_worker_control_and_refit(tmp_path):
    task, workspace, _, evaluator = setup(tmp_path)
    code = b"""class Model:
    classes_ = []
    def fit(self, X, y): self.mean = float(y.mean())
    def predict(self, X): return [self.mean] * len(X)
def build_model(*args): return Model()
"""
    digest, path = workspace.store.save_code(code)
    runner = SubprocessRunner()
    validation = runner.run(
        RunRequest(task, str(tmp_path / "search"), candidate_path=path, threads=1), 10
    )
    assert validation.status == ExperimentStatus.SUCCEEDED
    measured = evaluator.evaluate(task, validation.output)
    workspace.reports.append(
        {
            "id": "regularization",
            "status": "succeeded",
            "level": "full",
            "measured_by": "harness",
            "protocol_id": task.protocol.fingerprint,
            "full_score": asdict(measured),
            "code_path": path,
            "code_hash": digest,
            "prediction_path": str(tmp_path / "search/result.json"),
        }
    )
    result = finish(task, workspace, runner, evaluator)
    assert result["selected"] == "regularization" and result["refit_status"] == "succeeded"
    _, rows = read_csv(Path(task.spec.submission_path))
    assert [row["id"] for row in rows] == ["002", "001"]
    assert all(float(row["y"]) == 14.5 for row in rows)


def test_started_control_without_complete_result_is_not_reexecuted(tmp_path):
    task, workspace, runner, evaluator = setup(tmp_path)
    register(task, workspace, evaluator, "regularization", 1)
    runner.crash_control = True
    with pytest.raises(KeyboardInterrupt):
        finish(task, workspace, runner, evaluator)
    next((workspace.path / "finalization/control").glob("*/result.json")).unlink()
    runner.crash_control = False
    previous = Path(task.spec.submission_path).read_bytes()
    result = finish(task, workspace, runner, evaluator)
    assert result["refit_status"] == "failed_baseline_preserved"
    assert runner.calls == [("control", 1)]
    assert Path(task.spec.submission_path).read_bytes() == previous


def test_failed_best_refit_falls_back_only_to_another_checked_finalist(tmp_path):
    task, workspace, runner, evaluator = setup(tmp_path)
    register(task, workspace, evaluator, "regularization", 0)
    register(task, workspace, evaluator, "baseline", 2)
    original = runner.run

    def fail_best(request, timeout):
        if request.mode == "refit" and request.parameters["offset"] == 0:
            runner.calls.append(("refit", 0))
            return ExecutionResult(ExperimentStatus.FAILED, 0.01, detail="best failed")
        return original(request, timeout)

    runner.run = fail_best
    result = finish(task, workspace, runner, evaluator)
    assert result["selected"] == "baseline" and result["refit_status"] == "succeeded"
    assert len(result["control_scores"]) == 2
    assert runner.calls.count(("control", 0)) == runner.calls.count(("control", 2)) == 1


def test_no_holdout_uses_best_search_and_invalid_search_artifact_is_excluded(tmp_path):
    task, workspace, runner, evaluator = setup(tmp_path, control=False)
    register(task, workspace, evaluator, "baseline", 2)
    register(task, workspace, evaluator, "regularization", 0)
    invalid = register(task, workspace, evaluator, "unmeasured", -1)
    invalid["full_score"]["value"] = 0
    result = finish(task, workspace, runner, evaluator)
    assert result["selected"] == "regularization"
    assert result["selection_reason"] == "no_control_partition"
    assert not any(mode == "control" for mode, _ in runner.calls)
    assert not any(item["id"] == "unmeasured" for item in result["finalists"])


def test_worker_artifacts_without_runtime_state_are_recovered_without_control_rerun(tmp_path):
    task, workspace, runner, evaluator = setup(tmp_path)
    register(task, workspace, evaluator, "regularization", 1)
    runner.crash_control = True
    with pytest.raises(KeyboardInterrupt):
        finish(task, workspace, runner, evaluator)
    (workspace.path / "finalization-state.json").unlink()
    runner.crash_control = False
    result = finish(task, workspace, runner, evaluator)
    assert result["selected"] == "regularization"
    assert runner.calls.count(("control", 1)) == 1


def test_same_pid_with_different_create_time_is_not_a_live_worker(tmp_path):
    task, workspace, runner, evaluator = setup(tmp_path)
    register(task, workspace, evaluator, "regularization", 1)
    runner.crash_control = True
    with pytest.raises(KeyboardInterrupt):
        finish(task, workspace, runner, evaluator)
    directory = next((workspace.path / "finalization/control").iterdir())
    write_json(
        directory / "process.json",
        {"pid": os.getpid(), "created_at": psutil.Process().create_time() - 0.5},
    )
    runner.crash_control = False
    assert finish(task, workspace, runner, evaluator)["selected"] == "regularization"
    assert runner.calls.count(("control", 1)) == 1


def test_control_cannot_change_development_winner_or_frozen_weights(tmp_path):
    task, workspace, runner, evaluator = setup(tmp_path)
    register(task, workspace, evaluator, "baseline", -3)
    register(task, workspace, evaluator, "candidate", 1)
    original = runner.run

    def reversed_control(request, timeout):
        if request.mode == "control":
            runner.calls.append(("control", request.parameters["offset"]))
            # Make the baseline perfect and the development ensemble bad on control.
            offset = 0 if request.parameters["offset"] == -3 else 100
            return ExecutionResult(
                ExperimentStatus.SUCCEEDED, 0.01, predictions(request.task, offset)
            )
        return original(request, timeout)

    runner.run = reversed_control
    result = finish(task, workspace, runner, evaluator)
    assert result["selected"].startswith("ensemble:")
    ensemble = next(item for item in result["finalists"] if item["id"] == result["selected"])
    assert ensemble["weights"] == [0.75, 0.25]
    assert result["control_scores"]["baseline"]["value"] == 0
    assert result["selection_reason"] == "development_frozen_control_audit"


def test_checkpoint_publishes_improvement_without_control_or_freezing(tmp_path):
    from ml_harness.adapters.research.checkpoint import publish_checkpoint
    from ml_harness.domain.budget import WallClockBudget

    task, workspace, _, evaluator = setup(tmp_path)
    register(task, workspace, evaluator, "candidate", 1)

    class Clock:
        def now(self):
            return 0.0

    class Runner:
        def __init__(self):
            self.calls = []

        def run(self, request, timeout):
            self.calls.append(request.mode)
            assert request.mode == "refit"
            return ExecutionResult(
                ExperimentStatus.SUCCEEDED, 1, predictions(task, request.parameters["offset"], True)
            )

    runner = Runner()
    budget = WallClockBudget(1000, validation_seconds=2, refit_seconds=3)
    checkpoint = publish_checkpoint(
        task, workspace, runner, AtomicSubmissionWriter(), budget, Clock()
    )
    assert checkpoint["id"] == "candidate"
    assert runner.calls == ["refit"]
    assert not (workspace.path / "finalists.json").exists()
    assert (workspace.path / "checkpoint.json").exists()
    assert (
        publish_checkpoint(task, workspace, runner, AtomicSubmissionWriter(), budget, Clock())
        is None
    )
    register(task, workspace, evaluator, "improved", 0)
    constrained = WallClockBudget(20, validation_seconds=2, refit_seconds=3)
    previous = Path(task.spec.submission_path).read_bytes()
    assert (
        publish_checkpoint(task, workspace, runner, AtomicSubmissionWriter(), constrained, Clock())
        is None
    )
    assert runner.calls == ["refit"]
    assert Path(task.spec.submission_path).read_bytes() == previous
    checkpoint = publish_checkpoint(
        task, workspace, runner, AtomicSubmissionWriter(), budget, Clock()
    )
    assert checkpoint["id"] == "improved"
    assert runner.calls == ["refit", "refit"]


def test_first_checkpoint_requires_improvement_over_baseline(tmp_path):
    from ml_harness.adapters.research.checkpoint import publish_checkpoint
    from ml_harness.domain.budget import WallClockBudget

    task, workspace, runner, evaluator = setup(tmp_path)
    register(task, workspace, evaluator, "baseline", 1)
    register(task, workspace, evaluator, "worse", 2)

    class Clock:
        def now(self):
            return 0

    previous = Path(task.spec.submission_path).read_bytes()
    assert (
        publish_checkpoint(
            task, workspace, runner, AtomicSubmissionWriter(), WallClockBudget(1000), Clock()
        )
        is None
    )
    assert runner.calls == []
    assert Path(task.spec.submission_path).read_bytes() == previous
    assert not (workspace.path / "checkpoint.json").exists()


@pytest.mark.parametrize("matching_checkpoint", [True, False])
def test_refit_timeout_cannot_replace_verified_checkpoint_with_worse_baseline(
    tmp_path, matching_checkpoint
):
    from ml_harness.adapters.research.checkpoint import publish_checkpoint
    from ml_harness.domain.budget import WallClockBudget

    task, workspace, runner, evaluator = setup(tmp_path)
    register(task, workspace, evaluator, "baseline", 4)
    register(task, workspace, evaluator, "checkpoint-winner", 1)

    class Clock:
        def now(self):
            return 0.0

    class CheckpointRunner:
        def run(self, request, timeout):
            return ExecutionResult(
                ExperimentStatus.SUCCEEDED, 1, predictions(task, request.parameters["offset"], True)
            )

    checkpoint = publish_checkpoint(
        task,
        workspace,
        CheckpointRunner(),
        AtomicSubmissionWriter(),
        WallClockBudget(1000, validation_seconds=2, refit_seconds=3),
        Clock(),
    )
    assert checkpoint["id"] == "checkpoint-winner"
    previous = Path(task.spec.submission_path).read_bytes()
    if not matching_checkpoint:
        checkpoint["submission_hash"] = "stale"
        write_json(workspace.path / "checkpoint.json", checkpoint)
    register(task, workspace, evaluator, "later-best", 0)
    original = runner.run

    def timeout_improved(request, timeout):
        if request.mode == "refit" and request.parameters["offset"] < 4:
            runner.calls.append(("refit", request.parameters["offset"]))
            return ExecutionResult(ExperimentStatus.TIMED_OUT, 10, detail="deadline")
        return original(request, timeout)

    runner.run = timeout_improved
    result = finish(task, workspace, runner, evaluator)
    if matching_checkpoint:
        assert result["refit_status"] == "failed_checkpoint_preserved"
        assert result["selected"] == "checkpoint-winner"
        assert result["preserved_checkpoint"]["submission_hash"] == checkpoint["submission_hash"]
        assert ("refit", 4) not in runner.calls
        assert Path(task.spec.submission_path).read_bytes() == previous
        # Control still ran once for every frozen member before the preservation fallback.
        assert set(offset for mode, offset in runner.calls if mode == "control") == {0, 1, 4}
        from ml_harness.adapters.storage.run_result import save_run_result

        canonical = save_run_result(
            tmp_path,
            task,
            {"research": {"finalization": result, "reports": workspace.reports}},
            {},
        )
        assert canonical["selected"] == "checkpoint-winner"
        assert canonical["result_status"] == "preserved"
        assert canonical["score"] == checkpoint["score"]
    else:
        assert result["selected"] == "baseline"
        assert result["refit_status"] == "succeeded"
        assert ("refit", 4) in runner.calls
