"""Resume reuses measured work and validates saved predictions before research."""

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from ml_harness.adapters.storage.lease import RunLease
from ml_harness.adapters.storage.submission import AtomicSubmissionWriter
from ml_harness.application.use_cases.baseline import BaselineResult
from ml_harness.domain.models import (
    ContractError,
    Metric,
    PredictionKind,
    PreparedTask,
    Score,
    SplitKind,
    TaskKind,
    TaskSpec,
    ValidationProtocol,
)
from ml_harness.entrypoints import cli


def fixture_task(root, variant="regression"):
    train, test = root / "train.csv", root / "test.csv"
    train.write_text("id,x,y\n1,1,0\n2,2,1\n3,3,0\n4,4,1\n")
    test.write_text("id,x\n001,1\nNA,2\n")
    kind, metric, prediction, classes, columns = (
        TaskKind.REGRESSION,
        Metric.RMSE,
        PredictionKind.VALUES,
        (),
        ("id", "y"),
    )
    if variant == "labels":
        kind, metric, prediction, classes = (
            TaskKind.BINARY,
            Metric.ACCURACY,
            PredictionKind.LABELS,
            (0, 1),
        )
    elif variant == "binary":
        kind, metric, prediction, classes = (
            TaskKind.BINARY,
            Metric.LOG_LOSS,
            PredictionKind.PROBABILITIES,
            ("z", "a"),
        )
        columns = ("id", "positive")
    elif variant == "multiclass":
        kind, metric, prediction, classes = (
            TaskKind.MULTICLASS,
            Metric.LOG_LOSS,
            PredictionKind.PROBABILITIES,
            ("c", "a", "b"),
        )
        columns = ("id", "c", "a", "b")
    spec = TaskSpec(
        str(root / "task.md"),
        str(train),
        str(test),
        str(root / "dict.csv"),
        str(root / "sample.csv"),
        str(root / "submission.csv"),
        "y",
        "id",
        ("x",),
        kind,
        metric,
        prediction,
        columns,
        classes,
    )
    protocol = ValidationProtocol("fingerprint", "data", SplitKind.RANDOM, 42, (0, 1), (2, 3))
    return PreparedTask(spec, protocol, 4, 2)


def setup_cli(monkeypatch, tmp_path, variant="regression"):
    task = fixture_task(tmp_path, variant)
    run_dir = tmp_path / "run"
    counts = {"baseline": 0, "research": 0}
    monkeypatch.setattr(cli, "prepare_task", lambda *args: task)

    def baseline(prepared, runner, evaluator, writer, **kwargs):
        counts["baseline"] += 1
        spec = prepared.spec
        if spec.prediction_kind == PredictionKind.PROBABILITIES:
            row = [0.2, 0.8] if len(spec.classes) == 2 else [0.1, 0.3, 0.6]
            predictions = [row, row]
        else:
            predictions = [0, 1]
        AtomicSubmissionWriter().write(
            prepared,
            {
                "ids": ["001", "NA"],
                "classes": list(spec.classes),
                "predictions": predictions,
            },
        )
        return BaselineResult(
            Score(0.5, spec.metric, prepared.protocol.fingerprint, "full"),
            spec.submission_path,
            1.0,
            2.0,
        )

    def research(prepared, result, directory, **kwargs):
        counts["research"] += 1
        assert json.loads((run_dir / "baseline.json").read_text())["score"]["value"] == 0.5
        assert kwargs["max_sessions"] == 5
        with pytest.raises(ContractError, match="in use"):
            with RunLease(run_dir / ".run.lock"):
                pass
        return {"status": "fake"}

    monkeypatch.setattr(cli, "run_baseline", baseline)
    monkeypatch.setattr(cli, "research_after_baseline", research)

    def invoke(*extra):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "solution.py",
                str(tmp_path / "task.md"),
                "--run-dir",
                str(run_dir),
                "--research-sessions",
                "5",
                *extra,
            ],
        )
        return cli.main()

    return task, run_dir, counts, invoke


@pytest.mark.parametrize("variant", ["regression", "labels", "binary", "multiclass"])
def test_resume_reuses_baseline_and_validates_current_submission(monkeypatch, tmp_path, variant):
    task, run_dir, counts, invoke = setup_cli(monkeypatch, tmp_path, variant)
    assert invoke() == 0
    manifest = (run_dir / "prepared.json").read_bytes()
    baseline = (run_dir / "baseline.json").read_bytes()
    if variant == "regression":
        Path(task.spec.submission_path).write_text("id,y\n001,99\nNA,101\n")
    assert invoke("--resume") == 0
    assert counts == {"baseline": 1, "research": 2}
    assert (run_dir / "prepared.json").read_bytes() == manifest
    assert (run_dir / "baseline.json").read_bytes() == baseline
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["score"]["value"] == 0.5


@pytest.mark.parametrize(
    "damage",
    [
        "manifest_missing",
        "manifest_json",
        "fingerprint",
        "baseline_missing",
        "baseline_score",
        "submission_missing",
        "schema",
        "order",
        "nonfinite",
        "value",
    ],
)
def test_resume_rejects_invalid_saved_state_without_rerun(monkeypatch, tmp_path, damage, capsys):
    task, run_dir, counts, invoke = setup_cli(monkeypatch, tmp_path)
    assert invoke("--baseline-only") == 0
    manifest_path = run_dir / "prepared.json"
    baseline_path = run_dir / "baseline.json"
    submission = Path(task.spec.submission_path)
    if damage == "manifest_missing":
        manifest_path.unlink()
    elif damage == "manifest_json":
        manifest_path.write_text("broken")
    elif damage == "fingerprint":
        newer = replace(task, protocol=replace(task.protocol, fingerprint="changed"))
        monkeypatch.setattr(cli, "prepare_task", lambda *args: newer)
    elif damage == "baseline_missing":
        baseline_path.unlink()
    elif damage == "baseline_score":
        baseline = json.loads(baseline_path.read_text())
        baseline["score"]["protocol_id"] = "wrong"
        baseline_path.write_text(json.dumps(baseline))
    elif damage == "submission_missing":
        submission.unlink()
    else:
        submission.write_text(
            {
                "schema": "id,other\n001,1\nNA,2\n",
                "order": "id,y\nNA,1\n001,2\n",
                "nonfinite": "id,y\n001,NaN\nNA,2\n",
                "value": "id,y\n001,garbage\nNA,2\n",
            }[damage]
        )
    previous = manifest_path.read_bytes() if manifest_path.exists() else None
    assert invoke("--resume") == 2
    assert "Cannot resume saved baseline" in capsys.readouterr().err
    assert counts == {"baseline": 1, "research": 0}
    assert (manifest_path.read_bytes() if manifest_path.exists() else None) == previous


def test_existing_directory_requires_resume_and_lockfile_is_ignored(monkeypatch, tmp_path):
    _, run_dir, counts, invoke = setup_cli(monkeypatch, tmp_path)
    run_dir.mkdir()
    (run_dir / ".run.lock").touch()
    assert invoke("--baseline-only") == 0
    previous = (run_dir / "prepared.json").read_bytes()
    assert invoke("--baseline-only") == 2
    assert counts["baseline"] == 1
    assert (run_dir / "prepared.json").read_bytes() == previous


def test_lease_blocks_another_process_and_survives_release(tmp_path):
    path = tmp_path / ".run.lock"
    script = """import sys
from pathlib import Path
from ml_harness.adapters.storage.lease import RunLease
from ml_harness.domain.models import ContractError
try:
    with RunLease(Path(sys.argv[1])): pass
except ContractError:
    sys.exit(7)
"""
    command = [sys.executable, "-c", script, str(path)]
    with RunLease(path):
        result = subprocess.run(command, cwd=Path(__file__).resolve().parents[2], timeout=10)
        assert result.returncode == 7
    inode = path.stat().st_ino
    assert (
        subprocess.run(command, cwd=Path(__file__).resolve().parents[2], timeout=10).returncode == 0
    )
    assert path.exists() and path.stat().st_ino == inode


@pytest.mark.parametrize("arguments", [["--resume"], ["--research-sessions", "0"]])
def test_invalid_resume_arguments(monkeypatch, arguments):
    monkeypatch.setattr(sys, "argv", ["solution.py", *arguments])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2


@pytest.mark.parametrize(
    "variant,contents",
    [
        ("binary", "id,positive\n001,1.1\nNA,0.5\n"),
        ("multiclass", "id,c,a,b\n001,0.1,0.2,0.3\nNA,0.1,0.2,0.7\n"),
        ("labels", "id,y\n001,unknown\nNA,0\n"),
    ],
)
def test_resume_rejects_invalid_probability_or_class(monkeypatch, tmp_path, variant, contents):
    task, _, counts, invoke = setup_cli(monkeypatch, tmp_path, variant)
    assert invoke("--baseline-only") == 0
    Path(task.spec.submission_path).write_text(contents)
    assert invoke("--resume") == 2
    assert counts == {"baseline": 1, "research": 0}
