"""Real CLI/worker interruption preserves a ready CSV and hides control targets."""

import csv
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import Mock

import pytest

from ml_harness.adapters.data.task import CsvTaskData
from ml_harness.adapters.research.workspace import ResearchWorkspace
from ml_harness.adapters.storage.submission import load_submission

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]


def fixture_task(root: Path, count: int = 60) -> Path:
    files = {
        "train.csv": (["id", "x", "y"], [[str(i), i, i % 7] for i in range(count)]),
        "test.csv": (["id", "x"], [["002", 2], ["NA", 1]]),
        "sample_submission.csv": (["id", "y"], [["NA", 0], ["002", 0]]),
        "data_dictionary.csv": (
            ["feature", "role", "dtype"],
            [["id", "id", "object"], ["x", "feature", "float64"], ["y", "target", "float64"]],
        ),
    }
    for name, (header, rows) in files.items():
        with (root / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            writer.writerows(rows)
    path = root / "task.md"
    path.write_text('```task-json\n{"metric":"rmse"}\n```')
    return path


def launch_slow_cli(
    root: Path, path: Path, *, budget: float, worker_timeout: float, slow_stage: str = "fit"
) -> subprocess.Popen:
    candidate = root / "slow_candidate.py"
    candidate.write_text(
        """import os
import time
from pathlib import Path
class SlowModel:
    def fit(self, X, y):
        Path(__file__).with_name('worker_started').write_text(str(os.getpid()))
        if SLOW_STAGE == 'fit':
            time.sleep(120)
    def predict(self, X):
        Path(__file__).with_name('predict_started').write_text(str(os.getpid()))
        if SLOW_STAGE == 'predict':
            time.sleep(120)
        return [0] * len(X)
def build_model(kind, parameters, threads, seed):
    return SlowModel()
""".replace("SLOW_STAGE", repr(slow_stage))
    )
    # Inject only the deterministic slow candidate. CLI parsing, task preparation,
    # deadline handling, runner process tree and worker all execute their real code.
    bootstrap = """import time
started = time.monotonic()
from dataclasses import replace
from ml_harness.entrypoints import cli
from ml_harness.adapters.execution.runner import SubprocessRunner
class SlowRunner(SubprocessRunner):
    def run(self, request, timeout_seconds):
        request = replace(request, candidate_path=CANDIDATE)
        return super().run(request, timeout_seconds)
cli.SubprocessRunner = SlowRunner
raise SystemExit(cli.main(started_at=started))
""".replace("CANDIDATE", repr(str(candidate)))
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(("OPENAI_", "QWEN_")):
            del env[key]
    env["PYTHONPATH"] = str(ROOT)
    return subprocess.Popen(
        [
            sys.executable,
            "-c",
            bootstrap,
            str(path),
            "--run-dir",
            str(root / "run"),
            "--baseline-only",
            "--threads",
            "1",
            "--time-budget",
            str(budget),
            "--worker-timeout",
            str(worker_timeout),
        ],
        cwd=root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )


def wait_for(path: Path, process: subprocess.Popen, timeout: float = 15) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            pytest.fail(f"CLI exited before {path.name}: {stdout}\n{stderr}")
        time.sleep(0.02)
    pytest.fail(f"CLI did not create {path.name} within {timeout} seconds")


def cleanup(process: subprocess.Popen, root: Path) -> None:
    if process.poll() is None:
        process.kill()
    process.communicate(timeout=10)
    metadata = root / "run/validation/process.json"
    if metadata.exists():
        pid = json.loads(metadata.read_text())["pid"]
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


@pytest.mark.parametrize("slow_stage", ["fit", "predict"])
@pytest.mark.parametrize("budget,worker_timeout,preserve", [(5, 30, False), (15, 2, True)])
def test_cli_deadline_or_slow_worker_preserves_ready_csv(
    tmp_path: Path,
    budget: float,
    worker_timeout: float,
    preserve: bool,
    slow_stage: str,
) -> None:
    path = fixture_task(tmp_path)
    task = CsvTaskData().prepare(str(path), 42)
    destination = Path(task.spec.submission_path)
    if preserve:
        destination.write_bytes(b"id,y\n002,91.000\nNA,92.000\n")
    process = launch_slow_cli(
        tmp_path, path, budget=budget, worker_timeout=worker_timeout, slow_stage=slow_stage
    )
    started = time.monotonic()
    try:
        wait_for(tmp_path / "run/ready.json", process)
        ready_bytes = destination.read_bytes()
        wait_for(tmp_path / "worker_started", process)
        if slow_stage == "predict":
            wait_for(tmp_path / "predict_started", process)
        stdout, stderr = process.communicate(timeout=20)
        assert process.returncode == 0, (stdout, stderr)
        assert time.monotonic() - started < min(budget, worker_timeout) + 5
        assert destination.read_bytes() == ready_bytes
        assert load_submission(task)["ids"] == ["002", "NA"]
        summary = json.loads((tmp_path / "run/summary.json").read_text())
        assert summary["status"] == "fallback_preserved"
        assert not (tmp_path / "run/baseline.json").exists()
        worker_pid = int((tmp_path / "worker_started").read_text())
        with pytest.raises(ProcessLookupError):
            os.kill(worker_pid, 0)
        ready = json.loads((tmp_path / "run/ready.json").read_text())
        assert ready["status"] == (
            "preserved_existing" if preserve else "statistical_fallback_written"
        )
    finally:
        cleanup(process, tmp_path)


@pytest.mark.skipif(not hasattr(signal, "SIGKILL"), reason="Requires POSIX SIGKILL")
def test_sigkill_after_ready_leaves_exact_valid_csv(tmp_path: Path) -> None:
    path = fixture_task(tmp_path)
    task = CsvTaskData().prepare(str(path), 42)
    process = launch_slow_cli(tmp_path, path, budget=30, worker_timeout=30)
    try:
        wait_for(tmp_path / "run/ready.json", process)
        saved = Path(task.spec.submission_path).read_bytes()
        wait_for(tmp_path / "worker_started", process)
        process.kill()
        process.communicate(timeout=10)
        assert process.returncode == -signal.SIGKILL
        assert Path(task.spec.submission_path).read_bytes() == saved
        assert load_submission(task)["ids"] == ["002", "NA"]
        assert not (tmp_path / "run/baseline.json").exists()
        assert json.loads((tmp_path / "run/ready.json").read_text())["status"] == (
            "statistical_fallback_written"
        )
    finally:
        cleanup(process, tmp_path)


def test_inspect_data_never_reads_control_targets_or_control_feature_samples(
    tmp_path: Path,
) -> None:
    path = fixture_task(tmp_path, count=1200)
    task = CsvTaskData().prepare(str(path), 42)
    assert task.protocol.control_indices
    train_path = Path(task.spec.train_path)
    with train_path.open() as stream:
        rows = list(csv.DictReader(stream))
    control = set(task.protocol.control_indices)
    for i in control:
        # A nonnumeric sentinel would fail regression profiling if control rows leaked.
        rows[i]["y"] = "CONTROL_TARGET_SECRET"
        rows[i]["x"] = "CONTROL_FEATURE_SECRET"
    with train_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id", "x", "y"])
        writer.writeheader()
        writer.writerows(rows)
    workspace = ResearchWorkspace(task, tmp_path / "research", Mock(), Mock())
    profile = workspace.inspect_data()
    development = task.protocol.train_indices + task.protocol.validation_indices
    targets = [float(rows[i]["y"]) for i in development]
    assert profile["target_profile"] == {
        "min": min(targets),
        "max": max(targets),
        "mean": sum(targets) / len(targets),
    }
    assert profile["development_rows"] == len(development)
    assert profile["control_rows_withheld"] == len(control)
    assert "CONTROL_" not in json.dumps(profile)
