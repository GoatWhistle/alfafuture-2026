"""Actual process group deadlines and credential isolation, with tiny fake workers."""

import subprocess
import sys
from pathlib import Path

# Shared fixture generation only; no runtime test-only hooks in the worker.
from test_prepare_task import task_fixture

from ml_harness.adapters.data.task import CsvTaskData
from ml_harness.adapters.execution.runner import SubprocessRunner
from ml_harness.application.ports.execution import RunRequest
from ml_harness.domain.models import ExperimentStatus


def task(tmp_path):
    return CsvTaskData().prepare(str(task_fixture(tmp_path)), 42)


def test_timeout_kills_worker_group_and_preserves_submission(tmp_path):
    prepared = task(tmp_path)
    submission = Path(prepared.spec.submission_path)
    submission.write_text("previous answer")
    candidate = tmp_path / "slow.py"
    candidate.write_text("""
import subprocess, sys, time
from pathlib import Path

def build_model(kind, parameters, threads, seed):
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(90)'])
    Path(parameters['child_pid']).write_text(str(child.pid))
    time.sleep(90)
""")
    pid_file = tmp_path / "child.pid"
    result = SubprocessRunner().run(
        RunRequest(
            prepared,
            str(tmp_path / "timeout"),
            candidate_path=str(candidate),
            parameters={"child_pid": str(pid_file)},
            threads=1,
        ),
        3,
    )
    assert result.status == ExperimentStatus.TIMED_OUT
    assert result.elapsed_seconds < 6
    assert submission.read_text() == "previous answer"
    assert pid_file.exists(), "Candidate must spawn a child before timeout"
    import psutil

    pid = int(pid_file.read_text())
    assert not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE


def test_credentials_not_in_worker_environment(tmp_path, monkeypatch):
    prepared = task(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-secret")
    monkeypatch.setenv("QWEN_API_KEY", "synthetic-secret")
    candidate = tmp_path / "envcheck.py"
    candidate.write_text("""
import os

def build_model(kind, parameters, threads, seed):
    assert not any(key.startswith(('OPENAI_', 'QWEN_')) for key in os.environ)
    raise RuntimeError('credentials removed')
""")
    output_dir = tmp_path / "env"
    result = SubprocessRunner().run(
        RunRequest(
            prepared,
            str(output_dir),
            candidate_path=str(candidate),
            threads=1,
        ),
        10,
    )
    assert result.status == ExperimentStatus.FAILED
    assert "credentials removed" in (output_dir / "stderr.log").read_text()
    assert "synthetic-secret" not in (output_dir / "stderr.log").read_text()


def test_memory_limit_and_logs(tmp_path):
    prepared = task(tmp_path)
    output_dir = tmp_path / "memory"
    result = SubprocessRunner(memory_limit_mb=1).run(
        RunRequest(prepared, str(output_dir), threads=1), 10
    )
    assert result.status == ExperimentStatus.FAILED
    assert "memory limit" in result.detail
    assert (output_dir / "stdout.log").exists()
    assert (output_dir / "stderr.log").exists()


def test_cli_help_without_environment(tmp_path):
    solution = Path(__file__).resolve().parents[2] / "solution.py"
    process = subprocess.run(
        [sys.executable, str(solution), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert process.returncode == 0
    assert "--backend" in process.stdout


def test_cli_loads_dotenv_without_overriding_platform(tmp_path, monkeypatch):
    import ml_harness.entrypoints.cli as cli
    from ml_harness.domain.models import ContractError

    (tmp_path / ".env").write_text("OPENAI_API_KEY=local\nOPENAI_MODEL=provided-qwen\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "platform")
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.setattr(sys, "argv", ["solution.py", "missing.md"])

    def prepare(*args, **kwargs):
        import os

        assert os.environ["OPENAI_API_KEY"] == "platform"
        assert os.environ["OPENAI_MODEL"] == "provided-qwen"
        raise ContractError("deliberate stop before any training")

    monkeypatch.setattr(cli, "prepare_task", prepare)
    assert cli.main() == 2
    monkeypatch.delenv("OPENAI_MODEL")
