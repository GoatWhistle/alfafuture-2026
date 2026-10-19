"""Actual process group deadlines and credential isolation, with tiny fake workers."""

import subprocess
import sys
from pathlib import Path

import pytest

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


@pytest.mark.parametrize("exit_early", [False, True])
def test_timeout_kills_detached_session_descendant_with_explicit_test_cleanup(tmp_path, exit_early):
    """setsid changes process groups, but the timeout still owns observed descendants."""
    import json
    import math

    import psutil

    prepared = task(tmp_path)
    candidate = tmp_path / "detached_child.py"
    candidate.write_text("""import json,subprocess,sys,time
from pathlib import Path
import psutil

def build_model(kind, parameters, threads, seed):
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(90)"],
                             start_new_session=True)
    Path(parameters["child_info"]).write_text(json.dumps({
        "pid": child.pid, "created_at": psutil.Process(child.pid).create_time()}))
    time.sleep(0.3 if parameters["exit_early"] else 90)
    raise RuntimeError("parent exits before cleanup")
""")
    info_file = tmp_path / "detached.json"
    child = None
    try:
        result = SubprocessRunner().run(
            RunRequest(
                prepared,
                str(tmp_path / "detached-timeout"),
                candidate_path=str(candidate),
                parameters={"child_info": str(info_file), "exit_early": exit_early},
                threads=1,
            ),
            3,
        )
        assert result.status == (
            ExperimentStatus.FAILED if exit_early else ExperimentStatus.TIMED_OUT
        )
        assert info_file.exists(), "Detached descendant must exist before the deadline"
        info = json.loads(info_file.read_text())
        try:
            child = psutil.Process(info["pid"])
            same = math.isclose(child.create_time(), info["created_at"], rel_tol=0, abs_tol=0.001)
            alive = same and child.is_running() and child.status() != psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:
            alive = False
        assert not alive, "Detached start_new_session child survived the worker timeout"
    finally:
        # RED reproduction must never leave its own sleeping orphan behind.
        if info_file.exists():
            info = json.loads(info_file.read_text())
            try:
                child = psutil.Process(info["pid"])
                if math.isclose(child.create_time(), info["created_at"], rel_tol=0, abs_tol=0.001):
                    child.kill()
                    try:
                        child.wait(timeout=2)
                    except psutil.TimeoutExpired:
                        pass
            except psutil.NoSuchProcess:
                pass


def test_alarm_with_unblocked_background_thread_is_deferred_until_worker_cleanup(monkeypatch):
    import signal
    import threading
    import time

    import psutil

    import ml_harness.adapters.execution.runner as runner_module
    from ml_harness.adapters.execution.deadline import BudgetExpired, DeadlineGuard

    if not hasattr(runner_module, "_cleanup_worker"):
        pytest.skip("Descendant cleanup patch has not been applied to this module")
    original_handler = signal.getsignal(signal.SIGALRM)
    ready, stop = threading.Event(), threading.Event()

    def background():
        old = signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGALRM})
        try:
            ready.set()
            stop.wait(2)
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, old)

    thread = threading.Thread(target=background, daemon=True)
    thread.start()
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(90)"],
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    original_suspend = psutil.Process.suspend

    def slow_suspend(child):
        original_suspend(child)
        time.sleep(0.08)

    monkeypatch.setattr(psutil.Process, "suspend", slow_suspend)
    try:
        assert ready.wait(1)
        observed = {process.pid: psutil.Process(process.pid).create_time()}
        with pytest.raises(BudgetExpired):
            with DeadlineGuard(0.03):
                runner_module._cleanup_worker(process, observed)
        assert process.poll() is not None, "Alarm interrupted cleanup after suspend but before kill"
        assert signal.getsignal(signal.SIGALRM) == original_handler
    finally:
        # The old mask-based implementation leaves the dummy worker stopped at RED.
        if process.poll() is None:
            process.kill()
        process.wait(timeout=2)
        stop.set()
        thread.join(timeout=2)


def test_cleanup_in_background_thread_does_not_mutate_signal_handlers(monkeypatch):
    import signal
    import threading

    import psutil

    import ml_harness.adapters.execution.runner as runner_module

    if not hasattr(runner_module, "_cleanup_worker"):
        pytest.skip("Descendant cleanup patch has not been applied to this module")
    original_handler = signal.getsignal(signal.SIGALRM)
    original_signal = signal.signal
    mutations = []

    def tracked_signal(signum, handler):
        mutations.append(threading.current_thread())
        assert threading.current_thread() is threading.main_thread()
        return original_signal(signum, handler)

    monkeypatch.setattr(signal, "signal", tracked_signal)
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(90)"],
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    observed = {process.pid: psutil.Process(process.pid).create_time()}
    results, errors = [], []

    def cleanup():
        try:
            results.append(runner_module._cleanup_worker(process, observed))
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=cleanup, daemon=True)
    try:
        thread.start()
        thread.join(timeout=4)
        assert not thread.is_alive() and not errors
        assert results == [True] and process.poll() is not None
        assert not mutations and signal.getsignal(signal.SIGALRM) == original_handler
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=2)
        thread.join(timeout=4)
