"""One process group per training; bounded resources and no LLM credentials."""

import json
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import asdict
from math import isclose, isfinite
from pathlib import Path
from typing import Any

import psutil  # type: ignore[import-untyped]

from ml_harness.application.ports.execution import ExecutionResult, RunRequest
from ml_harness.domain.models import ContractError, ExperimentStatus


def _matching_process(pid: int, created_at: float) -> Any:
    """Never signal an observed PID after it has been reused by another process."""
    try:
        child = psutil.Process(pid)
        if (
            not child.is_running()
            or child.status() == psutil.STATUS_ZOMBIE
            or not isclose(child.create_time(), created_at, rel_tol=0, abs_tol=0.001)
        ):
            return None
        return child
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        return None


def _cleanup_worker(process: subprocess.Popen[bytes], observed: dict[int, float]) -> bool:
    """Best-effort owned-process cleanup, including observed children that used setsid.

    This is not OS containment: an unobserved process that forks/reparents between
    samples, or SIGKILL of the harness itself, cannot be guaranteed to be cleaned up.
    """
    alarm_pending = False
    previous_handler: Any = None

    def defer_alarm(signum: int, frame: Any) -> None:
        nonlocal alarm_pending
        alarm_pending = True

    # Masking the main thread alone is insufficient: another unblocked thread can
    # receive ITIMER_REAL, scheduling Python's handler back on the main thread.
    # The CLI contract runs here on main; never mutate handlers from other threads.
    if threading.current_thread() is threading.main_thread():
        previous_handler = signal.getsignal(signal.SIGALRM)
        if previous_handler is not None:
            signal.signal(signal.SIGALRM, defer_alarm)
    cleanup_deadline = time.monotonic() + 3.0
    complete = True
    try:
        pending = [(pid, created) for pid, created in observed.items() if pid != process.pid]
        if process.pid in observed:
            pending.append((process.pid, observed[process.pid]))
        visited: set[tuple[int, float]] = set()
        # Freeze each known parent before discovering its children, so cleanup does
        # not create a window where a running parent forks after its last snapshot.
        while pending and len(visited) < 4096 and time.monotonic() < cleanup_deadline:
            identity = pending.pop()
            if identity in visited:
                continue
            visited.add(identity)
            try:
                parent = _matching_process(*identity)
                if parent is None:
                    continue
                parent.suspend()
                for child in parent.children():
                    created_at = child.create_time()
                    observed[child.pid] = created_at
                    pending.append((child.pid, created_at))
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                pass
            except (psutil.Error, OSError):
                complete = False
        if pending:
            complete = False
        killed = []
        for pid, created_at in observed.items():
            if time.monotonic() >= cleanup_deadline:
                complete = False
                break
            if pid == process.pid:
                continue
            try:
                child = _matching_process(pid, created_at)
                if child is not None:
                    child.kill()
                    killed.append(child)
            except (psutil.Error, OSError):
                complete = False
        # Keep the original process-group fallback for group members whose leaders
        # have already exited; detached sessions are handled by identities above.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError:
            complete = False
        if process.poll() is None:
            process.kill()
        try:
            process.wait(timeout=max(0.001, min(2.0, cleanup_deadline - time.monotonic())))
        except subprocess.TimeoutExpired:
            complete = False
        psutil.wait_procs(killed, timeout=max(0.0, min(0.5, cleanup_deadline - time.monotonic())))
        for pid, created_at in observed.items():
            if time.monotonic() >= cleanup_deadline:
                complete = False
                break
            try:
                if _matching_process(pid, created_at) is not None:
                    complete = False
            except (psutil.Error, OSError):
                complete = False
        return complete
    finally:
        if previous_handler is not None:
            signal.signal(signal.SIGALRM, previous_handler)
            if alarm_pending:
                signal.raise_signal(signal.SIGALRM)


class SubprocessRunner:
    def __init__(self, *, memory_limit_mb: int = 28000) -> None:
        if memory_limit_mb <= 0:
            raise ContractError("Memory limit must be positive")
        self.memory_limit_mb = memory_limit_mb
        self._cleanup_failed = False

    def run(self, request: RunRequest, timeout_seconds: float) -> ExecutionResult:
        if self._cleanup_failed:
            raise ContractError(
                "Previous worker cleanup is incomplete; inspect remaining processes"
            )
        if not isfinite(timeout_seconds) or timeout_seconds <= 0 or not 1 <= request.threads <= 8:
            raise ContractError("Positive timeout and 1–8 threads required")
        output_dir = Path(request.output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        result_file = output_dir / "result.json"
        if result_file.exists():
            raise ContractError("Execution directory already contains a result; use a new version")
        payload = {
            "spec": asdict(request.task.spec),
            "protocol": asdict(request.task.protocol),
            "candidate_path": request.candidate_path,
            "parameters": request.parameters or {},
            "output_dir": str(output_dir),
            "threads": request.threads,
            "mode": request.mode,
            "backend": request.backend,
        }
        request_file = output_dir / "request.json"
        request_file.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")
        env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith(("OPENAI_", "QWEN_"))
        }
        package_root = str(Path(__file__).resolve().parents[3])
        env["PYTHONPATH"] = package_root
        for name in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
        ):
            env[name] = str(request.threads)
        observed: dict[int, float] = {}
        started = time.monotonic()
        status, detail = ExperimentStatus.FAILED, ""
        with (
            (output_dir / "stdout.log").open("wb") as stdout,
            (output_dir / "stderr.log").open("wb") as stderr,
        ):
            process = subprocess.Popen(
                [sys.executable, "-m", "ml_harness.adapters.execution.worker", str(request_file)],
                cwd=output_dir,
                env=env,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
            try:
                created_at = psutil.Process(process.pid).create_time()
                observed[process.pid] = created_at
                (output_dir / "process.json").write_text(
                    json.dumps(
                        {
                            "pid": process.pid,
                            "created_at": created_at,
                            "timeout_seconds": timeout_seconds,
                        }
                    ),
                    encoding="utf-8",
                )
                while process.poll() is None:
                    if time.monotonic() - started >= timeout_seconds:
                        status, detail = ExperimentStatus.TIMED_OUT, "Worker deadline exceeded"
                        break
                    try:
                        parent = psutil.Process(process.pid)
                        processes = [parent, *parent.children(recursive=True)]
                        for child in processes:
                            observed[child.pid] = child.create_time()
                        rss = sum(
                            child.memory_info().rss for child in processes if child.is_running()
                        )
                        if rss > self.memory_limit_mb * 1024 * 1024:
                            detail = "Worker process tree exceeded memory limit"
                            break
                    except psutil.NoSuchProcess:
                        pass
                    time.sleep(0.05)
                else:
                    if process.returncode == 0:
                        status = ExperimentStatus.SUCCEEDED
                    else:
                        detail = f"Worker exited with code {process.returncode}; see stderr.log"
            finally:
                self._cleanup_failed = True
                self._cleanup_failed = not _cleanup_worker(process, observed)
            if self._cleanup_failed:
                raise ContractError(
                    "Worker descendants may remain active; further execution refused"
                )
        elapsed = time.monotonic() - started
        if status != ExperimentStatus.SUCCEEDED:
            return ExecutionResult(status, elapsed, detail=detail)
        try:
            output = json.loads(result_file.read_text(encoding="utf-8"))
            if not isinstance(output, dict):
                raise ValueError("Worker result must be an object")
        except (OSError, ValueError) as exc:
            return ExecutionResult(ExperimentStatus.FAILED, elapsed, detail=str(exc))
        return ExecutionResult(status, elapsed, output)
