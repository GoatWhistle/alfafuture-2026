"""One process group per training; bounded resources and no LLM credentials."""

import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict
from math import isfinite
from pathlib import Path

import psutil  # type: ignore[import-untyped]

from ml_harness.application.ports.execution import ExecutionResult, RunRequest
from ml_harness.domain.models import ContractError, ExperimentStatus


class SubprocessRunner:
    def __init__(self, *, memory_limit_mb: int = 28000) -> None:
        if memory_limit_mb <= 0:
            raise ContractError("Memory limit must be positive")
        self.memory_limit_mb = memory_limit_mb

    def run(self, request: RunRequest, timeout_seconds: float) -> ExecutionResult:
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
                while process.poll() is None:
                    if time.monotonic() - started >= timeout_seconds:
                        status, detail = ExperimentStatus.TIMED_OUT, "Worker deadline exceeded"
                        break
                    try:
                        parent = psutil.Process(process.pid)
                        processes = [parent, *parent.children(recursive=True)]
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
                # The group can outlive its leader. Always terminate descendants as well.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
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
