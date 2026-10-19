"""Verify an unpacked source archive in isolated task/cwd directories without installations."""

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path, PurePosixPath
from typing import Any
from zipfile import ZipFile

import psutil  # type: ignore[import-untyped]


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _environment() -> dict[str, str]:
    return {
        key: value
        for key, value in os.environ.items()
        if key not in ("PYTHONPATH", "PYTHONHOME") and not key.startswith(("OPENAI_", "QWEN_"))
    }


def extract_archive(archive_path: Path, destination: Path) -> list[str]:
    if destination.exists():
        raise ValueError("Extraction destination must be fresh")
    with ZipFile(archive_path) as archive:
        names = []
        size = 0
        for entry in archive.infolist():
            name = entry.filename
            parts = PurePosixPath(name).parts
            mode = entry.external_attr >> 16
            if (
                not parts
                or name.startswith("/")
                or "\\" in name
                or name.endswith("/")
                or name != "/".join(parts)
                or any(part in (".", "..") or part.startswith(".") for part in parts)
                or any(
                    part in ("__pycache__", ".venv", "venv", "runs", "dist", "node_modules")
                    for part in parts
                )
                or not (
                    name == "solution.py"
                    or (len(parts) >= 2 and parts[0] == "ml_harness" and name.endswith(".py"))
                )
                or stat.S_IFMT(mode) not in (0, stat.S_IFREG)
                or name in names
            ):
                raise ValueError("Archive violates the regular Python source allowlist")
            size += entry.file_size
            if entry.file_size > 5_000_000 or size > 20_000_000:
                raise ValueError("Archive exceeds the source size limit")
            names.append(name)
        if not {"solution.py", "ml_harness/__init__.py"}.issubset(names):
            raise ValueError("Archive is missing the entrypoint or package")
        destination.mkdir(parents=True)
        for name in names:
            path = destination / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(archive.read(name))
    return names


def _probe(
    unpacked: Path, task: Path, cwd: Path, *, validate: bool = False, seed: int = 42
) -> dict[str, Any]:
    bootstrap = """import json, sys
from pathlib import Path
from dataclasses import asdict
sys.path.insert(0, sys.argv[1])
import ml_harness
from ml_harness.adapters.data.task import CsvTaskData
prepared = CsvTaskData().prepare(sys.argv[2], int(sys.argv[3]))
result = {'prepared': asdict(prepared), 'import_origin': str(Path(ml_harness.__file__).resolve())}
if sys.argv[4] == 'validate':
    from ml_harness.adapters.storage.submission import load_submission
    output = load_submission(prepared)
    result['validation'] = {'status': 'validated', 'rows': len(output['ids'])}
print(json.dumps(result, allow_nan=False))
"""
    checked = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            bootstrap,
            str(unpacked),
            str(task),
            str(seed),
            "validate" if validate else "prepare",
        ],
        cwd=cwd,
        env=_environment(),
        capture_output=True,
        text=True,
        timeout=30,
    )
    if checked.returncode:
        raise ValueError("Unpacked task parser/CSV validation failed: " + checked.stderr[-1500:])
    result: dict[str, Any] = json.loads(checked.stdout)
    if not Path(result["import_origin"]).is_relative_to(unpacked):
        raise ValueError("Import origin is outside the unpacked archive")
    return result


_INPUTS = ("train_path", "test_path", "dictionary_path", "sample_path")


def copy_task(
    unpacked: Path, source: Path, destination: Path, cwd: Path, seed: int
) -> dict[str, Any]:
    """Resolve with the archived parser, copy only required inputs, then recheck isolation."""
    source = source.resolve()
    parsed = _probe(unpacked, source, cwd, seed=seed)
    spec = parsed["prepared"]["spec"]
    source_root = source.parent
    for key in (*_INPUTS, "submission_path"):
        if not Path(spec[key]).is_relative_to(source_root):
            raise ValueError("External absolute/traversing task input or output is forbidden")
    if destination.exists():
        raise ValueError("Isolated task destination must be fresh")
    destination.mkdir()
    copied = [source.name]
    shutil.copyfile(source, destination / source.name)
    for key in _INPUTS:
        original = Path(spec[key])
        relative = original.relative_to(source_root)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, target)
        copied.append(relative.as_posix())
    isolated = _probe(unpacked, destination / source.name, cwd, seed=seed)
    if (
        parsed["prepared"]["protocol"]["fingerprint"]
        != isolated["prepared"]["protocol"]["fingerprint"]
    ):
        raise ValueError("Task inputs changed during isolated copying")
    isolated_spec = isolated["prepared"]["spec"]
    for key in (*_INPUTS, "submission_path"):
        if not Path(isolated_spec[key]).is_relative_to(destination):
            raise ValueError("Task uses absolute external paths after isolated copying")
    input_paths = {isolated_spec[key] for key in _INPUTS}
    if isolated_spec["submission_path"] in input_paths or isolated_spec["submission_path"] == str(
        destination / source.name
    ):
        raise ValueError("Submission output overlaps immutable task inputs")
    return {
        "task": str(destination / source.name),
        "copied_files": copied,
        "prepared": isolated["prepared"],
        "import_origin": isolated["import_origin"],
        "source_task_sha256": _hash(source),
        "copied_task_sha256": _hash(destination / source.name),
        "input_sha256": {
            Path(isolated_spec[key]).relative_to(destination).as_posix(): _hash(
                Path(isolated_spec[key])
            )
            for key in _INPUTS
        },
    }


def _cgroup(name: str) -> str | None:
    path = Path("/sys/fs/cgroup") / name
    return path.read_text().strip() if path.exists() else None


def _verify_archive(
    archive: Path,
    tasks: list[Path],
    output: Path,
    *,
    time_budget: float = 60,
    memory_limit_mb: int = 5000,
    threads: int = 8,
    backend: str = "catboost",
    seed: int = 42,
    cwd: Path,
) -> dict[str, Any]:
    if (
        not tasks
        or not math.isfinite(time_budget)
        or not 0 < time_budget <= 10800
        or not 1 <= threads <= 8
        or not 1 <= memory_limit_mb <= 28000
        or backend not in ("catboost", "lightgbm", "autogluon")
    ):
        raise ValueError("Invalid archive verification limits")
    output = output.resolve()
    if output.exists():
        raise ValueError("Verification output directory must be fresh")
    output.mkdir(parents=True)
    unpacked = output / "unpacked"
    names = extract_archive(archive.resolve(), unpacked)
    results = []
    for index, source in enumerate(tasks):
        directory = output / f"task-{index:02d}"
        isolated = copy_task(unpacked, source, directory, cwd, seed)
        command = [
            sys.executable,
            str(unpacked / "solution.py"),
            isolated["task"],
            "--baseline-only",
            "--backend",
            backend,
            "--seed",
            str(seed),
            "--threads",
            str(threads),
            "--time-budget",
            str(time_budget),
            "--worker-timeout",
            str(time_budget),
            "--memory-limit-mb",
            str(memory_limit_mb),
            "--run-dir",
            str(directory / "harness"),
        ]
        started = time.monotonic()
        timed_out = False
        with (
            (directory / "stdout.log").open("wb") as stdout,
            (directory / "stderr.log").open("wb") as stderr,
        ):
            process = subprocess.Popen(
                command,
                cwd=cwd,
                env=_environment(),
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
            try:
                process.wait(timeout=time_budget + 10)
            except subprocess.TimeoutExpired:
                timed_out = True
                try:
                    children = psutil.Process(process.pid).children(recursive=True)
                except psutil.NoSuchProcess:
                    children = []
                for child in reversed(children):
                    try:
                        child.kill()
                    except psutil.NoSuchProcess:
                        pass
                process.kill()
                process.wait()
                psutil.wait_procs(children, timeout=3)
        record = {
            **isolated,
            "command": command,
            "cwd": str(cwd),
            "returncode": process.returncode,
            "wall_seconds": time.monotonic() - started,
            "outer_timeout": timed_out,
        }
        summary = directory / "harness/summary.json"
        record["summary"] = json.loads(summary.read_text()) if summary.exists() else None
        try:
            validation = _probe(unpacked, Path(isolated["task"]), cwd, validate=True, seed=seed)[
                "validation"
            ]
            submission = Path(isolated["prepared"]["spec"]["submission_path"])
            record["submission"] = {
                **validation,
                "path": str(submission),
                "sha256": _hash(submission),
            }
        except (ValueError, OSError) as error:
            record["submission"] = {"status": "invalid_or_unavailable", "reason": str(error)}
        record["success"] = (
            process.returncode == 0 and record["submission"]["status"] == "validated"
        )
        results.append(record)
    result = {
        "archive": str(archive.resolve()),
        "archive_sha256": _hash(archive),
        "source_files": names,
        "unpacked_root": str(unpacked),
        "results": results,
        "environment": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "cgroup_cpu_max": _cgroup("cpu.max"),
            "cgroup_memory_max": _cgroup("memory.max"),
        },
        "limits": {
            "time_budget": time_budget,
            "memory_limit_mb": memory_limit_mb,
            "threads": threads,
            "backend": backend,
        },
        "all_successful": all(record["success"] for record in results),
    }
    (output / "report.json").write_text(
        json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8"
    )
    return result


def verify_archive(
    archive: Path,
    tasks: list[Path],
    output: Path,
    *,
    time_budget: float = 60,
    memory_limit_mb: int = 5000,
    threads: int = 8,
    backend: str = "catboost",
    seed: int = 42,
) -> dict[str, Any]:
    # The launch cwd remains outside the checkout even when output is under runs/.
    with tempfile.TemporaryDirectory(prefix="archive-verification-cwd-") as external:
        return _verify_archive(
            archive,
            tasks,
            output,
            time_budget=time_budget,
            memory_limit_mb=memory_limit_mb,
            threads=threads,
            backend=backend,
            seed=seed,
            cwd=Path(external),
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--time-budget", type=float, default=60)
    parser.add_argument("--memory-limit-mb", type=int, default=5000)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--backend", choices=("catboost", "lightgbm", "autogluon"), default="catboost"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = verify_archive(
        args.archive,
        args.tasks,
        args.output,
        time_budget=args.time_budget,
        memory_limit_mb=args.memory_limit_mb,
        threads=args.threads,
        backend=args.backend,
        seed=args.seed,
    )
    print(
        json.dumps(
            {"all_successful": result["all_successful"], "report": str(args.output / "report.json")}
        )
    )
    return 0 if result["all_successful"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
