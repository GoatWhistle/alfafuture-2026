"""Sequential baseline-only CLI measurements; results never imply an agent comparison."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import psutil  # type: ignore[import-untyped]

BACKENDS = ("catboost", "lightgbm", "autogluon")
ROOT = Path(__file__).resolve().parents[1]


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _runtime_versions() -> dict[str, str]:
    versions = {}
    for package in ("catboost", "lightgbm", "autogluon.tabular", "scikit-learn", "numpy", "psutil"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not_installed"
    return versions


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path.name}")
    return value


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _task_inputs(task: Path) -> dict[str, str]:
    text = task.read_text(encoding="utf-8")
    match = re.search(r"```(?:task-json|json)\s*\n(.*?)```", text, re.S)
    meta = json.loads(match.group(1)) if match else {}
    inputs = {str(task): _hash(task)}
    for key, name in (
        ("train_path", "train.csv"),
        ("test_path", "test.csv"),
        ("dictionary_path", "data_dictionary.csv"),
        ("sample_path", "sample_submission.csv"),
    ):
        value = meta.get(key)
        if value is None:
            matches = [
                item
                for item in re.findall(r"[\w./-]*" + re.escape(name), text)
                if Path(item).name == name
            ]
            qualified = [item for item in matches if "/" in item]
            options = list(dict.fromkeys(qualified or matches or [name]))
            if len(options) != 1:
                raise ValueError(f"Ambiguous benchmark input {key}")
            value = options[0]
        path = (task.parent / value).resolve()
        inputs[str(path)] = _hash(path)
    return inputs


def _csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = list(reader.fieldnames or [])
        rows = list(reader)
    if not columns or len(set(columns)) != len(columns) or not rows:
        raise ValueError("Empty or duplicate-column CSV")
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError("Malformed CSV")
    return columns, rows


def validate_submission(path: Path, spec: dict[str, Any]) -> dict[str, Any]:
    """Validate saved CSV independently using only public schema and test IDs."""
    columns, rows = _csv(path)
    _, test = _csv(Path(spec["test_path"]))
    if columns != list(spec["submission_columns"]):
        raise ValueError("Submission schema mismatch")
    ids = [row[spec["id_column"]] for row in rows]
    expected = [row[spec["id_column"]] for row in test]
    if ids != expected or len(set(ids)) != len(ids) or any(not item for item in ids):
        raise ValueError("Submission ID/count/order mismatch")
    for row in rows:
        values = [row[column] for column in columns[1:]]
        if spec["prediction_kind"] == "labels":
            classes = spec["classes"]
            if values[0] not in classes:
                if not classes or not all(type(label) in (int, float) for label in classes):
                    raise ValueError("Unknown submission class")
                if not math.isfinite(float(values[0])) or float(values[0]) not in classes:
                    raise ValueError("Unknown numeric submission class")
        else:
            numbers = [float(value) for value in values]
            if not all(math.isfinite(value) for value in numbers):
                raise ValueError("Nonfinite submission values")
            if spec["prediction_kind"] == "probabilities":
                if any(value < 0 or value > 1 for value in numbers):
                    raise ValueError("Invalid probability")
                if spec["kind"] != "binary" and (
                    len(numbers) != len(spec["classes"])
                    or not math.isclose(sum(numbers), 1, rel_tol=1e-6, abs_tol=1e-6)
                ):
                    raise ValueError("Multiclass probabilities do not normalize")
    return {
        "status": "validated",
        "rows": len(rows),
        "sha256": _hash(path),
        "path": str(path.resolve()),
    }


def _label_index(value: Any, classes: list[Any]) -> int:
    matches = [index for index, label in enumerate(classes) if value == label]
    if (
        not matches
        and isinstance(value, str)
        and all(type(label) in (int, float) for label in classes)
    ):
        try:
            numeric = float(value)
            matches = [index for index, label in enumerate(classes) if numeric == label]
        except ValueError:
            pass
    if len(matches) != 1:
        raise ValueError("Unknown or ambiguous validation class")
    return matches[0]


def _validate_worker_output(
    value: dict[str, Any], spec: dict[str, Any], protocol: dict[str, Any]
) -> None:
    _, train = _csv(Path(spec["train_path"]))
    expected = [train[index][spec["id_column"]] for index in protocol["validation_indices"]]
    ids, predictions, classes = value.get("ids"), value.get("predictions"), value.get("classes")
    if (
        ids != expected
        or not ids
        or len(set(ids)) != len(ids)
        or not isinstance(predictions, list)
        or len(predictions) != len(ids)
        or not isinstance(classes, list)
    ):
        raise ValueError("Invalid validation prediction artifact shape/IDs")
    if spec["prediction_kind"] == "values":
        if classes or any(
            type(item) not in (int, float) or not math.isfinite(item) for item in predictions
        ):
            raise ValueError("Invalid regression prediction artifact")
    else:
        if sorted(str(label) for label in classes) != sorted(
            str(label) for label in spec["classes"]
        ):
            raise ValueError("Validation artifact classes disagree")
        for row in predictions:
            if spec["prediction_kind"] == "labels":
                _label_index(row, spec["classes"])
            elif (
                not isinstance(row, list)
                or len(row) != len(classes)
                or any(
                    type(item) not in (int, float) or not math.isfinite(item) or not 0 <= item <= 1
                    for item in row
                )
                or not math.isclose(sum(row), 1, rel_tol=1e-6, abs_tol=1e-6)
            ):
                raise ValueError("Invalid validation probabilities")


def _rss(process: subprocess.Popen[bytes]) -> int:
    try:
        root = psutil.Process(process.pid)
        children = [root, *root.children(recursive=True)]
    except psutil.NoSuchProcess:
        return 0
    total = 0
    for child in children:
        try:
            total += child.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return total


def _terminate(process: subprocess.Popen[bytes]) -> None:
    try:
        children = psutil.Process(process.pid).children(recursive=True)
    except psutil.NoSuchProcess:
        children = []
    for child in reversed(children):
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass
    if process.poll() is None:
        process.kill()
    process.wait()
    psutil.wait_procs(children, timeout=2)


def _measure(
    command: list[str], directory: Path, ceiling: float, memory_mb: int, poll_seconds: float
) -> dict[str, Any]:
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith(("OPENAI_", "QWEN_"))
    }
    started, peak, termination = time.monotonic(), 0, "exited"
    with (
        (directory / "stdout.log").open("wb") as stdout,
        (directory / "stderr.log").open("wb") as stderr,
    ):
        process = subprocess.Popen(
            command,
            cwd=directory,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        try:
            while True:
                peak = max(peak, _rss(process))
                if process.poll() is not None:
                    break
                if time.monotonic() - started >= ceiling:
                    termination = "time_limit"
                    _terminate(process)
                    break
                if peak > memory_mb * 1024 * 1024:
                    termination = "memory_limit"
                    _terminate(process)
                    break
                time.sleep(min(poll_seconds, max(0.001, ceiling - (time.monotonic() - started))))
        finally:
            if process.poll() is None:
                _terminate(process)
    return {
        "returncode": process.returncode,
        "wall_seconds": time.monotonic() - started,
        "peak_rss_bytes": peak,
        "rss_sampling_seconds": poll_seconds,
        "rss_scope": "cli_and_descendants",
        "termination": termination,
    }


def _artifacts(directory: Path) -> dict[str, Any]:
    harness = directory / "harness"
    result: dict[str, Any] = {
        "score_status": "unavailable",
        "submission": {"status": "unavailable"},
    }
    summary = _json(harness / "summary.json") if (harness / "summary.json").exists() else {}
    result["summary"] = summary
    final = summary.get("finalization", summary.get("research", {}).get("finalization", {}))
    result["finalization"] = final
    result["control_scores"] = final.get("control_scores", {})
    result["fallback_status"] = (
        summary.get("status") == "fallback_preserved"
        or final.get("refit_status") == "failed_baseline_preserved"
    )
    if not (harness / "prepared.json").exists():
        return result
    prepared = _json(harness / "prepared.json")
    spec, protocol = prepared["spec"], prepared["protocol"]
    result["protocol"] = protocol
    baseline_file = harness / "baseline.json"
    score = _json(baseline_file).get("score") if baseline_file.exists() else summary.get("score")
    validation = harness / "validation/result.json"
    if (
        isinstance(score, dict)
        and type(score.get("value")) in (int, float)
        and math.isfinite(score["value"])
        and score.get("protocol_id") == protocol["fingerprint"]
        and score.get("metric") == spec["metric"]
        and score.get("level") == protocol.get("level", "full")
        and validation.exists()
    ):
        prediction = _json(validation)
        try:
            _validate_worker_output(prediction, spec, protocol)
            result.update(
                score_status="measured", score=score, validation_prediction_sha256=_hash(validation)
            )
        except (ValueError, KeyError, TypeError, IndexError) as error:
            result["validation_artifact_error"] = str(error)
    workers = []
    for request_path in sorted(harness.rglob("request.json")):
        request = _json(request_path)
        result_path = request_path.parent / "result.json"
        workers.append(
            {
                "path": str(request_path.parent),
                "mode": request.get("mode"),
                "prediction_status": "present" if result_path.exists() else "missing",
            }
        )
    result["worker_artifacts"] = workers
    try:
        source = Path(summary.get("submission_path", spec["submission_path"]))
        validated = validate_submission(source, spec)
        destination = directory / "submission.csv"
        temporary = destination.with_suffix(".csv.tmp")
        temporary.write_bytes(source.read_bytes())
        copied = validate_submission(temporary, spec)
        if copied["sha256"] != validated["sha256"]:
            raise ValueError("Submission changed during benchmark copy")
        temporary.replace(destination)
        result["submission"] = validate_submission(destination, spec)
    except (OSError, ValueError, KeyError, TypeError) as error:
        result["submission"] = {"status": "invalid", "error": str(error)}

    return result


def run_benchmarks(
    tasks: list[Path],
    output: Path,
    *,
    seeds: tuple[int, ...] = (7, 42, 2026),
    backends: tuple[str, ...] = BACKENDS,
    time_budget: float = 600,
    threads: int = 8,
    memory_limit_mb: int = 28000,
    solution: Path | None = None,
    poll_seconds: float = 0.02,
) -> dict[str, Any]:
    """Run one CLI at a time; completed manifests are verified and reused, never retrained."""
    if (
        not tasks
        or len({str(task.resolve()) for task in tasks}) != len(tasks)
        or len(set(seeds)) != len(seeds)
        or len(set(backends)) != len(backends)
        or not seeds
        or not backends
        or any(backend not in BACKENDS for backend in backends)
        or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds)
        or not math.isfinite(time_budget)
        or not 0 < time_budget <= 10800
        or not 1 <= threads <= 8
        or not 1 <= memory_limit_mb <= 28000
        or not math.isfinite(poll_seconds)
        or poll_seconds <= 0
    ):
        raise ValueError("Invalid benchmark configuration")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    solution = (solution or ROOT / "solution.py").resolve()
    code = {
        str(path.relative_to(ROOT)): _hash(path)
        for path in sorted((ROOT / "ml_harness").rglob("*.py"))
    }
    code.update(
        solution=_hash(solution),
        orchestrator=_hash(Path(__file__)),
        pyproject=_hash(ROOT / "pyproject.toml"),
        uv_lock=_hash(ROOT / "uv.lock"),
    )
    config = {
        "tasks": [
            {"path": str(task.resolve()), "inputs": _task_inputs(task.resolve())} for task in tasks
        ],
        "seeds": list(seeds),
        "backends": list(backends),
        "time_budget": time_budget,
        "threads": threads,
        "memory_limit_mb": memory_limit_mb,
        "baseline_only": True,
        "python": sys.executable,
        "python_version": sys.version,
        "runtime_versions": _runtime_versions(),
        "code": code,
    }
    configuration = output / "configuration.json"
    if configuration.exists() and _json(configuration) != config:
        raise ValueError(
            "Benchmark configuration differs from existing output; choose another directory"
        )
    _write(configuration, config)
    runs = []
    for task in tasks:
        task = task.resolve()
        task_key = (
            re.sub(r"[^\w.-]+", "_", task.parent.name)
            + "-"
            + hashlib.sha256(str(task).encode()).hexdigest()[:12]
        )
        for seed in seeds:
            for backend in backends:
                directory = output / task_key / f"{seed}-{backend}"
                directory.mkdir(parents=True, exist_ok=True)
                manifest = directory / "manifest.json"
                identity = {"task": str(task), "seed": seed, "backend": backend}
                if manifest.exists():
                    saved = _json(manifest)
                    if saved.get("identity") != identity or saved.get("status") != "completed":
                        raise ValueError(
                            "Incomplete/conflicting benchmark manifest; inspect before proceeding"
                        )
                    submission = saved.get("submission", {})
                    if (
                        submission.get("status") == "validated"
                        and _hash(Path(submission["path"])) != submission["sha256"]
                    ):
                        raise ValueError("Cached benchmark submission checksum mismatch")
                    runs.append(saved)
                    continue
                command = [
                    sys.executable,
                    str(solution),
                    str(task),
                    "--baseline-only",
                    "--seed",
                    str(seed),
                    "--backend",
                    backend,
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
                _write(manifest, {"status": "running", "identity": identity, "command": command})
                record = {
                    "identity": identity,
                    "command": command,
                    **_measure(command, directory, time_budget, memory_limit_mb, poll_seconds),
                }
                try:
                    record.update(_artifacts(directory))
                except (OSError, ValueError, KeyError, TypeError) as error:
                    record.update(
                        artifact_error=str(error),
                        submission={"status": "invalid"},
                        score_status="unavailable",
                    )
                record["valid_submission_success"] = (
                    record["returncode"] == 0 and record["submission"]["status"] == "validated"
                )
                record["measured_success"] = (
                    record["valid_submission_success"] and record["score_status"] == "measured"
                )
                record["status"] = "completed"
                _write(manifest, record)
                runs.append(record)
    protocols: dict[tuple[str, int], str] = {}
    for item in runs:
        fingerprint = item.get("protocol", {}).get("fingerprint")
        if fingerprint is None:
            continue
        group = (item["identity"]["task"], item["identity"]["seed"])
        if group in protocols and protocols[group] != fingerprint:
            raise ValueError(
                "Backend comparison protocol fingerprint mismatch for the same task/seed"
            )
        protocols[group] = fingerprint
    report = {
        "mode": "baseline_only",
        "protocol_consistency": "verified_for_available_prepared_artifacts",
        "runs": runs,
        "success_fraction": sum(item["measured_success"] for item in runs) / len(runs),
        "valid_submission_fraction": sum(item["valid_submission_success"] for item in runs)
        / len(runs),
        "full_agent_comparison": {
            "status": "blocked",
            "reason": (
                "Missing verified supplied-Qwen configuration for a full-agent comparison; "
                "baseline-only results do not measure agent benefit"
            ),
        },
    }
    _write(output / "report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tasks", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("runs/stage7/benchmarks"))
    parser.add_argument("--seeds", nargs="+", default=["7", "42", "2026"])
    parser.add_argument("--backends", nargs="+", default=list(BACKENDS))
    parser.add_argument("--time-budget", type=float, default=600)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--memory-limit-mb", type=int, default=28000)
    args = parser.parse_args()
    try:
        result = run_benchmarks(
            args.tasks,
            args.output,
            seeds=tuple(int(value) for item in args.seeds for value in item.split(",")),
            backends=tuple(value for item in args.backends for value in item.split(",")),
            time_budget=args.time_budget,
            threads=args.threads,
            memory_limit_mb=args.memory_limit_mb,
        )
        print(json.dumps({key: value for key, value in result.items() if key != "runs"}, indent=2))
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Benchmark failed: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
