"""Frozen, sequential real-Qwen acceptance and paired Qwen/Optuna comparisons."""

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from ml_harness.adapters.data.task import CsvTaskData, read_csv
from ml_harness.adapters.ml.ensemble import combine_predictions
from ml_harness.adapters.storage.submission import load_submission
from verification.run_benchmarks import (
    ROOT,
    _hash,
    _measure,
    _runtime_versions,
    _task_inputs,
    _write,
)
from verification.verify_archive import copy_task


def source_hashes() -> dict[str, str]:
    files = [
        ROOT / "solution.py",
        ROOT / "uv.lock",
        ROOT / "pyproject.toml",
        Path(__file__),
        ROOT / "verification/run_benchmarks.py",
        ROOT / "verification/verify_archive.py",
    ]
    files += sorted((ROOT / "ml_harness").rglob("*.py"))
    return {str(p.relative_to(ROOT)): _hash(p) for p in files}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def verify_run(directory: Path, task_path: Path, seed: int) -> dict[str, Any]:
    task = CsvTaskData().prepare(str(task_path), seed)
    summary = read_json(directory / "harness/summary.json")
    research = summary.get("research", {})
    final = research.get("finalization", {})
    saved = load_submission(task)
    winner = next((r for r in final.get("finalists", []) if r["id"] == final.get("selected")), None)
    if final.get("refit_status") != "succeeded" or winner is None:
        raise ValueError(
            "A valid full-data final refit is required; fallback is not measured success"
        )
    state = read_json(directory / "harness/research/finalization-state.json")
    outputs = []
    for member in winner["members"]:
        entry = state["refit"][member["id"]]
        result_path = Path(entry["output_dir"]) / "result.json"
        if _hash(result_path) != entry["result_hash"]:
            raise ValueError("Refit artifact checksum mismatch")
        outputs.append(read_json(result_path))
    combined = combine_predictions(task.spec, saved["ids"], outputs, tuple(winner["weights"]))
    if saved["ids"] != combined["ids"] or not same(saved["predictions"], combined["predictions"]):
        raise ValueError("CSV differs from selected full-data refit predictions")
    experiments = [r for r in research.get("reports", []) if not r["id"].startswith("baseline")]
    changed = [
        r
        for r in experiments
        if r.get("status") == "succeeded" and r.get("code_hash") != r.get("parent_code_hash")
    ]
    interpreted = [
        s
        for s in research.get("sessions", [])
        if s["reason"] == "model_finished"
        and s["summary"].strip()
        and any(r["status"] == "succeeded" for r in s["reports"])
    ]
    score = final.get("control_scores", {}).get(final["selected"])
    return {
        "protocol": task.protocol.fingerprint,
        "selected": final["selected"],
        "csv_rows": len(saved["ids"]),
        "csv_sha256": _hash(Path(task.spec.submission_path)),
        "refit_verified": True,
        "changed_successes": len(changed),
        "interpreted_successes": len(interpreted),
        "experiments": len(experiments),
        "research_status": research.get("status"),
        "reported_tokens": research.get("tokens_used", 0),
        "sessions": [
            {k: s[k] for k in ("reason", "steps", "tool_errors", "api_attempts")}
            for s in research.get("sessions", [])
        ],
        "quality": score,
        "prepared": asdict(task),
    }


def same(left: Any, right: Any) -> bool:
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(same(a, b) for a, b in zip(left, right, strict=True))
    if isinstance(left, str) or isinstance(right, str):
        return left == right
    return math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-12)


def paired_results(runs: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for run in runs:
        grouped.setdefault((run["task"], run["seed"]), {})[run["mode"]] = run
    pairs = []
    for (task, seed), modes in grouped.items():
        if set(modes) != {"qwen", "optuna"}:
            raise ValueError("Every task/seed requires a complete Qwen/Optuna pair")
        qwen, numeric = modes["qwen"], modes["optuna"]
        if qwen["protocol"] != numeric["protocol"]:
            raise ValueError("Pair validation protocols differ")
        a, b = qwen["quality"], numeric["quality"]
        if (
            a is None
            or b is None
            or a["metric"] != b["metric"]
            or a.get("protocol_id") != b.get("protocol_id")
        ):
            raise ValueError("Pair requires measured final quality on the same metric")
        gain = (
            (b["value"] - a["value"])
            if a["metric"] in ("rmse", "mae", "log_loss")
            else (a["value"] - b["value"])
        ) / max(abs(b["value"]), 1e-12)
        pairs.append(
            {
                "task": task,
                "seed": seed,
                "qwen": a["value"],
                "optuna": b["value"],
                "relative_gain": gain,
            }
        )
    means = {
        task: statistics.mean(p["relative_gain"] for p in pairs if p["task"] == task)
        for task, _ in grouped
    }
    overall = statistics.mean(means.values())
    return {
        "pairs": pairs,
        "task_mean_relative_gains": means,
        "overall_mean_relative_gain": overall,
        "benefit_gate_passed": sum(v > 0 for v in means.values()) >= 2 and overall > 0,
        "interpretation": (
            "Descriptive paired results; three seeds do not establish statistical significance"
        ),
    }


def complete_holdout(runs: list[dict[str, Any]], sources: list[Path], expected: int) -> None:
    """Only completed cohorts can open diabetes holdout labels; they never reach workers."""
    if len(runs) != expected or any(not r.get("refit_verified") for r in runs):
        raise ValueError("Holdout is unavailable until the entire cohort is verified")
    from sklearn.metrics import root_mean_squared_error

    for source in sources:
        if source.parent.name != "diabetes":
            continue
        _, targets = read_csv(source.parent / "holdout_targets.csv")
        for record in runs:
            if record["task"] != str(source):
                continue
            prepared = record["prepared"]
            id_column, target = prepared["spec"]["id_column"], prepared["spec"]["target"]
            _, rows = read_csv(Path(prepared["spec"]["submission_path"]))
            truth = {row[id_column]: float(row[target]) for row in targets}
            ids = [row[id_column] for row in rows]
            if len(truth) != len(ids) or set(ids) != set(truth):
                raise ValueError("Holdout IDs differ from predictions")
            record["quality"] = {
                "metric": "rmse",
                "value": float(
                    root_mean_squared_error(
                        [truth[i] for i in ids], [float(row[target]) for row in rows]
                    )
                ),
                "source": "posthoc_fixed_test_holdout",
            }


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    load_dotenv(ROOT / ".env", override=False)
    endpoint_fingerprint = hashlib.sha256(
        os.environ.get("OPENAI_BASE_URL", "").encode()
    ).hexdigest()
    sources = [p.resolve() for p in args.tasks]
    if (
        len(set(sources)) != len(sources)
        or len(set(args.seeds)) != len(args.seeds)
        or not 1 <= args.threads <= 8
        or not 0 < args.time_budget <= 10800
        or args.sessions < 1
    ):
        raise ValueError("Invalid or duplicate benchmark inputs")
    code = source_hashes()
    if not args.gate:
        if args.gate_report is None:
            raise ValueError("A passed live-gate report is required before comparisons")
        gate = read_json(args.gate_report)
        if (
            not gate.get("gate_passed")
            or gate.get("code") != code
            or gate.get("model") != os.environ.get("OPENAI_MODEL")
            or gate.get("endpoint_fingerprint") != endpoint_fingerprint
        ):
            raise ValueError("Live-gate must pass against exactly the current source code")
    names = ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL")
    if "qwen" in args.modes and not all(os.environ.get(k) for k in names):
        raise ValueError("Supplied Qwen credentials are not configured")
    config = {
        "tasks": [{"path": str(p), "inputs": _task_inputs(p)} for p in sources],
        "seeds": args.seeds,
        "modes": args.modes,
        "time_budget": args.time_budget,
        "threads": args.threads,
        "memory_limit_mb": 6000,
        "worker_timeout": 300,
        "token_limit": 32000,
        "sessions": args.sessions,
        "code": code,
        "model": os.environ.get("OPENAI_MODEL"),
        "runtime": _runtime_versions(),
        "gate": args.gate,
        "endpoint_fingerprint": endpoint_fingerprint,
    }
    configuration = output / "configuration.json"
    if configuration.exists() and read_json(configuration) != config:
        raise ValueError("Configuration changed; choose a new output directory")
    _write(configuration, config)
    runs = []
    for task in sources:
        key = task.parent.name + "-" + hashlib.sha256(str(task).encode()).hexdigest()[:8]
        for seed in args.seeds:
            for mode in args.modes:
                if source_hashes() != code:
                    raise ValueError("Source changed during the frozen cohort")
                directory = output / key / f"{seed}-{mode}"
                directory.mkdir(parents=True, exist_ok=True)
                manifest = directory / "manifest.json"
                if manifest.exists():
                    record = read_json(manifest)
                    if record.get("status") != "completed":
                        raise ValueError(
                            "Incomplete attempt requires inspection; choose a new cohort"
                        )
                    verified = verify_run(directory, directory / "inputs" / task.name, seed)
                    if record["csv_sha256"] != verified["csv_sha256"]:
                        raise ValueError("Cached CSV changed")
                    runs.append(record)
                    continue
                isolated = copy_task(ROOT, task, directory / "inputs", directory, seed)
                command = [
                    sys.executable,
                    str(ROOT / "solution.py"),
                    isolated["task"],
                    "--run-dir",
                    str(directory / "harness"),
                    "--backend",
                    "portfolio",
                    "--search-mode",
                    mode,
                    "--seed",
                    str(seed),
                    "--threads",
                    str(args.threads),
                    "--time-budget",
                    str(args.time_budget),
                    "--worker-timeout",
                    "300",
                    "--memory-limit-mb",
                    "6000",
                    "--research-sessions",
                    str(args.sessions),
                ]
                _write(manifest, {"status": "running", "command": command})
                print(f"START {task.parent.name} seed={seed} mode={mode}", flush=True)
                credentials = {k: os.environ[k] for k in names} if mode == "qwen" else None
                telemetry = _measure(
                    command, directory, args.time_budget, 6000, 0.05, llm_environment=credentials
                )
                if source_hashes() != code:
                    raise ValueError("Source changed while a run was executing")
                for relative, digest in isolated["input_sha256"].items():
                    if _hash(directory / "inputs" / relative) != digest:
                        raise ValueError("Worker changed an immutable task input")
                record = {
                    "task": str(task),
                    "seed": seed,
                    "mode": mode,
                    "command": command,
                    **telemetry,
                    **verify_run(directory, Path(isolated["task"]), seed),
                    "status": "completed",
                }
                _write(manifest, record)
                runs.append(record)
                _write(output / "progress.json", {"completed": len(runs), "runs": runs})
                print(
                    f"DONE {task.parent.name} seed={seed} mode={mode} "
                    f"edits={record['changed_successes']} status={record['research_status']} "
                    f"wall={record['wall_seconds']:.1f}",
                    flush=True,
                )
    report = {
        "model": config["model"],
        "endpoint_fingerprint": endpoint_fingerprint,
        "code": code,
        "runs": runs,
        "gate_passed": all(
            r["returncode"] == 0 and r["changed_successes"] > 0 and r["interpreted_successes"] > 0
            for r in runs
        ),
    }
    if not args.gate:
        report.pop("gate_passed")
        complete_holdout(runs, sources, len(sources) * len(args.seeds) * len(args.modes))
        report["comparison"] = paired_results(runs)
    _write(output / "report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tasks", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gate", action="store_true")
    parser.add_argument("--gate-report", type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 42, 2026])
    parser.add_argument(
        "--modes", nargs="+", choices=("qwen", "optuna"), default=["qwen", "optuna"]
    )
    parser.add_argument("--sessions", type=int, default=12)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--time-budget", type=float, default=1800)
    args = parser.parse_args()
    if args.gate and args.modes != ["qwen"]:
        parser.error("Live gate requires --modes qwen")
    try:
        report = run(args)
    except (OSError, ValueError, KeyError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0 if not args.gate or report["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
