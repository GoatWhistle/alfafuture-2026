"""Frozen five-mode cohorts with external targets opened only after all runs finish."""

import argparse
import csv
import hashlib
import json
import os
import random
import statistics
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from ml_harness.adapters.data.task import CsvTaskData
from verification.run_agent_comparison import source_hashes, validate_saved_gate
from verification.run_benchmarks import (
    ROOT,
    _hash,
    _label_index,
    _measure,
    _runtime_versions,
    _write,
)
from verification.verify_archive import copy_task

MODES = ("frozen", "optuna", "qwen", "autogluon", "flaml")


def write_csv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def benchmark_copy(source: Path, destination: Path, seed: int = 2026) -> Path:
    """Make new external test rows from labeled train; source materials stay untouched."""
    task = CsvTaskData().prepare(str(source), seed)
    spec = task.spec
    destination.mkdir(parents=True, exist_ok=False)
    with Path(spec.train_path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        columns, rows = list(reader.fieldnames or []), list(reader)
    rng = random.Random(seed)
    heldout: set[int] = set()
    buckets: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        buckets.setdefault(row[spec.target] if spec.kind != "regression" else "all", []).append(i)
    for bucket in buckets.values():
        rng.shuffle(bucket)
        heldout.update(bucket[: max(1, round(len(bucket) * 0.2))])
    train = [row for i, row in enumerate(rows) if i not in heldout]
    test = [row for i, row in enumerate(rows) if i in heldout]
    feature_columns = [column for column in columns if column != spec.target]
    write_csv(destination / "train.csv", columns, train)
    write_csv(
        destination / "test.csv",
        feature_columns,
        [{c: row[c] for c in feature_columns} for row in test],
    )
    write_csv(
        destination / "external_targets.csv",
        [spec.id_column, spec.target],
        [{spec.id_column: row[spec.id_column], spec.target: row[spec.target]} for row in test],
    )
    write_csv(
        destination / "sample_submission.csv",
        list(spec.submission_columns),
        [
            {
                column: row[spec.id_column]
                if column == spec.id_column
                else str(spec.classes[0])
                if spec.prediction_kind == "labels"
                else "0"
                for column in spec.submission_columns
            }
            for row in test
        ],
    )
    (destination / "data_dictionary.csv").write_bytes(Path(spec.dictionary_path).read_bytes())
    metadata = {
        key: value
        for key, value in asdict(spec).items()
        if key
        in (
            "kind",
            "metric",
            "prediction_kind",
            "target",
            "id_column",
            "classes",
            "time_column",
            "group_column",
        )
        and value is not None
    }
    metadata["submission_path"] = "submission.csv"
    (destination / "task.md").write_text(
        "# External benchmark copy\n"
        "Train train.csv; predict test.csv; schema sample_submission.csv.\n"
        "```task-json\n" + json.dumps(metadata) + "\n```\n"
    )
    _write(
        destination / "provenance.json",
        {
            "source": str(source.resolve()),
            "train_sha256": _hash(Path(spec.train_path)),
            "holdout_seed": seed,
            "external_indices": sorted(heldout),
        },
    )
    return destination / "task.md"


def snapshot_baseline(revision: str, destination: Path) -> str:
    """Extract only tracked Python sources, fixing the revision before execution."""
    commit = subprocess.check_output(["git", "rev-parse", revision], cwd=ROOT, text=True).strip()
    names = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", commit], cwd=ROOT, text=True
    ).splitlines()
    destination.mkdir(parents=True, exist_ok=False)
    for name in names:
        if name == "solution.py" or (name.startswith("ml_harness/") and name.endswith(".py")):
            path = destination / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=ROOT))
    return commit


def external_score(task_path: Path, submission: Path) -> dict[str, Any]:
    """Independent scorer consumes only completed CSVs; targets never enter a worker."""
    from sklearn import metrics  # type: ignore[import-untyped]

    task = CsvTaskData().prepare(str(task_path), 42)
    spec = task.spec
    from ml_harness.adapters.storage.submission import load_submission

    load_submission(replace(task, spec=replace(spec, submission_path=str(submission))))
    with (task_path.parent / "external_targets.csv").open(newline="") as stream:
        truth = {row[spec.id_column]: row[spec.target] for row in csv.DictReader(stream)}
    with submission.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    ids = [row[spec.id_column] for row in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(truth):
        raise ValueError("External prediction IDs differ")
    columns = [c for c in spec.submission_columns if c != spec.id_column]
    if spec.kind == "regression":
        y = [float(truth[i]) for i in ids]
        prediction = [float(row[columns[0]]) for row in rows]
        fn = (
            metrics.root_mean_squared_error
            if spec.metric == "rmse"
            else metrics.mean_absolute_error
        )
        value = fn(y, prediction)
    else:
        labels = list(spec.classes)
        y = [_label_index(truth[i], labels) for i in ids]
        if spec.prediction_kind == "probabilities":
            probability = [[float(row[c]) for c in columns] for row in rows]
            if len(spec.classes) == 2:
                probability = [[1 - p[0], p[0]] for p in probability]
            value = (
                metrics.log_loss(y, probability, labels=list(range(len(labels))))
                if spec.metric == "log_loss"
                else metrics.roc_auc_score(
                    y,
                    [p[1] for p in probability] if len(labels) == 2 else probability,
                    **({} if len(labels) == 2 else {"multi_class": "ovr"}),
                )
            )
        else:
            prediction = [_label_index(row[columns[0]], labels) for row in rows]
            value = (
                metrics.accuracy_score(y, prediction)
                if spec.metric == "accuracy"
                else metrics.f1_score(y, prediction, average="macro", zero_division=0)
            )
    return {"metric": str(spec.metric), "value": float(value), "source": "external_holdout"}


def run(args: argparse.Namespace) -> dict[str, Any]:
    if (
        not args.modes
        or not args.seeds
        or len({p.resolve() for p in args.tasks}) != len(args.tasks)
        or len(set(args.modes)) != len(args.modes)
        or len(set(args.seeds)) != len(args.seeds)
        or not args.tasks
        or not 0 < args.time_budget <= 10800
        or not 1 <= args.threads <= 8
    ):
        raise ValueError("Invalid cohort configuration")
    output = args.output.resolve()
    code = source_hashes()
    code["verification/run_hybrid_comparison.py"] = _hash(Path(__file__))
    if "qwen" in args.modes:
        if args.gate_report is None:
            raise ValueError("Qwen comparison requires the target provider functional gate")
        validate_saved_gate(
            args.gate_report,
            source_hashes(),
            os.getenv("OPENAI_MODEL"),
            hashlib.sha256(os.getenv("OPENAI_BASE_URL", "").encode()).hexdigest(),
        )
    output.mkdir(parents=True, exist_ok=False)
    baseline = output / "frozen-source"
    revision = snapshot_baseline(args.baseline_ref, baseline)
    baseline_hashes = {str(p.relative_to(baseline)): _hash(p) for p in baseline.rglob("*.py")}
    tasks = [
        benchmark_copy(p.resolve(), output / "tasks" / f"{i}-{p.parent.name}")
        for i, p in enumerate(args.tasks)
    ]
    task_hashes = {
        str(p.relative_to(output)): _hash(p)
        for task in tasks
        for p in task.parent.iterdir()
        if p.is_file()
    }
    config = {
        "version": 3,
        "code": code,
        "baseline_commit": revision,
        "baseline_hashes": baseline_hashes,
        "seeds": args.seeds,
        "modes": args.modes,
        "seconds": args.time_budget,
        "threads": args.threads,
        "memory_mb": 6000,
        "runtime": _runtime_versions(),
        "tasks": {str(p): _hash(p.parent / "provenance.json") for p in tasks},
        "task_hashes": task_hashes,
    }
    _write(output / "configuration.json", config)
    records = []
    for task in tasks:
        for seed in args.seeds:
            for mode in args.modes:
                if any(_hash(ROOT / p) != digest for p, digest in code.items()):
                    raise ValueError("Source changed during frozen cohort")
                if any(_hash(output / p) != digest for p, digest in task_hashes.items()):
                    raise ValueError("Frozen benchmark inputs changed during cohort")
                directory = output / "runs" / task.parent.name / f"{seed}-{mode}"
                directory.mkdir(parents=True)
                isolated = copy_task(ROOT, task, directory / "inputs", directory, seed)
                command = [
                    sys.executable,
                    str((baseline if mode == "frozen" else ROOT) / "solution.py"),
                    isolated["task"],
                    "--run-dir",
                    str(directory / "harness"),
                    "--seed",
                    str(seed),
                    "--threads",
                    str(args.threads),
                    "--time-budget",
                    str(args.time_budget),
                    "--memory-limit-mb",
                    "6000",
                    "--worker-timeout",
                    str(args.time_budget),
                    "--backend",
                    "autogluon-comparison"
                    if mode == "autogluon"
                    else "flaml"
                    if mode == "flaml"
                    else "portfolio",
                    "--search-mode",
                    "qwen"
                    if mode == "qwen"
                    else "none"
                    if mode in ("autogluon", "flaml")
                    else "optuna",
                ]
                credentials = (
                    {
                        k: os.environ[k]
                        for k in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL")
                    }
                    if mode == "qwen"
                    else None
                )
                _write(directory / "manifest.json", {"status": "running", "command": command})
                telemetry = _measure(
                    command, directory, args.time_budget, 6000, 0.05, llm_environment=credentials
                )
                if any(
                    _hash(directory / "inputs" / p) != digest
                    for p, digest in isolated["input_sha256"].items()
                ):
                    raise ValueError("Worker changed frozen input")
                if any(_hash(baseline / p) != digest for p, digest in baseline_hashes.items()):
                    raise ValueError("Frozen baseline source changed")
                prepared = CsvTaskData().prepare(isolated["task"], seed)
                record = {
                    "task": str(task),
                    "seed": seed,
                    "mode": mode,
                    **telemetry,
                    "submission": prepared.spec.submission_path,
                    "command": command,
                }
                summary_path = directory / "harness/summary.json"
                if summary_path.exists():
                    summary = json.loads(summary_path.read_text())
                    research = summary.get("research", {})
                    facts = summary.get("run_result", {})
                    reports = [r for r in research.get("reports", []) if r.get("hypothesis_source")]
                    record["search_measurements"] = {
                        "research_status": research.get("status"),
                        "completed_experiments": sum(
                            r.get("status") == "succeeded" for r in reports
                        ),
                        "improved_experiments": sum(
                            (
                                r.get("parent_delta", 0) < 0
                                if prepared.spec.metric in ("rmse", "mae", "log_loss")
                                else r.get("parent_delta", 0) > 0
                            )
                            for r in reports
                        ),
                        "reported_tokens": facts.get("usage", {}).get(
                            "reported_tokens", research.get("reported_tokens")
                        ),
                        "tokens_charged": facts.get("usage", {}).get(
                            "tokens_used", research.get("tokens_used")
                        ),
                        "usage_known": facts.get("usage", {}).get(
                            "usage_known", research.get("usage_known")
                        ),
                        "result_status": facts.get("result_status", summary.get("status")),
                        "refit_status": summary.get(
                            "finalization", research.get("finalization", {})
                        ).get("refit_status"),
                    }
                records.append(record)
                _write(directory / "manifest.json", {"status": "completed", **record})
                _write(output / "progress.json", {"runs": records})
    if any(_hash(ROOT / p) != digest for p, digest in code.items()):
        raise ValueError("Source changed during frozen cohort")
    if any(_hash(output / p) != digest for p, digest in task_hashes.items()):
        raise ValueError("Frozen benchmark inputs changed during cohort")
    # No external target is read before the entire frozen cohort has completed.
    for record in records:
        try:
            record["external_quality"] = external_score(
                Path(record["task"]), Path(record["submission"])
            )
            record["valid_csv"] = True
        except (OSError, ValueError, KeyError) as error:
            record.update(valid_csv=False, evaluation_error=str(error))
    report = {
        "configuration": config,
        "runs": records,
        "valid_csv_fraction": sum(r["valid_csv"] for r in records) / len(records),
    }
    paired = []
    for task in tasks:
        for seed in args.seeds:
            pair = {r["mode"]: r for r in records if r["task"] == str(task) and r["seed"] == seed}
            if all(mode in pair and pair[mode]["valid_csv"] for mode in ("qwen", "optuna")):
                qwen, numeric = pair["qwen"]["external_quality"], pair["optuna"]["external_quality"]
                sign = -1 if qwen["metric"] in ("rmse", "mae", "log_loss") else 1
                paired.append(
                    {
                        "task": str(task),
                        "seed": seed,
                        "relative_gain": sign
                        * (qwen["value"] - numeric["value"])
                        / max(abs(numeric["value"]), 1e-12),
                    }
                )
    means = {
        str(task): statistics.mean(p["relative_gain"] for p in paired if p["task"] == str(task))
        for task in tasks
        if any(p["task"] == str(task) for p in paired)
    }
    report["paired_hybrid_gain"] = {
        "pairs": paired,
        "task_means": means,
        "benefit_gate_passed": len(paired) == len(tasks) * len(args.seeds)
        and sum(value > 0 for value in means.values()) >= 2
        and statistics.mean(means.values()) > 0
        and all(r["valid_csv"] for r in records),
        "interpretation": (
            "Descriptive paired comparison; three seeds do not establish statistical significance"
        ),
    }
    _write(output / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tasks", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--baseline-ref", default="3327e07")
    parser.add_argument("--gate-report", type=Path)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 42, 2026])
    parser.add_argument("--time-budget", type=float, default=1800)
    parser.add_argument("--threads", type=int, default=4)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
