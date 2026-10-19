"""Descriptive baseline aggregates and gated post-hoc holdout evaluation; no model selection."""

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from sklearn import metrics  # type: ignore[import-untyped]


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def _csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns, rows = list(reader.fieldnames or []), list(reader)
    if (
        not columns
        or not rows
        or len(set(columns)) != len(columns)
        or any(None in row or any(value is None for value in row.values()) for row in rows)
    ):
        raise ValueError("Invalid CSV schema or rows")
    return columns, rows


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _stats(values: list[float]) -> dict[str, Any]:
    return {
        "n": len(values),
        "mean": statistics.mean(values) if values else None,
        "std": statistics.stdev(values) if len(values) > 1 else 0.0 if values else None,
    }


def _identity(run: dict[str, Any]) -> tuple[str, int, str]:
    item = run["identity"]
    return str(item["task"]), item["seed"], str(item["backend"])


def _label(value: Any, classes: list[Any]) -> int:
    matches = [i for i, label in enumerate(classes) if value == label]
    if not matches and isinstance(value, str) and all(type(c) in (int, float) for c in classes):
        try:
            number = float(value)
            if math.isfinite(number):
                matches = [i for i, label in enumerate(classes) if number == label]
        except ValueError:
            pass
    if len(matches) != 1:
        raise ValueError("Unknown or ambiguous class")
    return matches[0]


def _predictions(path: Path, spec: dict[str, Any]) -> tuple[list[str], list[Any]]:
    columns, rows = _csv(path)
    _, test = _csv(Path(spec["test_path"]))
    ids = [row[spec["id_column"]] for row in rows]
    expected = [row[spec["id_column"]] for row in test]
    if (
        columns != spec["submission_columns"]
        or ids != expected
        or len(set(ids)) != len(ids)
        or any(not value for value in ids)
    ):
        raise ValueError("Submission schema, IDs or order disagree")
    predictions = []
    for row in rows:
        if spec["prediction_kind"] == "labels":
            prediction: Any = _label(row[columns[1]], spec["classes"])
        else:
            values = [float(row[column]) for column in columns[1:]]
            if not all(math.isfinite(value) for value in values):
                raise ValueError("Nonfinite predictions")
            if spec["prediction_kind"] == "values":
                prediction = values[0]
            else:
                if any(value < 0 or value > 1 for value in values):
                    raise ValueError("Probability outside [0,1]")
                if spec["kind"] == "binary":
                    prediction = [1 - values[0], values[0]]
                else:
                    if len(values) != len(spec["classes"]) or not math.isclose(
                        sum(values),
                        1,
                        rel_tol=1e-6,
                        abs_tol=1e-6,
                    ):
                        raise ValueError("Invalid multiclass probabilities")
                    prediction = values
        predictions.append(prediction)
    return ids, predictions


def _same(left: Any, right: Any) -> bool:
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _same(a, b) for a, b in zip(left, right, strict=True)
        )
    return bool(math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-12))


def _attributed(
    run: dict[str, Any],
    directory: Path,
    spec: dict[str, Any],
    ids: list[str],
    predictions: list[Any],
) -> bool:
    backend = run["identity"]["backend"]
    final = run.get("finalization", {})
    if final.get("refit_status") == "succeeded":
        selected = next(
            (f for f in final.get("finalists", []) if f["id"] == final.get("selected")), None
        )
        if (
            selected
            and selected.get("members")
            and all(member.get("backend") == backend for member in selected["members"])
        ):
            return True
    # A failed final refit may preserve a successful early ML result. Attribute it
    # only if its backend and predictions match the saved CSV, not merely its filename.
    request_path, result_path = (
        directory / "harness/refit/request.json",
        directory / "harness/refit/result.json",
    )
    if (
        not request_path.exists()
        or not result_path.exists()
        or not (directory / "harness/baseline.json").exists()
    ):
        return False
    request, output = _json(request_path), _json(result_path)
    if (
        request.get("backend") != backend
        or request.get("mode") not in ("fallback", "refit")
        or output.get("ids") != ids
    ):
        return False
    raw = output.get("predictions", [])
    if len(raw) != len(ids):
        return False
    if spec["prediction_kind"] == "labels":
        normalized: list[Any] = [_label(value, spec["classes"]) for value in raw]
    elif spec["prediction_kind"] == "probabilities":
        order = [_label(value, spec["classes"]) for value in output.get("classes", [])]
        if sorted(order) != list(range(len(spec["classes"]))):
            return False
        normalized = [[row[order.index(i)] for i in range(len(order))] for row in raw]
    else:
        normalized = raw
    return _same(normalized, predictions)


def _posthoc(run: dict[str, Any]) -> dict[str, Any]:
    holdout = Path(run["identity"]["task"]).parent / "holdout_targets.csv"
    if not holdout.exists():
        return {
            "status": "unavailable",
            "reason": "No independent holdout_targets.csv for this task",
        }
    submission = run.get("submission", {})
    if submission.get("status") != "validated":
        return {"status": "unavailable", "reason": "No validated saved submission"}
    try:
        path = Path(submission["path"])
        if _hash(path) != submission["sha256"]:
            raise ValueError("Saved submission checksum changed")
        directory = path.parent
        spec = _json(directory / "harness/prepared.json")["spec"]
        ids, predictions = _predictions(path, spec)
        if not _attributed(run, directory, spec, ids, predictions):
            return {
                "status": "not_attributed",
                "reason": "Saved CSV lacks successful matching backend ML provenance",
            }
        holdout_hash = _hash(holdout)
        columns, rows = _csv(holdout)
        if columns != [spec["id_column"], spec["target"]]:
            raise ValueError("Holdout schema disagrees")
        by_id = {row[spec["id_column"]]: row[spec["target"]] for row in rows}
        if len(by_id) != len(rows) or set(by_id) != set(ids) or any(not value for value in by_id):
            raise ValueError("Holdout IDs must be unique and match all test IDs exactly")
        raw_targets = [by_id[identifier] for identifier in ids]
        if spec["metric"] == "rmse":
            targets: list[Any] = [float(value) for value in raw_targets]
            if not all(math.isfinite(value) for value in targets):
                raise ValueError("Nonfinite holdout target")
            score = metrics.root_mean_squared_error(targets, predictions)
        else:
            targets = [_label(value, spec["classes"]) for value in raw_targets]
            if spec["metric"] == "roc_auc" and spec["kind"] == "binary":
                score = metrics.roc_auc_score(targets, [row[1] for row in predictions])
            elif spec["metric"] == "roc_auc":
                score = metrics.roc_auc_score(
                    targets,
                    predictions,
                    multi_class="ovr",
                    average="macro",
                    labels=list(range(len(spec["classes"]))),
                )
            elif spec["metric"] == "macro_f1":
                score = metrics.f1_score(
                    targets,
                    predictions,
                    average="macro",
                    zero_division=0,
                    labels=list(range(len(spec["classes"]))),
                )
            else:
                raise ValueError("Unsupported post-hoc metric")
        if not math.isfinite(float(score)):
            raise ValueError("Nonfinite post-hoc metric")
        return {
            "status": "measured",
            "metric": spec["metric"],
            "value": float(score),
            "attributed_backend": run["identity"]["backend"],
            "holdout_sha256": holdout_hash,
            "submission_sha256": submission["sha256"],
            "holdout_path": str(holdout),
            "used_for_selection": False,
        }
    except (OSError, KeyError, TypeError, ValueError, IndexError, OverflowError) as error:
        return {"status": "invalid", "reason": str(error)}


def summarize_benchmarks(report_path: Path, output: Path) -> dict[str, Any]:
    report = _json(report_path)
    runs = report.get("runs", [])
    protocols: dict[tuple[str, int], str] = {}
    seen = set()
    task_metrics: dict[str, str] = {}
    for run in runs:
        task, seed, backend = _identity(run)
        if (task, seed, backend) in seen:
            raise ValueError("Duplicate task/seed/backend run")
        seen.add((task, seed, backend))
        score_metric = run.get("score", {}).get("metric")
        if run.get("score_status") == "measured" and isinstance(score_metric, str):
            if task in task_metrics and task_metrics[task] != score_metric:
                raise ValueError("Metric mismatch for the same task")
            task_metrics[task] = score_metric
        fingerprint = run.get("protocol", {}).get("fingerprint")
        if fingerprint:
            key = (task, seed)
            if key in protocols and protocols[key] != fingerprint:
                raise ValueError("Protocol mismatch for the same task and seed")
            protocols[key] = fingerprint
    configuration = report_path.parent / "configuration.json"
    complete = False
    expected: set[tuple[str, int, str]] = set()
    if configuration.exists():
        config = _json(configuration)
        expected = {
            (task["path"], seed, backend)
            for task in config["tasks"]
            for seed in config["seeds"]
            for backend in config["backends"]
        }
        complete = (
            bool(expected)
            and seen == expected
            and all(run.get("status") == "completed" for run in runs)
        )
    posthoc = [
        dict(
            identity=run["identity"],
            **(
                _posthoc(run)
                if complete
                else {
                    "status": "blocked",
                    "reason": "Expected cohort is not fully completed and verified",
                }
            ),
        )
        for run in runs
    ]
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        task, _, backend = _identity(run)
        groups[task, backend].append(run)
    for task, _, backend in expected:
        groups[task, backend]
    aggregates = []
    for (task, backend), members in sorted(groups.items()):
        measured = []
        score_metrics = set()
        for run in members:
            score = run.get("score", {})
            if (
                run.get("status") == "completed"
                and run.get("score_status") == "measured"
                and score.get("level") == "full"
                and score.get("protocol_id") == run.get("protocol", {}).get("fingerprint")
                and _number(score.get("value"))
            ):
                measured.append(float(score["value"]))
                score_metrics.add(score.get("metric"))
        if len(score_metrics) > 1:
            raise ValueError("Different metrics cannot be aggregated together")
        expected_count = sum(t == task and b == backend for t, _, b in expected)
        denominator = max(len(members), expected_count)
        external = [
            item["value"]
            for item in posthoc
            if item["identity"]["task"] == task
            and item["identity"]["backend"] == backend
            and item["status"] == "measured"
        ]
        aggregates.append(
            {
                "task": task,
                "backend": backend,
                "runs": len(members),
                "expected_runs": expected_count or None,
                "completed_runs": sum(run.get("status") == "completed" for run in members),
                "metric": next(iter(score_metrics), None),
                "search_score": _stats(measured),
                "wall_seconds": _stats(
                    [
                        float(run["wall_seconds"])
                        for run in members
                        if _number(run.get("wall_seconds"))
                    ]
                ),
                "peak_rss_mib": _stats(
                    [
                        float(run["peak_rss_bytes"]) / 1024**2
                        for run in members
                        if _number(run.get("peak_rss_bytes"))
                    ]
                ),
                "measured_fraction": sum(
                    run.get("status") == "completed" and run.get("measured_success") is True
                    for run in members
                )
                / denominator,
                "valid_csv_fraction": sum(
                    run.get("status") == "completed" and run.get("valid_submission_success") is True
                    for run in members
                )
                / denominator,
                "fallback_fraction": sum(
                    run.get("status") == "completed" and run.get("fallback_status") is True
                    for run in members
                )
                / denominator,
                "posthoc_score": _stats(external),
                "protocols_by_seed": {
                    str(run["identity"]["seed"]): run.get("protocol", {}).get("fingerprint")
                    for run in members
                },
            }
        )
    result = {
        "mode": report.get("mode", "baseline_only"),
        "cohort_completed": complete,
        "aggregation": (
            "Descriptive mean and sample standard deviation across seeds; "
            "no cross-protocol ranking or model selection."
        ),
        "report_sha256": _hash(report_path),
        "configuration_sha256": _hash(configuration) if configuration.exists() else None,
        "groups": aggregates,
        "posthoc": posthoc,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8"
    )
    table = output.with_suffix(".csv")
    if table == output:
        table = output.with_name(output.stem + ".summary.csv")
    with table.open("w", encoding="utf-8", newline="") as stream:
        fields = [
            "task",
            "backend",
            "runs",
            "completed_runs",
            "metric",
            "search_mean",
            "search_std",
            "wall_mean",
            "wall_std",
            "rss_mib_mean",
            "rss_mib_std",
            "measured_fraction",
            "valid_csv_fraction",
            "fallback_fraction",
            "posthoc_mean",
            "posthoc_std",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for group in aggregates:
            writer.writerow(
                {
                    **{key: group[key] for key in fields if key in group},
                    **{
                        f"{prefix}_{stat}": group[source][stat]
                        for prefix, source in (
                            ("search", "search_score"),
                            ("wall", "wall_seconds"),
                            ("rss_mib", "peak_rss_mib"),
                            ("posthoc", "posthoc_score"),
                        )
                        for stat in ("mean", "std")
                    },
                }
            )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize_benchmarks(args.report, args.output)
    print(
        json.dumps(
            {"cohort_completed": result["cohort_completed"], "groups": len(result["groups"])}
        )
    )


if __name__ == "__main__":
    main()
