"""Synthetic cohort summaries gate independent holdout scoring and preserve provenance."""

import csv
import hashlib
import importlib.util
import json
from math import sqrt
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


def summarizer():
    path = Path(__file__).resolve().parents[2] / "verification/summarize_benchmarks.py"
    spec = importlib.util.spec_from_file_location("benchmark_summary", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def write_csv(path: Path, columns: list[str], rows: list[list]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(rows)


def cohort(tmp_path: Path, kind: str = "regression", seeds=(7, 42)) -> tuple[Path, dict]:
    source = tmp_path / "task"
    source.mkdir()
    task_path = source / "task.md"
    task_path.write_text("synthetic fixture only")
    write_csv(source / "test.csv", ["id", "x"], [["002", 2], ["001", 1]])
    metric = {"regression": "rmse", "binary": "roc_auc", "multiclass": "macro_f1"}[kind]
    prediction = {"regression": "values", "binary": "probabilities", "multiclass": "labels"}[kind]
    classes = [] if kind == "regression" else ["z", "a"] if kind == "binary" else [3, 1, 2]
    targets = [1, 3] if kind == "regression" else ["a", "z"] if kind == "binary" else [3, 1]
    # Deliberately reverse holdout row order: scoring must align by unique test ID.
    write_csv(
        source / "holdout_targets.csv", ["id", "y"], [["001", targets[1]], ["002", targets[0]]]
    )
    records = []
    for i, seed in enumerate(seeds):
        directory = tmp_path / f"run-{seed}"
        directory.mkdir()
        spec = {
            "test_path": str(source / "test.csv"),
            "id_column": "id",
            "target": "y",
            "kind": kind,
            "metric": metric,
            "prediction_kind": prediction,
            "classes": classes,
            "submission_columns": ["id", "y"],
        }
        dump(directory / "harness/prepared.json", {"spec": spec})
        path = directory / "submission.csv"
        predictions = [2, 3] if kind == "regression" else [0.9, 0.1] if kind == "binary" else [3, 1]
        write_csv(path, ["id", "y"], [["002", predictions[0]], ["001", predictions[1]]])
        raw = predictions if kind != "binary" else [[0.9, 0.1], [0.1, 0.9]]
        raw_classes = classes if kind != "binary" else ["a", "z"]
        dump(directory / "harness/refit/request.json", {"backend": "catboost", "mode": "fallback"})
        dump(
            directory / "harness/refit/result.json",
            {"ids": ["002", "001"], "predictions": raw, "classes": raw_classes},
        )
        dump(directory / "harness/baseline.json", {"score": {"value": 1}})
        records.append(
            {
                "identity": {"task": str(task_path), "seed": seed, "backend": "catboost"},
                "status": "completed",
                "returncode": 0,
                "protocol": {"fingerprint": f"protocol-{seed}"},
                "score_status": "measured",
                "score": {
                    "value": i + 1,
                    "metric": metric,
                    "level": "full",
                    "protocol_id": f"protocol-{seed}",
                },
                "wall_seconds": 10 + i,
                "peak_rss_bytes": (100 + i) * 1024**2,
                "measured_success": True,
                "valid_submission_success": True,
                "fallback_status": False,
                "submission": {
                    "status": "validated",
                    "path": str(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                },
            }
        )
    report = {"mode": "baseline_only", "runs": records}
    path = tmp_path / "report.json"
    dump(path, report)
    dump(
        tmp_path / "configuration.json",
        {"tasks": [{"path": str(task_path)}], "seeds": list(seeds), "backends": ["catboost"]},
    )
    return path, report


@pytest.mark.parametrize(
    "kind,expected", [("regression", 1 / sqrt(2)), ("binary", 1), ("multiclass", 2 / 3)]
)
def test_completed_cohort_posthoc_metrics_and_descriptive_aggregation(
    tmp_path: Path,
    kind: str,
    expected: float,
) -> None:
    path, _ = cohort(tmp_path, kind)
    output = tmp_path / "summary.json"
    result = summarizer().summarize_benchmarks(path, output)
    assert result["cohort_completed"] is True
    group = result["groups"][0]
    assert group["search_score"] == {"n": 2, "mean": 1.5, "std": pytest.approx(1 / sqrt(2))}
    assert group["wall_seconds"]["mean"] == 10.5
    assert group["peak_rss_mib"]["mean"] == 100.5
    assert group["measured_fraction"] == group["valid_csv_fraction"] == 1
    assert group["fallback_fraction"] == 0
    assert all(item["value"] == pytest.approx(expected) for item in result["posthoc"])
    for item in result["posthoc"]:
        assert item["attributed_backend"] == "catboost"
        assert item["used_for_selection"] is False
        assert (
            item["holdout_sha256"]
            == hashlib.sha256(Path(item["holdout_path"]).read_bytes()).hexdigest()
        )
    assert json.loads(output.read_text())["mode"] == "baseline_only"
    assert output.with_suffix(".csv").exists()


@pytest.mark.parametrize("damage", ["running", "missing_run", "missing_configuration"])
def test_incomplete_cohort_never_opens_holdout(tmp_path: Path, monkeypatch, damage: str) -> None:
    path, report = cohort(tmp_path)
    if damage == "running":
        report["runs"][1]["status"] = "running"
    elif damage == "missing_run":
        report["runs"].pop()
    else:
        (tmp_path / "configuration.json").unlink()
    dump(path, report)
    module = summarizer()

    def forbidden(*args):
        raise AssertionError("Holdout scoring must wait for the complete cohort")

    monkeypatch.setattr(module, "_posthoc", forbidden)
    result = module.summarize_benchmarks(path, tmp_path / "summary.json")
    assert not result["cohort_completed"]
    assert all(item["status"] == "blocked" for item in result["posthoc"])
    if damage == "missing_run":
        assert result["groups"][0]["valid_csv_fraction"] == 0.5
        assert result["groups"][0]["expected_runs"] == 2


def test_public_task_has_explicit_unavailable_holdout(tmp_path: Path) -> None:
    path, _ = cohort(tmp_path)
    (tmp_path / "task/holdout_targets.csv").unlink()
    result = summarizer().summarize_benchmarks(path, tmp_path / "summary.json")
    assert all(item["status"] == "unavailable" for item in result["posthoc"])
    assert result["groups"][0]["posthoc_score"]["n"] == 0


@pytest.mark.parametrize("damage", ["duplicate", "missing", "extra"])
def test_holdout_ids_require_exact_unique_coverage(tmp_path: Path, damage: str) -> None:
    path, _ = cohort(tmp_path)
    rows = [["002", 1], ["001", 3]]
    if damage == "duplicate":
        rows.append(["002", 1])
    elif damage == "missing":
        rows.pop()
    else:
        rows.append(["extra", 1])
    write_csv(tmp_path / "task/holdout_targets.csv", ["id", "y"], rows)
    result = summarizer().summarize_benchmarks(path, tmp_path / "summary.json")
    assert all(item["status"] == "invalid" for item in result["posthoc"])


@pytest.mark.parametrize("damage", ["missing_early_ml", "other_backend", "different_predictions"])
def test_fallback_csv_without_matching_ml_is_not_attributed(tmp_path: Path, damage: str) -> None:
    path, report = cohort(tmp_path)
    for run in report["runs"]:
        run["fallback_status"] = True
        run["measured_success"] = False
        directory = Path(run["submission"]["path"]).parent
        if damage == "missing_early_ml":
            (directory / "harness/refit/result.json").unlink()
        elif damage == "other_backend":
            dump(directory / "harness/refit/request.json", {"backend": "lightgbm", "mode": "refit"})
        else:
            dump(
                directory / "harness/refit/result.json",
                {"ids": ["002", "001"], "predictions": [999, 999], "classes": []},
            )
    dump(path, report)
    result = summarizer().summarize_benchmarks(path, tmp_path / "summary.json")
    assert all(item["status"] == "not_attributed" for item in result["posthoc"])
    assert result["groups"][0]["fallback_fraction"] == 1
    assert result["groups"][0]["measured_fraction"] == 0


def test_protocol_mismatch_rejected_before_holdout(tmp_path: Path, monkeypatch) -> None:
    path, report = cohort(tmp_path)
    other = json.loads(json.dumps(report["runs"][0]))
    other["identity"]["backend"] = "lightgbm"
    other["protocol"]["fingerprint"] = "different"
    report["runs"].append(other)
    dump(path, report)
    module = summarizer()
    monkeypatch.setattr(module, "_posthoc", lambda *args: pytest.fail("must not read holdout"))
    with pytest.raises(ValueError, match="Protocol mismatch"):
        module.summarize_benchmarks(path, tmp_path / "summary.json")


def test_saved_csv_checksum_change_is_rejected(tmp_path: Path) -> None:
    path, report = cohort(tmp_path)
    Path(report["runs"][0]["submission"]["path"]).write_text("id,y\n002,99\n001,99\n")
    result = summarizer().summarize_benchmarks(path, tmp_path / "summary.json")
    assert result["posthoc"][0]["status"] == "invalid"
    assert "checksum" in result["posthoc"][0]["reason"]


@pytest.mark.parametrize(
    "backend,expected", [("catboost", "measured"), ("lightgbm", "not_attributed")]
)
def test_final_refit_requires_matching_selected_backend(
    tmp_path: Path,
    backend: str,
    expected: str,
) -> None:
    path, report = cohort(tmp_path)
    for run in report["runs"]:
        directory = Path(run["submission"]["path"]).parent
        (directory / "harness/refit/result.json").unlink()
        run["finalization"] = {
            "refit_status": "succeeded",
            "selected": "final",
            "finalists": [{"id": "final", "members": [{"backend": backend}]}],
        }
    dump(path, report)
    result = summarizer().summarize_benchmarks(path, tmp_path / "summary.json")
    assert all(item["status"] == expected for item in result["posthoc"])


def test_missing_entire_backend_group_is_visible_and_blocks_holdout(tmp_path: Path) -> None:
    path, _ = cohort(tmp_path)
    configuration = tmp_path / "configuration.json"
    config = json.loads(configuration.read_text())
    config["backends"].append("autogluon")
    dump(configuration, config)
    result = summarizer().summarize_benchmarks(path, tmp_path / "summary.json")
    assert not result["cohort_completed"]
    group = next(group for group in result["groups"] if group["backend"] == "autogluon")
    assert group["runs"] == 0
    assert group["expected_runs"] == 2
    assert group["measured_fraction"] == 0
    assert group["search_score"]["n"] == 0


def test_nonzero_exit_valid_csv_is_counted_separately_from_success(tmp_path: Path) -> None:
    path, report = cohort(tmp_path)
    for run in report["runs"]:
        run.update(
            returncode=7,
            execution_success=False,
            valid_submission_success=False,
            measured_success=False,
        )
    dump(path, report)
    result = summarizer().summarize_benchmarks(path, tmp_path / "summary.json")
    group = result["groups"][0]
    assert group["valid_csv_fraction"] == 1
    assert group["execution_success_fraction"] == 0
    assert group["measured_fraction"] == 0
    assert all(run["measured_success"] is False for run in report["runs"])
