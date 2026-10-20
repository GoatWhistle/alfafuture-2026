"""Publish improved development candidates without opening or freezing control."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ml_harness.adapters.data.task import read_csv
from ml_harness.adapters.ml.evaluation import validate_predictions
from ml_harness.adapters.research.finalization import _score
from ml_harness.adapters.research.workspace import ResearchWorkspace, write_json
from ml_harness.application.ports.clock import Clock
from ml_harness.application.ports.execution import ExperimentRunner, RunRequest, SubmissionWriter
from ml_harness.domain.budget import WallClockBudget
from ml_harness.domain.models import ExperimentStatus, PreparedTask


def publish_checkpoint(
    task: PreparedTask,
    workspace: ResearchWorkspace,
    runner: ExperimentRunner,
    writer: SubmissionWriter,
    wall: WallClockBudget,
    clock: Clock,
    *,
    threads: int = 8,
    timeout_seconds: float = 600,
) -> dict[str, Any] | None:
    """Best verified full pipeline, frozen parameters, and atomic submission writer."""
    state_path = workspace.path / "checkpoint.json"
    previous = json.loads(state_path.read_text()) if state_path.exists() else None
    reports = ([workspace.baseline_report] if workspace.baseline_report else []) + workspace.reports
    candidates = []
    for report in reports:
        if (
            report.get("status") != "succeeded"
            or report.get("measured_by") != "harness"
            or report.get("level") != "full"
            or report.get("protocol_id") != task.protocol.fingerprint
            or not report.get("prediction_path")
        ):
            continue
        try:
            raw = Path(report["prediction_path"]).read_bytes()
            if (
                report.get("prediction_hash")
                and hashlib.sha256(raw).hexdigest() != report["prediction_hash"]
            ):
                continue
            score = workspace.evaluator.evaluate(task, json.loads(raw))
            if score != _score(report["full_score"]):
                continue
            floor = previous or workspace.baseline_report
            if floor and not score.better_than(_score(floor.get("score", floor.get("full_score")))):
                continue
            candidates.append((score, report))
        except (OSError, ValueError, KeyError, TypeError):
            continue
    if not candidates:
        return None
    score, report = candidates[0]
    for alternative_score, alternative in candidates[1:]:
        if alternative_score.better_than(score):
            score, report = alternative_score, alternative
    available = wall.available(clock.now())
    # Refit the full data; use a conservative measured cost rather than smoke quality.
    estimate = 1.5 * max(wall.refit_seconds, wall.validation_seconds) + wall.write_seconds
    if estimate > available or estimate > timeout_seconds or available <= 0:
        return None
    parameters = report.get("refit_parameters", report.get("parameters", {}))
    if report.get("tuning_report") and "refit_parameters" not in report:
        return None
    code_path = report.get("refit_code_path", report.get("code_path"))
    code_hash = report.get("refit_code_hash", report.get("code_hash"))
    if not code_path or hashlib.sha256(Path(code_path).read_bytes()).hexdigest() != code_hash:
        return None
    directory = workspace.path / "checkpoints" / report["id"]
    # Failed/interrupted attempts are not reexecuted during the same search.
    if directory.exists():
        return None
    directory.mkdir(parents=True)
    result = runner.run(
        RunRequest(
            task,
            str(directory),
            mode="refit",
            backend=report.get("backend", "catboost"),
            candidate_path=code_path,
            parameters=parameters,
            threads=threads,
        ),
        min(timeout_seconds, available),
    )
    if result.status != ExperimentStatus.SUCCEEDED or result.output is None:
        return None
    _, rows = read_csv(Path(task.spec.test_path))
    validate_predictions(task.spec, [row[task.spec.id_column] for row in rows], result.output)
    writer.write(task, result.output)
    checkpoint = {
        "id": report["id"],
        "score": asdict(score),
        "protocol_id": task.protocol.fingerprint,
        "prediction_path": str(directory / "result.json"),
        "code_hash": code_hash,
        "parameters": parameters,
        "submission_path": task.spec.submission_path,
        "submission_hash": hashlib.sha256(Path(task.spec.submission_path).read_bytes()).hexdigest(),
    }
    write_json(state_path, checkpoint)
    workspace.store.append(
        {
            "event_id": "checkpoint:" + report["id"],
            "kind": "checkpoint_published",
            "payload": checkpoint,
        }
    )
    return checkpoint
