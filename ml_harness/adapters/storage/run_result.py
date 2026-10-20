"""Build the version 3 summary from persisted harness measurements."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ml_harness.adapters.research.workspace import write_json
from ml_harness.adapters.storage.submission import load_submission
from ml_harness.domain.models import PreparedTask
from ml_harness.domain.run_result import RunResult


def _object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _recovered_research(directory: Path) -> dict[str, Any]:
    """Recover measured facts without truncating or repairing an interrupted journal."""
    journal = directory / "research/memory/events.jsonl"
    events = {}
    try:
        for line in journal.read_bytes().splitlines(keepends=True):
            if not line.endswith(b"\n"):
                break
            try:
                event = json.loads(line)
                if isinstance(event, dict) and isinstance(event.get("payload"), dict):
                    events[event["event_id"]] = event
            except (ValueError, KeyError, TypeError):
                continue
    except OSError:
        pass
    budget = _object(directory / "budget.json")
    tokens = reported = budget.get("preparation_tokens", 0)
    known = budget.get("preparation_usage_known", True) is True
    accounted = known
    started, finished = set(), set()
    reports = []
    for event in events.values():
        payload = event["payload"]
        if event.get("kind") == "session_started":
            started.add(payload.get("id"))
        elif event.get("kind") == "session_finished":
            finished.add(payload.get("id"))
            usage = payload.get("usage", {})
            measured = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
            reported += measured
            tokens += max(measured, usage.get("tokens_charged", measured))
            known = known and usage.get("usage_known") is True
            accounted = (
                accounted and usage.get("budget_accounted", usage.get("usage_known")) is True
            )
        elif event.get("kind") in ("experiment_finished", "baseline_registered"):
            reports.append(payload)
    return {
        "reports": reports,
        "tokens_used": tokens,
        "reported_tokens": reported,
        "usage_known": known and not (started - finished),
        "budget_accounted": accounted and not (started - finished),
        "status": "interrupted" if events else "disabled",
    }


def save_run_result(
    directory: Path, task: PreparedTask, details: dict[str, Any], resources: dict[str, Any]
) -> dict[str, Any]:
    research = details.get("research") or _recovered_research(directory)
    final = research.get("finalization", details.get("finalization", {}))
    checkpoint = {}
    checkpoint_path = directory / "research/checkpoint.json"
    if final.get("refit_status") != "succeeded" and checkpoint_path.exists():
        candidate = _object(checkpoint_path)
        submission = Path(task.spec.submission_path)
        if (
            submission.exists()
            and candidate.get("protocol_id") == task.protocol.fingerprint
            and candidate.get("submission_hash")
            == hashlib.sha256(submission.read_bytes()).hexdigest()
        ):
            checkpoint = candidate
    selected = checkpoint.get("id", final.get("selected", "baseline"))

    report: dict[str, Any] = next(
        (r for r in research.get("reports", []) if r["id"] == selected), {}
    )
    finalist: dict[str, Any] = next(
        (r for r in final.get("finalists", []) if r["id"] == selected), {}
    )
    try:
        load_submission(task)
        result_status = "refitted" if final.get("refit_status") == "succeeded" else "preserved"
    except (OSError, ValueError):
        result_status = "unavailable"
    recipe = report.get("executed_recipe") or report.get("resolved_spec")
    if recipe is None and report.get("backend"):
        fitted = report.get("refit_parameters", report.get("parameters", {}))
        recipe = {
            "task_kind": task.spec.kind.value,
            "backend": report["backend"],
            "parameters": {
                key: value for key, value in fitted.items() if key not in ("backend", "tuning")
            },
            "transforms": [],
            "python_transform": "",
            "tuning": {},
        }
    origin = report.get("hypothesis_source", "baseline")
    if len(finalist.get("members", [])) > 1:
        members = []
        for member in finalist["members"]:
            source: dict[str, Any] = next(
                (item for item in research.get("reports", []) if item["id"] == member["id"]), {}
            )
            fitted_recipe = source.get("executed_recipe") or source.get("resolved_spec")
            members.append(
                {
                    "id": member["id"],
                    "recipe": fitted_recipe,
                    "backend": member.get("backend"),
                    "parameters": member.get("parameters", {}),
                    "candidate_path": member.get("candidate_path"),
                    "code_hash": member.get("code_hash"),
                }
            )
        recipe = {
            "members": members,
            "weights": finalist.get("weights", []),
            "prediction_rule": (
                "weighted mean of values"
                if task.spec.kind.value == "regression"
                else "weighted mean of class probabilities then argmax in class order"
                if task.spec.prediction_kind.value == "labels"
                else "weighted mean of class probabilities"
            ),
            "classes": list(task.spec.classes),
        }
        origin = "ensemble"
    if not report and not finalist and not checkpoint and not details.get("score"):
        origin = "fallback"
    value = asdict(
        RunResult(
            result_status=result_status,
            search_status=research.get("status", "disabled"),
            selected=selected,
            recipe=recipe,
            origin=origin,
            score=checkpoint.get("score", finalist.get("score", details.get("score"))),
            protocol=task.protocol.fingerprint,
            stop_reason=details.get("reason") or research.get("status"),
            submission_path=task.spec.submission_path,
            usage={
                "tokens_used": research.get("tokens_used", 0),
                "reported_tokens": research.get("reported_tokens", 0),
                "usage_known": research.get("usage_known", True),
                "budget_accounted": research.get("budget_accounted", True),
            },
            resources=resources,
            rejections=tuple(
                {"id": r["id"], "error": r.get("error"), "admission": r.get("admission")}
                for r in research.get("reports", [])
                if r.get("status") != "succeeded"
            ),
        )
    )
    write_json(directory / "run-result.json", value)
    return value
