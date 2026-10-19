"""Choose an evidence-backed parent and retain segment-specific alternatives."""

import json
from dataclasses import dataclass
from math import isfinite
from typing import Any

from ml_harness.domain.models import ContractError, Metric, PreparedTask, Score


def diagnostic_focus(diagnostic: dict[str, Any], limit: int = 4) -> list[dict[str, Any]]:
    """Small measured summaries; interpretations never affect ranking."""
    focuses: list[dict[str, Any]] = []
    for row in diagnostic.get("class_report", []):
        if row.get("sufficient_support"):
            focuses.append(
                {
                    "key": "class:" + json.dumps(row["class"], ensure_ascii=False),
                    "class": row["class"],
                    "support": row["support"],
                    "loss": 1 - row["f1"],
                    "f1": row["f1"],
                    "recall": row["recall"],
                    "delta": row.get("delta"),
                }
            )
    for row in diagnostic.get("segments", []):
        focuses.append(
            {
                "key": "segment:"
                + json.dumps(row["definition"], sort_keys=True, ensure_ascii=False),
                "definition": row["definition"],
                "support": row["support"],
                "loss": row["metrics"]["mae"],
                "metrics": row["metrics"],
                "delta": row.get("delta"),
            }
        )
    return sorted(focuses, key=lambda item: (-item["loss"], -item["support"], item["key"]))[:limit]


def measured_score(task: PreparedTask, report: dict[str, Any]) -> Score | None:
    if (
        report.get("status") != "succeeded"
        or report.get("measured_by") != "harness"
        or report.get("level") != "full"
        or report.get("protocol_id") != task.protocol.fingerprint
    ):
        return None
    try:
        raw = report["full_score"]
        if isinstance(raw["value"], bool) or not isinstance(raw["value"], int | float):
            return None
        score = Score(raw["value"], Metric(raw["metric"]), raw["protocol_id"], raw["level"])
        if (score.metric, score.protocol_id, score.level) != (
            task.spec.metric,
            task.protocol.fingerprint,
            "full",
        ):
            return None
        return score
    except (KeyError, TypeError, ValueError):
        return None


@dataclass(frozen=True)
class ResearchDirection:
    parent_id: str
    best_id: str
    alternatives: tuple[str, ...]
    target_errors: tuple[dict[str, Any], ...]
    reason: str
    recent_failures: tuple[str, ...]


def choose_direction(
    task: PreparedTask,
    baseline_score: Score,
    reports: list[dict[str, Any]],
    *,
    baseline_report: dict[str, Any] | None = None,
    session_index: int = 0,
) -> ResearchDirection:
    baseline_score.better_than(
        Score(baseline_score.value, task.spec.metric, task.protocol.fingerprint, "full")
    )
    candidates = {
        report["id"]: report for report in reports if measured_score(task, report) is not None
    }
    if baseline_report is not None:
        candidates["baseline"] = baseline_report
    scores = {
        identifier: score
        for identifier, report in candidates.items()
        if (score := measured_score(task, report)) is not None
    }
    scores["baseline"] = baseline_score
    best_id = "baseline"
    best = baseline_score
    for identifier, score in scores.items():
        if score.better_than(best):
            best_id, best = identifier, score
    # Keep candidates that lead on at least one supported class/segment, even when overall worse.
    per_focus: dict[str, tuple[float, str]] = {}
    for identifier, report in candidates.items():
        diagnostic = report.get("error_report", {})
        if (
            diagnostic.get("measured_by") != "harness"
            or diagnostic.get("protocol_id") != task.protocol.fingerprint
            or diagnostic.get("level") != "full"
        ):
            continue
        for focus in diagnostic_focus(diagnostic, limit=1000):
            loss = focus["loss"]
            if not isinstance(loss, int | float) or not isfinite(loss):
                continue
            key = focus["key"]
            if key not in per_focus or loss < per_focus[key][0]:
                per_focus[key] = (loss, identifier)
    alternatives = tuple(
        sorted({identifier for _, identifier in per_focus.values() if identifier != best_id})
    )
    # Every third hypothesis explores a preserved alternative instead of greedily extending best.
    parent_id = best_id
    reason = "Continue the best comparable full-validation candidate"
    if alternatives and session_index % 3 == 2:
        parent_id = alternatives[(session_index // 3) % len(alternatives)]
        reason = "Explore an alternative with a measured advantage on a supported error segment"
    diagnostic = candidates.get(parent_id, {}).get("error_report", {})
    failures = tuple(
        str(report["id"])
        for report in reports[-10:]
        if report.get("status") in ("failed", "timed_out")
    )[-3:]
    return ResearchDirection(
        parent_id, best_id, alternatives, tuple(diagnostic_focus(diagnostic)), reason, failures
    )


def validate_parent(task: PreparedTask, report: dict[str, Any]) -> None:
    if measured_score(task, report) is None:
        raise ContractError(
            "Parent must be a successful harness-measured full result on this protocol"
        )
