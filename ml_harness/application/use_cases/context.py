"""Build a bounded research context from comparable measured evidence."""

import json
from dataclasses import asdict
from math import isfinite
from typing import Any

from ml_harness.application.use_cases.directions import ResearchDirection, diagnostic_focus
from ml_harness.domain.models import ContractError, Metric, PreparedTask, Score


def _text(value: Any, limit: int) -> str:
    text = value if isinstance(value, str) else ""
    return text if len(text) <= limit else text[: max(0, limit - 1)] + "…"


def _encode(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _measured(report: dict[str, Any], task: PreparedTask) -> Score | None:
    raw = report.get("full_score")
    if report.get("measured_by") != "harness" or not isinstance(raw, dict):
        return None
    try:
        value = raw["value"]
        if isinstance(value, bool) or not isinstance(value, int | float):
            return None
        score = Score(value, Metric(raw["metric"]), raw["protocol_id"], raw["level"])
        if (score.metric, score.protocol_id, score.level) != (
            task.spec.metric,
            task.protocol.fingerprint,
            "full",
        ):
            return None
        return score
    except (KeyError, TypeError, ValueError, OverflowError):
        return None


def diagnostic_summary(diagnostic: dict[str, Any], limit: int = 3) -> dict[str, Any]:
    items = []
    for focus in diagnostic_focus(diagnostic, limit):
        items.append(
            {
                "error": _text(focus["key"], 240),
                "support": focus["support"],
                "loss": focus["loss"],
                "metrics": focus.get(
                    "metrics", {"f1": focus.get("f1"), "recall": focus.get("recall")}
                ),
                "delta": focus.get("delta"),
            }
        )
    rows = diagnostic.get("class_report", [])
    scope = {"selected": len(items), "exhaustive": False}
    if rows:
        scope["classes_total"] = len(rows)
        scope["exhaustive"] = len(items) == len(rows) and all(
            item["error"].startswith("class:") for item in items
        )
    result = {
        "measured_by": diagnostic.get("measured_by"),
        "overall_delta": diagnostic.get("delta"),
        "target_errors": items,
        "focus_scope": scope,
        "interpretation": "Observations on validation; no causal claim",
    }
    if any(isinstance(row.get("delta"), dict) for row in rows):
        changes = {}
        for metric in ("precision", "recall", "f1"):
            counts = dict(increased=0, decreased=0, unchanged=0, unavailable=0)
            for row in rows:
                value = (row.get("delta") or {}).get(metric)
                if (
                    isinstance(value, bool)
                    or not isinstance(value, int | float)
                    or not isfinite(value)
                ):
                    counts["unavailable"] += 1
                else:
                    counts[
                        "increased" if value > 0 else "decreased" if value < 0 else "unchanged"
                    ] += 1
            changes[metric] = counts
        result["class_metric_changes"] = changes
    return result


def build_context(
    task: PreparedTask,
    baseline_score: Score,
    reports: list[dict[str, Any]],
    *,
    parent_id: str | None = None,
    max_chars: int = 12000,
    budget: dict[str, Any] | None = None,
    direction: ResearchDirection | None = None,
    baseline_diagnostics: dict[str, Any] | None = None,
) -> str:
    """Prioritize parent, best full result, recent failures and alternative successes."""
    if isinstance(max_chars, bool) or not isinstance(max_chars, int) or max_chars < 512:
        raise ContractError("Context limit must be at least 512 characters")
    spec = task.spec
    feature_limit, class_limit, text_limit = 12, 8, 80

    def base() -> dict[str, Any]:
        return {
            "task": {
                "target": _text(spec.target, text_limit),
                "kind": spec.kind.value,
                "metric": spec.metric.value,
                "features": [_text(name, text_limit) for name in spec.features[:feature_limit]],
                "features_omitted": max(0, len(spec.features) - feature_limit),
                "classes": [
                    (_text(c, text_limit) if isinstance(c, str) else c)
                    for c in spec.classes[:class_limit]
                ],
                "classes_omitted": max(0, len(spec.classes) - class_limit),
            },
            "baseline_measured": {
                "value": baseline_score.value,
                "metric": baseline_score.metric.value,
                "protocol_id": _text(baseline_score.protocol_id, text_limit),
                "level": baseline_score.level,
                "measured_by": "harness",
            },
            "reports": [],
        }

    context = base()
    while len(_encode(context)) > max_chars:
        if feature_limit or class_limit:
            feature_limit //= 2
            class_limit //= 2
        elif text_limit > 8:
            text_limit //= 2
        else:
            raise ContractError("Task summary cannot fit the context limit")
        context = base()
    additions: dict[str, Any] = {}
    if baseline_diagnostics:
        additions["baseline_errors"] = diagnostic_summary(baseline_diagnostics)
    if direction is not None:
        additions["direction"] = {
            "parent_id": _text(direction.parent_id, 64),
            "best_id": _text(direction.best_id, 64),
            "alternatives": [_text(value, 64) for value in direction.alternatives[:5]],
            "reason": _text(direction.reason, 240),
            "recent_failures": [_text(value, 64) for value in direction.recent_failures],
        }
    for key, value in additions.items():
        context[key] = value
        if len(_encode(context)) > max_chars:
            del context[key]
    if budget:
        safe_budget = {
            _text(key, 40): (_text(value, 80) if isinstance(value, str) else value)
            for key, value in list(budget.items())[:10]
            if isinstance(value, str | int | float | bool)
            and (not isinstance(value, float) or isfinite(value))
        }
        context["budget"] = safe_budget
        if len(_encode(context)) > max_chars:
            del context["budget"]
    eligible = [
        report
        for report in reports
        if (
            report.get("protocol_id") == task.protocol.fingerprint and report.get("level") == "full"
        )
    ]
    successful = [
        (report, score)
        for report in eligible
        if report.get("status") == "succeeded" and (score := _measured(report, task)) is not None
    ]
    selected: list[tuple[str, dict[str, Any]]] = []
    parent = next((report for report in reversed(eligible) if report.get("id") == parent_id), None)
    if parent_id is not None and parent is not None:
        selected.append(("parent", parent))
    if successful:
        best, _ = min(
            successful, key=lambda item: item[1].value if spec.metric.minimize else -item[1].value
        )
        selected.append(("best", best))
    selected.extend(
        ("failure", report)
        for report in reversed(eligible)
        if report.get("status") in ("failed", "timed_out")
    )
    selected = selected[:4]
    selected.extend(("alternative", report) for report, _ in reversed(successful[-3:]))
    seen: set[int] = set()
    for role, report in selected:
        if id(report) in seen:
            continue
        seen.add(id(report))
        score = _measured(report, task)
        summary: dict[str, Any] = {
            "role": role,
            "id": _text(report.get("id"), 64),
            "status": _text(report.get("status"), 24),
            "hypothesis": _text(report.get("hypothesis"), 400),
            "expected_effect": _text(report.get("expected_effect"), 240),
            "measured": asdict(score) if score is not None else None,
            "error": _text(report.get("error"), 300),
            "interpretation": _text(report.get("llm_commentary"), 400),
            "measured_errors": diagnostic_summary(report.get("error_report", {})),
        }
        context["reports"].append(summary)
        if len(_encode(context)) > max_chars:
            for key in ("hypothesis", "expected_effect", "error", "interpretation"):
                summary[key] = _text(summary[key], 80)
            summary["measured_errors"] = diagnostic_summary(report.get("error_report", {}), 1)
            if len(_encode(context)) > max_chars:
                context["reports"].pop()
    return _encode(context)
