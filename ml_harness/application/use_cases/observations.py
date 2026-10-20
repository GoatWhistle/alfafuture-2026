"""Small, lossless measured observations shared by tools and context compaction."""

import json
from typing import Any

from ml_harness.application.use_cases.context import diagnostic_summary


def encoded(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def experiment_observation(report: dict[str, Any]) -> dict[str, Any]:
    if not ("id" in report or "status" in report):
        return report
    result = {
        key: report[key]
        for key in (
            "id",
            "parent_id",
            "status",
            "full_score",
            "smoke_score",
            "protocol_id",
            "level",
            "failure_phase",
            "measured_by",
            "code_hash",
            "parent_code_hash",
            "parent_backend",
            "hypothesis_source",
            "elapsed_seconds",
            "parent_score",
            "delta",
            "reused_from",
            "diagnostic_status",
        )
        if key in report
    }
    for key in ("hypothesis", "expected_effect", "error", "diagnostic_error"):
        if report.get(key):
            result[key] = str(report[key])[:400]
    if report.get("error_report"):
        result["measured_errors"] = diagnostic_summary(report["error_report"], 3)
    if report.get("artifacts"):
        result["artifacts"] = list(report["artifacts"])[:3]
    # Diagnostics are optional; measured status/scores/identity are never sliced.
    if len(encoded(result)) > 5500:
        result.pop("measured_errors", None)
    if len(encoded(result)) > 5500:
        result["artifacts"] = []
    return result


def compact_observation(value: dict[str, Any]) -> dict[str, Any]:
    if len(encoded(value)) <= 6000:
        return value
    result = value.get("result")
    if isinstance(result, dict) and ("full_score" in result or "status" in result):
        return {"ok": value.get("ok", False), "result": experiment_observation(result)}
    if isinstance(result, dict) and "text" in result:
        return value
    if isinstance(result, dict):
        return {
            "ok": value.get("ok", False),
            "result": {
                key: (item[:1200] if isinstance(item, str) else item)
                for key, item in result.items()
                if key in ("text", "truncated", "next_offset", "task", "protocol")
            },
        }
    return {
        "ok": value.get("ok", False),
        "error": {
            "type": "large_observation",
            "message": "Read the paginated artifact for details",
        },
    }
