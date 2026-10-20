"""Read-only view of harness artifacts; never repairs or rewrites journals."""

import json
import math
import re
import time
from pathlib import Path
from typing import Any

_LOWER_IS_BETTER = re.compile(r"rmse|mae|mse|logloss|log_loss|rmsle|mape|error", re.IGNORECASE)
_CODE_LIMIT = 20_000

TASK_KINDS = {
    "binary": "Бинарная классификация",
    "multiclass": "Мультикласс",
    "regression": "Регрессия",
}


def direction_for(metric: str) -> str:
    return "lower" if _LOWER_IS_BETTER.search(metric) else "higher"


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def read_events(path: Path) -> list[dict[str, Any]]:
    """Parse complete lines only. A torn final line is skipped, not truncated."""
    try:
        data = path.read_bytes()
    except OSError:
        return []
    events: dict[str, dict[str, Any]] = {}
    for line in data.splitlines(keepends=True):
        if not line.endswith(b"\n"):
            break
        try:
            event = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            continue
        if isinstance(event, dict) and isinstance(event.get("payload"), dict):
            events[str(event.get("event_id"))] = event
    return list(events.values())


def _score(report: dict[str, Any], key: str = "full_score") -> float | None:
    value = report.get(key)
    if isinstance(value, dict):
        value = value.get("value")
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        return None
    return float(value)


def _fmt(value: float, metric: str) -> str:
    return f"{value:.2f}" if re.search(r"rmse|mae|mse", metric, re.IGNORECASE) else f"{value:.4f}"


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("hypothesis", "rationale", "summary", "description"):
            if isinstance(value.get(key), str):
                return value[key]
    return "—"


def _params(value: Any) -> dict[str, str | int | float | bool]:
    if not isinstance(value, dict):
        return {}
    return {
        str(k): v if isinstance(v, str | int | float | bool) else json.dumps(v, ensure_ascii=False)
        for k, v in value.items()
    }


def _read_code(path: Any, base: Path) -> str | None:
    """Only files inside the run directory are exposed."""
    if not isinstance(path, str):
        return None
    try:
        resolved = Path(path).resolve()
        resolved.relative_to(base.resolve())
        return resolved.read_text(encoding="utf-8")[:_CODE_LIMIT]
    except (OSError, ValueError):
        return None


def _changes(recipe: dict[str, Any]) -> list[str]:
    out = []
    for t in recipe.get("transforms") or []:
        if isinstance(t, dict):
            columns = ", ".join(str(c) for c in t.get("columns") or [])
            out.append(f"{t.get('operation', '?')}: {columns}")
    if recipe.get("python_transform"):
        out.append("python_transform: build_transformer()")
    return out


def _status(report: dict[str, Any]) -> str:
    state = str(report.get("status", ""))
    if state == "running":
        return "running"
    if state == "succeeded":
        return "ok"
    error = str(report.get("error") or "").lower()
    if "timeout" in error or "deadline" in error:
        return "timeout"
    if "reject" in error or "admission" in error or "отклон" in error:
        return "rejected"
    return "failed"


def experiments(
    events: list[dict[str, Any]], metric: str, baseline: float | None, base: Path
) -> list[dict[str, Any]]:
    """One row per experiment; the finished event replaces the started one."""
    reports: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for event in events:
        if event.get("kind") not in ("experiment_started", "experiment_finished"):
            continue
        payload = event["payload"]
        identifier = str(payload.get("id"))
        if identifier not in reports:
            order.append(identifier)
        if event["kind"] == "experiment_finished" or identifier not in reports:
            reports[identifier] = payload
    scores = {i: _score(r) for i, r in reports.items()}
    rows = []
    for identifier in order:
        r = reports[identifier]
        parent = r.get("parent_id")
        parent_score = baseline if parent in (None, "baseline") else scores.get(str(parent))
        score = scores[identifier]
        delta = None if score is None or parent_score is None else score - parent_score
        status = _status(r)
        proposal = (
            r.get("requested_proposal") if isinstance(r.get("requested_proposal"), dict) else {}
        )
        recipe = r.get("executed_recipe") if isinstance(r.get("executed_recipe"), dict) else {}
        admission = r.get("admission") if isinstance(r.get("admission"), dict) else {}
        refit = r.get("refit_parameters") if isinstance(r.get("refit_parameters"), dict) else {}
        # Optuna: найденные параметры сохраняются для refit, служебные ключи не показываем
        refit = {k: v for k, v in refit.items() if k not in ("backend", "tuning")}
        executed = _params(recipe.get("parameters") or refit or r.get("parameters"))
        requested = _params(
            proposal.get("parameters") or r.get("requested_parameters") or r.get("parameters")
        )
        if proposal.get("tune"):
            requested["tune"] = True
        if recipe.get("tuning"):
            executed["tune"] = True
        changes = _changes(recipe or proposal) or [str(c) for c in r.get("feature_changes") or []]
        if admission.get("allowed") is False:
            status = "rejected"
        if status == "ok" and score is not None:
            measured = f"{metric} {_fmt(score, metric)}"
            if delta is not None:
                sign = "+" if delta >= 0 else "−"
                measured += f": {sign}{_fmt(abs(delta), metric)} к родителю"
        elif status == "running":
            measured = "Выполняется…"
        elif status == "rejected" and "estimated_full_seconds" in admission:
            measured = (
                f"Отклонено по ресурсам: прогноз {admission['estimated_full_seconds']:.0f} с, "
                f"доступно {admission.get('available_seconds', 0):.0f} с. Параметры не уменьшались."
            )
        else:
            measured = str(r.get("error") or "Результата нет.")
        executed_text = (
            ", ".join(f"{k} {v}" for k, v in executed.items()) or "Параметры по умолчанию"
        )
        if status == "rejected":
            executed_text = "Не запускалось: отклонено до обучения."
        code = recipe.get("python_transform") or _read_code(r.get("code_path"), base)
        rows.append(
            {
                "id": identifier,
                "parentId": None if parent is None else str(parent),
                "level": "full" if r.get("level", "full") == "full" else "smoke",
                "status": status,
                "observed": _text(
                    proposal.get("target_error") or r.get("target_error") or r.get("observed")
                ),
                "proposed": _text(proposal.get("rationale") or r.get("hypothesis")),
                "executed": executed_text,
                "measured": measured,
                "proposedParams": requested,
                "executedParams": executed,
                "featureChanges": changes,
                "score": score,
                "delta": delta,
                "durationSec": r.get("elapsed_seconds")
                if isinstance(r.get("elapsed_seconds"), int | float)
                else None,
                "error": r.get("error") if isinstance(r.get("error"), str) else None,
                "code": code if isinstance(code, str) and code else None,
                "backend": str(r.get("backend", "")),
                "kind": "experiment",
            }
        )
    rows.extend(_failed_sessions(events))
    return rows


_SESSION_OK = {"completed", "experiment_completed", "success", "succeeded", "budget_exhausted"}


def _failed_sessions(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """LLM sessions that failed before any experiment: the error belongs to that attempt."""
    rows = []
    for event in events:
        if event.get("kind") != "session_finished":
            continue
        session = event["payload"]
        reason = str(session.get("reason") or "")
        if reason in _SESSION_OK or session.get("experiment_ids"):
            continue
        detail = session.get("error_detail")
        rows.append(
            {
                "id": "qwen-" + str(session.get("id", ""))[:8],
                "parentId": None,
                "level": "full",
                "status": "failed",
                "observed": "—",
                "proposed": "Предложение не получено.",
                "executed": "Эксперимент не запускался.",
                "measured": f"Сессия Qwen завершилась: {reason or 'ошибка'}.",
                "proposedParams": {},
                "executedParams": {},
                "featureChanges": [],
                "score": None,
                "delta": None,
                "durationSec": None,
                "error": str(detail) if detail else reason or None,
                "code": None,
                "backend": "qwen",
                "kind": "llm_request",
            }
        )
    return rows


def _selected(research: dict[str, Any], metric: str) -> dict[str, Any] | None:
    """Score of the finalist written to the CSV, only if measured as full on this metric."""
    final = research.get("finalization")
    if not isinstance(final, dict):
        return None
    chosen = final.get("selected") or research.get("selected")
    for item in final.get("finalists") or []:
        if not isinstance(item, dict) or item.get("id") != chosen:
            continue
        score = item.get("score")
        if (
            isinstance(score, dict)
            and score.get("level") == "full"
            and score.get("metric") == metric
        ):
            value = _score(item, "score")
            if value is not None:
                return {"id": str(chosen), "score": value}
    return None


def stage_of(harness: Path) -> str:
    if (harness / "summary.json").exists():
        return "result"
    research = harness / "research"
    if (research / "finalists.json").exists() or (research / "finalization-state.json").exists():
        return "refit"
    if (research / "memory" / "events.jsonl").exists() or (harness / "baseline.json").exists():
        return "research" if (harness / "baseline.json").exists() else "baseline"
    if (harness / "prepared.json").exists():
        return "baseline"
    return "prepare"


def run_facts(run_dir: Path) -> dict[str, Any]:
    """Facts derived from artifacts only: no process state, no side effects."""
    harness = run_dir / "harness"
    prepared = read_json(harness / "prepared.json") or {}
    spec = prepared.get("spec", {}) if isinstance(prepared, dict) else {}
    baseline_json = read_json(harness / "baseline.json") or {}
    summary = read_json(harness / "summary.json") or {}
    budget = read_json(harness / "budget.json") or {}
    metric = str((baseline_json.get("score") or {}).get("metric") or spec.get("metric") or "metric")
    baseline = _score(baseline_json, "score")
    events = read_events(harness / "research" / "memory" / "events.jsonl")
    rows = experiments(events, metric, baseline, run_dir)
    direction = direction_for(metric)
    best = baseline
    for row in rows:
        if row["level"] == "full" and row["status"] == "ok" and row["score"] is not None:
            if best is None or (
                row["score"] > best if direction == "higher" else row["score"] < best
            ):
                best = row["score"]
    research = summary.get("research") if isinstance(summary, dict) else None
    research = research if isinstance(research, dict) else {}
    selected = _selected(research, metric)
    if selected is not None:
        # В итоговый CSV входит выбранный финалист; показываем его измеренную метрику
        best = selected["score"]
    tokens = research.get("tokens_used")
    if not isinstance(tokens, int):
        tokens = int(budget.get("preparation_tokens", 0)) if isinstance(budget, dict) else 0
    uncertain = research.get("usage_known") is False or (
        isinstance(budget, dict) and budget.get("preparation_usage_known") is False
    )
    submission = spec.get("submission_path")
    return {
        "metric": metric,
        "direction": direction,
        "taskKind": TASK_KINDS.get(str(spec.get("kind")), "Табличная задача"),
        "baseline": baseline,
        "best": best,
        "experiments": rows,
        "stage": stage_of(harness),
        "tokens": tokens,
        "tokensUncertain": bool(uncertain),
        "submission": Path(submission) if isinstance(submission, str) else None,
        "researchStatus": research.get("status"),
        "selected": selected,
        "hasSummary": bool(summary),
    }


def now() -> float:
    return time.time()
