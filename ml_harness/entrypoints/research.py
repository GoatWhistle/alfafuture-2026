"""Wire bounded research, then irreversibly leave search for frozen finalization."""

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any
from uuid import uuid4

from ml_harness.adapters.execution.budget import BudgetedRunner, BudgetedSubmissionWriter
from ml_harness.adapters.execution.clock import SystemClock
from ml_harness.adapters.execution.runner import SubprocessRunner
from ml_harness.adapters.llm.proposals import ProposalTools
from ml_harness.adapters.llm.qwen import QwenLanguageModel
from ml_harness.adapters.research.finalization import FinalizationWorkspace
from ml_harness.adapters.research.workspace import ResearchWorkspace, write_json
from ml_harness.adapters.storage.submission import AtomicSubmissionWriter, IndependentEvaluator
from ml_harness.application.ports.execution import ExperimentRunner
from ml_harness.application.ports.language_model import ModelError
from ml_harness.application.use_cases.baseline import BaselineResult
from ml_harness.application.use_cases.context import build_context
from ml_harness.application.use_cases.directions import choose_direction
from ml_harness.application.use_cases.proposal_session import run_proposal_session
from ml_harness.domain.budget import WallClockBudget
from ml_harness.domain.models import PreparedTask, RunBudget


def _workspace(
    task: PreparedTask,
    baseline: BaselineResult,
    directory: Path,
    threads: int,
    timeout_seconds: float,
    runtime_budget: WallClockBudget | None,
    base_runner: ExperimentRunner | None,
) -> tuple[ResearchWorkspace, WallClockBudget, BudgetedRunner]:
    clock = SystemClock()
    wall = runtime_budget or WallClockBudget(clock.now() + 1800)
    wall.observe("validation", baseline.validation_seconds)
    wall.observe("refit", baseline.refit_seconds)
    base = base_runner or SubprocessRunner()
    search_runner = BudgetedRunner(base, clock, wall)
    workspace = ResearchWorkspace(
        task,
        directory,
        search_runner,
        IndependentEvaluator(),
        threads=threads,
        timeout_seconds=timeout_seconds,
    )
    validation = (
        Path(baseline.validation_dir)
        if baseline.validation_dir
        else directory.parent / "validation"
    )
    baseline_output = validation / "result.json"
    request_file = validation / "request.json"
    request = json.loads(request_file.read_text()) if request_file.exists() else {}
    workspace.register_baseline(
        baseline.score,
        baseline_output if baseline_output.exists() else None,
        backend=request.get("backend", "catboost"),
        parameters=request.get("parameters", {}),
    )
    for item in baseline.candidates:
        if item["backend"] == request.get("backend", baseline.backend):
            continue
        workspace.register_starting_point(item)
    return workspace, wall, BudgetedRunner(base, clock, wall, finalizing=True)


def _finish(
    task: PreparedTask,
    workspace: ResearchWorkspace,
    runner: BudgetedRunner,
    threads: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    return FinalizationWorkspace(
        task,
        workspace,
        runner,
        IndependentEvaluator(),
        BudgetedSubmissionWriter(AtomicSubmissionWriter(), runner.clock, runner.budget),
        threads=threads,
        timeout_seconds=timeout_seconds,
    ).finish()


def finalize_baseline(
    task: PreparedTask,
    baseline: BaselineResult,
    directory: Path,
    *,
    threads: int,
    timeout_seconds: float,
    runtime_budget: WallClockBudget | None = None,
    base_runner: ExperimentRunner | None = None,
) -> dict[str, Any]:
    workspace, _, runner = _workspace(
        task, baseline, directory, threads, timeout_seconds, runtime_budget, base_runner
    )
    return _finish(task, workspace, runner, threads, timeout_seconds)


def research_after_baseline(
    task: PreparedTask,
    baseline: BaselineResult,
    directory: Path,
    *,
    threads: int,
    timeout_seconds: float,
    max_steps: int,
    token_limit: int,
    max_sessions: int = 3,
    search_mode: str = "qwen",
    initial_usage: dict[str, Any] | None = None,
    runtime_budget: WallClockBudget | None = None,
    base_runner: ExperimentRunner | None = None,
) -> dict[str, Any]:
    version_file = directory / "pipeline-version.json"
    if version_file.exists():
        if json.loads(version_file.read_text()) != {"version": 2}:
            raise ValueError("Incompatible research pipeline; use a new run directory")
    else:
        journal = directory / "memory/events.jsonl"
        if journal.exists() and any(
            json.loads(line).get("kind") in ("session_started", "session_finished")
            for line in journal.read_text().splitlines()
            if line.strip()
        ):
            raise ValueError("Legacy research sessions cannot resume under pipeline 2")
    workspace, wall, final_runner = _workspace(
        task, baseline, directory, threads, timeout_seconds, runtime_budget, base_runner
    )
    write_json(version_file, {"version": 2})
    max_sessions = min(3, max_sessions)
    clock = SystemClock()
    initial_usage = initial_usage or {}
    budget = RunBudget(
        wall.deadline,
        token_limit,
        2 + max_steps * 3 * max_sessions,
        initial_usage.get("tokens", 0),
        initial_usage.get("attempts", 0),
    )
    completed = 0
    usage_known = initial_usage.get("known", True) is True
    budget_accounted = usage_known
    reported_tokens = initial_usage.get("tokens", 0)
    started_sessions: set[str] = set()
    finished_sessions: set[str] = set()
    for event in workspace.store.read_events():
        if event["kind"] == "session_started":
            started_sessions.add(event["payload"]["id"])
        if event["kind"] == "session_finished":
            finished_sessions.add(event["payload"]["id"])
            usage = event["payload"]["usage"]
            budget.tokens_used += usage.get(
                "tokens_charged", usage["input_tokens"] + usage["output_tokens"]
            )
            reported_tokens += usage["input_tokens"] + usage["output_tokens"]
            budget.attempts_used += usage.get("api_attempts", 0)
            usage_known = usage_known and usage["usage_known"]
            budget_accounted = budget_accounted and usage.get(
                "budget_accounted", usage["usage_known"]
            )
            completed += 1
    # A killed request may have consumed tokens without returning usage. Never reset that cost.
    usage_known = usage_known and not (started_sessions - finished_sessions)
    budget_accounted = budget_accounted and not (started_sessions - finished_sessions)
    sessions = []
    status = "budget_exhausted"
    names = ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL")
    model = None
    if (directory / "finalists.json").exists():
        status = "finalization_resumed"
    elif search_mode == "optuna":
        from ml_harness.application.use_cases.numeric_search import run_numeric_search

        status = run_numeric_search(workspace, wall, clock)
    elif not budget_accounted:
        status = "unknown_usage"
    elif not all(os.environ.get(name) for name in names):
        status = "not_configured"
    else:
        try:
            model = QwenLanguageModel(
                os.environ[names[0]],
                os.environ[names[1]],
                os.environ[names[2]],
            )
            if hasattr(model, "trace_dir"):
                model.trace_dir = directory / "api-trace"
        except ModelError:
            status = "configuration_error"
    if model is not None:
        for _ in range(max(0, max_sessions - completed)):
            if not budget.can_search(clock.now(), wall.reserve_seconds()):
                break
            identifier = uuid4().hex
            tools = ProposalTools(workspace, wall, clock, identifier, baseline.validation_seconds)
            session_index = sum(
                event["kind"] == "session_finished" for event in workspace.store.read_events()
            )
            direction = choose_direction(
                task,
                baseline.score,
                workspace.reports,
                baseline_report=workspace.baseline_report,
                session_index=session_index,
            )
            workspace.prepare_hypothesis(direction, identifier)
            context = build_context(
                task,
                baseline.score,
                workspace.reports,
                parent_id=direction.parent_id,
                direction=direction,
                baseline_diagnostics=(workspace.baseline_report or {}).get("error_report"),
                budget={
                    "tokens_remaining": max(0, token_limit - budget.tokens_used),
                    "seconds_remaining": wall.available(clock.now()),
                    "reserved_seconds": wall.reserve_seconds(),
                    "steps_per_hypothesis": max_steps,
                    "worker_timeout_seconds": timeout_seconds,
                    "smoke_timeout_seconds": min(60, timeout_seconds),
                    "threads": threads,
                    "baseline_validation_seconds": baseline.validation_seconds,
                    "baseline_refit_seconds": baseline.refit_seconds,
                },
            )
            context_value = json.loads(context)
            parent = next(r for r in workspace.read_experiments() if r["id"] == direction.parent_id)
            context_value["parent_recipe"] = parent.get("executed_recipe") or parent.get(
                "resolved_spec",
                {
                    "backend": parent.get("backend"),
                    "parameters": parent.get("parameters", {}),
                },
            )
            context_value["parent_id"] = direction.parent_id
            context_value["task_kind"] = task.spec.kind.value
            # Preserve feature names: semantic proposals must address actual columns.
            context_value["features"] = list(task.spec.features)
            context_value["task_overview"] = (directory / "task.md").read_text()[:1200]
            context_value["last_measurements"] = [
                {
                    k: r[k]
                    for k in (
                        "id",
                        "status",
                        "backend",
                        "full_score",
                        "parent_delta",
                        "admission",
                        "resolved_spec",
                        "executed_recipe",
                        "error",
                        "error_report",
                    )
                    if k in r
                }
                for r in workspace.read_experiments()[-2:]
            ]
            # Diagnostics are summarized in build_context; avoid duplicating large row reports.
            for measurement in context_value["last_measurements"]:
                measurement.pop("error_report", None)
            context_value["recent_tool_failures"] = [
                failure
                for event in workspace.store.read_events()
                if event["kind"] == "session_finished"
                for failure in event["payload"].get("tool_failures", [])
            ][-1:]
            context = json.dumps(context_value, ensure_ascii=False)
            workspace.store.append(
                {
                    "event_id": "session-start:" + identifier,
                    "kind": "session_started",
                    "payload": {"id": identifier},
                }
            )
            session = run_proposal_session(
                model,
                tools,
                context,
                budget,
                clock,
                max_steps=max_steps,
                reserve_seconds=wall.reserve_seconds,
            )
            workspace.register_session(identifier, asdict(session))
            write_json(directory / "sessions" / (identifier + ".json"), asdict(session))
            sessions.append(asdict(session))
            status = session.reason
            reported_tokens += session.input_tokens + session.output_tokens
            usage_known = usage_known and session.usage_known
            budget_accounted = budget_accounted and session.budget_accounted
            if not budget_accounted or status in (
                "model_error",
                "budget_exhausted",
                "unknown_usage",
                "protocol_error",
            ):
                break
    # All LLM calls have ended. A frozen plan on disk prohibits future search after restart as well.
    final = _finish(task, workspace, final_runner, threads, timeout_seconds)
    report = {
        "pipeline_version": 2,
        "status": status,
        "sessions": sessions,
        "reports": workspace.read_experiments(),
        "tokens_used": budget.tokens_used,
        "reported_tokens": reported_tokens,
        "usage_known": usage_known,
        "budget_accounted": budget_accounted,
        "finalization": final,
        "selected": final.get("selected", "baseline"),
        "refit_status": final.get("refit_status", "not_needed"),
    }
    write_json(directory / "session.json", report)
    return report
