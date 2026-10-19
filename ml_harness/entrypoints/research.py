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
from ml_harness.adapters.llm.qwen import QwenLanguageModel
from ml_harness.adapters.llm.tools import ResearchTools
from ml_harness.adapters.research.finalization import FinalizationWorkspace
from ml_harness.adapters.research.workspace import ResearchWorkspace, write_json
from ml_harness.adapters.storage.submission import AtomicSubmissionWriter, IndependentEvaluator
from ml_harness.application.ports.execution import ExperimentRunner
from ml_harness.application.ports.language_model import ModelError
from ml_harness.application.use_cases.baseline import BaselineResult
from ml_harness.application.use_cases.context import build_context
from ml_harness.application.use_cases.directions import choose_direction
from ml_harness.application.use_cases.research import run_session
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
    baseline_output = directory.parent / "validation/result.json"
    request_file = directory.parent / "validation/request.json"
    request = json.loads(request_file.read_text()) if request_file.exists() else {}
    workspace.register_baseline(
        baseline.score,
        baseline_output if baseline_output.exists() else None,
        backend=request.get("backend", "catboost"),
        parameters=request.get("parameters", {}),
    )
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
    initial_usage: dict[str, Any] | None = None,
    runtime_budget: WallClockBudget | None = None,
    base_runner: ExperimentRunner | None = None,
) -> dict[str, Any]:
    workspace, wall, final_runner = _workspace(
        task, baseline, directory, threads, timeout_seconds, runtime_budget, base_runner
    )
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
    started_sessions: set[str] = set()
    finished_sessions: set[str] = set()
    for event in workspace.store.read_events():
        if event["kind"] == "session_started":
            started_sessions.add(event["payload"]["id"])
        if event["kind"] == "session_finished":
            finished_sessions.add(event["payload"]["id"])
            usage = event["payload"]["usage"]
            budget.tokens_used += usage["input_tokens"] + usage["output_tokens"]
            budget.attempts_used += usage.get("api_attempts", 0)
            usage_known = usage_known and usage["usage_known"]
            completed += 1
    # A killed request may have consumed tokens without returning usage. Never reset that cost.
    usage_known = usage_known and not (started_sessions - finished_sessions)
    sessions = []
    status = "budget_exhausted"
    names = ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL")
    model = None
    if (directory / "finalists.json").exists():
        status = "finalization_resumed"
    elif not usage_known:
        status = "unknown_usage"
    elif not all(os.environ.get(name) for name in names):
        status = "not_configured"
    else:
        try:
            model = QwenLanguageModel(
                os.environ[names[0]], os.environ[names[1]], os.environ[names[2]]
            )
        except ModelError:
            status = "configuration_error"
    if model is not None:
        for _ in range(max(0, max_sessions - completed)):
            if not budget.can_search(clock.now(), wall.reserve_seconds()):
                break
            tools = ResearchTools(
                directory,
                workspace.inspect_data,
                workspace.run_experiment,
                workspace.read_experiments,
            )
            identifier = uuid4().hex
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
                },
            )
            workspace.store.append(
                {
                    "event_id": "session-start:" + identifier,
                    "kind": "session_started",
                    "payload": {"id": identifier},
                }
            )
            session = run_session(
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
            if status in ("model_error", "budget_exhausted", "unknown_usage"):
                break
    # All LLM calls have ended. A frozen plan on disk prohibits future search after restart as well.
    final = _finish(task, workspace, final_runner, threads, timeout_seconds)
    report = {
        "status": status,
        "sessions": sessions,
        "reports": workspace.read_experiments(),
        "tokens_used": budget.tokens_used,
        "finalization": final,
        "selected": final.get("selected", "baseline"),
        "refit_status": final.get("refit_status", "not_needed"),
    }
    write_json(directory / "session.json", report)
    return report
