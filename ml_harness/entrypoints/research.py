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
from ml_harness.adapters.research.finalization import FinalizationWorkspace
from ml_harness.adapters.research.session import ProposalSessionWorkspace
from ml_harness.adapters.research.workspace import ResearchWorkspace, write_json
from ml_harness.adapters.storage.submission import AtomicSubmissionWriter, IndependentEvaluator
from ml_harness.application.ports.execution import ExperimentRunner
from ml_harness.application.ports.language_model import ModelError
from ml_harness.application.use_cases.baseline import BaselineResult
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
    max_sessions: int = 12,
    search_mode: str = "qwen",
    initial_usage: dict[str, Any] | None = None,
    runtime_budget: WallClockBudget | None = None,
    base_runner: ExperimentRunner | None = None,
) -> dict[str, Any]:
    version_file = directory / "pipeline-version.json"
    if version_file.exists():
        if json.loads(version_file.read_text()) != {"version": 3}:
            raise ValueError("Incompatible research pipeline; use a new run directory")
    else:
        journal = directory / "memory/events.jsonl"
        if journal.exists() and any(
            json.loads(line).get("kind") in ("session_started", "session_finished")
            for line in journal.read_text().splitlines()
            if line.strip()
        ):
            raise ValueError("Legacy research sessions cannot resume under pipeline 3")
    workspace, wall, final_runner = _workspace(
        task, baseline, directory, threads, timeout_seconds, runtime_budget, base_runner
    )
    write_json(version_file, {"version": 3})
    max_sessions = min(12, max_sessions)
    clock = SystemClock()
    initial_usage = initial_usage or {}
    budget = RunBudget(
        wall.deadline,
        token_limit,
        2 + max_steps * 3 * max_sessions,
        initial_usage.get("tokens", 0),
        initial_usage.get("attempts", 0),
    )
    completed = stale = 0
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
            stale = 0 if event["payload"].get("improved", False) else stale + 1
    # A killed request may have consumed tokens without returning usage. Never reset that cost.
    usage_known = usage_known and not (started_sessions - finished_sessions)
    budget_accounted = budget_accounted and not (started_sessions - finished_sessions)
    status = "budget_exhausted"
    names = ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL")
    model = None
    if (directory / "finalists.json").exists():
        status = "finalization_resumed"
    elif search_mode == "optuna":
        status = "numeric_search"
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
    from ml_harness.adapters.data.profile import dataset_profile
    from ml_harness.application.use_cases.hybrid_search import run_hybrid_search

    profile = dataset_profile(task)

    def checkpoint() -> None:
        from ml_harness.adapters.research.checkpoint import publish_checkpoint

        successful = [r for r in workspace.read_experiments() if r.get("status") == "succeeded"]
        wall.finalists = min(3, len(successful))
        try:
            publish_checkpoint(
                task,
                workspace,
                workspace.runner,
                BudgetedSubmissionWriter(AtomicSubmissionWriter(), clock, wall),
                wall,
                clock,
                threads=threads,
                timeout_seconds=timeout_seconds,
            )
        except (OSError, ValueError) as error:
            workspace.store.append(
                {
                    "event_id": "checkpoint-failed:" + uuid4().hex,
                    "kind": "checkpoint_failed",
                    "payload": {"error": str(error)[:500]},
                }
            )

    hypothesis = ProposalSessionWorkspace(
        task=task,
        baseline=baseline,
        workspace=workspace,
        wall=wall,
        clock=clock,
        model=model,
        budget=budget,
        profile=profile,
        token_limit=token_limit,
        max_steps=max_steps,
        timeout_seconds=timeout_seconds,
        threads=threads,
        reported_tokens=reported_tokens,
        usage_known=usage_known,
        budget_accounted=budget_accounted,
        llm_stop_reason=status,
    )

    if not (directory / "finalists.json").exists() and search_mode == "optuna":
        from ml_harness.application.use_cases.numeric_search import run_numeric_search

        status = run_numeric_search(workspace, wall, clock, checkpoint=checkpoint)
    if not (directory / "finalists.json").exists() and search_mode == "qwen":
        status = run_hybrid_search(
            wall,
            clock,
            hypothesis,
            lambda index: workspace.numeric_batch("best", index),
            max_hypotheses=max_sessions,
            completed=completed,
            initial_stale=stale,
            numeric_completed=workspace.numeric_completed(),
            checkpoint=checkpoint,
            llm_available=model is not None and usage_known and budget_accounted,
            local_reason=hypothesis.llm_stop_reason,
            observe=lambda decision: workspace.store.append(
                {
                    "event_id": "search-decision:" + uuid4().hex,
                    "kind": "search_decision",
                    "payload": asdict(decision),
                }
            ),
        )
    # All LLM calls have ended. A frozen plan on disk prohibits future search after restart as well.
    final = _finish(task, workspace, final_runner, threads, timeout_seconds)
    report = {
        "pipeline_version": 3,
        "status": status,
        "llm_stop_reason": hypothesis.llm_stop_reason,
        "sessions": hypothesis.sessions,
        "reports": workspace.read_experiments(),
        "tokens_used": budget.tokens_used,
        "reported_tokens": hypothesis.reported_tokens,
        "usage_known": hypothesis.usage_known,
        "budget_accounted": hypothesis.budget_accounted,
        "finalization": final,
        "selected": final.get("selected", "baseline"),
        "refit_status": final.get("refit_status", "not_needed"),
    }
    write_json(directory / "session.json", report)
    return report
