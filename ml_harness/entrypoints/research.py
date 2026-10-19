"""Wire the first research session after a valid baseline answer exists."""

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any
from uuid import uuid4

from ml_harness.adapters.execution.clock import SystemClock
from ml_harness.adapters.execution.runner import SubprocessRunner
from ml_harness.adapters.llm.qwen import QwenLanguageModel
from ml_harness.adapters.llm.tools import ResearchTools
from ml_harness.adapters.research.workspace import ResearchWorkspace, write_json
from ml_harness.adapters.storage.submission import AtomicSubmissionWriter, IndependentEvaluator
from ml_harness.application.ports.execution import RunRequest
from ml_harness.application.ports.language_model import ModelError
from ml_harness.application.use_cases.baseline import BaselineResult
from ml_harness.application.use_cases.context import build_context
from ml_harness.application.use_cases.directions import choose_direction
from ml_harness.application.use_cases.research import run_session
from ml_harness.domain.models import (
    ContractError,
    ExperimentStatus,
    Metric,
    PreparedTask,
    RunBudget,
    Score,
)


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
) -> dict[str, Any]:
    names = ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL")
    if not all(os.environ.get(name) for name in names):
        return {"status": "not_configured", "reason": "Supplied Qwen API configuration is missing"}
    clock = SystemClock()
    runner, evaluator, writer = SubprocessRunner(), IndependentEvaluator(), AtomicSubmissionWriter()
    try:
        model = QwenLanguageModel(
            os.environ["OPENAI_API_KEY"], os.environ["OPENAI_BASE_URL"], os.environ["OPENAI_MODEL"]
        )
    except ModelError:
        return {"status": "configuration_error", "reason": "Invalid supplied Qwen configuration"}
    workspace = ResearchWorkspace(
        task, directory, runner, evaluator, threads=threads, timeout_seconds=timeout_seconds
    )
    baseline_output = directory.parent / "validation/result.json"
    baseline_request = directory.parent / "validation/request.json"
    request = json.loads(baseline_request.read_text()) if baseline_request.exists() else {}
    workspace.register_baseline(
        baseline.score,
        baseline_output if baseline_output.exists() else None,
        backend=request.get("backend", "catboost"),
        parameters=request.get("parameters", {}),
    )
    budget = RunBudget(clock.now() + 1800, token_limit, max_steps * 3 * max_sessions)
    # Usage survives restart. Unknown usage is kept explicit rather than called zero cost.
    for event in workspace.store.read_events():
        if event["kind"] == "session_finished":
            usage = event["payload"]["usage"]
            budget.tokens_used += usage["input_tokens"] + usage["output_tokens"]
            budget.attempts_used += usage.get("api_attempts", 0)
    sessions = []
    for _ in range(max_sessions):
        if not budget.can_search(clock.now(), 30):
            break
        tools = ResearchTools(
            directory, workspace.inspect_data, workspace.run_experiment, workspace.read_experiments
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
                "steps_per_hypothesis": max_steps,
            },
        )
        session = run_session(model, tools, context, budget, clock, max_steps=max_steps)
        workspace.register_session(identifier, asdict(session))
        write_json(directory / "sessions" / (identifier + ".json"), asdict(session))
        sessions.append(asdict(session))
        if session.reason in ("model_error", "budget_exhausted"):
            break
    report: dict[str, Any] = {
        "status": sessions[-1]["reason"] if sessions else "budget_exhausted",
        "sessions": sessions,
        "reports": workspace.read_experiments(),
        "selected": "baseline",
        "refit_status": "not_needed",
        "tokens_used": budget.tokens_used,
    }
    winner: dict[str, Any] | None = None
    best = baseline.score
    for experiment in workspace.reports:
        if experiment["status"] != "succeeded":
            continue
        measured = experiment["full_score"]
        score = Score(
            measured["value"],
            Metric(measured["metric"]),
            measured["protocol_id"],
            measured["level"],
        )
        if score.better_than(best):
            winner, best = experiment, score
    if winner is not None:
        try:
            outcome = runner.run(
                RunRequest(
                    task,
                    str(directory / ("selected-refit-" + uuid4().hex)),
                    mode="refit",
                    threads=threads,
                    candidate_path=str(directory / "experiments" / winner["id"] / "candidate.py"),
                ),
                timeout_seconds,
            )
            report["refit_status"] = outcome.status.value
            if outcome.status == ExperimentStatus.SUCCEEDED and outcome.output is not None:
                writer.write(task, outcome.output)
                report.update(selected=winner["id"], selected_score=asdict(best))
        except (ContractError, OSError, ValueError):
            report["refit_status"] = "failed_baseline_preserved"
    write_json(directory / "session.json", report)
    return report
