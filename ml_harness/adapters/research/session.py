"""Execute one semantic hypothesis and persist its evidence and usage facts."""

import json
from dataclasses import asdict, dataclass, field
from typing import Any
from uuid import uuid4

from ml_harness.adapters.llm.proposals import ProposalTools
from ml_harness.adapters.research.workspace import ResearchWorkspace, write_json
from ml_harness.application.ports.clock import Clock
from ml_harness.application.ports.language_model import LanguageModel
from ml_harness.application.use_cases.baseline import BaselineResult
from ml_harness.application.use_cases.context import bound_context, build_context
from ml_harness.application.use_cases.directions import choose_direction
from ml_harness.application.use_cases.hybrid_search import HypothesisOutcome
from ml_harness.application.use_cases.proposal_session import run_proposal_session
from ml_harness.domain.budget import WallClockBudget
from ml_harness.domain.models import PreparedTask, RunBudget


@dataclass
class ProposalSessionWorkspace:
    task: PreparedTask
    baseline: BaselineResult
    workspace: ResearchWorkspace
    wall: WallClockBudget
    clock: Clock
    model: LanguageModel | None
    budget: RunBudget
    profile: dict[str, Any]
    token_limit: int
    max_steps: int
    timeout_seconds: float
    threads: int
    reported_tokens: int = 0
    usage_known: bool = True
    budget_accounted: bool = True
    llm_stop_reason: str = "budget_exhausted"
    sessions: list[dict[str, Any]] = field(default_factory=list)

    def __call__(self, index: int) -> HypothesisOutcome:
        task, baseline, workspace = self.task, self.baseline, self.workspace
        wall, clock, model, budget = self.wall, self.clock, self.model, self.budget
        token_limit, max_steps = self.token_limit, self.max_steps
        timeout_seconds, threads = self.timeout_seconds, self.threads
        directory, profile = workspace.path, self.profile
        if model is None or not budget.can_search(clock.now(), wall.reserve_seconds()):
            return HypothesisOutcome(False, False, "llm_budget_exhausted")
        before = choose_direction(
            task, baseline.score, workspace.reports, baseline_report=workspace.baseline_report
        ).best_id
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
        context_value["dataset_profile"] = profile
        context_value["selected_evidence"] = list(direction.target_errors)
        context = bound_context(context_value)
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
        after = choose_direction(
            task, baseline.score, workspace.reports, baseline_report=workspace.baseline_report
        ).best_id
        persisted = {**asdict(session), "improved": before != after}
        workspace.register_session(identifier, persisted)
        write_json(directory / "sessions" / (identifier + ".json"), persisted)
        self.sessions.append(asdict(session))
        self.llm_stop_reason = session.reason
        self.reported_tokens += session.input_tokens + session.output_tokens
        self.usage_known = self.usage_known and session.usage_known
        self.budget_accounted = self.budget_accounted and session.budget_accounted
        after = choose_direction(
            task, baseline.score, workspace.reports, baseline_report=workspace.baseline_report
        ).best_id
        return HypothesisOutcome(
            before != after,
            self.budget_accounted
            and self.usage_known
            and session.reason in ("experiment_completed", "correction_limit"),
            session.reason,
        )
