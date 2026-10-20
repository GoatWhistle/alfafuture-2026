"""One hypothesis, at most two repairs; persist measurements without requiring LLM prose."""

import json
from collections.abc import Callable
from typing import Any

from ml_harness.application.ports.clock import Clock
from ml_harness.application.ports.language_model import (
    BudgetAwareLanguageModel,
    LanguageModel,
    ModelError,
    ToolExecutor,
)
from ml_harness.application.use_cases.research import SessionResult
from ml_harness.domain.models import RunBudget

PROMPT = (
    "You are the Qwen researcher. Propose one specific experiment addressing measured errors. "
    "Use propose_experiment exactly once per reply. Harness validates and trains it. "
    "Prefer a concrete feature/imbalance/model-family hypothesis; numerical search uses tune=true. "
    "target_error identifies supplied evidence; prefix exploratory: if unmeasured. "
    "Choose a small executable change within the fresh budget. Never invent results. "
    "Do not override evaluation, splits, paths, threads or seed. Python is only for feature "
    "transformers, fit(X) receives training features only. Never read files/network/targets. "
    "The tool result is the measured fact. No final interpretation or review is required."
)


def run_proposal_session(
    model: LanguageModel,
    tools: ToolExecutor,
    context: str,
    budget: RunBudget,
    clock: Clock,
    *,
    max_steps: int = 3,
    reserve_seconds: Callable[[], float],
) -> SessionResult:
    initial = json.loads(context)
    reports: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    input_tokens = output_tokens = attempts = steps = 0
    initial_tokens = budget.tokens_used
    known = accounted = True
    reason, detail = "correction_limit", None
    for _ in range(min(3, max_steps)):
        if not budget.can_search(clock.now(), reserve_seconds()):
            reason = "budget_exhausted"
            break
        remaining = max(0, budget.deadline - clock.now() - reserve_seconds())
        state = {
            **initial,
            "budget": {
                **initial.get("budget", {}),
                "seconds_remaining": remaining,
                "reserved_seconds": reserve_seconds(),
                "tokens_remaining": max(0, budget.token_limit - budget.tokens_used),
            },
            "previous_attempts": failures[-2:],
        }
        messages = [
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": json.dumps(state, ensure_ascii=False)},
        ]
        try:
            if isinstance(model, BudgetAwareLanguageModel):
                model.set_limits(
                    remaining,
                    max(0, budget.attempt_limit - budget.attempts_used),
                    max(0, budget.token_limit - budget.tokens_used),
                )
            reply = model.complete(messages, tools.schemas)
        except ModelError as exc:
            budget.attempts_used += exc.attempts
            attempts += exc.attempts
            known = known and exc.attempts == 0
            accounted = accounted and exc.attempts == 0
            reason, detail = exc.reason, str(exc)[:500]
            break
        steps += 1
        attempts += reply.attempts
        budget.record_attempt(
            reply.input_tokens + reply.output_tokens + reply.unreported_tokens_upper_bound
        )
        budget.attempts_used += max(0, reply.attempts - 1)
        input_tokens += reply.input_tokens
        output_tokens += reply.output_tokens
        known = known and reply.usage_known
        accounted = accounted and (reply.usage_known or reply.unreported_tokens_upper_bound > 0)
        if not accounted:
            reason = "unknown_usage"
            break
        if len(reply.tool_calls) != 1 or reply.finish_reason == "length":
            failures.append({"error": "Return exactly one complete propose_experiment call"})
            continue
        if not budget.can_search(clock.now(), reserve_seconds()):
            reason = "budget_exhausted"
            break
        response = tools.execute(reply.tool_calls[0])
        report = response.get("result")
        if isinstance(report, dict) and "id" in report:
            reports.append(report)
        if response.get("ok") and report and report.get("status") == "succeeded":
            reason = "experiment_completed"
            break
        failures.append(response)
    return SessionResult(
        reason,
        "",
        steps,
        len(failures),
        input_tokens,
        output_tokens,
        known,
        tuple(reports),
        attempts,
        tuple(failures),
        detail,
        budget.tokens_used - initial_tokens,
        accounted,
    )
