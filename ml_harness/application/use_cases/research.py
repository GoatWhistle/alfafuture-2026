"""Bounded ReAct: tool execution is ordinary code, training never polls the LLM."""

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ml_harness.application.ports.clock import Clock
from ml_harness.application.ports.language_model import (
    BudgetAwareLanguageModel,
    LanguageModel,
    ModelError,
    ToolExecutor,
)
from ml_harness.domain.models import RunBudget

SYSTEM_PROMPT = """You are an autonomous tabular ML researcher using only the provided Qwen model.
Use inspect_data and read_file to understand the task and candidate.py. State a specific hypothesis
and expected measured effect, edit candidate.py, then run_experiment. The harness executes smoke
and full validation and measures the declared metric independently. Do not claim an improvement
without measured evidence. Read failures, repair the candidate within the available tool budget,
and interpret observations without claiming causality. Never modify the evaluator, split or
submission format. Do not download data, weights or packages. Do not request other language models.
Use a new experiment ID for new code. One successful full experiment ends this hypothesis session.
Finish with a brief interpretation of the measured result, including failures. A smoke result is
never a full result. The previously saved baseline remains available if the experiment fails.
"""


@dataclass(frozen=True)
class SessionResult:
    reason: str
    summary: str
    steps: int
    tool_errors: int
    input_tokens: int
    output_tokens: int
    usage_known: bool
    reports: tuple[dict[str, Any], ...]
    api_attempts: int


def run_session(
    model: LanguageModel,
    tools: ToolExecutor,
    context: str,
    budget: RunBudget,
    clock: Clock,
    *,
    max_steps: int = 12,
    max_tool_errors: int = 3,
    reserve_seconds: float | Callable[[], float] = 30,
    history_char_limit: int = 48000,
) -> SessionResult:
    if history_char_limit < 4096:
        raise ValueError("History limit must be at least 4096 characters")
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": context[: history_char_limit // 2]},
    ]
    errors = input_tokens = output_tokens = steps = api_attempts = 0
    usage_known = True
    reports: list[dict[str, Any]] = []
    reason, summary = "step_limit", ""
    awaiting_interpretation = False

    def reserve() -> float:
        return reserve_seconds() if callable(reserve_seconds) else reserve_seconds

    for _ in range(max_steps):
        if not budget.can_search(clock.now(), reserve()):
            reason = "budget_exhausted"
            break
        if len(json.dumps(messages, ensure_ascii=False)) > history_char_limit:
            # Rebuild without orphan tool replies; full history stays on disk.
            observations = [message for message in messages[2:] if message["role"] == "tool"][-3:]
            recent = [
                {"tool_call_id": message["tool_call_id"], "observation": message["content"][:2000]}
                for message in observations
            ]
            messages = messages[:2] + [
                {
                    "role": "user",
                    "content": "History shortened for context budget; recent observations: "
                    + json.dumps(recent, ensure_ascii=False),
                }
            ]
        while len(json.dumps(messages, ensure_ascii=False)) > history_char_limit:
            largest = max(range(1, len(messages)), key=lambda i: len(messages[i]["content"]))
            text = messages[largest]["content"]
            messages[largest]["content"] = text[: len(text) // 2]
        try:
            if isinstance(model, BudgetAwareLanguageModel):
                model.set_limits(
                    max(0, budget.deadline - clock.now() - reserve()),
                    max(0, budget.attempt_limit - budget.attempts_used),
                    max(0, budget.token_limit - budget.tokens_used),
                )
            reply = model.complete(messages, [] if awaiting_interpretation else tools.schemas)
        except ModelError as exc:
            if exc.attempts > 0:
                budget.record_attempt(0)
                budget.attempts_used += max(0, exc.attempts - 1)
            api_attempts += exc.attempts
            usage_known = usage_known and exc.attempts == 0
            reason = "model_error"
            break
        steps += 1
        api_attempts += reply.attempts
        budget.record_attempt(reply.input_tokens + reply.output_tokens)
        budget.attempts_used += max(0, reply.attempts - 1)
        input_tokens += reply.input_tokens
        output_tokens += reply.output_tokens
        usage_known = usage_known and reply.usage_known
        assistant: dict[str, Any] = {"role": "assistant", "content": reply.content or None}
        if reply.tool_calls:
            assistant["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": call.arguments,
                    },
                }
                for call in reply.tool_calls
            ]
        messages.append(assistant)
        if not reply.tool_calls or awaiting_interpretation:
            summary = reply.content[:8000]
            reason = "model_finished" if usage_known else "unknown_usage"
            break
        for call in reply.tool_calls:
            if not budget.can_search(clock.now(), reserve()):
                reason = "budget_exhausted"
                break
            result = tools.execute(call)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(result, ensure_ascii=False, allow_nan=False),
                }
            )
            if not result.get("ok"):
                errors += 1
            elif call.name == "run_experiment":
                reports.append(result["result"])
                if result["result"].get("status") != "succeeded":
                    errors += 1
                else:
                    awaiting_interpretation = True
            if errors >= max_tool_errors:
                reason = "tool_error_limit"
                break
            if awaiting_interpretation:
                break
        completed_ids = {
            message["tool_call_id"] for message in messages if message["role"] == "tool"
        }
        assistant["tool_calls"] = [
            item for item in assistant.get("tool_calls", []) if item["id"] in completed_ids
        ]
        if not usage_known:
            reason = "unknown_usage"
            break
        if reason in ("budget_exhausted", "tool_error_limit"):
            break
    return SessionResult(
        reason,
        summary,
        steps,
        errors,
        input_tokens,
        output_tokens,
        usage_known,
        tuple(reports),
        api_attempts,
    )
