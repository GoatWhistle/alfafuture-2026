"""Bounded ReAct: tool execution is ordinary code, training never polls the LLM."""

import json
from dataclasses import dataclass
from typing import Any

from ml_harness.application.ports.clock import Clock
from ml_harness.application.ports.language_model import LanguageModel, ModelError, ToolExecutor
from ml_harness.domain.models import RunBudget

SYSTEM_PROMPT = """You are an autonomous tabular ML researcher using only the provided Qwen model.
Use inspect_data and read_file to understand the task and candidate.py. State a specific hypothesis
and expected measured effect, edit candidate.py, then run_experiment. The harness executes smoke
and full validation and measures the declared metric independently. Do not claim an improvement
without measured evidence. Read failures, repair the candidate within the available tool budget,
and interpret observations without claiming causality. Never modify the evaluator, split or
submission format. Do not download data, weights or packages. Do not request other language models.
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


def run_session(
    model: LanguageModel,
    tools: ToolExecutor,
    context: str,
    budget: RunBudget,
    clock: Clock,
    *,
    max_steps: int = 12,
    max_tool_errors: int = 3,
    reserve_seconds: float = 30,
) -> SessionResult:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": context},
    ]
    errors = input_tokens = output_tokens = steps = 0
    usage_known = True
    reports: list[dict[str, Any]] = []
    reason, summary = "step_limit", ""
    for _ in range(max_steps):
        if not budget.can_search(clock.now(), reserve_seconds):
            reason = "budget_exhausted"
            break
        try:
            reply = model.complete(messages, tools.schemas)
        except ModelError:
            budget.record_attempt(0)
            usage_known = False
            reason = "model_error"
            break
        steps += 1
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
        if not reply.tool_calls:
            summary, reason = reply.content[:8000], "model_finished"
            break
        for call in reply.tool_calls:
            if not budget.can_search(clock.now(), reserve_seconds):
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
            if errors >= max_tool_errors:
                reason = "tool_error_limit"
                break
        if reason in ("budget_exhausted", "tool_error_limit"):
            break
    return SessionResult(
        reason, summary, steps, errors, input_tokens, output_tokens, usage_known, tuple(reports)
    )
