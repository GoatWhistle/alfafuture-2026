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
from ml_harness.application.use_cases.observations import compact_observation, encoded
from ml_harness.domain.models import RunBudget

SYSTEM_PROMPT = """You are an autonomous tabular ML researcher using only the provided Qwen model.
Workspace-readable files: task.md, data_dictionary.csv, candidate_api.md, candidate.py.
The context includes task_overview, candidate_contract and current candidate_code.
Use files only for missing details. inspect_data(columns=...) gives selected statistics.
Inspect once, implement one concrete hypothesis, then run_experiment.
Do not reread unchanged pages. Prefer concise code reusing the documented base model.
Do not try to read package source or absolute dataset paths. Use read_file offset for more pages.
State a specific hypothesis
and expected measured effect, edit candidate.py, then run_experiment. The harness executes smoke
and full validation and measures the declared metric independently. Do not claim an improvement
without measured evidence. Read failures, repair the candidate within the available tool budget,
and interpret observations without claiming causality. Never modify the evaluator, split or
submission format. Do not download data, weights or packages. Do not request other language models.
Use a new experiment ID for new code. One successful full experiment ends this hypothesis session.
Finish with a brief interpretation of the measured result, including failures. A smoke result is
never a full result. The previously saved baseline remains available if the experiment fails.
"""


ACTION_SYSTEM_PROMPT = (
    "You are an autonomous tabular ML researcher using only the provided Qwen model. "
    "Follow the current phase and only the tools provided in this request. "
    "Use candidate_code and candidate_contract already supplied in context. "
    "Do not restart file navigation. State a concrete hypothesis and test the written code. "
    "The harness measures smoke and full validation independently; never invent scores. "
    "Never modify the evaluator, split or submission format. Do not download data, weights "
    "or packages or request other language models. "
)


INTERPRETATION_INSTRUCTION = (
    "In at most 120 words, compare full measured quality with the parent. "
    "Compare the hypothesis with candidate_code, which is the executed source. "
    "If the code does not implement the hypothesis, explicitly say the hypothesis was not tested; "
    "do not attribute the score to unimplemented changes. A single split does not prove causality. "
    "Do not propose tool calls. "
)


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
    tool_failures: tuple[dict[str, Any], ...] = ()
    error_detail: str | None = None
    tokens_charged: int = 0
    budget_accounted: bool = True


def _request_size(messages: list[dict[str, Any]], schemas: list[dict[str, Any]]) -> int:
    return len(
        json.dumps({"messages": messages, "tools": schemas}, ensure_ascii=False).encode("utf-8")
    )


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
    usage_known = budget_accounted = True
    initial_tokens_used = budget.tokens_used
    reports: list[dict[str, Any]] = []
    reason, summary = "step_limit", ""
    awaiting_interpretation = False
    correction_used = False
    failures: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    read_files: dict[str, dict[str, Any]] = {}
    try:
        initial_state = json.loads(context)
        working_code = (
            initial_state.get("candidate_code", "") if isinstance(initial_state, dict) else ""
        )
    except (ValueError, TypeError):
        working_code = ""
    navigation_steps = 0
    pending_action: str | None = None
    error_detail: str | None = None

    def reserve() -> float:
        return reserve_seconds() if callable(reserve_seconds) else reserve_seconds

    for _ in range(max_steps):
        if not budget.can_search(clock.now(), reserve()):
            reason = "budget_exhausted"
            break
        available_tokens = max(0, budget.token_limit - budget.tokens_used)
        schemas = [] if awaiting_interpretation else tools.schemas
        if not awaiting_interpretation:
            if navigation_steps >= 3 and pending_action is None:
                pending_action = "edit_file"
            if pending_action:
                required = [
                    schema for schema in schemas if schema["function"]["name"] == pending_action
                ]
                schemas = required or schemas
        if pending_action or awaiting_interpretation:
            phase = "interpretation" if awaiting_interpretation else pending_action
            messages[0] = {
                "role": "system",
                "content": ACTION_SYSTEM_PROMPT
                + f"Current phase: {phase}. "
                + (
                    INTERPRETATION_INSTRUCTION
                    if awaiting_interpretation
                    else "Return one complete call to the current phase tool."
                ),
            }
        request_limit = (
            min(history_char_limit, max(0, available_tokens - 1800))
            if isinstance(model, BudgetAwareLanguageModel)
            else history_char_limit
        )

        action_request = isinstance(model, BudgetAwareLanguageModel) and (
            pending_action is not None or awaiting_interpretation
        )
        if action_request:
            # Navigation transcripts can dominate a code-generation request. Retain
            # the structured research state, not the entire sequence of file pages.
            request_limit = min(request_limit, 12000)
        if (
            action_request
            or len(json.dumps(messages, ensure_ascii=False)) > history_char_limit
            or _request_size(messages, schemas) > request_limit
        ):
            try:
                state = json.loads(context)
                if not isinstance(state, dict):
                    state = {"task": str(state)[:500]}
            except (ValueError, TypeError):
                state = {"task": context[:500]}
            state["candidate_code"] = working_code[:4000]
            state["read_files"] = (
                {
                    path: {"offset": info["offset"], "next_offset": info["next_offset"]}
                    for path, info in read_files.items()
                }
                if action_request
                else read_files
            )
            state["next_action"] = (
                "interpretation" if awaiting_interpretation else pending_action or "research"
            )
            selected_reports = state.pop("reports", [])
            state["selected_reports"] = [
                {
                    key: report[key]
                    for key in ("id", "role", "measured", "measured_errors")
                    if key in report
                }
                for report in selected_reports
                if report.get("role") in ("parent", "best")
            ][:2]
            baseline_errors = state.get("baseline_errors")
            if isinstance(baseline_errors, dict):
                baseline_errors["target_errors"] = baseline_errors.get("target_errors", [])[:2]
            task_info = state.get("task")
            if isinstance(task_info, dict):
                task_info.pop("features", None)
                task_info.pop("classes", None)
            latest = next(
                (item for item in reversed(observations) if item["tool"] == "run_experiment"), None
            )
            if pending_action == "run_experiment" or awaiting_interpretation:
                # Executing written code and interpreting a measured report do not
                # need the task pages or the model-construction contract again.
                state = {
                    key: value
                    for key, value in state.items()
                    if key
                    in (
                        "task",
                        "baseline_measured",
                        "direction",
                        "candidate_code",
                        "next_action",
                        "budget",
                    )
                }
            recent = {
                "phase": "interpretation" if awaiting_interpretation else "research",
                "last_experiment": latest,
                "recent_tool_failures": failures[-3:],
                "recent_observations": (
                    observations[-2:]
                    if latest is None and pending_action != "run_experiment"
                    else []
                ),
            }
            phase_instruction = (
                "Experiment complete; tools unavailable. " + INTERPRETATION_INSTRUCTION
                if awaiting_interpretation
                else (
                    (
                        "Use run_experiment now. It executes candidate_code unchanged. "
                        "The hypothesis must describe this written code, "
                        "not intended future edits. "
                        if pending_action == "run_experiment"
                        else f"Use {pending_action} now. Prefer concise code reusing base_model. "
                    )
                    if pending_action
                    else "Continue research using the available tools. "
                )
            )
            if correction_used:
                phase_instruction += (
                    "The prior response was incomplete. Return one complete tool call, "
                    "or a concise interpretation if tools are unavailable. "
                )
            messages = [
                messages[0],
                {"role": "user", "content": encoded(state)},
                {"role": "user", "content": phase_instruction + encoded(recent)},
            ]
            # Drop optional reads first; measured latest experiment is retained intact.
            if _request_size(messages, schemas) > request_limit:
                recent["recent_observations"] = []
                messages[-1]["content"] = phase_instruction + encoded(recent)
            if _request_size(messages, schemas) > request_limit:
                state["read_files"] = {
                    path: {"offset": info["offset"]} for path, info in read_files.items()
                }
                messages[1]["content"] = encoded(state)
        if _request_size(messages, schemas) > request_limit:
            reason = "budget_exhausted"
            break
        try:
            if isinstance(model, BudgetAwareLanguageModel):
                model.set_limits(
                    max(0, budget.deadline - clock.now() - reserve()),
                    max(0, budget.attempt_limit - budget.attempts_used),
                    max(0, budget.token_limit - budget.tokens_used),
                )
            reply = model.complete(messages, schemas)
        except ModelError as exc:
            if exc.attempts > 0:
                budget.record_attempt(0)
                budget.attempts_used += max(0, exc.attempts - 1)
            api_attempts += exc.attempts
            usage_known = usage_known and exc.attempts == 0
            budget_accounted = budget_accounted and exc.attempts == 0
            reason = exc.reason
            error_detail = str(exc)[:500]
            break
        steps += 1
        api_attempts += reply.attempts
        budget.record_attempt(
            reply.input_tokens + reply.output_tokens + reply.unreported_tokens_upper_bound
        )
        budget.attempts_used += max(0, reply.attempts - 1)
        input_tokens += reply.input_tokens
        output_tokens += reply.output_tokens
        usage_known = usage_known and reply.usage_known
        budget_accounted = budget_accounted and (
            reply.usage_known or reply.unreported_tokens_upper_bound > 0
        )
        if awaiting_interpretation and reply.tool_calls:
            reason = "protocol_error"
            break
        invalid_reply = (
            (pending_action is not None and not reply.tool_calls)
            or reply.finish_reason == "length"
            or (not reply.content.strip() and not reply.tool_calls)
        )
        if invalid_reply:
            if not budget_accounted:
                reason = "unknown_usage"
                break
            if correction_used:
                reason = "protocol_error"
                break
            correction_used = True
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "The previous response was empty or truncated. Return one complete short "
                        "tool call, or a concise final interpretation if the experiment is "
                        "complete. Do not emit reasoning or XML tool calls."
                    ),
                }
            )
            continue
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
            reason = "model_finished" if budget_accounted else "unknown_usage"
            break
        for call in reply.tool_calls:
            if not budget.can_search(clock.now(), reserve()):
                reason = "budget_exhausted"
                break
            result: dict[str, Any]
            allowed_names = {schema["function"]["name"] for schema in schemas}
            if call.name not in allowed_names:
                result = {
                    "ok": False,
                    "error": {
                        "type": "tool_not_allowed",
                        "message": (
                            f"Tool {call.name!r} is unavailable in this phase. "
                            f"Use one of: {', '.join(sorted(allowed_names))}."
                        ),
                    },
                }
            else:
                result = tools.execute(call)
            if result.get("ok"):
                try:
                    arguments = json.loads(call.arguments)
                except (ValueError, TypeError):
                    arguments = {}
                if call.name == "read_file":
                    path = arguments.get("path", "")
                    read_files[path] = {
                        "offset": arguments.get("offset", 0),
                        "next_offset": result.get("result", {}).get("next_offset"),
                        "excerpt": str(result.get("result", {}).get("text", ""))[:500],
                    }
                    navigation_steps += 1
                elif call.name in ("inspect_data", "read_experiments"):
                    navigation_steps += 1
                elif call.name == "edit_file":
                    pending_action = "run_experiment"
                    working_code = arguments.get("content", working_code)
                    navigation_steps = 0
                elif call.name == "run_experiment":
                    pending_action = None
                    navigation_steps = 0
            observation = compact_observation(result)
            observations.append({"tool": call.name, "observation": observation})
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": encoded(observation),
                }
            )
            if not result.get("ok"):
                errors += 1
                if result.get("error", {}).get("type") == "candidate_unchanged":
                    pending_action = "edit_file"
                failures.append(
                    {
                        "tool": call.name,
                        "arguments": call.arguments[:250],
                        "error": result.get("error"),
                    }
                )
            elif call.name == "run_experiment":
                reports.append(result["result"])
                if result["result"].get("status") != "succeeded":
                    errors += 1
                    failures.append(
                        {
                            "tool": call.name,
                            "error": result["result"].get("error"),
                            "experiment_id": result["result"].get("id"),
                        }
                    )
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
        if navigation_steps >= 3 and not awaiting_interpretation:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "The task and candidate contract have been inspected. Stop repeating "
                        "navigation. Implement one hypothesis in candidate.py with edit_file, "
                        "then run_experiment. Current code is in candidate_code."
                    ),
                }
            )
        if awaiting_interpretation:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "The experiment has finished. This is the final interpretation phase; "
                        "tools are unavailable. " + INTERPRETATION_INSTRUCTION
                    ),
                }
            )
        if not budget_accounted:
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
        tuple(failures[-3:]),
        error_detail,
        budget.tokens_used - initial_tokens_used,
        budget_accounted,
    )
