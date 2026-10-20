"""Retry allocation must reserve complete requests and expose unknown prior usage.

QWEN_RETRY_BUDGET_MODULE optionally selects a temporary patched module for pre-apply
verification. Ordinary collection exercises the repository implementation directly.
"""

import importlib
import importlib.util
import json
import os
from types import ModuleType

import httpx
import pytest
from openai import OpenAI

from ml_harness.application.ports.language_model import ModelError
from ml_harness.application.use_cases.research import run_session
from ml_harness.domain.models import RunBudget

MESSAGES = [{"role": "user", "content": "Проверь бюджет"}]
TOOLS = [
    {
        "type": "function",
        "function": {"name": "inspect_data", "parameters": {"type": "object", "properties": {}}},
    }
]


def prompt_bound(messages, tools):
    return (
        len(json.dumps({"messages": messages, "tools": tools}, ensure_ascii=False).encode()) + 1024
    )


@pytest.fixture
def qwen_module(monkeypatch) -> ModuleType:
    target = os.environ.get("QWEN_RETRY_BUDGET_MODULE")
    if target:
        spec = importlib.util.spec_from_file_location("qwen_retry_budget_candidate", target)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module("ml_harness.adapters.llm.qwen")
    monkeypatch.setattr(module.time, "sleep", lambda seconds: None)
    return module


def success(tool_call=False):
    message = {"role": "assistant", "content": "done"}
    if tool_call:
        message["tool_calls"] = [
            {
                "id": "inspect",
                "type": "function",
                "function": {"name": "inspect_data", "arguments": "{}"},
            }
        ]
    return httpx.Response(
        200,
        json={
            "id": "reply",
            "object": "chat.completion",
            "created": 1,
            "model": "Qwen/supplied",
            "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9},
        },
    )


def model(module, handler):
    client = OpenAI(
        api_key="local-test-token",
        base_url="https://unused.test/v1",
        max_retries=9,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    return module.QwenLanguageModel(
        "local-test-token", "https://unused.test/v1", "Qwen/supplied", client=client, max_attempts=3
    )


@pytest.mark.parametrize("failures", [[429], ["timeout"], [503], [429, "timeout"]])
def test_retry_worst_case_sum_is_bounded_and_success_usage_is_unknown(qwen_module, failures):
    sent = []

    def handler(request):
        sent.append(json.loads(request.content))
        if len(sent) > len(failures):
            return success()
        failure = failures[len(sent) - 1]
        if failure == "timeout":
            raise httpx.ReadTimeout("local test", request=request)
        return httpx.Response(failure, json={"error": {"message": "local test"}})

    bound = prompt_bound(MESSAGES, TOOLS)
    tokens = 3 * (bound + 512)
    candidate = model(qwen_module, handler)
    candidate.set_limits(100, 3, tokens)
    reply = candidate.complete(MESSAGES, TOOLS)
    assert sum(bound + request["max_tokens"] for request in sent) <= tokens
    assert all(1 <= request["max_tokens"] <= 2048 for request in sent)
    assert reply.attempts == len(failures) + 1
    assert reply.usage_known is False
    assert (reply.input_tokens, reply.output_tokens) == (7, 2)
    assert reply.unreported_tokens_upper_bound == sum(
        bound + request["max_tokens"] for request in sent[:-1]
    )


@pytest.mark.parametrize("affordable", [0, 1, 2])
def test_retry_count_is_reduced_when_not_all_prompts_fit(qwen_module, affordable):
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(429, json={"error": {"message": "local test"}})

    bound = prompt_bound(MESSAGES, TOOLS)
    tokens = affordable * (bound + 512) if affordable else bound
    candidate = model(qwen_module, handler)
    candidate.set_limits(100, 3, tokens)
    with pytest.raises(ModelError) as error:
        candidate.complete(MESSAGES, TOOLS)
    assert error.value.attempts == len(requests) == affordable
    assert sum(bound + request["max_tokens"] for request in requests) <= tokens
    if requests:
        assert all(request["max_tokens"] == 512 for request in requests)


def test_first_attempt_success_still_has_known_usage(qwen_module):
    sent = []

    def handler(request):
        sent.append(json.loads(request.content))
        return success()

    candidate = model(qwen_module, handler)
    candidate.set_limits(100, 3, 10000)
    reply = candidate.complete(MESSAGES, TOOLS)
    assert reply.attempts == 1 and reply.usage_known is True
    assert len(sent) == 1


@pytest.mark.parametrize("failure", [429, "timeout"])
def test_successful_retry_charges_upper_bound_and_allows_followup_api(qwen_module, failure):
    calls, executed = [], []

    def handler(request):
        calls.append(json.loads(request.content))
        if len(calls) == 1:
            if failure == "timeout":
                raise httpx.ReadTimeout("local test", request=request)
            return httpx.Response(failure, json={"error": {"message": "local test"}})
        return success(tool_call=len(calls) == 2)

    class Clock:
        def now(self):
            return 0

    class Tools:
        schemas = TOOLS

        def execute(self, call):
            executed.append(call)
            return {"ok": True, "result": {"already_returned": True}}

    budget = RunBudget(100, 20000, 10)
    result = run_session(
        model(qwen_module, handler),
        Tools(),
        "task",
        budget,
        Clock(),
        max_steps=4,
        reserve_seconds=10,
    )
    assert result.reason == "model_finished" and result.usage_known is False
    assert result.budget_accounted
    assert result.api_attempts == budget.attempts_used == 3
    assert len(calls) == 3 and len(executed) == 1
    charged = prompt_bound(calls[0]["messages"], calls[0]["tools"]) + calls[0]["max_tokens"] + 18
    assert result.tokens_charged == budget.tokens_used == charged
    assert charged <= budget.token_limit


def test_edit_request_reserves_enough_output_before_allocating_retries(qwen_module):
    sent = []

    def handler(request):
        sent.append(json.loads(request.content))
        return httpx.Response(503, json={"error": {"message": "local test"}})

    tools = [{**TOOLS[0], "function": {**TOOLS[0]["function"], "name": "edit_file"}}]
    bound = prompt_bound(MESSAGES, tools)
    candidate = model(qwen_module, handler)
    candidate.set_limits(100, 3, 2 * (bound + 512))
    with pytest.raises(ModelError):
        candidate.complete(MESSAGES, tools)
    assert len(sent) == 1
    assert sent[0]["max_tokens"] >= 1536


def test_missing_usage_is_accounted_only_with_a_bounded_request(qwen_module):
    sent = []

    def handler(request):
        sent.append(json.loads(request.content))
        body = success().json()
        body.pop("usage")
        return httpx.Response(200, json=body)

    candidate = model(qwen_module, handler)
    candidate.set_limits(100, 1, 5000)
    reply = candidate.complete(MESSAGES, TOOLS)
    assert not reply.usage_known
    assert reply.unreported_tokens_upper_bound == (
        prompt_bound(MESSAGES, TOOLS) + sent[0]["max_tokens"]
    )
    assert reply.unreported_tokens_upper_bound <= 5000


def test_ample_budget_does_not_expand_provider_generation_window(qwen_module):
    sent = []

    def handler(request):
        sent.append(json.loads(request.content))
        return success()

    candidate = model(qwen_module, handler)
    candidate.set_limits(100, 1, 32000)
    candidate.complete(MESSAGES, TOOLS)
    # The saved live payload produced invalid tools/503 at 4096, valid edits at 1536.
    assert sent[0]["max_tokens"] == 1536
