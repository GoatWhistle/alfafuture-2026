"""Clock-controlled budget tests never train models or contact a real provider."""

import json
from types import SimpleNamespace

import httpx
import pytest
from openai import OpenAI

from ml_harness.adapters.execution.budget import BudgetedRunner
from ml_harness.adapters.llm.qwen import QwenLanguageModel
from ml_harness.application.ports.execution import ExecutionResult
from ml_harness.application.ports.language_model import ModelError, ModelReply, ToolCall
from ml_harness.application.use_cases.research import run_session
from ml_harness.domain.budget import WallClockBudget
from ml_harness.domain.models import ContractError, ExperimentStatus, RunBudget


class Clock:
    value = 0.0

    def now(self):
        return self.value


def test_dynamic_reserve_observes_maximum_and_finalization_uses_remaining():
    budget = WallClockBudget(1000)
    assert budget.reserve_seconds() == 15
    budget.observe("validation", 20)
    budget.observe("control", 10)
    budget.observe("refit", 30)
    budget.observe("predict", 4)
    assert budget.reserve_seconds() == 197
    assert budget.available(100) == 703
    assert budget.available(100, finalizing=True) == 900
    assert budget.available(2000) == 0
    budget.observe("write", 8)
    assert budget.reserve_seconds() == 200


@pytest.mark.parametrize(
    "kwargs",
    [{"deadline": float("nan")}, {"refit_seconds": -1}, {"finalists": -1}, {"finalists": True}],
)
def test_invalid_wallclock_budget(kwargs):
    with pytest.raises(ContractError):
        WallClockBudget(**({"deadline": 100} | kwargs))


@pytest.mark.parametrize("mode", ["validation", "control", "refit"])
def test_runner_clamps_timeout_observes_and_never_launches_when_exhausted(mode):
    clock = Clock()
    budget = WallClockBudget(100)
    calls = []

    def run(request, timeout):
        calls.append(timeout)
        clock.value += 5
        return ExecutionResult(ExperimentStatus.SUCCEEDED, 5, {"predict_seconds": 2})

    wrapped = BudgetedRunner(SimpleNamespace(run=run), clock, budget)
    request = SimpleNamespace(mode=mode)
    assert wrapped.run(request, 1000).status == ExperimentStatus.SUCCEEDED
    assert calls == [85]
    assert getattr(budget, "refit_seconds" if mode == "refit" else "validation_seconds") == 5
    assert budget.predict_seconds == 2
    clock.value = 100
    assert wrapped.run(request, 30).status == ExperimentStatus.TIMED_OUT
    assert calls == [85]
    clock.value = 98
    final = BudgetedRunner(SimpleNamespace(run=run), clock, budget, finalizing=True)
    final.run(request, 30)
    assert calls[-1] == 2


def response(usage=None, calls=False):
    message = {"role": "assistant", "content": "done"}
    if calls:
        message["tool_calls"] = [
            {
                "id": "call",
                "type": "function",
                "function": {"name": "inspect_data", "arguments": "{}"},
            }
        ]
    result = {
        "id": "reply",
        "object": "chat.completion",
        "created": 1,
        "model": "qwen-test",
        "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
    }
    if usage is not None:
        result["usage"] = usage
    return httpx.Response(200, json=result)


def qwen(handler):
    client = OpenAI(
        api_key="test-secret",
        base_url="https://unused.test/v1",
        max_retries=9,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    return QwenLanguageModel("test-secret", "https://unused.test/v1", "qwen-test", client=client)


@pytest.mark.parametrize(
    "seconds,attempts,tokens", [(0, 3, 10000), (100, 0, 10000), (100, 3, 1024)]
)
def test_qwen_exhaustion_before_api_has_zero_attempts(seconds, attempts, tokens):
    calls = []
    model = qwen(lambda request: calls.append(request))
    model.set_limits(seconds, attempts, tokens)
    with pytest.raises(ModelError) as error:
        model.complete([], [])
    assert error.value.attempts == 0
    assert not calls


def test_qwen_attempt_cap_and_prompt_aware_max_tokens(monkeypatch):
    monkeypatch.setattr("ml_harness.adapters.llm.qwen.time.sleep", lambda seconds: None)
    calls = []

    def handler(request):
        calls.append(json.loads(request.content))
        return httpx.Response(429, json={"error": {"message": "test-secret"}})

    model = qwen(handler)
    model.set_limits(100, 2, 2000)
    messages = [{"role": "user", "content": "я" * 10}]
    with pytest.raises(ModelError) as error:
        model.complete(messages, [])
    assert error.value.attempts == 1 and len(calls) == 1
    expected = (
        2000
        - len(json.dumps({"messages": messages, "tools": []}, ensure_ascii=False).encode())
        - 1024
    )
    assert all(call["max_tokens"] == expected for call in calls)


def test_qwen_remaining_deadline_bounds_each_timeout_and_retry_sleep(monkeypatch):
    clock = Clock()
    monkeypatch.setattr("ml_harness.adapters.llm.qwen.time.monotonic", clock.now)
    monkeypatch.setattr(
        "ml_harness.adapters.llm.qwen.time.sleep",
        lambda seconds: setattr(clock, "value", clock.value + seconds),
    )
    timeouts = []

    def handler(request):
        timeouts.append(request.extensions["timeout"]["read"])
        clock.value += 0.4
        return httpx.Response(503, json={"error": {"message": "test-secret"}})

    model = qwen(handler)
    model.set_limits(1, 5, 10000)
    with pytest.raises(ModelError) as error:
        model.complete([], [])
    assert error.value.attempts == 2
    assert timeouts == pytest.approx([1, 0.35])


@pytest.mark.parametrize(
    "usage,expected",
    [
        (None, (0, 0)),
        ({"prompt_tokens": 5}, (5, 0)),
        ({"prompt_tokens": -1, "completion_tokens": 7}, (0, 7)),
        ({"prompt_tokens": True, "completion_tokens": "bad"}, (0, 0)),
    ],
)
def test_invalid_or_partial_usage_is_unknown_but_preserves_valid_counts(usage, expected):
    reply = qwen(lambda request: response(usage)).complete([], [])
    assert reply.usage_known is False
    assert (reply.input_tokens, reply.output_tokens) == expected


def test_session_budget_hook_dynamic_reserve_unknown_usage_and_zero_attempt_error():
    clock = Clock()
    reserve = [10]
    calls, limits = [], []

    class Model:
        def set_limits(self, *args):
            limits.append(args)

        def complete(self, messages, tools):
            calls.append(messages)
            return ModelReply(
                "inspect",
                (ToolCall("call", "inspect_data", "{}"),),
                5,
                2,
                usage_known=False,
                attempts=2,
            )

    class Tools:
        schemas = []

        def execute(self, call):
            reserve[0] = 80
            return {"ok": True, "result": {}}

    budget = RunBudget(100, 10000, 3)
    result = run_session(
        Model(), Tools(), "task", budget, clock, reserve_seconds=lambda: reserve[0]
    )
    assert result.reason == "unknown_usage"
    assert len(calls) == 1 and limits == [(90, 3, 10000)]
    assert result.api_attempts == budget.attempts_used == 2
    assert budget.tokens_used == 7

    class Rejected(Model):
        def complete(self, messages, tools):
            raise ModelError("No room", attempts=0)

    zero = RunBudget(100, 10000, 3)
    result = run_session(Rejected(), Tools(), "task", zero, clock)
    assert result.api_attempts == zero.attempts_used == 0
    assert result.usage_known


def test_dynamic_reserve_changes_stop_search_after_tool():
    reserve = [10]
    calls = []

    class Model:
        def complete(self, messages, tools):
            calls.append(messages)
            return ModelReply("", (ToolCall("call", "inspect_data", "{}"),))

    class Tools:
        schemas = [{"type": "function", "function": {"name": "inspect_data"}}]

        def execute(self, call):
            reserve[0] = 101
            return {"ok": True, "result": {}}

    result = run_session(
        Model(),
        Tools(),
        "task",
        RunBudget(100, 10000, 3),
        Clock(),
        reserve_seconds=lambda: reserve[0],
    )
    assert result.reason == "budget_exhausted" and len(calls) == 1
