"""Deterministic model scripts prove bounded ReAct semantics without an API."""

import json

from ml_harness.application.ports.language_model import ModelError, ModelReply, ToolCall
from ml_harness.application.use_cases.research import run_session
from ml_harness.domain.models import RunBudget


class FakeLanguageModel:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.messages = []

    def complete(self, messages, tools):
        self.messages.append(list(messages))
        reply = next(self.replies)
        if isinstance(reply, Exception):
            raise reply
        return reply


class FakeClock:
    def now(self):
        return 0


class FakeTools:
    schemas = []

    def __init__(self, results):
        self.results = iter(results)
        self.calls = []

    def execute(self, call):
        self.calls.append(call)
        return next(self.results)


def session(model, tools, **kwargs):
    return run_session(model, tools, "task", RunBudget(1000, 10000, 10), FakeClock(), **kwargs)


def test_tool_observation_returns_to_model_and_usage_is_counted():
    call = ToolCall("call-1", "run_experiment", "{}")
    model = FakeLanguageModel(
        [
            ModelReply("hypothesis", (call,), 5, 3, attempts=2),
            ModelReply("measured summary", (), 6, 4),
        ]
    )
    tools = FakeTools([{"ok": True, "result": {"status": "succeeded", "full_score": 1}}])
    result = session(model, tools)
    assert result.reason == "model_finished"
    assert (result.input_tokens, result.output_tokens) == (11, 7)
    assert len(result.reports) == 1
    assistant, observation = model.messages[1][-2:]
    assert assistant["tool_calls"][0]["id"] == "call-1"
    assert observation["tool_call_id"] == "call-1"
    assert json.loads(observation["content"])["result"]["full_score"] == 1


def test_broken_candidate_attempts_are_bounded():
    calls = tuple(ToolCall(str(i), "run_experiment", "{}") for i in range(5))
    model = FakeLanguageModel([ModelReply("", calls)])
    tools = FakeTools([{"ok": True, "result": {"status": "failed"}}] * 5)
    result = session(model, tools)
    assert result.reason == "tool_error_limit" and result.tool_errors == 3
    assert len(tools.calls) == 3


def test_api_error_returns_control_with_unknown_usage():
    result = session(FakeLanguageModel([ModelError("API failed")]), FakeTools([]))
    assert result.reason == "model_error" and not result.usage_known


def test_exhausted_budget_never_calls_model():
    result = run_session(FakeLanguageModel([]), FakeTools([]), "", RunBudget(0, 0, 0), FakeClock())
    assert result.reason == "budget_exhausted" and result.steps == 0


def test_no_tools_means_finished_without_fabricated_experiments():
    result = session(FakeLanguageModel([ModelReply("could not improve")]), FakeTools([]))
    assert result.reason == "model_finished" and result.reports == ()
