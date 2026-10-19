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
    assistant, observation = model.messages[1][-3:-1]
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


def test_tool_history_is_bounded_even_for_escaped_large_strings():
    model = FakeLanguageModel(
        [
            ModelReply("", (ToolCall("read", "read_file", "{}"),)),
            ModelReply("done"),
        ]
    )
    tools = FakeTools([{"ok": True, "result": {"text": "\x00" * 100000}}])
    result = run_session(
        model,
        tools,
        "\x00" * 10000,
        RunBudget(1000, 10000, 10),
        FakeClock(),
        history_char_limit=8000,
    )
    assert result.reason == "model_finished"
    assert all(len(json.dumps(messages, ensure_ascii=False)) <= 8000 for messages in model.messages)


def test_new_hypothesis_is_not_executed_in_same_session():
    model = FakeLanguageModel(
        [
            ModelReply(
                "", (ToolCall("run", "run_experiment", "{}"), ToolCall("extra", "edit_file", "{}"))
            ),
            ModelReply("", (ToolCall("new", "run_experiment", "{}"),)),
        ]
    )
    tools = FakeTools([{"ok": True, "result": {"status": "succeeded"}}])
    result = session(model, tools)
    assert len(tools.calls) == 1 and len(result.reports) == 1
    # The skipped tool must not leave an orphan call in the next provider request.
    assert len(model.messages[1][-3]["tool_calls"]) == 1


def test_large_measured_report_keeps_score_after_compaction():
    report = {
        "id": "changed",
        "status": "succeeded",
        "full_score": {"value": 12.5},
        "parent_score": {"value": 13.0},
        "hypothesis": "x" * 51000,
        "code_hash": "a" * 64,
        "error_report": {"segments": []},
    }
    model = FakeLanguageModel(
        [
            ModelReply("", (ToolCall("run", "run_experiment", "{}"),)),
            ModelReply("Measured gain 0.5"),
        ]
    )
    result = session(model, FakeTools([{"ok": True, "result": report}]), history_char_limit=4096)
    assert result.reason == "model_finished"
    request = json.dumps(model.messages[-1])
    assert "12.5" in request and "13.0" in request and "succeeded" in request
    assert len(request) <= 4096


def test_truncated_reply_is_repaired_and_usage_is_counted():
    model = FakeLanguageModel(
        [
            ModelReply("unfinished", input_tokens=10, output_tokens=20, finish_reason="length"),
            ModelReply("done", input_tokens=12, output_tokens=3),
        ]
    )
    result = session(model, FakeTools([]))
    assert result.reason == "model_finished"
    assert result.steps == 2 and result.output_tokens == 23


def test_repeated_incomplete_reply_is_not_finished():
    model = FakeLanguageModel(
        [ModelReply("", finish_reason="length"), ModelReply("", finish_reason="length")]
    )
    result = session(model, FakeTools([]))
    assert result.reason == "protocol_error" and not result.reports
