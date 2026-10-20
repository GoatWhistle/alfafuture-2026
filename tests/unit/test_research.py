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
    schemas = [
        {"type": "function", "function": {"name": name}}
        for name in ("read_file", "inspect_data", "read_experiments", "edit_file", "run_experiment")
    ]

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
            ModelReply("Reviewed measured summary."),
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
            ModelReply("Reviewed measured gain 0.5."),
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


def test_compaction_remembers_read_materials_and_current_code():
    context = json.dumps(
        {
            "task_overview": "Annual building energy use",
            "candidate_contract": "fit(X, y), predict(X)",
            "candidate_code": (
                "def build_model(kind, parameters, threads, seed): "
                "return base_model(kind, parameters, threads, seed)"
            ),
        }
    )
    model = FakeLanguageModel(
        [
            ModelReply("", (ToolCall("task", "read_file", '{"path":"task.md"}'),)),
            ModelReply("ready"),
        ]
    )
    tools = FakeTools([{"ok": True, "result": {"text": "condition " * 4000, "next_offset": 4000}}])
    result = run_session(
        model, tools, context, RunBudget(1000, 32000, 10), FakeClock(), history_char_limit=8000
    )
    assert result.reason == "model_finished"
    compacted = json.dumps(model.messages[-1])
    assert "candidate_code" in compacted and "build_model" in compacted
    assert "fit(X, y)" in compacted and "Annual building energy use" in compacted
    assert "read_files" in compacted and "task.md" in compacted


def test_navigation_transitions_to_edit_then_validation_then_interpretation():
    class Tools(FakeTools):
        schemas = [
            {"type": "function", "function": {"name": name}}
            for name in ("read_file", "inspect_data", "edit_file", "run_experiment")
        ]

    class Model(FakeLanguageModel):
        def __init__(self, replies):
            super().__init__(replies)
            self.allowed = []

        def set_limits(self, *limits):
            pass

        def complete(self, messages, tools):
            self.allowed.append([tool["function"]["name"] for tool in tools])
            return super().complete(messages, tools)

    model = Model(
        [
            ModelReply(
                "", tuple(ToolCall(str(i), "read_file", '{"path":"task.md"}') for i in range(3))
            ),
            ModelReply("Hypothesis stated, but edit still missing"),
            ModelReply(
                "", (ToolCall("edit", "edit_file", '{"path":"candidate.py","content":"changed"}'),)
            ),
            ModelReply("", (ToolCall("run", "run_experiment", "{}"),)),
            ModelReply("Measured result received"),
            ModelReply("Reviewed measured result."),
        ]
    )
    tools = Tools(
        [{"ok": True, "result": {"text": "task"}}] * 3
        + [
            {"ok": True, "result": {"written": True}},
            {"ok": True, "result": {"status": "succeeded", "full_score": 1}},
        ]
    )
    result = session(model, tools)
    assert result.reason == "model_finished" and len(result.reports) == 1
    assert model.allowed[1:] == [["edit_file"], ["edit_file"], ["run_experiment"], [], []]
    assert "prior response was incomplete" in json.dumps(model.messages[2])
    assert all(message["role"] != "tool" for message in model.messages[3])
    assert "changed" in json.dumps(model.messages[3])
    assert "full_score" in json.dumps(model.messages[4])
    assert "executes candidate_code unchanged" in json.dumps(model.messages[3])
    assert "Compare the hypothesis with candidate_code" in json.dumps(model.messages[4])
    assert "hypothesis was not tested" in json.dumps(model.messages[4])
    assert "Inspect once" not in model.messages[1][0]["content"]
    assert "Current phase: edit_file" in model.messages[1][0]["content"]
    assert "Current phase: run_experiment" in model.messages[3][0]["content"]
    assert "Current phase: interpretation" in model.messages[4][0]["content"]


def test_action_request_discards_navigation_history_but_keeps_research_facts():
    class Tools(FakeTools):
        schemas = [
            {"type": "function", "function": {"name": name}}
            for name in ("read_file", "edit_file", "run_experiment")
        ]

    class Model(FakeLanguageModel):
        def set_limits(self, *limits):
            pass

    context = json.dumps(
        {
            "task": {
                "target": "energy",
                "metric": "rmse",
                "classes": ["A", "B"],
                "classes_omitted": 8,
            },
            "candidate_code": "def build_model(): pass",
            "candidate_contract": "fit(X,y), predict(X)",
            "baseline_measured": {"value": 42.2},
            "direction": {"parent_id": "baseline", "hypothesis": "reduce outlier sensitivity"},
        }
    )
    model = Model(
        [
            ModelReply(
                "",
                tuple(
                    ToolCall(str(i), "read_file", json.dumps({"path": f"file-{i}"}))
                    for i in range(3)
                ),
            ),
            ModelError("stop", attempts=0),
        ]
    )
    result = run_session(
        model,
        Tools([{"ok": True, "result": {"text": "navigation " * 700}}] * 3),
        context,
        RunBudget(1000, 32000, 10),
        FakeClock(),
    )
    request = json.dumps(model.messages[-1], ensure_ascii=False)
    assert len(request.encode()) < 12000
    assert "42.2" in request and "reduce outlier sensitivity" in request
    assert "def build_model" in request and "file-2" in request
    assert all(message["role"] != "tool" for message in model.messages[-1])
    assert result.reason == "model_error"
    task_state = json.loads(model.messages[-1][1]["content"])["task"]
    assert task_state["class_count"] == 10
    assert "classes_omitted" not in task_state


def test_out_of_phase_tool_is_rejected_before_execution_and_can_be_corrected():
    class Tools(FakeTools):
        schemas = [
            {"type": "function", "function": {"name": name}}
            for name in ("read_file", "inspect_data", "edit_file", "run_experiment")
        ]

    class Model(FakeLanguageModel):
        def set_limits(self, *limits):
            pass

    model = Model(
        [
            ModelReply("", tuple(ToolCall(str(i), "read_file", "{}") for i in range(3))),
            # Replay the provider symptom: inspect_data despite edit_file-only schema.
            ModelReply("", (ToolCall("wrong", "inspect_data", "{}"),), 12, 4),
            ModelReply("", (ToolCall("edit", "edit_file", '{"content":"changed"}'),)),
            ModelReply("", (ToolCall("run", "run_experiment", "{}"),)),
            ModelReply("The measured score is 1."),
            ModelReply("Reviewed: the measured score is 1."),
        ]
    )
    tools = Tools(
        [{"ok": True, "result": {"text": "task"}}] * 3
        + [
            {"ok": True, "result": {"written": True}},
            {"ok": True, "result": {"status": "succeeded", "full_score": 1}},
        ]
    )
    result = session(model, tools)
    assert [call.name for call in tools.calls] == ["read_file"] * 3 + [
        "edit_file",
        "run_experiment",
    ]
    assert result.reason == "model_finished" and len(result.reports) == 1
    assert result.tool_errors == 1
    assert result.input_tokens == 12 and result.output_tokens == 4
    corrective = json.dumps(model.messages[2])
    assert "tool_not_allowed" in corrective and "edit_file" in corrective


def test_interpretation_draft_is_audited_before_session_finishes():
    model = FakeLanguageModel(
        [
            ModelReply("", (ToolCall("run", "run_experiment", "{}"),), 2, 1),
            ModelReply("Draft incorrectly claims all classes improved.", (), 5, 3),
            ModelReply(
                "Only the selected classes improved; one split is not causal evidence.", (), 7, 4
            ),
        ]
    )
    tools = FakeTools([{"ok": True, "result": {"status": "succeeded", "full_score": 1}}])
    result = session(model, tools)
    assert result.summary == "Only the selected classes improved; one split is not causal evidence."
    assert result.reason == "model_finished" and result.steps == 3
    assert (result.input_tokens, result.output_tokens, result.api_attempts) == (14, 8, 3)
    assert model.messages[2][-2]["content"].startswith("Draft incorrectly")
    assert "Audit the previous draft" in model.messages[2][-1]["content"]


def test_unreviewed_draft_does_not_count_as_finished_at_step_limit():
    model = FakeLanguageModel(
        [
            ModelReply("", (ToolCall("run", "run_experiment", "{}"),)),
            ModelReply("An unreviewed draft."),
        ]
    )
    result = session(
        model,
        FakeTools([{"ok": True, "result": {"status": "succeeded"}}]),
        max_steps=2,
    )
    assert result.reason == "step_limit" and result.summary == ""
    assert len(result.reports) == 1


def test_review_compaction_retains_draft_code_and_latest_measurement():
    class Model(FakeLanguageModel):
        def set_limits(self, *limits):
            pass

    code = (
        "def build_model(kind, parameters, threads, seed): "
        "return base_model(kind, parameters, threads, seed)"
    )
    model = Model(
        [
            ModelReply("", (ToolCall("run", "run_experiment", "{}"),)),
            ModelReply("unreviewed draft " * 800),
            ModelReply("Reviewed: RMSE 12.5 vs 13.0 on one split."),
        ]
    )
    tools = FakeTools(
        [
            {
                "ok": True,
                "result": {
                    "id": "changed",
                    "status": "succeeded",
                    "full_score": {"value": 12.5},
                    "parent_score": {"value": 13.0},
                },
            }
        ]
    )
    result = run_session(
        model,
        tools,
        json.dumps({"candidate_code": code}),
        RunBudget(1000, 32000, 10),
        FakeClock(),
        history_char_limit=6000,
    )
    assert result.reason == "model_finished" and result.steps == 3
    state = json.loads(model.messages[2][1]["content"])
    assert state["candidate_code"] == code and not state["candidate_code_truncated"]
    review = model.messages[2][-1]["content"]
    assert "Audit the previous draft" in review and "draft_interpretation" in review
    assert '"draft_truncated":true' in review
    assert "12.5" in review and "13.0" in review
    assert len(json.dumps(model.messages[2], ensure_ascii=False).encode()) < 6000


def test_factory_runtime_contract_survives_run_interpretation_and_review_compaction():
    class Model(FakeLanguageModel):
        def set_limits(self, *limits):
            pass

    contract = {
        "kind_selects": "task_type",
        "backend_option": "backend",
        "default_backend": "catboost",
    }
    model = Model(
        [
            ModelReply("", (ToolCall("edit", "edit_file", '{"content":"changed"}'),)),
            ModelReply("", (ToolCall("run", "run_experiment", "{}"),)),
            ModelReply("Measured draft."),
            ModelReply("Reviewed measured interpretation."),
        ]
    )
    result = run_session(
        model,
        FakeTools(
            [
                {"ok": True, "result": {"written": True}},
                {"ok": True, "result": {"status": "succeeded", "full_score": 1}},
            ]
        ),
        json.dumps({"candidate_runtime_contract": contract}),
        RunBudget(1000, 32000, 10),
        FakeClock(),
    )
    assert result.reason == "model_finished"
    for messages in model.messages[1:]:
        assert json.loads(messages[1]["content"])["candidate_runtime_contract"] == contract
        assert "candidate_runtime_contract" in messages[0]["content"]
