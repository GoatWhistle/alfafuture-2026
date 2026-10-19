"""Transport-level checks use the installed SDK and never contact a provider."""

import json
import traceback

import httpx
import pytest
from openai import OpenAI

from ml_harness.adapters.llm.qwen import QwenLanguageModel
from ml_harness.application.ports.language_model import ModelError, ToolCall

SECRET = "sentinel-secret-never-expose"
TOOLS = [
    {
        "type": "function",
        "function": {"name": "inspect_data", "parameters": {"type": "object", "properties": {}}},
    }
]


def completion(usage=True):
    value = {
        "id": "reply",
        "object": "chat.completion",
        "created": 1,
        "model": "Qwen/test",
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "inspect_data", "arguments": "{}"},
                        }
                    ],
                },
            }
        ],
    }
    if usage:
        value["usage"] = {"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16}
    return value


def adapter(handler, **kwargs):
    client = OpenAI(
        api_key=SECRET,
        base_url="https://unused.test/v1",
        max_retries=9,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    return QwenLanguageModel(
        SECRET, "https://unused.test/v1", "Qwen/exact-given", client=client, **kwargs
    )


@pytest.mark.parametrize("usage", [True, False])
def test_tool_calls_usage_exact_request_and_timeout(usage):
    def handler(request):
        body = json.loads(request.content)
        assert body == {
            "model": "Qwen/exact-given",
            "messages": [{"role": "user", "content": "inspect"}],
            "tools": TOOLS,
        }
        assert request.extensions["timeout"]["read"] == 7
        return httpx.Response(200, json=completion(usage))

    model = adapter(handler, timeout_seconds=7)
    assert model.client.max_retries == 0
    reply = model.complete([{"role": "user", "content": "inspect"}], TOOLS)
    assert reply.content == ""
    assert reply.tool_calls == (ToolCall("call-1", "inspect_data", "{}"),)
    assert reply.usage_known is usage
    assert (reply.input_tokens, reply.output_tokens) == ((12, 4) if usage else (0, 0))
    assert reply.attempts == 1


@pytest.mark.parametrize("failure", [429, 500, 503, "timeout", "connection"])
@pytest.mark.parametrize("recover", [False, True])
def test_retry_bound_and_actual_attempts(monkeypatch, failure, recover):
    monkeypatch.setattr("ml_harness.adapters.llm.qwen.time.sleep", lambda seconds: None)
    calls = []

    def handler(request):
        calls.append(request)
        if recover and len(calls) == 3:
            return httpx.Response(200, json=completion())
        if failure == "timeout":
            raise httpx.ReadTimeout(SECRET, request=request)
        if failure == "connection":
            raise httpx.ConnectError(SECRET, request=request)
        return httpx.Response(failure, json={"error": {"message": SECRET}})

    model = adapter(handler, max_attempts=3)
    if recover:
        assert model.complete([], []).attempts == 3
    else:
        with pytest.raises(ModelError) as error:
            model.complete([], [])
        assert SECRET not in str(error.value)
        assert SECRET not in "".join(traceback.format_exception(error.value))
    assert len(calls) == 3
    assert all(json.loads(call.content)["model"] == "Qwen/exact-given" for call in calls)
    assert all("tools" not in json.loads(call.content) for call in calls)


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_permanent_errors_are_not_retried(status):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(status, json={"error": {"message": SECRET}})

    with pytest.raises(ModelError) as error:
        adapter(handler).complete([], [])
    assert len(calls) == 1
    assert SECRET not in str(error.value)


@pytest.mark.parametrize(
    "overrides",
    [
        {"api_key": ""},
        {"base_url": ""},
        {"base_url": "ftp://example.test"},
        {"base_url": "https://user:secret@example.test"},
        {"model": "other-model"},
        {"timeout_seconds": 0},
        {"timeout_seconds": float("nan")},
        {"max_attempts": 0},
        {"max_attempts": True},
        {"max_attempts": 1.5},
    ],
)
def test_invalid_configuration(overrides):
    config = dict(api_key=SECRET, base_url="https://example.test/v1", model="Qwen/given")
    config.update(overrides)
    with pytest.raises(ModelError, match="Invalid Qwen configuration") as error:
        QwenLanguageModel(**config)
    assert error.value.attempts == 0


def test_malformed_reply_is_sanitized():
    with pytest.raises(ModelError, match="Malformed Qwen response"):
        adapter(lambda request: httpx.Response(200, json={"choices": []})).complete([], [])
