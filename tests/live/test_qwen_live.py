"""Explicit opt-in provider probe; offline test collection never sends requests."""

import os

import pytest
from openai import OpenAI


@pytest.mark.live
def test_supplied_qwen_forced_tool_and_usage():
    if os.environ.get("RUN_QWEN_LIVE") != "1":
        pytest.skip("Set RUN_QWEN_LIVE=1 for the explicit provider check")
    names = ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL")
    if not all(os.environ.get(name) for name in names):
        pytest.skip("Explicit Qwen connection configuration is missing")
    model = os.environ["OPENAI_MODEL"]
    assert "qwen" in model.lower(), "Only the supplied Qwen model is permitted"
    with OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_BASE_URL"],
        timeout=30,
        max_retries=0,
    ) as client:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Call ping with no arguments."}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "ping",
                        "parameters": {"type": "object", "properties": {}, "required": []},
                    },
                }
            ],
            tool_choice={"type": "function", "function": {"name": "ping"}},
        )
    calls = response.choices[0].message.tool_calls
    assert calls and calls[0].function.name == "ping"
    assert response.usage is not None
    assert response.usage.prompt_tokens > 0 and response.usage.completion_tokens > 0
