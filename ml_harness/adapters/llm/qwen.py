"""Bounded OpenAI-compatible transport for the explicitly supplied Qwen model."""

import json
import math
import time
from typing import Any, cast
from urllib.parse import urlsplit

from openai import NOT_GIVEN, APIConnectionError, APIError, APIStatusError, OpenAI
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam

from ml_harness.application.ports.language_model import ModelError, ModelReply, ToolCall


class QwenLanguageModel:
    """Keep SDK retries disabled so the harness owns the actual attempt bound."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        timeout_seconds: float = 60,
        max_attempts: int = 3,
        client: OpenAI | None = None,
    ) -> None:
        try:
            url = urlsplit(base_url)
            valid_url = (
                url.scheme in ("https", "http")
                and bool(url.hostname)
                and not url.username
                and not url.password
                and not url.query
                and not url.fragment
            )
        except ValueError:
            valid_url = False
        if (
            not api_key.strip()
            or not model.strip()
            or "qwen" not in model.lower()
            or not valid_url
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
            or isinstance(max_attempts, bool)
            or not isinstance(max_attempts, int)
            or max_attempts < 1
        ):
            raise ModelError("Invalid Qwen configuration", attempts=0)
        self.model = model
        self.max_attempts = max_attempts
        self.timeout_seconds = timeout_seconds
        self._limits: tuple[float, int, int] | None = None
        self.client = (
            client.with_options(max_retries=0, timeout=timeout_seconds)
            if client is not None
            else OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_seconds, max_retries=0)
        )

    def set_limits(
        self, remaining_seconds: float, remaining_attempts: int, remaining_tokens: int
    ) -> None:
        if (
            not math.isfinite(remaining_seconds)
            or isinstance(remaining_attempts, bool)
            or not isinstance(remaining_attempts, int)
            or remaining_attempts < 0
            or isinstance(remaining_tokens, bool)
            or not isinstance(remaining_tokens, int)
            or remaining_tokens < 0
        ):
            raise ModelError("Invalid Qwen request budget", attempts=0)
        self._limits = (
            time.monotonic() + max(0, remaining_seconds),
            remaining_attempts,
            remaining_tokens,
        )

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> ModelReply:
        deadline = math.inf
        attempts_allowed = self.max_attempts
        max_tokens: Any = NOT_GIVEN
        if self._limits is not None:
            deadline, remaining_attempts, remaining_tokens = self._limits
            attempts_allowed = min(attempts_allowed, remaining_attempts)
            try:
                prompt = json.dumps({"messages": messages, "tools": tools}, ensure_ascii=False)
                prompt_bound = len(prompt.encode("utf-8")) + 1024
                # Reserve a complete prompt and at least one output token for every retry.
                # A later successful response cannot reveal tokens spent by failed requests.
                attempts_allowed = min(attempts_allowed, remaining_tokens // (prompt_bound + 1))
                if attempts_allowed <= 0:
                    raise ModelError("Qwen request budget exhausted", attempts=0)
                max_tokens = min(2048, remaining_tokens // attempts_allowed - prompt_bound)
            except (ValueError, TypeError, UnicodeError):
                raise ModelError("Cannot serialize Qwen request", attempts=0) from None
            if max_tokens <= 0 or attempts_allowed <= 0 or time.monotonic() >= deadline:
                raise ModelError("Qwen request budget exhausted", attempts=0)
        for attempt in range(1, attempts_allowed + 1):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ModelError("Qwen request deadline reached", attempts=attempt - 1)
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=cast(list[ChatCompletionMessageParam], messages),
                    tools=cast(list[ChatCompletionToolParam], tools) if tools else NOT_GIVEN,
                    max_tokens=max_tokens,
                    timeout=min(self.timeout_seconds, remaining)
                    if self._limits is not None
                    else NOT_GIVEN,
                )
                message = response.choices[0].message
                calls = tuple(
                    ToolCall(call.id, call.function.name, call.function.arguments)
                    for call in (message.tool_calls or [])
                )
                usage = response.usage
                input_count = getattr(usage, "prompt_tokens", None)
                output_count = getattr(usage, "completion_tokens", None)
                valid_input = type(input_count) is int and input_count >= 0
                valid_output = type(output_count) is int and output_count >= 0
                return ModelReply(
                    content=message.content or "",
                    tool_calls=calls,
                    input_tokens=cast(int, input_count) if valid_input else 0,
                    output_tokens=cast(int, output_count) if valid_output else 0,
                    usage_known=valid_input and valid_output and attempt == 1,
                    attempts=attempt,
                )
            except APIError as error:
                retryable = isinstance(error, APIConnectionError) or (
                    isinstance(error, APIStatusError)
                    and (error.status_code == 429 or error.status_code >= 500)
                )
                if not retryable or attempt == attempts_allowed:
                    raise ModelError(
                        f"Qwen request failed after {attempt} attempt(s)", attempts=attempt
                    ) from None
                delay = min(0.25 * 2 ** (attempt - 1), 2.0)
                remaining = deadline - time.monotonic()
                if remaining <= delay:
                    raise ModelError("Qwen request deadline reached", attempts=attempt) from None
                time.sleep(delay)
            except (AttributeError, IndexError, TypeError, ValueError):
                raise ModelError("Malformed Qwen response", attempts=attempt) from None
        raise ModelError("Qwen request exhausted attempts", attempts=0)
