"""Bounded OpenAI-compatible transport for the explicitly supplied Qwen model."""

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
            raise ModelError("Invalid Qwen configuration")
        self.model = model
        self.max_attempts = max_attempts
        self.client = (
            client.with_options(max_retries=0, timeout=timeout_seconds)
            if client is not None
            else OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_seconds, max_retries=0)
        )

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> ModelReply:
        for attempt in range(1, self.max_attempts + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=cast(list[ChatCompletionMessageParam], messages),
                    tools=cast(list[ChatCompletionToolParam], tools) if tools else NOT_GIVEN,
                )
                message = response.choices[0].message
                calls = tuple(
                    ToolCall(call.id, call.function.name, call.function.arguments)
                    for call in (message.tool_calls or [])
                )
                usage = response.usage
                known = (
                    usage is not None
                    and isinstance(usage.prompt_tokens, int)
                    and isinstance(usage.completion_tokens, int)
                )
                return ModelReply(
                    content=message.content or "",
                    tool_calls=calls,
                    input_tokens=usage.prompt_tokens if known and usage is not None else 0,
                    output_tokens=usage.completion_tokens if known and usage is not None else 0,
                    usage_known=known,
                    attempts=attempt,
                )
            except APIError as error:
                retryable = isinstance(error, APIConnectionError) or (
                    isinstance(error, APIStatusError)
                    and (error.status_code == 429 or error.status_code >= 500)
                )
                if not retryable or attempt == self.max_attempts:
                    raise ModelError(f"Qwen request failed after {attempt} attempt(s)") from None
                time.sleep(min(0.25 * 2 ** (attempt - 1), 2.0))
            except (AttributeError, IndexError, TypeError, ValueError):
                raise ModelError("Malformed Qwen response") from None
        raise ModelError("Qwen request exhausted attempts")
