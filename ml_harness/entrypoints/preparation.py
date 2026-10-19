"""Defer Qwen configuration until a verified task ambiguity actually needs a request."""

import os
from typing import Any

from ml_harness.adapters.llm.qwen import QwenLanguageModel
from ml_harness.application.ports.language_model import ModelError, ModelReply


class ConfiguredTaskModel:
    def __init__(self) -> None:
        self.model: QwenLanguageModel | None = None
        self.limits = (0.0, 0, 0)

    def set_limits(
        self, remaining_seconds: float, remaining_attempts: int, remaining_tokens: int
    ) -> None:
        self.limits = (remaining_seconds, remaining_attempts, remaining_tokens)

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> ModelReply:
        if self.model is None:
            names = ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL")
            if not all(os.environ.get(name) for name in names):
                raise ModelError(
                    "Task clarification requires issued Qwen configuration", attempts=0
                )
            self.model = QwenLanguageModel(
                os.environ[names[0]], os.environ[names[1]], os.environ[names[2]]
            )
        self.model.set_limits(*self.limits)
        return self.model.complete(messages, tools)
