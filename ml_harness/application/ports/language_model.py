"""Provider-independent messages, tool calls and usage."""

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


class ModelError(RuntimeError):
    """A bounded model request failed; previously saved results remain usable."""

    def __init__(self, message: str, *, attempts: int = 1) -> None:
        super().__init__(message)
        self.attempts = attempts


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class ModelReply:
    content: str
    tool_calls: tuple[ToolCall, ...] = ()
    input_tokens: int = 0
    output_tokens: int = 0
    usage_known: bool = True
    attempts: int = 1


class LanguageModel(Protocol):
    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> ModelReply: ...


@runtime_checkable
class BudgetAwareLanguageModel(Protocol):
    def set_limits(
        self, remaining_seconds: float, remaining_attempts: int, remaining_tokens: int
    ) -> None: ...


class ToolExecutor(Protocol):
    @property
    def schemas(self) -> list[dict[str, Any]]: ...

    def execute(self, call: ToolCall) -> dict[str, Any]: ...
