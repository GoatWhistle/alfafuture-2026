"""Durable experiment facts and immutable candidate versions."""

from typing import Any, Protocol


class ExperimentStore(Protocol):
    def append(self, event: dict[str, Any]) -> None: ...

    def read_events(self) -> list[dict[str, Any]]: ...

    def save_code(self, code: bytes) -> tuple[str, str]: ...
