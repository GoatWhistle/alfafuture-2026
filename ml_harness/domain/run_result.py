"""Versioned facts consumed by presentation surfaces independently of search success."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RunResult:
    version: int = 3
    result_status: str = "unavailable"
    search_status: str = "not_started"
    selected: str = "baseline"
    recipe: dict[str, Any] | None = None
    origin: str | None = None
    score: dict[str, Any] | None = None
    protocol: str | None = None
    stop_reason: str | None = None
    submission_path: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    resources: dict[str, Any] = field(default_factory=dict)
    rejections: tuple[dict[str, Any], ...] = ()
