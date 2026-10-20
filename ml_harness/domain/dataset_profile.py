"""Dataset facts available to research, without dataframe or file dependencies."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DatasetProfile:
    fit_partition: str
    columns: dict[str, dict[str, Any]]
    target: dict[str, Any]
    duplicate_features: list[list[str]]
    schema: dict[str, list[str]]
