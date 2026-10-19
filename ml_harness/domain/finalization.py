"""Frozen candidate identities and comparable finalist rankings, without I/O."""

from dataclasses import dataclass
from typing import Any

from ml_harness.domain.models import Score


@dataclass(frozen=True)
class CandidateRef:
    id: str
    candidate_path: str | None
    code_hash: str | None
    backend: str = "catboost"
    parameters: dict[str, Any] | None = None


@dataclass(frozen=True)
class Finalist:
    id: str
    members: tuple[CandidateRef, ...]
    score: Score
    weights: tuple[float, ...] = (1.0,)


def rank_finalists(candidates: list[Finalist], limit: int = 3) -> tuple[Finalist, ...]:
    ranked: list[Finalist] = []
    for candidate in candidates:
        position = next(
            (
                index
                for index, previous in enumerate(ranked)
                if candidate.score.better_than(previous.score)
            ),
            len(ranked),
        )
        ranked.insert(position, candidate)
    return tuple(ranked[:limit])
