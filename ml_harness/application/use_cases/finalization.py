"""One-way finalist ranking and control selection independent of persistence."""

from ml_harness.domain.finalization import CandidateRef, Finalist, rank_finalists
from ml_harness.domain.models import Score

__all__ = ["CandidateRef", "Finalist", "rank_finalists", "select_finalists"]


def select_finalists(finalists: tuple[Finalist, ...], scores: dict[str, Score]) -> tuple[str, ...]:
    """Rank only finalists with measured scores; never resume hypothesis search."""
    checked = [
        Finalist(item.id, item.members, scores[item.id], item.weights)
        for item in finalists
        if item.id in scores
    ]
    return tuple(item.id for item in rank_finalists(checked, len(checked)))
