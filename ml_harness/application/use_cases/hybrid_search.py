"""Budget-driven research policy shared by the CLI and scripted executions."""

from collections.abc import Callable
from dataclasses import dataclass

from ml_harness.application.ports.clock import Clock
from ml_harness.domain.budget import WallClockBudget


@dataclass(frozen=True)
class SearchDecision:
    action: str
    reason: str
    hypothesis_index: int


@dataclass(frozen=True)
class HypothesisOutcome:
    improved: bool
    llm_available: bool = True
    reason: str = "experiment_completed"


def run_hybrid_search(
    wall: WallClockBudget,
    clock: Clock,
    hypothesis: Callable[[int], HypothesisOutcome],
    numeric: Callable[[int], bool],
    *,
    max_hypotheses: int = 12,
    completed: int = 0,
    numeric_completed: int = 0,
    initial_stale: int = 0,
    llm_available: bool = True,
    local_reason: str = "qwen_unavailable",
    observe: Callable[[SearchDecision], None] = lambda decision: None,
    checkpoint: Callable[[], None] = lambda: None,
) -> str:
    """Alternate semantic and numeric steps; provider failures disable only semantic work."""
    stale, numeric_failures = initial_stale, 0
    index, batch = completed, numeric_completed
    llm_available = llm_available and stale < 3 and index < min(12, max_hypotheses)
    if not llm_available:
        observe(SearchDecision("local_only", "stagnation" if stale >= 3 else local_reason, index))
    while wall.available(clock.now()) > 1:
        if llm_available:
            observe(SearchDecision("qwen", "semantic_hypothesis", index))
            outcome = hypothesis(index)
            checkpoint()
            index += 1
            stale = 0 if outcome.improved else stale + 1
            llm_available = outcome.llm_available and stale < 3 and index < min(12, max_hypotheses)
            if not llm_available:
                observe(
                    SearchDecision(
                        "local_only", outcome.reason if stale < 3 else "stagnation", index
                    )
                )
        if wall.available(clock.now()) <= 1:
            break
        observe(SearchDecision("numeric", "continue_best_pipeline", index))
        succeeded = numeric(batch)
        checkpoint()
        batch += 1
        numeric_failures = 0 if succeeded else numeric_failures + 1
        if numeric_failures >= 3:
            return "numeric_error_limit"
    return "budget_exhausted"
