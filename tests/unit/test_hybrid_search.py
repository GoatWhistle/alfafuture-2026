from ml_harness.application.use_cases.hybrid_search import HypothesisOutcome, run_hybrid_search
from ml_harness.domain.budget import WallClockBudget


class Clock:
    time = 0.0

    def now(self) -> float:
        return self.time


def test_hybrid_continues_beyond_three_and_alternates():
    clock = Clock()
    calls = []

    def hypothesis(index):
        calls.append(("qwen", index))
        clock.time += 2
        return HypothesisOutcome(True)

    def numeric(index):
        calls.append(("numeric", index))
        clock.time += 2
        return True

    run_hybrid_search(WallClockBudget(60), clock, hypothesis, numeric, max_hypotheses=5)
    assert [i for action, i in calls if action == "qwen"] == list(range(5))
    assert calls[:8] == [(action, i) for i in range(4) for action in ("qwen", "numeric")]
    assert calls[-1][0] == "numeric"


def test_provider_failure_disables_only_llm_and_still_uses_local_budget():
    clock = Clock()
    calls = []

    def hypothesis(index):
        calls.append("qwen")
        return HypothesisOutcome(False, False, "model_error")

    def numeric(index):
        calls.append("numeric")
        clock.time += 5
        return True

    run_hybrid_search(WallClockBudget(40), clock, hypothesis, numeric)
    assert calls == ["qwen"] + ["numeric"] * 5


def test_three_unimproved_hypotheses_switch_to_local_search():
    clock = Clock()
    calls = []
    decisions = []

    def hypothesis(index):
        calls.append("qwen")
        clock.time += 1
        return HypothesisOutcome(False)

    def numeric(index):
        calls.append("numeric")
        clock.time += 2
        return True

    run_hybrid_search(WallClockBudget(40), clock, hypothesis, numeric, observe=decisions.append)
    assert calls.count("qwen") == 3
    assert any(d.action == "local_only" and d.reason == "stagnation" for d in decisions)


def test_resume_stagnation_keeps_local_only_policy_and_publishes_checkpoints():
    clock = Clock()
    semantic = []
    checkpoints = []

    def numeric(index):
        clock.time += 5
        return True

    run_hybrid_search(
        WallClockBudget(40),
        clock,
        lambda index: semantic.append(index),
        numeric,
        completed=3,
        initial_stale=3,
        checkpoint=lambda: checkpoints.append(clock.time),
    )
    assert semantic == []
    assert checkpoints == [5, 10, 15, 20, 25]


def test_numeric_mode_runs_checkpoint_after_each_batch():
    from ml_harness.application.use_cases.numeric_search import run_numeric_search

    clock = Clock()
    checkpoints = []

    class Workspace:
        def numeric_completed(self):
            return 0

        def numeric_batch(self, backend, index):
            clock.time += 5
            return True

    run_numeric_search(
        Workspace(),
        WallClockBudget(40),
        clock,
        checkpoint=lambda: checkpoints.append(clock.time),
    )
    assert checkpoints == [5, 10, 15, 20, 25]
