"""Adaptive research must develop alternatives without leaking validation labels."""
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fe_dsl
import research_search as rs
from agent_utils import StageTimeout


def context(n=100):
    rng = np.random.RandomState(7)
    df = pd.DataFrame(dict(a=rng.randn(n), b=rng.randn(n), y=rng.randn(n)))
    return dict(df=df, y=df.y.to_numpy(), features=["a", "b"], classes=None,
                folds=[(np.arange(n // 2), np.arange(n // 2, n))], facts={},
                metric="rmse", ptype="regression", prof={"log_target": False},
                target="y", cv={"kind": "kfold"}, fold_scheme="kfold",
                deadline=time.time() + 30, importances={"a": .5, "b": .5}, period=None, entity=None)


def spec(col):
    return fe_dsl.validate(dict(op="log", cols=[col], params={}, prior=1., source="test"), ["a", "b"])


class FakeEvaluator:
    last_seconds = .001

    def __init__(self, ctx, scores):
        self.ctx, self.scores = ctx, scores
        self.calls = []

    def fits(self, deadline):
        return time.time() < deadline

    def evaluate(self, specs, deadline, screen=False):
        key = tuple(sorted(s["cols"][0] for s in specs))
        self.calls.append((key, screen))
        err = self.scores[key]
        if isinstance(err, BaseException):
            raise err
        y = self.ctx["y"]
        return dict(scores=[err] if screen else [err, err], rows=np.arange(len(y)),
                    oof=y - err, seconds=.001)


def test_losing_alternative_can_develop_into_winner_and_refresh_diagnostics(tmp_path):
    ctx = context()
    ctx["prof"]["workdir"] = tmp_path
    ev = FakeEvaluator(ctx, {(): 10., ("a",): 10.2, ("b",): 10.1, ("a", "b"): 7.})
    observed = []

    def propose(diag, deadline):
        observed.append((diag["facts"]["current_recipe"]["id"], float(diag["resid"].mean())))
        return [spec("a"), spec("b")]

    search = rs.Research(ctx, evaluator=ev, proposer=propose, batch_size=4)
    winner = search.run([spec("a"), spec("b")], time.time() + 10)
    assert {s["cols"][0] for s in winner} == {"a", "b"}
    assert any(parent != search.baseline and err > 10 for parent, err in observed)
    assert any(parent == search.champion and err == pytest.approx(7) for parent, err in observed)
    assert len(search.portfolio) <= 4
    assert search.stop_reason == "no_new_experiments"
    # Canonical recipe identities prevent repeating full training across parent branches.
    calls = [key for key, screen in ev.calls if not screen]
    assert len(calls) == len(set(calls)) == 4
    report = json.loads(next((tmp_path / "research").glob('*.json')).read_text())
    assert report["champion"] == search.champion
    assert report["history"][-1]["specs"] == winner
    assert all("oof" not in c for c in report["portfolio"])


def test_deadline_keeps_completed_winner():
    ctx = context()
    ev = FakeEvaluator(ctx, {(): 10., ("a",): 8., ("b",): StageTimeout(), ("a", "b"): StageTimeout()})
    search = rs.Research(ctx, evaluator=ev, proposer=lambda *a: [])
    winner = search.run([spec("a"), spec("b")], time.time() + 10)
    assert winner == [spec("a")]
    assert search.stop_reason == "budget_exhausted"
    assert ctx["search_final"] == 8


def test_one_fold_screen_cannot_promote_without_full_validation():
    ctx = context()
    ev = FakeEvaluator(ctx, {(): 10., ("a",): 10.1})
    original = ev.evaluate

    def misleading(specs, deadline, screen=False):
        result = original(specs, deadline, screen)
        if screen:
            result["scores"] = [.1]
        return result

    ev.evaluate = misleading
    search = rs.Research(ctx, evaluator=ev, proposer=lambda *a: [], max_rounds=3)
    assert search.run([spec("a")], time.time() + 10) == []
    assert (("a",), False) in ev.calls


def test_null_calibration_rejects_gains_below_measured_noise():
    ctx = context()

    class NoisyEvaluator(FakeEvaluator):
        def null_feats(self, col, seed):
            return "noise"

        def evaluate(self, specs, deadline, screen=False, extra=None):
            if extra is not None:
                return dict(scores=[11., 11.], rows=np.arange(len(ctx["y"])), oof=ctx["y"]-11., seconds=.001)
            return super().evaluate(specs, deadline, screen)

    ev = NoisyEvaluator(ctx, {(): 10., ("a",): 9.5})
    search = rs.Research(ctx, evaluator=ev, proposer=lambda *a: [], max_rounds=3)
    assert search.run([spec("a")], time.time() + 10) == []
    assert search.rel_min == pytest.approx(.1)


def test_only_admissible_oof_rows_enter_diagnostics():
    ctx = context()
    rows = np.array([80, 90, 99])
    candidate = dict(id="temporal_recipe", rows=rows, oof=ctx["y"][rows] - 2, specs=[])
    ctx["facts"].update(tail={"old": True}, segments=["old"], perm_importance=["old"], temporal_candidates=["period"])
    diag = rs.diagnostic_context(ctx, candidate)
    pd.testing.assert_frame_equal(diag["df"], ctx["df"].iloc[rows].reset_index(drop=True))
    np.testing.assert_allclose(diag["resid"], 2)
    assert all(k not in diag["facts"] for k in ("tail", "segments", "perm_importance"))
    assert diag["facts"]["temporal_candidates"] == ["period"]
    assert ctx["facts"]["tail"] == {"old": True}


def test_validation_labels_never_reach_model_fit_or_early_stopping(monkeypatch):
    ctx = context(400)
    ctx["y"] = 3 * ctx["df"].a.to_numpy() + ctx["df"].b.to_numpy()
    before = rs.RecipeEvaluator(ctx).evaluate([], time.time() + 30)
    changed = {**ctx, "y": ctx["y"].copy()}
    changed["y"][ctx["folds"][0][1]] += 1000
    after = rs.RecipeEvaluator(changed).evaluate([], time.time() + 30)
    np.testing.assert_array_equal(before["oof"], after["oof"])
    assert after["scores"][0] > before["scores"][0] * 100


def test_classification_diagnostics_use_aligned_probability_columns():
    ctx = context(4)
    ctx.update(classes=["a", "b"], y=np.array(["a", "b", "a", "b"]), ptype="binary", metric="log_loss")
    candidate = dict(id="c", rows=np.array([1, 2]), specs=[], oof=np.array([[.2, .8], [.7, .3]]))
    diag = rs.diagnostic_context(ctx, candidate)
    np.testing.assert_allclose(diag["resid"], [.2, .3])
    np.testing.assert_array_equal(diag["y_idx"], [1, 0])


def test_model_operation_is_replaced_instead_of_conflicting():
    a = dict(op="objective", cols=[], params={"name": "huber", "alpha": 1})
    b = dict(op="objective", cols=[], params={"name": "fair", "alpha": 2})
    result = rs.extend([a, spec("a")], b)
    assert len(result) == 2 and a not in result and b in result


def test_repeated_llm_suggestions_do_not_repeat_training():
    ctx = context()
    ev = FakeEvaluator(ctx, {(): 10., ("a",): 10.1})
    calls = []

    def llm(diag, feedback, planned, seconds):
        calls.append(feedback)
        return [spec("a")]

    search = rs.Research(ctx, evaluator=ev, proposer=lambda *a: [], llm=llm, max_rounds=20)
    search.run([], time.time() + 30)
    assert 0 < len(calls) <= 8
    assert [c for c in ev.calls if c == (("a",), False)] == [(("a",), False)]
    assert calls[-1]["recent_trials"]


def test_fresh_error_facts_precede_long_history_in_llm_prompt(monkeypatch):
    import task_profile
    prompts = []
    monkeypatch.setattr(task_profile, "_llm_json", lambda prompt, *a, **kw: prompts.append(prompt))
    task_profile.llm_research_specs("task", {"current_recipe": {}, "tail": {"fresh_error": 17}},
                                    {"very_long_history": "x" * 10000}, ["a"], [], 100)
    assert '"fresh_error": 17' in prompts[0]
    assert prompts[0].index('"fresh_error"') < prompts[0].index('"very_long_history"')


def test_provider_reasoning_budget_and_token_limit(monkeypatch):
    from types import SimpleNamespace as NS
    import task_profile
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return NS(usage=None, choices=[NS(message=NS(content='{"specs": []}'), finish_reason="stop")])

    monkeypatch.setitem(sys.modules, "openai", NS(OpenAI=lambda **kw: NS(chat=NS(completions=NS(create=create)))))
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AGENT_LLM_MAX_TOKENS", "8192")
    monkeypatch.setenv("AGENT_LLM_REASONING_EFFORT", "low")
    monkeypatch.setenv("AGENT_LLM_REASONING_TOKENS", "1024")
    assert task_profile._llm_json_call("prompt", "test", 1, 0) == {"specs": []}
    assert calls[0]["extra_body"] == {"reasoning": {"max_tokens": 1024}}
    assert calls[0]["max_tokens"] == 8192
