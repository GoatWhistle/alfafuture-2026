"""Критерии приёмки #9, #11, #12, #13, #14, #15, #16, #17, #18, не покрытые другими тестами."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.pop("OPENAI_API_KEY", None)

import fe_dsl  # noqa: E402
import models  # noqa: E402
import pool  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_pool import _prof, cls_frames, reg_frames  # noqa: E402


# --- #9: Tweedie / двухчастная модель ---------------------------------------

def test_twopart_not_planned_for_all_zero_target():
    tr, te = reg_frames()
    tr["y"] = 0.0
    names = [r["name"] for r in pool.plan(_prof(tr, te, "y", "regression", "rmse"), "y")]
    assert "twopart" not in names and "lgbm_tweedie" not in names


def test_twopart_event_probability_matches_rate():
    tr, te = reg_frames(n=2000, zeros=0.4, seed=3)
    fd = pool.FoldData(_prof(tr, te, "y", "regression", "rmse"), "y")
    fam = pool.TwoPartFamily(fd, {})
    y = fd.y_orig.astype(float)
    Xall, _ = fd.full()
    c, _ = fam.refit(Xall, y, None, (100, 100), 1e18)
    p = c.predict_proba(Xall)[:, list(c.classes_).index(1)]
    assert abs(p.mean() - (y > 0).mean()) < 0.03       # вероятность события согласована с долей в train


def test_twopart_skips_fold_without_positives_and_pool_continues(monkeypatch):
    tr, te = reg_frames(n=600, zeros=0.97, seed=1)      # почти все нули: в фолде < 50 положительных
    prof = _prof(tr, te, "y", "regression", "rmse")
    journal = []
    monkeypatch.setattr(pool, "plan", lambda p, t: [{"name": "twopart", "family": "twopart", "why": "w"},
                                                    {"name": "lgbm", "family": "lgbm", "why": "b"}])
    monkeypatch.setattr(pool, "SEARCH_TRIALS", 0)
    monkeypatch.setattr(pool, "EXTRA_SEEDS", 0)
    out = pool.run_pool(prof, "y", 1e18, 1.0, register=lambda *a, **k: journal.append((a, k)))
    assert [c["name"] for c in out] == ["lgbm"]
    assert any(a[0] == "pool:twopart" and "error" in k for a, k in journal)


# --- #14: преобразования таргета --------------------------------------------

def test_log_scale_win_can_lose_on_rmse():
    """y=[0, 99]: среднее даёт RMSE 49.5, обратное среднее log1p — около 63.957: выбор — по RMSE в исходной шкале."""
    y = np.array([0.0, 99.0])
    mean_pred = np.full(2, y.mean())
    log_pred = np.full(2, pool._inverse(np.log1p(y).mean(), "log1p"))
    assert models.score("rmse", y, mean_pred) == pytest.approx(49.5)
    assert models.score("rmse", y, log_pred) == pytest.approx(63.957, abs=1e-3)


def test_inverse_guards_overflow_and_negative_sqrt():
    assert np.isfinite(pool._inverse(np.array([1e6]), "log1p")).all()
    assert (pool._inverse(np.array([-3.0, 2.0]), "sqrt") == np.array([0.0, 4.0])).all()


# --- #11: остановка только по метрике задачи ---------------------------------

def test_f1_stop_has_no_hidden_logloss_condition():
    tr, te = cls_frames()
    fd = pool.FoldData(_prof(tr, te, "t", "multiclass", "f1_macro"), "t")
    fam = pool.LGBMFamily(fd, {"f1_stop": True})
    fit_i, es_i, va, Xf, Xes, Xva = fd.fold(0)
    pos = {c: i for i, c in enumerate(fd.classes)}
    yi = np.array([pos[v] for v in fd.y_orig])
    keep = np.isin(yi[es_i], np.unique(yi[fit_i]))
    m, _ = fam.fit(Xf, yi[fit_i], Xes[keep], yi[es_i][keep], None, None, 1e18)
    assert list(m.evals_result_["valid_0"]) == ["task_f1"]


# --- #12: параметры каталога реально доходят до модели ------------------------

def test_catalog_params_reach_model_and_row_sampling_changes_predictions():
    tr, te = reg_frames()
    fd = pool.FoldData(_prof(tr, te, "y", "regression", "rmse"), "y")
    Xall, Xt = fd.full()
    y = fd.y_orig.astype(float)
    p_on = {"subsample": 0.6, "subsample_freq": 1}
    m_on = pool.LGBMFamily(fd, {"params": p_on}).refit(Xall, y, None, 100, 1e18)
    m_off = pool.LGBMFamily(fd, {"params": {"subsample": 0.6, "subsample_freq": 0}}).refit(Xall, y, None, 100, 1e18)
    assert m_on.get_params()["subsample"] == 0.6 and m_on.get_params()["subsample_freq"] == 1
    m_full = pool.LGBMFamily(fd, {"params": {"subsample": 1.0, "subsample_freq": 0}}).refit(Xall, y, None, 100, 1e18)
    assert np.allclose(m_off.predict(Xt), m_full.predict(Xt))     # без частоты сэмплирование не включается
    assert not np.allclose(m_on.predict(Xt), m_full.predict(Xt))


# --- #13: выравнивание кандидатов перед ансамблем -----------------------------

def test_ensemble_after_row_and_class_permutation():
    import solution

    rng = np.random.RandomState(0)
    n = 400
    classes = ["a", "b", "c"]
    y = rng.choice(classes, n)
    rows = np.arange(n)
    p1 = rng.dirichlet(np.ones(3), n)
    perm = rng.permutation(n)
    c1 = {"name": "m1", "rows": rows, "y": y, "oof": p1, "test": p1[:10], "classes": classes, "score": 0.5}
    # тот же прогноз, строки и классы переставлены
    c2 = {"name": "m2", "rows": rows[perm], "y": y[perm], "oof": p1[perm][:, [2, 0, 1]],
          "test": p1[:10][:, [2, 0, 1]], "classes": ["c", "a", "b"], "score": 0.5}
    al = solution.align_candidates([c1, c2])
    assert np.allclose(al[0]["oof"], al[1]["oof"]) and np.allclose(al[0]["test"], al[1]["test"])
    bad = {**c2, "classes": ["c", "a", "zzz"], "name": "bad"}
    assert [c["name"] for c in solution.align_candidates([c1, bad])] == ["m1"]   # несовместимый исключён


# --- #15: новые категории и строковые метки в XGBoost/ExtraTrees --------------

@pytest.mark.parametrize("family", ["xgboost", "extratrees"])
def test_new_category_in_test(family):
    tr, te = cls_frames()
    te = te.copy()
    te.loc[:5, "c"] = "unseen_level"
    fd = pool.FoldData(_prof(tr, te, "t", "multiclass", "f1_macro"), "t")
    c = pool.run_candidate(fd, {"name": family, "family": family}, 1e18)
    assert c is not None and np.isfinite(c["test"]).all() and len(c["test"]) == len(te)


# --- #16: диагностика сдвига ---------------------------------------------------

def _shift_ctx(train, test):
    feats = [c for c in train.columns if c not in ("id", "y")]
    return {"df": train, "test": test, "features": feats, "period": None, "facts": {}}


def test_shift_detects_known_shift_and_ignores_labels():
    import eda_detectors

    rng = np.random.RandomState(0)
    tr = pd.DataFrame({"x": rng.randn(3000), "z": rng.randn(3000), "y": rng.randn(3000)})
    te = pd.DataFrame({"x": rng.randn(3000) + 1.5, "z": rng.randn(3000)})
    ctx = _shift_ctx(tr, te)
    eda_detectors.detect_shift(ctx)
    auc = ctx["facts"]["adversarial"]["auc"]
    assert auc > 0.8 and ctx["facts"]["adversarial"]["top"][0][0] == "x"
    ctx2 = _shift_ctx(tr.assign(y=rng.randn(3000) * 100), te)      # другие (недоступные) метки
    eda_detectors.detect_shift(ctx2)
    assert ctx2["facts"]["adversarial"]["auc"] == auc


def test_shift_same_distribution_separate_ids():
    import eda_detectors

    rng = np.random.RandomState(1)
    tr = pd.DataFrame({"x": rng.randn(3000), "z": rng.randn(3000)})
    te = pd.DataFrame({"x": rng.randn(3000), "z": rng.randn(3000)})
    ctx = _shift_ctx(tr, te)
    out = eda_detectors.detect_shift(ctx)
    assert ctx["facts"]["adversarial"]["auc"] < 0.6 and out == []


# --- #17: история объекта не видит будущих меток ------------------------------

def test_history_does_not_see_future_labels():
    rng = np.random.RandomState(0)
    rows = [{"obj": o, "year": t, "y": float(rng.rand())} for t in range(1, 5) for o in range(50)]
    fit = pd.DataFrame(rows)
    spec = fe_dsl.validate({"op": "lag", "cols": [], "params": {"keys": ["obj"], "period": "year",
                                                                "aggs": ["last", "mean", "count"]}},
                           fit.columns, {"year"})
    cv = {"kind": "temporal", "period": "year"}
    f1, _ = fe_dsl.apply(spec, fit, fit["y"].to_numpy(), fit.iloc[:5], None, cv)
    y2 = fit["y"].to_numpy().copy()
    y2[fit["year"].to_numpy() == 4] += 100                      # меняются только метки последнего года
    f2, _ = fe_dsl.apply(spec, fit, y2, fit.iloc[:5], None, cv)
    early = (fit["year"] <= 4).to_numpy()                       # история любой строки — строго прошлые годы
    assert np.allclose(f1[early].fillna(-1), f2[early].fillna(-1))


# --- #18: пустой словарь ------------------------------------------------------

def test_tfidf_empty_vocabulary_fails_cleanly(monkeypatch):
    tr, te = cls_frames()
    tr["txt"] = ""
    te["txt"] = ""
    prof = _prof(tr, te, "t", "multiclass", "f1_macro")
    journal = []
    monkeypatch.setattr(pool, "plan", lambda p, t: [
        {"name": "tfidf_linear", "family": "tfidf", "text_cols": ["txt"], "why": "w"},
        {"name": "lgbm", "family": "lgbm", "why": "b"}])
    monkeypatch.setattr(pool, "SEARCH_TRIALS", 0)
    monkeypatch.setattr(pool, "EXTRA_SEEDS", 0)
    out = pool.run_pool(prof, "t", 1e18, 1.0, register=lambda *a, **k: journal.append((a, k)))
    assert [c["name"] for c in out] == ["lgbm"]                 # сбой текста не мешает остальным
    assert any(a[0] == "pool:tfidf_linear" and "error" in k for a, k in journal)
