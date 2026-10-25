"""Пул рецептов (#9, #11, #13, #14, #15, #20) и калибровка log loss (#10)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.pop("OPENAI_API_KEY", None)

import models  # noqa: E402
import pool  # noqa: E402


def _prof(train, test, target, ptype, metric):
    feats = [c for c in train.columns if c not in ("id", target)]
    return {"train": train, "test": test, "features": feats, "problem_type": ptype, "metric": metric,
            "log_target": metric == "rmsle", "target_specs": [], "cv": {"kind": "kfold"}}


def reg_frames(n=900, zeros=0.0, seed=0):
    rng = np.random.RandomState(seed)
    x = rng.randn(n + 200, 3)
    y = np.exp(1 + x[:, 0] + 0.3 * rng.randn(n + 200))
    y[rng.rand(n + 200) < zeros] = 0.0
    cat = np.where(x[:, 1] > 0, "a", "b")
    df = pd.DataFrame({"id": np.arange(n + 200), "x0": x[:, 0], "x1": x[:, 1], "x2": x[:, 2], "c": cat, "y": y})
    return df.iloc[:n].reset_index(drop=True), df.iloc[n:].drop(columns="y").reset_index(drop=True)


def cls_frames(n=900, seed=0):
    rng = np.random.RandomState(seed)
    x = rng.randn(n + 200, 3)
    lab = np.where(x[:, 0] > 0.8, "hi", np.where(x[:, 0] < -0.8, "lo", "mid"))
    lab[:12] = "rare"  # редкий класс
    df = pd.DataFrame({"id": np.arange(n + 200), "x0": x[:, 0], "x1": x[:, 1], "x2": x[:, 2],
                       "c": np.where(x[:, 1] > 0, "p", "q"), "t": lab})
    return df.iloc[:n].reset_index(drop=True), df.iloc[n:].drop(columns="t").reset_index(drop=True)


@pytest.mark.parametrize("family", ["lgbm", "catboost", "xgboost", "extratrees"])
def test_families_regression(family):
    tr, te = reg_frames()
    fd = pool.FoldData(_prof(tr, te, "y", "regression", "rmse"), "y")
    c = pool.run_candidate(fd, {"name": family, "family": family}, deadline_ts=1e18)
    assert c is not None and np.isfinite(c["oof"]).all() and len(c["test"]) == len(te)
    assert c["score"] == pytest.approx(models.score("rmse", c["y"], c["oof"]))


@pytest.mark.parametrize("family", ["lgbm", "catboost", "xgboost", "extratrees"])
def test_families_multiclass_string_labels(family):
    tr, te = cls_frames()
    fd = pool.FoldData(_prof(tr, te, "t", "multiclass", "f1_macro"), "t")
    c = pool.run_candidate(fd, {"name": family, "family": family}, deadline_ts=1e18)
    assert c is not None
    assert c["classes"] == sorted(tr["t"].unique().tolist())
    assert c["oof"].shape[1] == 4 and np.allclose(c["test"].sum(axis=1), 1)


def test_f1_stop_variant_runs():
    tr, te = cls_frames()
    fd = pool.FoldData(_prof(tr, te, "t", "multiclass", "f1_macro"), "t")
    c = pool.run_candidate(fd, {"name": "f1", "family": "lgbm", "f1_stop": True}, deadline_ts=1e18)
    assert c is not None and 0 < c["score"] <= 1


def test_plan_applicability():
    tr, te = reg_frames(zeros=0.3)
    names = [r["name"] for r in pool.plan(_prof(tr, te, "y", "regression", "rmse"), "y")]
    assert {"lgbm_log1p", "lgbm_sqrt", "lgbm_tweedie", "twopart"} <= set(names)
    tr2 = tr.assign(y=tr["y"] - 5)  # отрицательные значения: ни преобразований, ни Tweedie
    names2 = [r["name"] for r in pool.plan(_prof(tr2, te, "y", "regression", "rmse"), "y")]
    assert not {"lgbm_log1p", "lgbm_sqrt", "lgbm_tweedie", "twopart"} & set(names2)
    names3 = [r["name"] for r in pool.plan(_prof(tr, te, "y", "regression", "mae"), "y")]
    assert "lgbm_l1" in names3


def test_twopart_and_transforms_in_original_scale():
    tr, te = reg_frames(zeros=0.3)
    fd = pool.FoldData(_prof(tr, te, "y", "regression", "rmse"), "y")
    for r in pool.plan(fd.prof, "y")[:4]:
        c = pool.run_candidate(fd, r, deadline_ts=1e18)
        assert c is not None, r["name"]
        assert (c["test"] >= 0).all() and np.isfinite(c["test"]).all()
        # обратное преобразование ровно один раз: шкала прогнозов как у таргета
        assert 0.2 < np.mean(c["oof"]) / np.mean(c["y"]) < 5


def test_greedy_picks_model_outside_top2():
    """Третья по одиночному скору модель с противоположной ошибкой улучшает смесь."""
    rng = np.random.RandomState(0)
    y = rng.randn(2000)
    e = rng.randn(2000)
    a = {"name": "a", "oof": y + 0.5 * e + 0.1 * rng.randn(2000)}
    b = {"name": "b", "oof": y + 0.5 * e + 0.12 * rng.randn(2000)}   # почти копия ошибки a
    c = {"name": "c", "oof": y - 0.6 * e + 0.1 * rng.randn(2000)}    # хуже поодиночке, ошибка противоположна
    cands = [a, b, c]
    for x in cands:
        x.update(y=y, classes=None, score=models.score("rmse", y, x["oof"]))
    assert c["score"] > a["score"] and c["score"] > b["score"]
    w, s = pool.greedy_ensemble(cands, "rmse")
    assert w[2] > 0 and s < min(x["score"] for x in cands)


def test_greedy_skips_duplicates():
    rng = np.random.RandomState(1)
    y = rng.randn(500)
    p = y + rng.randn(500)
    cands = [{"name": n, "oof": p.copy(), "y": y, "classes": None, "score": models.score("rmse", y, p)}
             for n in ("a", "b")]
    w, _ = pool.greedy_ensemble(cands, "rmse")
    assert w[1] == 0


def test_calibration_accepts_overconfident_and_keeps_good():
    rng = np.random.RandomState(0)
    n = 4000
    logit = rng.randn(n) * 2
    p_true = 1 / (1 + np.exp(-logit))
    y = np.where(rng.rand(n) < p_true, "yes", "no")
    classes = ["no", "yes"]
    over = 1 / (1 + np.exp(-3 * logit))  # чрезмерно уверенная модель
    proba = np.column_stack([1 - over, over])
    res = models.calibrate_proba(y, proba, proba[:100], classes)
    assert res["test"] is not None and res["new"] < res["base"]
    assert np.allclose(res["test"].sum(axis=1), 1) and ((res["test"] >= 0) & (res["test"] <= 1)).all()
    good = np.column_stack([1 - p_true, p_true])
    res2 = models.calibrate_proba(y, good, good[:100], classes)
    assert res2["test"] is None
    res3 = models.calibrate_proba(y, proba, proba[:100], classes, from_autogluon=True)
    assert res3["test"] is None


def test_temperature_multiclass():
    rng = np.random.RandomState(0)
    n, k = 3000, 4
    z = rng.randn(n, k)
    p = np.exp(z) / np.exp(z).sum(1, keepdims=True)
    y = np.array([rng.choice(k, p=row) for row in p])
    sharp = np.exp(4 * z) / np.exp(4 * z).sum(1, keepdims=True)
    res = models.calibrate_proba(y, sharp, sharp[:50], list(range(k)))
    assert res["test"] is not None and "temp" in res["info"]


def test_tfidf_text_route():
    rng = np.random.RandomState(0)
    n = 900
    good = ["great product works well", "excellent quality love it", "very good would buy again"]
    bad = ["terrible broke after a day", "awful waste of money", "bad quality do not buy"]
    lab = rng.rand(n + 100) > 0.5
    text = [rng.choice(good if l else bad) + f" item {rng.randint(1000)}" for l in lab]
    text[5] = None  # пропуск
    df = pd.DataFrame({"id": np.arange(n + 100), "review": text, "code": [f"ID{i}" for i in range(n + 100)],
                       "x": rng.randn(n + 100), "t": np.where(lab, "pos", "neg")})
    tr, te = df.iloc[:n].reset_index(drop=True), df.iloc[n:].drop(columns="t").reset_index(drop=True)
    te.loc[0, "review"] = "brand new unseen words here"
    prof = _prof(tr, te, "t", "binary", "f1")
    assert pool.text_columns(prof) == ["review"]          # ID без пробелов текстом не считается
    r = [r for r in pool.plan(prof, "t") if r["family"] == "tfidf"][0]
    c = pool.run_candidate(pool.FoldData(prof, "t"), r, deadline_ts=1e18)
    assert c is not None and c["score"] > 0.95 and np.allclose(c["test"].sum(axis=1), 1)


def test_delta_from_history_temporal():
    """#17: объекты повторяются по годам, таргет = прошлый + шум -> прогноз изменения работает, без утечки."""
    import fe_dsl

    rng = np.random.RandomState(0)
    n_obj, years = 300, 5
    level = rng.gamma(2, 50, n_obj)
    rows = []
    for t in range(1, years + 2):
        for o in range(n_obj):
            rows.append({"obj": o, "year": t, "x": rng.randn(), "y": level[o] * (1 + 0.05 * t) + rng.randn()})
    df = pd.DataFrame(rows)
    df["id"] = np.arange(len(df))
    tr = df[df.year <= years].reset_index(drop=True)
    te = df[df.year > years].drop(columns="y").reset_index(drop=True)
    prof = _prof(tr, te, "y", "regression", "rmse")
    prof["cv"] = {"kind": "temporal", "period": "year"}
    spec = fe_dsl.validate({"op": "lag", "cols": [], "params": {"keys": ["obj"], "period": "year",
                                                                "aggs": ["last", "count"]}}, tr.columns, {"year"})
    prof["target_specs"] = [spec]
    r = [r for r in pool.plan(prof, "y") if r["family"] == "delta"][0]
    fd = pool.FoldData(prof, "y")
    c = pool.run_candidate(fd, r, deadline_ts=1e18)
    base = pool.run_candidate(fd, {"name": "lgbm", "family": "lgbm"}, deadline_ts=1e18)
    assert c is not None and base is not None and c["score"] < base["score"]
    # изменение таргета test-периода не влияет на признаки (опора строится по прошлому)
    assert np.isfinite(c["test"]).all()


def test_run_pool_propagates_transform_and_adds_seeds(monkeypatch):
    tr, te = reg_frames(n=1200)
    prof = _prof(tr, te, "y", "regression", "rmse")
    log_journal = []
    monkeypatch.setattr(pool, "plan", lambda p, t: [
        {"name": "lgbm_log1p", "family": "lgbm", "tf": "log1p", "why": "t"},
        {"name": "extratrees", "family": "extratrees", "why": "e"},
    ])
    out = pool.run_pool(prof, "y", 1e18, 1.0, register=lambda *a, **k: log_journal.append(a[0]),
                        base_score=1e9)  # исходная шкала заведомо хуже
    names = [c["name"] for c in out]
    assert "extratrees_log1p" in names                       # лучшая шкала перенесена на другое семейство
    assert any(n.endswith("_s1") for n in names)              # доп. seed лучшего рецепта
    assert len(log_journal) == len(out)
