"""Проверки P0 (issue #2): независимость target-признаков и единая схема валидации."""

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
import solution  # noqa: E402
from eda_probe import row_keys  # noqa: E402


def panel(n_obj=120, n_per=5, seed=0):
    """Объекты × периоды: таргет зависит от объекта и периода."""
    rng = np.random.RandomState(seed)
    obj = np.repeat(np.arange(n_obj), n_per)
    per = np.tile(np.arange(n_per), n_obj)
    x = rng.randn(len(obj))
    level = rng.randn(n_obj)[obj]
    y = level * 3 + 0.5 * per + x + 0.1 * rng.randn(len(obj))
    return pd.DataFrame({"id": np.arange(len(obj)), "obj": obj, "per": per.astype(float), "x": x,
                         "x2": rng.randn(len(obj)), "y": y})


SPECS = [
    fe_dsl.validate({"op": "group_stat", "cols": [], "params": {"keys": ["obj"], "src": "target",
                                                                "aggs": ["mean", "std", "count"]}}, ["obj", "per", "x", "x2"]),
    fe_dsl.validate({"op": "knn_target", "cols": ["x", "x2"], "params": {"k": 5}}, ["obj", "per", "x", "x2"]),
    fe_dsl.validate({"op": "lag", "cols": [], "params": {"keys": ["obj"], "period": "per", "src": "target",
                                                         "aggs": ["last", "mean", "count"]}}, ["obj", "per", "x", "x2"]),
]
CVS = [{"kind": "kfold"}, {"kind": "temporal", "period": "per"}, {"kind": "group", "keys": ["obj"]}]


def prof_for(df, cv, specs=SPECS, ptype="regression"):
    return {"train": df, "test": df.iloc[:0].drop(columns="y"), "problem_type": ptype, "metric": "rmse",
            "target_specs": specs, "cv": cv, "features": ["x", "x2", "obj", "per"], "log_target": False,
            "model_ops": {}}


@pytest.mark.parametrize("cv", CVS, ids=lambda c: c["kind"])
@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s["op"])
def test_outer_val_targets_do_not_change_features(cv, spec):
    df = panel()
    prof = prof_for(df, cv, [spec])
    folds = models.outer_folds(prof, df, df["y"].to_numpy())
    for tr, va in folds:
        fit_i, es_i = models.es_split(prof, df, tr)
        app = np.concatenate([es_i, va])
        F1, A1 = models.target_features(prof, "y", df.iloc[fit_i], df.iloc[app])
        changed = df.copy()
        changed.loc[changed.index[va], "y"] = np.random.RandomState(9).randn(len(va)) * 100
        F2, A2 = models.target_features(prof, "y", changed.iloc[fit_i], changed.iloc[app])
        if F1 is None:
            # group-схема: статистика по собственному объекту пуста — как для новых объектов test.
            assert F2 is None and cv["kind"] == "group" and spec["op"] == "group_stat"
            continue
        pd.testing.assert_frame_equal(F1, F2)
        pd.testing.assert_frame_equal(A1, A2)


def test_old_full_train_materialize_leaked():
    """Регрессионный контроль исходной проблемы: признак на всём train зависит от таргета валидации."""
    df = panel()
    spec = SPECS[2]
    va = df.index[df["per"] == 2]
    f1 = fe_dsl.apply(spec, df, df["y"].to_numpy(), df.iloc[:0])[0]
    changed = df["y"].to_numpy().copy()
    changed[va] += 100
    f2 = fe_dsl.apply(spec, df, changed, df.iloc[:0])[0]
    outer_train_rows = (df["per"] > 2).to_numpy()
    assert not np.allclose(f1.loc[outer_train_rows].fillna(0), f2.loc[outer_train_rows].fillna(0))


def test_temporal_folds_and_inner_splits_use_only_past():
    df = panel()
    prof = prof_for(df, {"kind": "temporal", "period": "per"})
    folds = models.outer_folds(prof, df, df["y"].to_numpy())
    assert len(folds) == 2
    for tr, va in folds:
        assert df["per"].iloc[tr].max() < df["per"].iloc[va].min()
        fit_i, es_i = models.es_split(prof, df, tr)
        assert df["per"].iloc[fit_i].max() < df["per"].iloc[es_i].min()
    for src, dst in fe_dsl.inner_splits(df, prof["cv"]):
        assert df["per"].iloc[src].max() < df["per"].iloc[dst].min()

    # Target-статистика строки периода t = статистика строк того же объекта с периодом < t.
    F, _ = models.target_features(prof_for(df, prof["cv"], [SPECS[0]]), "y", df, df.iloc[:0])
    col = [c for c in F.columns if c.endswith("__mean")][0]
    i = df.index[(df["obj"] == 7) & (df["per"] == 3)][0]
    past = df[(df["obj"] == 7) & (df["per"] < 3)]["y"].mean()
    assert F.loc[i, col] == pytest.approx(past)
    assert F.loc[(df["per"] == 0).to_numpy(), col].isna().all()


def test_group_folds_disjoint_and_inner_respects_groups():
    df = panel()
    prof = prof_for(df, {"kind": "group", "keys": ["obj"]})
    g = row_keys(df, ["obj"])
    for tr, va in models.outer_folds(prof, df, df["y"].to_numpy()):
        assert not set(g[tr]) & set(g[va])
        fit_i, es_i = models.es_split(prof, df, tr)
        assert not set(g[fit_i]) & set(g[es_i])
    for src, dst in fe_dsl.inner_splits(df, prof["cv"]):
        assert not set(g[src]) & set(g[dst])
    # Статистика по собственному объекту при group-схеме недоступна — как для новых объектов test.
    F, _ = models.target_features(prof_for(df, prof["cv"], [SPECS[0]]), "y", df, df.iloc[:0])
    assert F is None or F.filter(like="__mean").isna().all().all()


# --- выравнивание OOF кандидатов --------------------------------------------

def _cand(name, rows, y, oof, classes, test=None):
    return {"name": name, "score": 0.0, "rows": np.asarray(rows), "y": np.asarray(y, dtype=object),
            "oof": np.asarray(oof), "test": np.zeros((3, len(classes))) if test is None else test, "classes": classes}


def test_align_candidates_permutation_invariant():
    rng = np.random.RandomState(0)
    rows = np.arange(100, 160)
    y = rng.choice(["a", "b", "c"], len(rows))
    p1, p2 = rng.dirichlet(np.ones(3), len(rows)), rng.dirichlet(np.ones(3), len(rows))
    base = solution.align_candidates([_cand("m1", rows, y, p1, ["a", "b", "c"]),
                                      _cand("m2", rows, y, p2, ["a", "b", "c"])])
    perm = rng.permutation(len(rows))
    shuffled = solution.align_candidates([
        _cand("m1", rows, y, p1, ["a", "b", "c"]),
        _cand("m2", rows[perm], y[perm], p2[perm][:, [2, 0, 1]], ["c", "a", "b"]),
    ])
    for c0, c1 in zip(base, shuffled):
        assert np.array_equal(c0["rows"], c1["rows"])
        assert np.allclose(c0["oof"], c1["oof"])
        assert c1["classes"] == ["a", "b", "c"]


def test_align_candidates_drops_rows_without_oof_and_detects_mismatch():
    rows = np.arange(10)
    y = np.array(["a", "b"] * 5)
    p = np.full((10, 2), 0.5)
    out = solution.align_candidates([_cand("m1", rows, y, p, ["a", "b"]),
                                     _cand("m2", rows[3:], y[3:], p[3:], ["a", "b"])])
    assert all(len(c["rows"]) == 7 for c in out)
    bad = y.copy()
    bad[5] = "a"
    with pytest.raises(ValueError):
        solution.align_candidates([_cand("m1", rows, y, p, ["a", "b"]), _cand("m2", rows, bad, p, ["a", "b"])])


# --- k-fold LightGBM по схеме --------------------------------------------------

@pytest.mark.parametrize("cv", CVS, ids=lambda c: c["kind"])
def test_quick_oof_follows_scheme(cv):
    df = panel(n_obj=150)
    test = panel(n_obj=20, seed=1).drop(columns="y")
    test["per"] = 5.0
    prof = {**prof_for(df, cv), "test": test}
    kq = models.quick_oof(prof, "y")
    folds = models.outer_folds(prof, df, df["y"].to_numpy())
    covered = np.unique(np.concatenate([va for _, va in folds]))
    assert np.array_equal(np.sort(kq["rows"]), df.index[covered].to_numpy())
    assert np.isfinite(kq["oof"]).all() and len(kq["test"]) == len(test)
    if cv["kind"] == "temporal":
        assert set(df.loc[kq["rows"], "per"]) == {3.0, 4.0}


def test_segment_without_rare_class_does_not_crash():
    """Сегмент, в train которого нет редкого класса, а в val он есть: обе ветки (валидатор и k-fold)."""
    import fe_search

    rng = np.random.RandomState(0)
    n = 1200
    seg = (np.arange(n) % 2).astype(float)
    y = np.where(rng.rand(n) < 0.5, "a", "b").astype(object)
    rare = np.flatnonzero(seg == 1)[:3]
    y[rare] = "rare"
    X = pd.DataFrame({"x": rng.randn(n), "seg": seg})
    ctx = {"ptype": "multiclass", "metric": "f1_macro", "classes": ["a", "b", "rare"], "deadline": 1e12}
    tr = np.setdiff1d(np.arange(n), rare)
    va = np.concatenate([rare, np.arange(0, 400, 2)])
    out = fe_search.fit_predict_segmented(ctx, X.iloc[tr].reset_index(drop=True), y[tr], X.iloc[va].reset_index(drop=True),
                                          y[va], seg[tr], seg[va])
    assert out.shape == (len(va), 3)
