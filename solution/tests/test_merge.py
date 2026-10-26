"""Объединение веток поиска (#21)."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.pop("OPENAI_API_KEY", None)

import fe_dsl  # noqa: E402
import fe_search  # noqa: E402

COLS = ["a", "b", "c", "g"]


def S(op, cols, params=None, source="x"):
    v = fe_dsl.validate({"op": op, "cols": cols, "params": params or {}, "source": source}, COLS + ["t"])
    assert v is not None
    return v


def test_merge_dedups_common_and_keeps_order():
    common = S("ratio", ["a", "b"])
    a = [common, S("log", ["a"])]
    b = [S("count_enc", ["g"]), common]
    m, why = fe_search.Search.merge_specs(a, b)
    assert why == "" and [s["op"] for s in m] == ["ratio", "log", "count_enc"]   # общая операция один раз


def test_merge_conflicts():
    m, why = fe_search.Search.merge_specs([S("drop", ["a"])], [S("ratio", ["a", "b"])])
    assert m is None and "удаляемую" in why
    seg1 = S("segment_model", ["g"], {"rule": "level"})
    seg2 = S("segment_model", ["c"], {"rule": "zero"})
    m, why = fe_search.Search.merge_specs([seg1], [seg2])
    assert m is None and "модельные" in why


def test_order_and_params_distinguish_recipes():
    k1 = frozenset(fe_dsl.spec_key(s) for s in [S("bin", ["a"], {"bins": 5})])
    k2 = frozenset(fe_dsl.spec_key(s) for s in [S("bin", ["a"], {"bins": 10})])
    assert k1 != k2


def test_merge_of_two_useful_branches_becomes_leader():
    """Ветка 1 улучшает отношение a/b, ветка 2 — статистику по группе g; вместе — лучше обеих и лидера."""
    rng = np.random.RandomState(0)
    n = 6000
    a, b = rng.lognormal(0, 1, n), rng.lognormal(0, 1, n)
    g = rng.randint(0, 600, n)
    eff = rng.randn(600) * 3
    y = 2 * np.log(a / b) ** 3 + eff[g] + 0.3 * rng.randn(n)
    df = pd.DataFrame({"a": a, "b": b, "c": rng.randn(n), "g": g, "t": y})
    te = df.drop(columns="t").iloc[:200]
    prof = {"train": df, "test": te, "features": COLS, "problem_type": "regression", "metric": "rmse"}
    ctx = fe_search.build_context(prof, "t", time.time() + 600)
    fe_search.build_folds(ctx)
    log = []
    srch = fe_search.Search(ctx, lambda *x, **k: log.append((x, k)))
    srch.calibrate()
    br1 = [S("ratio", ["a", "b"], source="transforms"), S("diff", ["a", "b"], source="transforms")]
    br2 = [S("group_stat", [], {"keys": ["g"], "src": "target", "aggs": ["mean"]}, source="categoricals")]
    srch.stage_a(br1 + br2, time.time() + 300)
    assert {s['source'] for s in srch.passed} == {'transforms', 'categoricals'}
    leader = [s for s in srch.passed if s['source'] == 'transforms']                       # лидер — одна ветка (как если бы этап B взял только её)
    srch.final_scores = srch._scores(leader)
    out = srch.merge_branches(leader, time.time() + 300)
    assert {s['source'] for s in out} == {'transforms', 'categoricals'}
    merges = [x for x in log if str(x[0][0]).startswith("merge:")]
    assert merges and merges[0][0][3] is True and "parents" in merges[0][1]
    # повтор того же объединения не запускается заново
    out2 = srch.merge_branches(out, time.time() + 300)
    assert any(k.get("reason") == "дубль" for _, k in log[len(merges):]) or out2 == out


def _search_for(df, te, feats):
    prof = {"train": df, "test": te, "features": feats, "problem_type": "regression", "metric": "rmse"}
    ctx = fe_search.build_context(prof, "t", time.time() + 600)
    fe_search.build_folds(ctx)
    return fe_search.Search(ctx)


def test_provenance_strong_legit_vs_leak():
    """#2 P1: сильный допустимый признак принимается, утечка через идентичность объектов — нет."""
    rng = np.random.RandomState(0)
    n = 3000
    g = rng.randint(0, 100, n)
    df = pd.DataFrame({"a": rng.randn(n), "g": g, "uid": np.arange(n), "t": rng.randn(100)[g] * 10})
    te_known = df.drop(columns="t").iloc[:300].assign(uid=np.arange(n, n + 300))   # группы g известны
    srch = _search_for(df, te_known, ["a", "g", "uid"])
    legit = fe_dsl.validate({"op": "group_stat", "cols": [], "params": {"keys": ["g"], "src": "target"}},
                            ["a", "g", "uid", "t"])
    leak = fe_dsl.validate({"op": "group_stat", "cols": [], "params": {"keys": ["uid"], "src": "target"}},
                           ["a", "g", "uid", "t"])
    plain = fe_dsl.validate({"op": "log", "cols": ["a"]}, ["a", "g", "uid", "t"])
    assert srch.provenance(legit)[0] is False
    assert srch.provenance(leak)[0] is True             # uid test — новые: запоминание строк
    assert srch.provenance(plain)[0] is False
