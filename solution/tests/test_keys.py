"""Точные ключи объектов (issue #4): группировка, поиск дублей, перенос меток."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.pop("OPENAI_API_KEY", None)

import fe_dsl  # noqa: E402
import models  # noqa: E402
from eda_probe import normalize_keys, row_key_strings, row_keys  # noqa: E402

BIG_A, BIG_B = 9007199254740992, 9007199254740993  # через float совпадают


def test_big_ints_stay_distinct():
    df = pd.DataFrame({"k": np.array([BIG_A, BIG_B], dtype=np.int64)})
    k = row_keys(df, ["k"])
    assert k[0] != k[1]
    s = row_key_strings(df, ["k"])
    assert s[0] != s[1]


def test_int_and_float_equal_values_match():
    tr = pd.DataFrame({"k": np.array([1942, 7], dtype=np.int64)})
    te = pd.DataFrame({"k": [1942.0, np.nan]})
    assert row_keys(tr, ["k"])[0] == row_keys(te, ["k"])[0]
    assert row_key_strings(tr, ["k"])[0] == row_key_strings(te, ["k"])[0]
    # пропуск — отдельное значение, не совпадает со строкой "nan"
    obj = pd.DataFrame({"k": ["nan", None]})
    n = normalize_keys(obj, ["k"])["k"].tolist()
    assert n[0] != n[1]


def test_composite_keys_keep_components():
    df = pd.DataFrame({"a": ["x|y", "x"], "b": ["z", "y|z"]})
    assert row_keys(df, ["a", "b"])[0] != row_keys(df, ["a", "b"])[1]
    s = row_key_strings(df, ["a", "b"])
    assert s[0] != s[1]


def test_group_stat_separates_big_int_keys():
    n = 40
    fit = pd.DataFrame({"k": np.array([BIG_A] * n + [BIG_B] * n, dtype=np.int64)})
    y = np.r_[np.zeros(n), np.full(n, 10.0)]
    app = pd.DataFrame({"k": np.array([BIG_A, BIG_B], dtype=np.int64)})
    spec = fe_dsl.validate({"op": "group_stat", "cols": [], "params": {"keys": ["k"], "src": "target",
                                                                       "aggs": ["mean"]}}, ["k"])
    _, a = fe_dsl.apply(spec, fit, y, app)
    col = [c for c in a.columns if c.endswith("mean")][0]
    assert a[col].tolist() == [0.0, 10.0]
    # смена таргета одного объекта не меняет статистику другого
    y2 = y.copy()
    y2[:n] = 5.0
    _, a2 = fe_dsl.apply(spec, fit, y2, app)
    assert a2[col].iloc[1] == 10.0


def test_label_copy_true_and_false_duplicates():
    """Настоящий дубль получает метку train, ложный (соседнее большое целое) — прогноз модели."""
    rng = np.random.RandomState(0)
    n = 300
    big = np.array([BIG_A + 2 * i for i in range(n)], dtype=np.int64)  # чётные: все различимы
    lab = np.where(rng.rand(n) > 0.5, "a", "b")
    # каждый ключ встречается дважды с одной меткой — правило дублей подтверждается LOO
    train = pd.DataFrame({"id": np.arange(2 * n), "k": np.r_[big, big], "target": np.r_[lab, lab]})
    test = pd.DataFrame({"id": [10_000, 10_001], "k": np.array([big[0], big[0] + 1], dtype=np.int64)})
    prof = {"train": train, "test": test, "features": ["k"]}
    y = train["target"].to_numpy()
    oof_pred = np.where(rng.rand(2 * n) > 0.5, "a", "b")  # модель ~ случайна, копия точнее
    model_test = np.array(["model", "model"], dtype=object)
    out, base, new, k = models.copy_duplicate_labels(prof, "target", y, oof_pred, model_test, "accuracy")
    assert out is not None and k == 1
    assert out[0] == lab[0]           # настоящий дубль
    assert out[1] == "model"          # ложный дубль: BIG+1 не равен BIG
    assert len(out) == len(test)


def test_as_category_op_levels_nan_unknown():
    """#8: код -> категория с точной записью; пропуск и неизвестный уровень — явные значения."""
    fit = pd.DataFrame({"code": [3.0, 7.0, np.nan, 3.0]})
    app = pd.DataFrame({"code": [7.0, 11.0, np.nan]})
    spec = fe_dsl.validate({"op": "as_category", "cols": ["code"]}, ["code"], {"code"})
    f, a = fe_dsl.apply(spec, fit, np.zeros(4), app)
    col = f.columns[0]
    assert f[col].tolist()[:2] == ["3", "7"] and a[col].iloc[0] == "7"
    assert a[col].iloc[1] == "11"                       # неизвестный уровень — свой код
    assert f[col].iloc[2] == a[col].iloc[2] != "nan"    # пропуск — отдельный согласованный уровень
    assert fe_dsl.validate({"op": "as_category", "cols": ["s"]}, ["s"], set()) is None  # только числа


def test_integer_code_candidates_skip_quantities_and_ids():
    import eda_detectors

    rng = np.random.RandomState(0)
    n = 2000
    df = pd.DataFrame({
        "region_code": rng.choice([17, 3, 42, 8, 99], n),     # код
        "item_count": rng.poisson(3, n),                     # количество
        "row_no": np.arange(n),                              # ID
        "x": rng.randn(n),                                   # не целое
    })
    ctx = {"df": df, "features": list(df.columns), "prof": {"data_dictionary": None}, "period": None}
    cols = [c for c, _, _ in eda_detectors.integer_code_candidates(ctx)]
    assert cols == ["region_code"]


def _gs(params):
    return fe_dsl.validate({"op": "group_stat", "cols": [], "params": {"src": "target", "aggs": ["mean"], **params}},
                           ["a", "b"])


def test_smoothing_formula_and_fallback():
    """#7: (S + m·P)/(n + m); неизвестный составной ключ -> общий ключ -> общее среднее."""
    fit = pd.DataFrame({"a": ["x", "x", "x", "y"], "b": ["p", "p", "q", "q"]})
    y = np.array([1.0, 3.0, 8.0, 10.0])
    app = pd.DataFrame({"a": ["x", "x", "z"], "b": ["p", "r", "r"]})
    spec = _gs({"keys": ["a", "b"], "smooth": 2.0, "fallback": [["a"]]})
    _, out = fe_dsl.apply(spec, fit, y, app)
    col = [c for c in out.columns if c.endswith("mean")][0]
    glob = y.mean()                                  # 5.5
    p_x = (12.0 + 2 * glob) / (3 + 2)                # уровень a=x, сглаженный к общему
    assert out[col].iloc[0] == np.float64((4.0 + 2 * p_x) / (2 + 2))
    assert np.isclose(out[col].iloc[1], p_x)         # (x, r) неизвестен -> уровень a=x
    assert np.isclose(out[col].iloc[2], glob)        # z неизвестен совсем -> общее среднее
    # нулевая сила для известной группы — несглаженное среднее
    _, raw = fe_dsl.apply(_gs({"keys": ["a", "b"], "fallback": [["a"]]}), fit, y, app)
    rc = [c for c in raw.columns if c.endswith("mean")][0]
    assert raw[rc].iloc[0] == 2.0 and np.isclose(raw[rc].iloc[1], 12.0 / 3)
    assert rc != col                                 # разные настройки — разные имена


def test_smoothing_classification_shares_sum_to_one():
    fit = pd.DataFrame({"a": ["x", "x", "y", "y", "y"], "b": ["p"] * 5})
    y = np.array(["u", "v", "u", "u", "w"])
    app = pd.DataFrame({"a": ["x", "q"], "b": ["p", "p"]})
    spec = _gs({"keys": ["a"], "smooth": 3.0, "fallback": [["b"]]})
    f, a = fe_dsl.apply(spec, fit, y, app, classes=["u", "v", "w"])
    pc = [c for c in a.columns if "__p_" in c]
    assert len(pc) == 3
    for frame in (a, f.dropna()):
        v = frame[pc].to_numpy()
        assert np.allclose(v.sum(axis=1), 1) and (v >= 0).all() and (v <= 1).all()


def test_recipe_ops_missing_dates_cyclic():
    """#19: число пропусков до импутации, календарные части с явным разбором, sin/cos периода."""
    fit = pd.DataFrame({"a": [1.0, np.nan, np.nan], "b": [np.nan, 2.0, np.nan],
                        "d": ["2024-01-15", "2024-03-02 13:30", None], "w": [0.0, 90.0, 360.0]})
    app = pd.DataFrame({"a": [np.nan], "b": [np.nan], "d": ["garbage"], "w": [180.0]})
    mc = fe_dsl.validate({"op": "missing_count", "cols": ["a", "b"]}, fit.columns)
    f, a = fe_dsl.apply(mc, fit, np.zeros(3), app)
    assert f.iloc[:, 0].tolist() == [1.0, 1.0, 2.0] and a.iloc[0, 0] == 2.0
    dp = fe_dsl.validate({"op": "date_parts", "cols": ["d"]}, fit.columns)
    f, a = fe_dsl.apply(dp, fit, np.zeros(3), app)
    month = [c for c in f.columns if c.endswith("__month")][0]
    assert f[month].iloc[0] == 1 and f[month].iloc[1] == 3 and np.isnan(f[month].iloc[2])
    assert a.isna().all(axis=None)                      # неразбираемая дата -> пропуски
    assert any(c.endswith("__hour") for c in f.columns)
    cy = fe_dsl.validate({"op": "cyclic", "cols": ["w"], "params": {"period": 360}}, fit.columns, {"w"})
    f, a = fe_dsl.apply(cy, fit, np.zeros(3), app)
    cos = [c for c in f.columns if c.endswith("__cos")][0]
    assert np.isclose(f[cos].iloc[0], f[cos].iloc[2])   # 0° и 360° — одно направление
    assert np.isclose(a[cos].iloc[0], -1.0)
    assert fe_dsl.validate({"op": "cyclic", "cols": ["w"], "params": {}}, fit.columns) is None
