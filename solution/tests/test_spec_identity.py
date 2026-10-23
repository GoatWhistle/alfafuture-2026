"""Issue #3: варианты признаков не теряются, эквивалентные спеки не обучаются повторно."""

import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import fe_dsl
import fe_search
import models
import solution


def spec(op, cols, **params):
    columns = set(cols) | set(params.get("keys", [])) | {params.get("period", "")}
    result = fe_dsl.validate({"op": op, "cols": cols, "params": params}, columns)
    assert result is not None
    return result


@pytest.mark.parametrize("op,params", [("prod", {}), ("interaction", {"bins": 4}),
                                      ("agg_cols", {"aggs": ["sum", "mean"]}),
                                      ("knn_target", {"k": 3})])
def test_commutative_operations_have_same_identity_and_values(op, params):
    left, right = spec(op, ["a", "b"], **params), spec(op, ["b", "a"], **params)
    assert fe_dsl.spec_key(left) == fe_dsl.spec_key(right)
    rng = np.random.RandomState(3)
    df = pd.DataFrame(rng.randn(80, 2), columns=["a", "b"])
    first = fe_dsl.apply(left, df.iloc[:60], df.a[:60], df.iloc[60:])
    second = fe_dsl.apply(right, df.iloc[:60], df.a[:60], df.iloc[60:])
    for a, b in zip(first, second):
        pd.testing.assert_frame_equal(a, b)


@pytest.mark.parametrize("op", ["diff", "ratio"])
def test_ordered_operations_remain_distinct(op):
    assert fe_dsl.spec_key(spec(op, ["a", "b"])) != fe_dsl.spec_key(spec(op, ["b", "a"]))


def test_group_keys_aggregates_and_metadata_are_canonical():
    a = spec("group_stat", ["x"], keys=["a", "b"], aggs=["mean", "count"])
    b = spec("group_stat", ["x"], keys=["b", "a"], aggs=["count", "mean"])
    b = {**b, "source": "qwen", "prior": 12, "why": "another explanation"}
    b["params"] = dict(reversed(list(b["params"].items())))
    assert fe_dsl.spec_key(a) == fe_dsl.spec_key(b)
    df = pd.DataFrame({"a": np.tile([0, 1], 20), "b": np.tile([1, 2, 3, 4], 10), "x": np.arange(40)})
    for left, right in zip(fe_dsl.apply(a, df[:30], df.x[:30], df[30:]),
                           fe_dsl.apply(b, df[:30], df.x[:30], df[30:])):
        pd.testing.assert_frame_equal(left, right)
    # Delimiter characters in column names cannot change argument boundaries.
    assert fe_dsl.spec_key(spec("prod", ["a,b", "c"])) != fe_dsl.spec_key(spec("prod", ["a", "b,c"]))


@pytest.mark.parametrize("op,params", [("bin", [{"bins": 2}, {"bins": 10}]),
                                      ("flag", [{"rule": "threshold", "value": 30},
                                                {"rule": "threshold", "value": 70}]),
                                      ("knn_target", [{"k": 2}, {"k": 10}])])
def test_parameters_produce_separate_features_in_validation_and_final_fit(op, params):
    rng = np.random.RandomState(11)
    df = pd.DataFrame({"x": rng.uniform(0, 100, 120), "y": rng.randn(120)})
    specs = [spec(op, ["x"], **p) for p in params]
    assert fe_dsl.spec_key(specs[0]) != fe_dsl.spec_key(specs[1])
    ctx = {"df": df, "features": ["x"], "y": df.y.to_numpy(), "classes": None,
           "folds": [(np.arange(90), np.arange(90, 120))]}
    ev = fe_search.Evaluator(ctx)
    train, val = ev.matrices(0, specs)
    assert len(ev.cache) == 2
    width = 2 if op == "knn_target" else 1
    assert train.shape[1] == val.shape[1] == 1 + 2 * width
    assert not np.allclose(train.iloc[:, 1], train.iloc[:, 1 + width])
    prof = {"train": df[:90], "test": df[90:].drop(columns="y"), "problem_type": "regression",
            "target_specs": specs, "cv": {"kind": "kfold"}}
    if op == "knn_target":
        f, a = models.target_features(prof, "y", prof["train"], prof["test"])
    else:
        parts = [fe_dsl.materialize(prof, s, "y") for s in specs]
        f = pd.concat([p[0] for p in parts], axis=1, verify_integrity=True)
        a = pd.concat([p[1] for p in parts], axis=1, verify_integrity=True)
    assert f.shape[1] == a.shape[1] == 2 * width
    assert list(f.columns) == list(train.columns[1:])


def test_names_are_model_safe_and_preserve_original_text_and_class():
    names = [fe_dsl._name(spec("log", [col]), suffix)
             for col in ["цена", "дата", "a-b", "a:b", "a_b", "x" * 150 + "a", "x" * 150 + "b"]
             for suffix in ["p_да!", "p_да?", "mean"]]
    assert len(names) == len(set(names))
    assert all(re.fullmatch(r"[A-Za-z0-9_]+", name) for name in names)


def test_detected_collision_fails_explicitly(monkeypatch):
    df = pd.DataFrame({"x": np.arange(100), "y": np.arange(100)})
    specs = [spec("bin", ["x"], bins=n) for n in [2, 10]]
    ctx = {"df": df, "features": ["x"], "y": df.y.to_numpy(), "classes": None,
           "folds": [(np.arange(80), np.arange(80, 100))]}
    monkeypatch.setattr(fe_dsl, "_name", lambda *a: "forced_collision")
    with pytest.raises(ValueError, match="overlapping"):
        fe_search.Evaluator(ctx).matrices(0, specs)
    target_specs = [spec("knn_target", ["x"], k=k) for k in [2, 10]]
    with pytest.raises(ValueError, match="overlapping"):
        models.target_features({"target_specs": target_specs, "problem_type": "regression"},
                               "y", df[:80], df[80:])


def test_equivalent_repeats_do_not_retrain_across_rounds_or_sets(monkeypatch):
    rng = np.random.RandomState(4)
    df = pd.DataFrame(rng.randn(400, 2), columns=["a", "b"])
    ctx = {"df": df, "features": ["a", "b"], "y": df.a.to_numpy() * df.b.to_numpy(),
           "classes": None, "metric": "rmse", "ptype": "regression", "deadline": time.time() + 60,
           "folds": [(np.arange(300), np.arange(300, 400))]}
    monkeypatch.setattr(models, "N_CPUS", 2)
    search = fe_search.Search(ctx)
    search.base, search.rel_min = search.ev.run([]), 0.002
    a, b = spec("prod", ["a", "b"]), spec("prod", ["b", "a"])
    assert len(search.stage_a([a, b], ctx["deadline"])) == 1
    count = search.ev.n_evals
    assert search.stage_a([b], ctx["deadline"]) == []
    assert search.ev.run([a, b]) == search.ev.run([b])
    assert search.ev.n_evals == count == 2
    assert len(search.ev.cache) == 1


def test_two_thresholds_survive_search_materialization_and_prediction(monkeypatch):
    rng = np.random.RandomState(23)
    x = rng.uniform(0, 100, 1800)
    y = 6 * (x > 30) + 4 * (x > 70) + rng.randn(len(x)) * 0.8
    train = pd.DataFrame({"x": x[:1500], "y": y[:1500]})
    test = pd.DataFrame({"x": x[1500:]})
    prof = {"train": train, "test": test, "features": ["x"], "problem_type": "regression",
            "metric": "rmse", "log_target": False, "output_proba": False}
    a, b = [spec("flag", ["x"], rule="threshold", value=v) for v in [30, 70]]
    duplicate = {**a, "source": "qwen", "why": "same threshold"}
    rows = rng.permutation(len(train))
    ctx = {"df": train, "features": ["x"], "y": train.y.to_numpy(), "classes": None,
           "metric": "rmse", "ptype": "regression", "importances": {"x": 1}, "facts": {},
           "period": None, "entity": None, "label_copy": False, "cv": {"kind": "kfold"},
           "fold_scheme": "kfold", "deadline": time.time() + 60,
           "folds": [(rows[:1000], rows[1000:])]}
    monkeypatch.setattr(fe_search, "build_context", lambda *args: ctx)
    monkeypatch.setattr(fe_search, "build_folds", lambda ctx: None)
    monkeypatch.setattr(solution.eda_detectors, "run_all", lambda *args: [a, b, duplicate])
    monkeypatch.setattr(fe_search, "perm_importance", lambda *args: [])
    monkeypatch.setattr(solution, "time_left", lambda: 0)
    monkeypatch.setattr(models, "N_CPUS", 2)
    original = fe_search._lgb_params
    # Coarse numeric bins make both exact threshold indicators useful to the
    # validator; the final model below uses its normal configuration.
    monkeypatch.setattr(fe_search, "_lgb_params", lambda ctx, fast=False:
                        {**original(ctx, fast), "max_bin": 2, "n_estimators": 100})
    result = solution.run_autoeda(prof, "y", 60)
    assert len(result["accepted"]) == 2
    expected = {fe_dsl._name(s, "threshold") for s in [a, b]}
    assert expected <= set(prof["features"])
    assert expected <= set(prof["train"].columns) & set(prof["test"].columns)
    model = models.quick_model(prof, "y")
    assert expected <= set(model["importances"])
    assert all(model["importances"][c] > 0 for c in expected)
    assert len(model["test_pred"]) == len(test)
    assert np.sqrt(np.mean((model["test_pred"] - y[1500:]) ** 2)) < 1.5


@pytest.mark.parametrize("reverse", [False, True])
def test_segment_model_and_flag_share_one_materialized_indicator(monkeypatch, reverse):
    train = pd.DataFrame({"x": np.arange(100), "y": np.arange(100)})
    prof = {"train": train, "test": train.iloc[:10].drop(columns="y"), "features": ["x"]}
    flag = spec("flag", ["x"], rule="threshold", value=30)
    segment = spec("segment_model", ["x"], rule="threshold", value=30)
    accepted = [flag, segment] if reverse else [segment, flag]
    ctx = {"df": train, "features": ["x"], "folds": [], "classes": None, "metric": "rmse",
           "ptype": "regression", "facts": {}, "importances": {}, "period": None,
           "entity": None, "label_copy": False, "cv": {"kind": "kfold"}, "fold_scheme": "kfold"}
    monkeypatch.setattr(fe_search, "build_context", lambda *a: ctx)
    monkeypatch.setattr(fe_search, "build_folds", lambda *a: None)
    monkeypatch.setattr(fe_search, "perm_importance", lambda *a: [])
    monkeypatch.setattr(solution.eda_detectors, "run_all", lambda *a: accepted)
    monkeypatch.setattr(solution, "time_left", lambda: 0)
    monkeypatch.setattr(fe_search.Search, "calibrate", lambda *a: None)
    monkeypatch.setattr(fe_search.Search, "stage_a", lambda *a: [])
    monkeypatch.setattr(fe_search.Search, "stage_b", lambda *a: accepted)
    solution.run_autoeda(prof, "y", 60)
    assert prof["features"] == ["x", fe_dsl._name(flag, "threshold")]
    assert prof["train"].columns.is_unique and prof["test"].columns.is_unique
    assert prof["model_ops"]["segment_model"] == segment
