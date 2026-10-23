"""Family-aware selection, fold isolation, inference and incumbent preservation."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import linear_search
import models
import solution
from agent_utils import StageTimeout


def profile(kind="log", n=800):
    rng = np.random.RandomState(41)
    x = np.exp(rng.uniform(-3, 6, size=(n + 200, 3)))
    value = np.log1p(x) if kind == "log" else x
    y = value @ np.array([3., -2., 1.]) + rng.normal(0, .1, len(x))
    df = pd.DataFrame(x, columns=["a", "b", "c"])
    df["y"] = y
    return dict(train=df.iloc[:n].copy(), test=df.iloc[n:].drop(columns="y"),
                features=["a", "b", "c"], base_features=["a", "b", "c"],
                problem_type="regression", metric="rmse", log_target=False,
                cv={"kind": "kfold"}), y[n:]


def test_log_recipe_selected_and_refitted_for_inference():
    prof, truth = profile()
    records = []
    result = linear_search.search(prof, "y", 20, lambda *r: records.append(r))
    assert result["name"] == "ridge_raw_log"
    assert result["log_score"] < .1 * result["raw_score"]
    assert models.score("rmse", truth, result["test"]) < .3
    expected, _ = linear_search.fit_predict(result["recipe"], prof["train"], prof["train"].y,
                                           prof["test"], False)
    np.testing.assert_array_equal(result["test"], expected)
    assert any(r[0] == "ridge_transform" and r[3] for r in records)


def test_raw_linear_control():
    prof, truth = profile("linear")
    result = linear_search.search(prof, "y", 20)
    assert result["name"] == "ridge_raw"
    assert models.score("rmse", truth, result["test"]) < 1


def test_recipe_does_not_depend_on_tree_acceptance():
    prof, _ = profile()
    prof["features"] = ["tree_only"]
    prof["train"]["tree_only"] = 1
    prof["test"]["tree_only"] = 1
    recipes, _ = linear_search.recipes(prof, "y")
    assert recipes[0]["columns"] == ["a", "b", "c"]
    assert len({linear_search.recipe_key(r) for r in recipes}) == 2
    assert linear_search.recipe_key({**recipes[0], "family": "tree"}) != linear_search.recipe_key(recipes[0])
    assert linear_search.search(prof, "y", 20)["name"] == "ridge_raw_log"


def test_log_helps_ridge_when_tree_rejects_it():
    from lightgbm import LGBMRegressor

    rng = np.random.RandomState(17)
    x = np.exp(rng.uniform(-3, 6, 1200))[:, None]
    y = 3 * np.log1p(x[:, 0]) + rng.normal(0, .05, len(x))
    errors = []
    for values in (x, np.column_stack([x, np.log1p(x)])):
        tree = LGBMRegressor(n_estimators=80, num_leaves=15, min_child_samples=10,
                            verbosity=-1, n_jobs=2, random_state=7)
        tree.fit(values[:800], y[:800])
        errors.append(models.score("rmse", y[800:], tree.predict(values[800:])))
    assert errors[1] >= errors[0] * .998
    frame = pd.DataFrame({"x": x[:, 0], "y": y})
    prof = dict(train=frame.iloc[:800], test=frame.iloc[800:].drop(columns="y"),
                features=["x"], problem_type="regression", metric="rmse", log_target=False)
    result = linear_search.search(prof, "y", 20)
    assert result["name"] == "ridge_raw_log"
    assert result["log_score"] < result["raw_score"] * .1
    assert models.score("rmse", y[800:], result["test"]) < errors[0]


def test_fitted_preprocessing_uses_training_rows_only():
    train = pd.DataFrame({"a": [1., 3., np.nan, np.inf], "b": [np.nan] * 4})
    test = pd.DataFrame({"a": [1e9, np.nan], "b": [99., np.nan]})
    recipe = dict(columns=["a", "b"], transform="raw", alpha=1.)
    original = train.copy(deep=True)
    pred, pipe = linear_search.fit_predict(recipe, train, [1., 2., 3., 4.], test, False)
    assert pipe.named_steps["simpleimputer"].statistics_[0] == 2
    assert pipe.named_steps["standardscaler"].mean_[0] == 2
    assert np.isfinite(pred).all()
    pd.testing.assert_frame_equal(train, original)


@pytest.mark.parametrize("kind", ["temporal", "group"])
def test_recipes_follow_fold_boundaries(monkeypatch, kind):
    prof, _ = profile(n=1000)
    prof["train"]["period"] = np.repeat(np.arange(5), 200)
    prof["train"]["entity"] = np.arange(1000) % 20
    prof["cv"] = {"kind": kind, "period": "period", "keys": ["entity"]}
    original = linear_search.fit_predict
    fold_calls = []

    def checked(recipe, train, y, test, log_target):
        if len(train) < 1000:
            assert not set(train.index) & set(test.index)
            if kind == "temporal":
                assert train.period.max() < test.period.min()
            else:
                assert not set(train.entity) & set(test.entity)
            fold_calls.append(recipe["transform"])
        return original(recipe, train, y, test, log_target)

    monkeypatch.setattr(linear_search, "fit_predict", checked)
    result = linear_search.search(prof, "y", 20)
    assert result is not None
    assert fold_calls.count("raw") == fold_calls.count("raw_log") > 0


def test_validation_labels_cannot_change_fold_predictions():
    prof, _ = profile()
    train = prof["train"]
    fit, val = models.outer_folds(prof, train, train.y)[0]
    recipe = linear_search.recipes(prof, "y")[0][1]
    first, _ = linear_search.fit_predict(recipe, train.iloc[fit], train.y.iloc[fit], train.iloc[val], False)
    changed = train.copy()
    changed.loc[val, "y"] = 1e12
    second, _ = linear_search.fit_predict(recipe, changed.iloc[fit], changed.y.iloc[fit], changed.iloc[val], False)
    np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize("failure", [RuntimeError("failure"), StageTimeout()])
def test_failed_search_returns_no_candidate(monkeypatch, failure):
    prof, _ = profile()
    def fail(*args, **kwargs):
        raise failure
    monkeypatch.setattr(linear_search, "fit_predict", fail)
    assert linear_search.search(prof, "y", 20) is None


def test_budget_and_unsupported_tasks_skip(monkeypatch):
    prof, _ = profile()
    monkeypatch.setattr(linear_search, "_search", lambda *a: pytest.fail("must not train"))
    assert linear_search.search(prof, "y", 0) is None
    prof["problem_type"] = "classification"
    assert linear_search.search(prof, "y", 20) is None


def test_rmsle_roundtrip():
    prof, _ = profile()
    prof["train"]["y"] = np.expm1(np.log1p(prof["train"].a) * .3)
    prof.update(metric="rmsle", log_target=True)
    result = linear_search.search(prof, "y", 20)
    truth = np.expm1(np.log1p(prof["test"].a) * .3)
    assert models.score("rmsle", truth, result["test"]) < .02


def candidate(name, y, error, test):
    return dict(name=name, rows=np.arange(len(y)), y=y, oof=y + error,
                test=np.asarray(test, float), score=float(abs(error)), classes=None)


@pytest.mark.parametrize("case", ["worse", "tie", "tiny_gain", "partial", "better"])
def test_optional_family_preserves_incumbent_or_wins(monkeypatch, case):
    y = np.arange(20, dtype=float)
    base = candidate("tree", y, 1., [7., 8.])
    extra = candidate("ridge", y, {"better": .1, "tiny_gain": .999}.get(case, 1. if case == "tie" else 2.), [4., 5.])
    if case == "partial":
        for key in ("rows", "y", "oof"):
            extra[key] = extra[key][1:]
    prof = dict(metric="rmse", problem_type="regression", train=pd.DataFrame({"y": y}))
    written = []
    monkeypatch.setattr(solution, "write_submission", lambda p, c: written.append(c["y"].copy()))
    result = solution.select_final(prof, "y", {}, None, [base], base["test"], extra_cands=[extra])
    np.testing.assert_array_equal(result, extra["test"] if case == "better" else base["test"])
    np.testing.assert_array_equal(result, written[-1])


def test_existing_blend_is_still_compared(monkeypatch):
    y = np.arange(20, dtype=float)
    a, b = candidate("tree", y, 1., [6., 8.]), candidate("ag", y, -1., [8., 6.])
    extra = candidate("ridge", y, .1, [4., 5.])
    prof = dict(metric="rmse", problem_type="regression", train=pd.DataFrame({"y": y}))
    monkeypatch.setattr(solution, "write_submission", lambda *a: None)
    result = solution.select_final(prof, "y", {}, None, [a, b], a["test"], extra_cands=[extra])
    np.testing.assert_array_equal(result, [7., 7.])
