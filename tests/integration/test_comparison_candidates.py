"""Comparison engines preserve training boundaries and frozen refit recipes."""

import importlib.util

import numpy as np
import pandas as pd
import pytest

from ml_harness.adapters.ml.flaml import FlamlCandidate
from ml_harness.adapters.ml.linear import LinearCandidate
from ml_harness.domain.models import ContractError


def test_linear_handles_missing_mixed_categories_and_unknowns():
    X = pd.DataFrame(
        {
            "x": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0],
            "category": ["a", 2, None, "b", "a", 2],
            "empty": [np.nan] * 6,
        }
    )
    model = LinearCandidate("binary", {}, 1, 42).fit(X, pd.Series(["z", "a"] * 3))
    test = pd.DataFrame({"x": [np.nan, 10.0], "category": ["new", None], "empty": [1.0, np.nan]})
    probabilities = model.predict_proba(test)
    assert model.classes_.tolist() == ["a", "z"]
    assert probabilities.shape == (2, 2)
    assert np.allclose(probabilities.sum(axis=1), 1)
    assert set(model.predict(test)) <= {"a", "z"}
    assert (
        "new"
        not in model.model.named_steps["preprocess"]
        .named_transformers_["category"]
        .named_steps["encode"]
        .categories_[0]
    )


@pytest.mark.parametrize("value", [True, 0, float("nan"), -1])
def test_flaml_rejects_invalid_time_limits(value):
    with pytest.raises(ContractError):
        FlamlCandidate("regression", {"time_limit": value}, 1, 42)


@pytest.mark.skipif(importlib.util.find_spec("flaml") is None, reason="optional automl extra")
@pytest.mark.parametrize("kind,metric", [("regression", "rmse"), ("multiclass", "roc_auc")])
def test_actual_flaml_refits_full_outer_train_and_frozen_recipe_skips_search(
    monkeypatch, kind, metric
):
    import flaml
    from flaml.automl.data import DataTransformer

    X = pd.DataFrame({"x": np.arange(90), "category": ["a", "b", "c"] * 30})
    y = pd.Series(np.arange(90) % 3 if kind == "multiclass" else np.arange(90) % 7 + 0.1)
    fitted_rows = []
    original = DataTransformer.fit_transform

    def spy(self, features, targets, task):
        fitted_rows.append(len(features))
        return original(self, features, targets, task)

    monkeypatch.setattr(DataTransformer, "fit_transform", spy)
    candidate = FlamlCandidate(kind, {"time_limit": 1}, 1, 42)
    candidate.configure_validation(
        {"kind": "stratified" if kind != "regression" else "random", "metric": metric}
    )
    candidate.fit(X, y)
    assert fitted_rows[0] == 72 and fitted_rows[-1] == 90
    assert candidate.tuning_report_["refit_rows"] == 90
    predictions = candidate.predict(X[:4])
    assert len(predictions) == 4 and np.isfinite(predictions).all()
    if kind == "multiclass":
        assert candidate.classes_.tolist() == [0, 1, 2]
        assert candidate.predict_proba(X[:4]).shape == (4, 3)

    def forbidden(*args, **kwargs):
        raise AssertionError("frozen refit must not search")

    monkeypatch.setattr(flaml.AutoML, "fit", forbidden)
    frozen = FlamlCandidate(kind, candidate.refit_parameters_, 1, 42).fit(X, y)
    assert np.allclose(frozen.predict(X[:4]), predictions)


@pytest.mark.skipif(importlib.util.find_spec("flaml") is None, reason="optional automl extra")
@pytest.mark.parametrize("strategy", ["group", "time"])
def test_flaml_search_receives_safe_inner_partitions(monkeypatch, strategy):
    import flaml

    X = pd.DataFrame(
        {
            "x": np.arange(60),
            "group": np.repeat(np.arange(6), 10),
            "time": np.repeat(np.arange(30), 2),
        }
    )
    y = pd.Series(np.arange(60) % 7 + 0.1)
    original = flaml.AutoML.fit
    observed = []

    def spy(self, *args, **kwargs):
        train, valid = kwargs["X_train"], kwargs["X_val"]
        if strategy == "group":
            assert set(train.group).isdisjoint(set(valid.group))
        else:
            assert train.time.max() < valid.time.min()
        observed.append((len(train), len(valid)))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(flaml.AutoML, "fit", spy)
    candidate = FlamlCandidate("regression", {"time_limit": 0.3}, 1, 42)
    candidate.configure_validation(
        {"kind": strategy, "group_column": "group", "time_column": "time", "metric": "rmse"}
    )
    candidate.fit(X, y)
    assert sum(observed[0]) == len(X)
    assert candidate.tuning_report_["refit_rows"] == len(X)


@pytest.mark.skipif(importlib.util.find_spec("flaml") is None, reason="optional automl extra")
def test_frozen_flaml_catboost_does_not_reserve_another_holdout(monkeypatch):
    from catboost import CatBoostRegressor

    original = CatBoostRegressor.fit
    rows = []

    def spy(self, X, y=None, **kwargs):
        rows.append(len(X))
        assert kwargs.get("eval_set") is None
        assert kwargs["use_best_model"] is False
        return original(self, X, y, **kwargs)

    monkeypatch.setattr(CatBoostRegressor, "fit", spy)
    X = pd.DataFrame({"x": np.arange(30)})
    model = FlamlCandidate(
        "regression",
        {
            "selected_estimator": "catboost",
            "selected_config": {"n_estimators": 4},
        },
        1,
        42,
    ).fit(X, pd.Series(np.arange(30) % 5 + 0.1))
    assert rows == [30]
    assert len(model.predict(X[:2])) == 2
