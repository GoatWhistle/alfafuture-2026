"""Small real fits verify bounded inner tuning and the optional CPU AutoGluon path."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml_harness.adapters.ml.automl import AutoGluonCandidate
from ml_harness.adapters.ml.candidates import TabularCandidate, build_model
from ml_harness.adapters.ml.tuning import TunedCandidate
from ml_harness.domain.models import ContractError


def require_autogluon():
    try:
        available = importlib.util.find_spec("autogluon.tabular") is not None
    except ModuleNotFoundError:
        available = False
    if not available:
        pytest.skip("Pinned optional AutoGluon extra is not installed")


def dataset():
    frame = pd.DataFrame(
        {
            "x": [float(i) if i % 9 else np.nan for i in range(100)],
            "cat": ["a", "b", None, "c"] * 25,
            "group": [i // 10 for i in range(100)],
            "time": [i // 2 for i in range(100)],
        },
        index=range(1000, 1100),
    )
    return frame, pd.Series(["NA", "red"] * 50, index=frame.index)


@pytest.mark.parametrize("backend", ["catboost", "lightgbm"])
@pytest.mark.parametrize("metric", ["rmse", "mae", "accuracy", "macro_f1", "log_loss", "roc_auc"])
def test_real_tuning_metrics_final_refit_and_json_report(backend, metric):
    X, y = dataset()
    regression = metric in ("rmse", "mae")
    if regression:
        y = pd.Series(np.arange(len(X)) / 3, index=X.index)
    options = {
        "backend": backend,
        "tuning": {"enabled": True, "trials": 2, "time_limit": 20},
        "iterations" if backend == "catboost" else "n_estimators": 3,
    }
    model = build_model("regression" if regression else "binary", options, 1, 42)
    assert isinstance(model, TunedCandidate)
    model.configure_validation({"kind": "random" if regression else "stratified", "metric": metric})
    model.fit(X, y)
    report = json.loads(json.dumps(model.tuning_report_, allow_nan=False))
    assert len(report["trials"]) == 2
    assert all(trial["state"] == "COMPLETE" for trial in report["trials"])
    assert report["best"]["value"] is not None
    assert len(report["inner_split_fingerprint"]) == 64
    assert model.model.medians["x"] == X.x.median()
    unseen = X.iloc[:2].copy()
    unseen["cat"] = "external-never-fit"
    assert model.predict(unseen).shape == (2,)
    assert "external-never-fit" not in model.model.categories["cat"]
    if not regression:
        assert model.classes_.tolist() == ["NA", "red"]
        assert model.predict_proba(unseen).shape == (2, 2)


@pytest.mark.parametrize("strategy", ["group", "time", "stratified", "random"])
def test_internal_partitions_fit_only_passed_training_rows(monkeypatch, strategy):
    X, y = dataset()
    fitted = []
    original = TabularCandidate.fit

    def record(self, frame, target, **kwargs):
        result = original(self, frame, target, **kwargs)
        fitted.append((frame.index.tolist(), self.medians["x"]))
        return result

    monkeypatch.setattr(TabularCandidate, "fit", record)
    model = build_model("binary", {"iterations": 3, "tuning": {"trials": 2}}, 1, 42)
    model.configure_validation(
        {"kind": strategy, "group_column": "group", "time_column": "time", "metric": "accuracy"}
    )
    model.fit(X, y)
    split = model.tuning_report_["inner_split"]
    train, valid = split["train_indices"], split["validation_indices"]
    assert not set(train) & set(valid)
    assert set(train) | set(valid) == set(range(len(X)))
    assert fitted[0][0] == fitted[1][0] == X.iloc[train].index.tolist()
    assert fitted[0][1] == X.iloc[train].x.median()
    assert fitted[-1][0] == X.index.tolist()
    if strategy == "group":
        assert not set(X.iloc[train].group) & set(X.iloc[valid].group)
    if strategy == "time":
        assert X.iloc[train].time.max() < X.iloc[valid].time.min()
    repeat = build_model("binary", {"iterations": 3, "tuning": {"trials": 1}}, 1, 42)
    repeat.configure_validation(model.context)
    repeat.fit(X, y)
    assert (
        repeat.tuning_report_["inner_split_fingerprint"]
        == model.tuning_report_["inner_split_fingerprint"]
    )


@pytest.mark.parametrize("strategy,column", [("group", "group"), ("time", "time")])
def test_impossible_group_or_time_split_fails_explicitly(strategy, column):
    X, y = dataset()
    X[column] = 1
    model = build_model("binary", {"tuning": {"trials": 1}}, 1, 42)
    model.configure_validation({"kind": strategy, strategy + "_column": column})
    with pytest.raises(ContractError, match="inner training split"):
        model.fit(X, y)


@pytest.mark.parametrize(
    "options", [{"trials": 11}, {"trials": 0}, {"time_limit": 301}, {"time_limit": float("nan")}]
)
def test_tuning_budgets_are_bounded(options):
    with pytest.raises(ContractError):
        build_model("regression", {"tuning": options}, 1, 42)


def test_disabled_tuning_preserves_plain_model_behavior():
    assert isinstance(
        build_model("regression", {"tuning": {"enabled": False}}, 1, 42), TabularCandidate
    )


@pytest.mark.parametrize(
    "options",
    [
        {"hyperparameters": {"NN_TORCH": {}}},
        {"hyperparameters": {"GBM": {"ag_args_fit": {}}}},
        {"time_limit": 601},
        {"presets": "best_quality"},
    ],
)
def test_autogluon_disallows_unapproved_families_and_overrides(options):
    with pytest.raises(ContractError):
        AutoGluonCandidate("binary", options, 1, 42)


@pytest.mark.parametrize(
    "family,options,prefix",
    [
        ("GBM", {"num_boost_round": 3, "min_data_in_leaf": 1}, "LightGBM"),
        ("CAT", {"iterations": 3}, "CatBoost"),
        ("XGB", {"n_estimators": 3}, "XGBoost"),
    ],
    ids=["lightgbm", "catboost", "xgboost"],
)
def test_optional_autogluon_real_cpu_smoke(monkeypatch, tmp_path, family, options, prefix):
    monkeypatch.chdir(tmp_path)
    require_autogluon()
    X, y = dataset()
    model = build_model(
        "binary",
        {
            "backend": "autogluon",
            "time_limit": 30,
            "hyperparameters": {family: options},
        },
        1,
        42,
    )
    assert isinstance(model, AutoGluonCandidate)
    model.fit(X, y)
    predictions = model.predict(X.iloc[:3])
    assert predictions.shape == (3,)
    assert set(predictions) <= set(y)
    probability = model.predict_proba(X.iloc[:3])
    assert probability.shape == (3, 2)
    assert np.allclose(probability.sum(axis=1), 1)
    assert set(model.classes_) == set(y)
    assert all(name.startswith(prefix) for name in model.predictor.model_names())


@pytest.mark.parametrize("backend", ["catboost", "lightgbm"])
@pytest.mark.parametrize("metric", ["accuracy", "macro_f1"])
def test_tuning_accepts_noninteger_numeric_class_labels(backend, metric):
    X, _ = dataset()
    y = pd.Series([1.5, 3.5] * 50, index=X.index)
    model = build_model(
        "binary",
        {
            "backend": backend,
            "tuning": {"trials": 1},
            "iterations" if backend == "catboost" else "n_estimators": 3,
        },
        1,
        42,
    )
    model.configure_validation({"kind": "stratified", "metric": metric})
    model.fit(X, y)
    assert model.classes_.tolist() == [1.5, 3.5]
    assert set(model.predict(X.iloc[:4])) <= {1.5, 3.5}


def test_failed_trials_remain_in_report(monkeypatch):
    X, y = dataset()
    model = build_model("binary", {"iterations": 2, "tuning": {"trials": 2}}, 1, 42)

    def fail(*args):
        raise ValueError("synthetic bad candidate")

    monkeypatch.setattr(model, "_score", fail)
    with pytest.raises(ContractError, match="No inner tuning trial"):
        model.fit(X, y)
    assert len(model.tuning_report_["trials"]) == 2
    assert all(trial["state"] == "FAIL" for trial in model.tuning_report_["trials"])
    assert model.tuning_report_["best"] is None
    json.dumps(model.tuning_report_, allow_nan=False)


@pytest.mark.parametrize(
    "kind,metric,expected",
    [
        ("regression", "rmse", "root_mean_squared_error"),
        ("regression", "mae", "mean_absolute_error"),
        ("binary", "macro_f1", "f1_macro"),
        ("binary", "accuracy", "accuracy"),
        ("binary", "log_loss", "log_loss"),
        ("binary", "roc_auc", "roc_auc"),
        ("multiclass", "roc_auc", "harness_roc_auc_ovr"),
    ],
)
@pytest.mark.parametrize("strategy", ["group", "time"])
def test_autogluon_context_split_metric_and_refit_resources(
    monkeypatch, tmp_path, kind, metric, expected, strategy
):
    import types

    from ml_harness.adapters.ml import automl

    monkeypatch.chdir(tmp_path)
    observed = {}
    original_import = automl.importlib.import_module

    class FakePredictor:
        def __init__(self, label, **kwargs):
            observed["init"] = kwargs
            self.label = label
            self.model_best = "Selected"

        def fit(self, train_data, **kwargs):
            observed["train"] = train_data
            observed["fit"] = kwargs
            valid = kwargs["tuning_data"]
            self.class_labels = sorted(set(train_data[self.label]) | set(valid[self.label]))

        def refit_full(self, **kwargs):
            observed["refit"] = kwargs
            self.model_best = "Selected_FULL"
            return {"Selected": "Selected_FULL"}

    monkeypatch.setattr(
        automl.importlib,
        "import_module",
        lambda name: (
            types.SimpleNamespace(TabularPredictor=FakePredictor)
            if name == "autogluon.tabular"
            else types.SimpleNamespace(make_scorer=lambda **kwargs: types.SimpleNamespace(**kwargs))
            if name == "autogluon.core.metrics"
            else original_import(name)
        ),
    )
    X, y = dataset()
    if kind == "regression":
        y = pd.Series(np.arange(len(X)) / 3, index=X.index)
    if kind == "multiclass":
        y = pd.Series(["a", "b", "c", "a"] * 25, index=X.index)
    model = AutoGluonCandidate(kind, {"hyperparameters": {"GBM": {}}, "comparison": True}, 1, 42)
    model.configure_validation(
        {"kind": strategy, "group_column": "group", "time_column": "time", "metric": metric}
    )
    model.fit(X, y)
    selected_metric = observed["init"]["eval_metric"]
    assert (
        selected_metric if isinstance(selected_metric, str) else selected_metric.name
    ) == expected
    assert observed["fit"]["fit_weighted_ensemble"] is True
    assert observed["fit"]["num_stack_levels"] == 0
    assert observed["fit"]["num_cpus"] == observed["refit"]["num_cpus"] == 1
    assert observed["fit"]["num_gpus"] == observed["refit"]["num_gpus"] == 0
    assert observed["refit"]["model"] == "best"
    assert observed["refit"]["set_best_to_refit_full"] is True
    assert observed["refit"]["fit_strategy"] == "sequential"
    train, valid = observed["train"], observed["fit"]["tuning_data"]
    assert not set(train.index) & set(valid.index)
    assert set(train.index) | set(valid.index) == set(X.index)
    if strategy == "group":
        assert not set(train.group) & set(valid.group)
    else:
        assert train.time.max() < valid.time.min()
    assert Path(model.path_).parent == tmp_path


@pytest.mark.parametrize(
    "kind,metric,strategy",
    [
        ("regression", "mae", "group"),
        ("binary", "log_loss", "time"),
        ("multiclass", "roc_auc", "time"),
    ],
)
def test_autogluon_real_metric_context_and_full_refit(
    monkeypatch, tmp_path, kind, metric, strategy
):
    require_autogluon()
    monkeypatch.chdir(tmp_path)
    X, y = dataset()
    if kind == "regression":
        y = pd.Series(np.arange(len(X)) / 3, index=X.index)
    elif kind == "multiclass":
        y = pd.Series(["a", "b", "c", "a"] * 25, index=X.index)
    model = AutoGluonCandidate(
        kind,
        {
            "time_limit": 30,
            "hyperparameters": {"GBM": {"num_boost_round": 3, "min_data_in_leaf": 1}},
        },
        1,
        42,
    )
    model.configure_validation(
        {"kind": strategy, "group_column": "group", "time_column": "time", "metric": metric}
    )
    model.fit(X, y)
    assert model.predictor.model_best.endswith("_FULL")
    expected_metric = {
        "mae": "mean_absolute_error",
        "log_loss": "log_loss",
        "roc_auc": "harness_roc_auc_ovr",
    }[metric]
    assert model.predictor.eval_metric.name == expected_metric
    info = model.predictor.info()["model_info"][model.predictor.model_best]
    assert info["num_samples"] == len(X)
    assert model.predict(X.iloc[:3]).shape == (3,)
    assert Path(model.path_).parent == tmp_path


def test_comparison_autogluon_has_full_allowed_cpu_families_and_ensemble():
    model = build_model("regression", {"backend": "autogluon-comparison"}, 4, 42)
    assert set(model.hyperparameters) == {"CAT", "GBM", "XGB"}
    assert model.hyperparameters["CAT"]["thread_count"] == 4
    assert model.hyperparameters["XGB"]["device"] == "cpu"
    assert model.comparison is True and model.time_limit == 600
    legacy = build_model("regression", {"backend": "autogluon"}, 4, 42)
    assert set(legacy.hyperparameters) == {"CAT"}
    assert legacy.comparison is False and legacy.time_limit == 60
