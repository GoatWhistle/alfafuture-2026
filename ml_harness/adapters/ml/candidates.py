"""Small train-fitted tabular candidates with a shared dataframe interface."""

from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from catboost import CatBoostClassifier, CatBoostRegressor  # type: ignore[import-untyped]
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.preprocessing import LabelEncoder  # type: ignore[import-untyped]


class TabularCandidate:
    """Fit all imputation and category vocabularies on training rows alone."""

    def __init__(self, kind: str, parameters: dict[str, Any], threads: int, seed: int):
        self.regression = kind == "regression"
        options = dict(parameters)
        self.backend = options.pop("backend", "catboost")
        self.categories: dict[str, list[str]] = {}
        self.medians: dict[str, float] = {}
        self.encoder = LabelEncoder()
        self.classes_: Any = np.array([])
        if self.backend == "catboost":
            defaults: dict[str, Any] = dict(
                iterations=200, depth=6, verbose=False, allow_writing_files=False
            )
            defaults.update(options)
            defaults.update(thread_count=threads, random_seed=seed, task_type="CPU")
            cls = CatBoostRegressor if self.regression else CatBoostClassifier
        elif self.backend == "lightgbm":
            defaults = dict(n_estimators=200, verbosity=-1)
            defaults.update(options)
            defaults.update(n_jobs=threads, random_state=seed, device_type="cpu")
            cls = LGBMRegressor if self.regression else LGBMClassifier
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
        self.model: Any = cls(**defaults)

    def _transform(self, frame: Any) -> Any:
        result = frame.copy()
        for column, vocabulary in self.categories.items():
            values = frame[column].fillna("__missing__").astype(str)
            if self.backend == "catboost":
                result[column] = values
            else:
                result[column] = pd.Categorical(values, categories=vocabulary)
        for column, median in self.medians.items():
            result[column] = (
                pd.to_numeric(frame[column], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .fillna(median)
            )
        return result

    def fit(self, X: Any, y: Any, validation: Any = None) -> "TabularCandidate":
        self.categories.clear()
        self.medians.clear()
        for column in X.columns:
            if not pd.api.types.is_numeric_dtype(X[column]):
                self.categories[column] = sorted(
                    X[column].fillna("__missing__").astype(str).unique().tolist()
                )
            else:
                median = X[column].replace([np.inf, -np.inf], np.nan).median()
                self.medians[column] = 0.0 if pd.isna(median) else float(median)
        labels = y if self.regression else self.encoder.fit_transform(y)
        if not self.regression:
            self.classes_ = self.encoder.classes_
        kwargs: dict[str, Any] = (
            {"cat_features": list(self.categories)} if self.backend == "catboost" else {}
        )
        if validation is not None:
            valid_X, valid_y = validation
            valid_labels = valid_y if self.regression else self.encoder.transform(valid_y)
            kwargs["eval_set"] = (self._transform(valid_X), valid_labels)
            if self.backend == "catboost":
                kwargs.update(early_stopping_rounds=20, use_best_model=True)
            else:
                from lightgbm import early_stopping

                kwargs["callbacks"] = [early_stopping(20, verbose=False)]
        self.model.fit(self._transform(X), labels, **kwargs)
        self.selected_iterations_ = (
            int(self.model.tree_count_)
            if self.backend == "catboost"
            else int(self.model.best_iteration_ or self.model.n_estimators_)
        )
        return self

    def predict(self, X: Any) -> Any:
        predictions = np.asarray(self.model.predict(self._transform(X))).reshape(-1)
        if not self.regression:
            predictions = self.encoder.inverse_transform(predictions.astype(int))
        return predictions

    def predict_proba(self, X: Any) -> Any:
        return self.model.predict_proba(self._transform(X))


def build_model(kind: str, parameters: dict[str, Any], threads: int, seed: int) -> Any:
    """Build an unfitted CPU candidate; optional engines are imported only on demand."""
    if parameters.get("backend") == "linear":
        from ml_harness.adapters.ml.linear import LinearCandidate

        return LinearCandidate(kind, parameters, threads, seed)
    if parameters.get("backend") == "flaml":
        from ml_harness.adapters.ml.flaml import FlamlCandidate

        return FlamlCandidate(kind, parameters, threads, seed)
    tuning = parameters.get("tuning", {})
    if tuning and tuning.get("enabled", True):
        from ml_harness.adapters.ml.tuning import TunedCandidate

        return TunedCandidate(kind, parameters, threads, seed)
    if parameters.get("backend") in ("autogluon", "autogluon-comparison"):
        from ml_harness.adapters.ml.automl import AutoGluonCandidate

        options = dict(parameters)
        options.pop("tuning", None)
        if options.get("backend") == "autogluon-comparison":
            options["backend"] = "autogluon"
            options["hyperparameters"] = {"CAT": {}, "GBM": {}, "XGB": {}}
            options["comparison"] = True
        return AutoGluonCandidate(kind, options, threads, seed)
    options = dict(parameters)
    options.pop("tuning", None)
    return TabularCandidate(kind, options, threads, seed)
