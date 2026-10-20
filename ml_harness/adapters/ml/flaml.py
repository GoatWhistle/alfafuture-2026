"""Optional FLAML search, followed by a frozen full outer-train refit."""

import importlib
import math
from typing import Any

import numpy as np

from ml_harness.domain.models import ContractError

_ESTIMATORS = ("lgbm", "catboost", "xgboost")


class FlamlCandidate:
    def __init__(self, kind: str, parameters: dict[str, Any], threads: int, seed: int):
        options = dict(parameters)
        options.pop("backend", None)
        self.time_limit = options.pop("time_limit", 600)
        self.selected_estimator = options.pop("selected_estimator", None)
        self.selected_config = options.pop("selected_config", None)
        if (
            options
            or isinstance(self.time_limit, bool)
            or not isinstance(self.time_limit, int | float)
            or not math.isfinite(self.time_limit)
            or not 0 < self.time_limit <= 600
            or (self.selected_estimator is None) != (self.selected_config is None)
            or (
                self.selected_estimator is not None
                and (
                    self.selected_estimator not in _ESTIMATORS
                    or not isinstance(self.selected_config, dict)
                )
            )
        ):
            raise ContractError("Invalid FLAML comparison parameters")
        self.kind, self.threads, self.seed = kind, threads, seed
        self.context: dict[str, Any] = {}
        self.classes_: Any = np.array([])
        self.model: Any = None
        self.transformer: Any = None

    def configure_validation(self, context: dict[str, Any]) -> None:
        self.context = dict(context)

    def fit(self, X: Any, y: Any) -> "FlamlCandidate":
        from ml_harness.adapters.ml.tuning import TunedCandidate

        try:
            automl_class = importlib.import_module("flaml").AutoML
            transformer_class = importlib.import_module("flaml.automl.data").DataTransformer
            task_factory = importlib.import_module("flaml.automl.task.factory").task_factory
        except ImportError:
            raise ContractError("Optional pinned FLAML extra is not installed") from None
        task = task_factory("regression" if self.kind == "regression" else "classification", X, y)
        if self.selected_estimator is None:
            splitter = TunedCandidate(self.kind, {"backend": "catboost"}, self.threads, self.seed)
            splitter.configure_validation(self.context)
            train, valid = splitter._split(X, y)
            metric = self.context.get("metric", "rmse" if self.kind == "regression" else "macro_f1")
            if metric == "roc_auc" and self.kind == "multiclass":
                metric = "roc_auc_ovr"
            if metric not in {
                "macro_f1",
                "accuracy",
                "roc_auc",
                "roc_auc_ovr",
                "log_loss",
                "rmse",
                "mae",
            }:
                raise ContractError("Unsupported FLAML comparison metric")
            search = automl_class()
            search.fit(
                X_train=X.iloc[train],
                y_train=y.iloc[train],
                X_val=X.iloc[valid],
                y_val=y.iloc[valid],
                task=task,
                estimator_list=list(_ESTIMATORS),
                metric=metric,
                time_budget=self.time_limit,
                n_jobs=self.threads,
                seed=self.seed,
                eval_method="holdout",
                retrain_full=False,
                verbose=0,
                model_history=False,
                keep_search_state=False,
                auto_augment=False,
            )
            self.selected_estimator = search.best_estimator
            self.selected_config = search.best_config
            if self.selected_estimator not in _ESTIMATORS or not isinstance(
                self.selected_config, dict
            ):
                raise ContractError("FLAML completed without a fitted candidate")
            # Resource callbacks may stop a trial before its configured tree count.
            fitted_estimator = search.model.estimator
            rounds = None
            if self.selected_estimator == "catboost":
                rounds = fitted_estimator.tree_count_
            elif self.selected_estimator == "lgbm":
                rounds = fitted_estimator.n_estimators_
            elif self.selected_estimator == "xgboost":
                rounds = fitted_estimator.get_booster().num_boosted_rounds()
            if rounds is not None and rounds > 0:
                self.selected_config = {**self.selected_config, "n_estimators": int(rounds)}
        # Explicit refit is required: FLAML retrain_full does not include an explicit X_val.
        # New preprocessing is fitted on every supplied outer-train row, never outer validation.
        self.transformer = transformer_class()
        features, targets = self.transformer.fit_transform(X, y, task)
        estimator_class = task.estimator_class_from_str(self.selected_estimator)
        seed_parameter = "random_seed" if self.selected_estimator == "catboost" else "random_state"
        self.model = estimator_class(
            task=task, **{**self.selected_config, "n_jobs": self.threads, seed_parameter: self.seed}
        )
        if self.selected_estimator == "catboost":
            # FLAML's wrapper otherwise reserves another 10% and early-stops again.
            self.model.fit(features, targets, use_best_model=False)
        else:
            self.model.fit(features, targets)
        if self.kind != "regression":
            self.classes_ = np.asarray(self.transformer.label_transformer.classes_)
        self.refit_parameters_ = {
            "backend": "flaml",
            "time_limit": self.time_limit,
            "selected_estimator": self.selected_estimator,
            "selected_config": self.selected_config,
        }
        self.tuning_report_ = {
            "engine": "flaml",
            "selected_estimator": self.selected_estimator,
            "selected_parameters": self.selected_config,
            "refit_rows": len(X),
        }
        return self

    def predict(self, X: Any) -> Any:
        predictions = self.model.predict(self.transformer.transform(X))
        if self.kind != "regression":
            return self.transformer.label_transformer.inverse_transform(
                np.asarray(predictions, dtype=int)
            )
        return predictions

    def predict_proba(self, X: Any) -> Any:
        return self.model.predict_proba(self.transformer.transform(X))
