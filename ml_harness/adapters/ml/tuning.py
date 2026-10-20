"""Bounded tuning on an inner split of the training rows supplied by the worker."""

import hashlib
import json
import math
from typing import Any

import numpy as np
import optuna
import pandas as pd  # type: ignore[import-untyped]
from sklearn import metrics  # type: ignore[import-untyped]
from sklearn.model_selection import (  # type: ignore[import-untyped]
    GroupShuffleSplit,
    train_test_split,
)

from ml_harness.domain.models import ContractError
from ml_harness.domain.proposals import CLASSIFICATION_CHOICES, RANGES


class TunedCandidate:
    def __init__(self, kind: str, parameters: dict[str, Any], threads: int, seed: int):
        self.kind, self.threads, self.seed = kind, threads, seed
        self.parameters = dict(parameters)
        options = self.parameters.pop("tuning", {})
        self.storage = options.get("storage")
        self.study_name = options.get("study_name")
        self.trials = options.get("trials", 5)
        self.time_limit = options.get("time_limit", 60)
        if (
            isinstance(self.trials, bool)
            or not isinstance(self.trials, int)
            or not 1 <= self.trials <= 10
            or not isinstance(self.time_limit, int | float)
            or not math.isfinite(self.time_limit)
            or not 0 < self.time_limit <= 300
        ):
            raise ContractError("Tuning requires 1..10 trials and a time limit in (0, 300]")
        if self.parameters.get("backend", "catboost") not in ("catboost", "lightgbm"):
            raise ContractError("Tuning supports CatBoost and LightGBM only")
        self.transforms: list[dict[str, Any]] = []
        self.python_source = ""
        self.context: dict[str, Any] = {}
        self.tuning_report_: dict[str, Any] = {}
        self.model: Any = None
        self.classes_: Any = np.array([])

    def set_feature_pipeline(self, transforms: list[dict[str, Any]], source: str) -> None:
        self.transforms, self.python_source = transforms, source

    def _candidate(self, parameters: dict[str, Any]) -> Any:
        from ml_harness.adapters.ml.candidates import TabularCandidate
        from ml_harness.adapters.ml.feature_recipe import FeatureRecipe

        if self.transforms or self.python_source:
            return FeatureRecipe(
                self.kind,
                {**parameters, "tuning": {"enabled": False}},
                self.threads,
                self.seed,
                self.transforms,
                self.python_source,
            )
        return TabularCandidate(self.kind, parameters, self.threads, self.seed)

    def configure_validation(self, context: dict[str, Any]) -> None:
        self.context = dict(context)

    def _split(self, X: Any, y: Any) -> tuple[Any, Any]:
        indices = np.arange(len(X))
        strategy = self.context.get("kind", "random" if self.kind == "regression" else "stratified")
        try:
            if len(X) < 5:
                raise ValueError("At least five training rows are required")
            if strategy == "group":
                column = self.context.get("group_column")
                if column not in X or X[column].isna().any() or X[column].nunique() < 2:
                    raise ValueError("Group split needs a complete group column and two groups")
                inner_train, validation = next(
                    GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=self.seed).split(
                        X, y, X[column]
                    )
                )
            elif strategy == "time":
                column = self.context.get("time_column")
                if column not in X or X[column].isna().any():
                    raise ValueError("Time split needs a complete time column")
                times = pd.to_numeric(X[column], errors="coerce")
                if times.isna().any():
                    times = pd.to_datetime(X[column], errors="raise", utc=True)
                ordered = np.argsort(np.asarray(times), kind="stable")
                boundary = times.iloc[ordered[min(len(X) - 1, max(1, int(len(X) * 0.8)))]]
                inner_train = indices[np.asarray(times < boundary)]
                validation = indices[np.asarray(times >= boundary)]
            elif strategy in ("random", "stratified"):
                inner_train, validation = train_test_split(
                    indices,
                    test_size=0.2,
                    random_state=self.seed,
                    stratify=y if strategy == "stratified" else None,
                )
            else:
                raise ValueError("Unknown inner split strategy")
            if not len(inner_train) or not len(validation):
                raise ValueError("Inner partitions must be nonempty")
            if self.kind != "regression" and set(y.iloc[inner_train]) != set(y):
                raise ValueError("Inner training partition is missing classes")
            return inner_train, validation
        except (ValueError, KeyError, TypeError) as error:
            raise ContractError(f"Cannot construct inner training split: {error}") from error

    def _score(self, model: Any, X: Any, y: Any, metric: str) -> float:
        if metric in ("roc_auc", "log_loss"):
            probability = model.predict_proba(X)
            classes = model.classes_.tolist()
            labels = [classes.index(value) for value in y]
            if metric == "log_loss":
                return float(
                    metrics.log_loss(labels, probability, labels=list(range(len(classes))))
                )
            if len(classes) == 2:
                return float(metrics.roc_auc_score(labels, probability[:, 1]))
            return float(
                metrics.roc_auc_score(
                    labels, probability, multi_class="ovr", labels=list(range(len(classes)))
                )
            )
        prediction = model.predict(X)
        if metric == "rmse":
            return float(metrics.root_mean_squared_error(y, prediction))
        if metric == "mae":
            return float(metrics.mean_absolute_error(y, prediction))
        if metric in ("accuracy", "macro_f1"):
            classes = model.classes_.tolist()
            encoded_y = [classes.index(value) for value in y]
            encoded_prediction = [classes.index(value) for value in prediction]
        if metric == "accuracy":
            return float(metrics.accuracy_score(encoded_y, encoded_prediction))
        if metric == "macro_f1":
            return float(
                metrics.f1_score(
                    encoded_y,
                    encoded_prediction,
                    average="macro",
                    zero_division=0,
                    labels=list(range(len(classes))),
                )
            )
        raise ContractError("Unsupported tuning metric")

    def fit(self, X: Any, y: Any) -> "TunedCandidate":
        inner_train, validation = self._split(X, y)
        metric = self.context.get("metric", "rmse" if self.kind == "regression" else "macro_f1")
        if metric not in ("rmse", "mae", "macro_f1", "accuracy", "roc_auc", "log_loss"):
            raise ContractError("Unsupported tuning metric")
        split = {
            "kind": self.context.get(
                "kind", "random" if self.kind == "regression" else "stratified"
            ),
            "train_indices": inner_train.tolist(),
            "validation_indices": validation.tolist(),
        }
        fingerprint = hashlib.sha256(json.dumps(split, sort_keys=True).encode())
        fingerprint.update(pd.util.hash_pandas_object(X, index=True).values.tobytes())
        fingerprint.update(pd.util.hash_pandas_object(y, index=True).values.tobytes())
        compatibility = {
            "version": 3,
            "kind": self.kind,
            "metric": metric,
            "parameters": self.parameters,
            "transforms": self.transforms,
            "source": self.python_source,
            "space": {
                k: [low, high, typ.__name__]
                for k, (low, high, typ) in RANGES[
                    self.parameters.get("backend", "catboost")
                ].items()
            },
            "classification_choices": CLASSIFICATION_CHOICES if self.kind != "regression" else {},
            "seed": self.seed,
            "context": self.context,
            "columns": [(str(c), str(X[c].dtype)) for c in X.columns],
        }
        study_fingerprint = hashlib.sha256(
            fingerprint.digest() + json.dumps(compatibility, sort_keys=True).encode()
        ).hexdigest()
        study = optuna.create_study(
            storage=self.storage,
            study_name=(
                str(self.study_name)
                + ":"
                + self.context.get("level", "full")
                + ":"
                + study_fingerprint
            )
            if self.storage
            else None,
            load_if_exists=bool(self.storage),
            direction="minimize" if metric in ("rmse", "mae", "log_loss") else "maximize",
            sampler=optuna.samplers.TPESampler(seed=self.seed),
        )

        def objective(trial: optuna.Trial) -> float:
            params = dict(self.parameters)
            backend = params.get("backend", "catboost")
            for name, (low, high, kind) in RANGES[backend].items():
                if name in ("iterations", "n_estimators"):
                    high = min(high, params.get(name, 200))
                    low = min(low, high)
                if kind is int:
                    params[name] = trial.suggest_int(name, int(low), int(high))
                else:
                    params[name] = trial.suggest_float(name, low, high, log=low > 0)
            if self.kind != "regression":
                for name, choices in CLASSIFICATION_CHOICES[backend].items():
                    params[name] = trial.suggest_categorical(name, choices)
            model = self._candidate(params)
            valid = (X.iloc[validation], y.iloc[validation])
            if hasattr(model, "fit_inner"):
                model.fit_inner(X.iloc[inner_train], y.iloc[inner_train], valid)
            else:
                model.fit(X.iloc[inner_train], y.iloc[inner_train], validation=valid)
            trial.set_user_attr("selected_iterations", model.selected_iterations_)
            value = self._score(model, X.iloc[validation], y.iloc[validation], metric)
            if not math.isfinite(value):
                raise ContractError("Inner metric must be finite")
            return value

        study.optimize(
            objective,
            n_trials=self.trials,
            timeout=self.time_limit,
            n_jobs=1,
            show_progress_bar=False,
            catch=(Exception,),
        )
        self.tuning_report_ = {
            "trials": [
                {
                    "number": trial.number,
                    "params": trial.params,
                    "value": trial.value,
                    "state": trial.state.name,
                }
                for trial in study.trials
            ],
            "best": None,
            "inner_split_fingerprint": fingerprint.hexdigest(),
            "study_fingerprint": study_fingerprint,
            "inner_split": split,
            "metric": metric,
        }
        try:
            best = study.best_trial
        except ValueError:
            raise ContractError("No inner tuning trial completed successfully") from None
        self.tuning_report_["best"] = {
            "number": best.number,
            "params": best.params,
            "value": best.value,
        }
        selected = dict(best.params)
        key = (
            "iterations"
            if self.parameters.get("backend", "catboost") == "catboost"
            else "n_estimators"
        )
        if best.user_attrs.get("selected_iterations"):
            selected[key] = best.user_attrs["selected_iterations"]
        self.tuning_report_["best"]["refit_params"] = selected
        self.model = self._candidate({**self.parameters, **selected})
        self.model.fit(X, y)
        self.refit_parameters_ = {**self.parameters, **selected, "tuning": {"enabled": False}}
        self.classes_ = self.model.classes_
        return self

    def predict(self, X: Any) -> Any:
        return self.model.predict(X)

    def predict_proba(self, X: Any) -> Any:
        return self.model.predict_proba(X)
