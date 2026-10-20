"""Optional AutoGluon restricted to explicitly allowed tabular CPU learners."""

import copy
import importlib
import math
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from sklearn import metrics  # type: ignore[import-untyped]
from sklearn.model_selection import (  # type: ignore[import-untyped]
    GroupShuffleSplit,
    train_test_split,
)

from ml_harness.domain.models import ContractError


def _multiclass_auc(y_true: Any, y_pred: Any) -> float:
    """Match the independent evaluator's macro OVR AUC and encoded class order."""
    return float(
        metrics.roc_auc_score(
            y_true,
            y_pred,
            multi_class="ovr",
            average="macro",
            labels=list(range(np.asarray(y_pred).shape[1])),
        )
    )


class AutoGluonCandidate:
    def __init__(self, kind: str, parameters: dict[str, Any], threads: int, seed: int):
        options = dict(parameters)
        options.pop("backend", None)
        self.comparison = options.pop("comparison", False)
        self.time_limit = options.pop("time_limit", 600 if self.comparison else 60)
        self.hyperparameters = copy.deepcopy(options.pop("hyperparameters", {"CAT": {}}))
        if (
            options
            or type(self.comparison) is not bool
            or not isinstance(self.time_limit, int | float)
            or not math.isfinite(self.time_limit)
            or not 0 < self.time_limit <= 600
        ):
            raise ContractError(
                "AutoGluon requires a time limit in (0, 600] and explicit parameters"
            )
        if (
            not isinstance(self.hyperparameters, dict)
            or not self.hyperparameters
            or set(self.hyperparameters) - {"CAT", "GBM", "XGB"}
        ):
            raise ContractError("AutoGluon allows CAT/GBM/XGB hyperparameters only")
        for config in self.hyperparameters.values():
            configs = config if isinstance(config, list) else [config]
            if not configs or any(
                not isinstance(item, dict) or any(str(key).startswith("ag_args") for key in item)
                for item in configs
            ):
                raise ContractError("AutoGluon custom resource/model overrides are not allowed")
        # Native backend options cannot override the CPU resource boundary.
        for family, config in self.hyperparameters.items():
            for item in config if isinstance(config, list) else [config]:
                if family == "CAT":
                    item.update(task_type="CPU", thread_count=threads, random_seed=seed)
                elif family == "GBM":
                    item.update(device="cpu", device_type="cpu", num_threads=threads, seed=seed)
                else:
                    item.update(device="cpu", tree_method="hist", n_jobs=threads, random_state=seed)
                    item.pop("predictor", None)
        self.kind, self.threads, self.seed = kind, threads, seed
        self.context: dict[str, Any] = {}
        self.inner_split_: dict[str, Any] = {}
        self.predictor: Any = None
        self.classes_: Any = np.array([])
        self.path_: str | None = None

    def configure_validation(self, context: dict[str, Any]) -> None:
        self.context = dict(context)

    def _split(self, X: Any, y: Any) -> tuple[Any, Any]:
        strategy = self.context.get("kind", "random" if self.kind == "regression" else "stratified")
        indices = np.arange(len(X))
        try:
            if len(X) < 5:
                raise ValueError("At least five training rows are required")
            if strategy == "group":
                column = self.context.get("group_column")
                if column not in X or X[column].isna().any() or X[column].nunique() < 2:
                    raise ValueError("Group split requires a complete column and two groups")
                train, valid = next(
                    GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=self.seed).split(
                        X, y, X[column]
                    )
                )
            elif strategy == "time":
                column = self.context.get("time_column")
                if column not in X or X[column].isna().any():
                    raise ValueError("Time split requires a complete time column")
                times = pd.to_numeric(X[column], errors="coerce")
                if times.isna().any():
                    times = pd.to_datetime(X[column], errors="raise", utc=True)
                order = np.argsort(np.asarray(times), kind="stable")
                boundary = times.iloc[order[min(len(X) - 1, max(1, int(len(X) * 0.8)))]]
                train, valid = (
                    indices[np.asarray(times < boundary)],
                    indices[np.asarray(times >= boundary)],
                )
            elif strategy in ("random", "stratified"):
                train, valid = train_test_split(
                    indices,
                    test_size=0.2,
                    random_state=self.seed,
                    stratify=y if strategy == "stratified" else None,
                )
            else:
                raise ValueError("Unknown inner split strategy")
            if not len(train) or not len(valid):
                raise ValueError("Inner partitions must be nonempty")
            if self.kind != "regression":
                if set(y.iloc[train]) != set(y):
                    raise ValueError("Inner training split is missing classes")
                if self.context.get("metric") == "roc_auc" and set(y.iloc[valid]) != set(y):
                    raise ValueError("AUC inner validation split is missing classes")
            self.inner_split_ = {
                "kind": strategy,
                "train_indices": train.tolist(),
                "validation_indices": valid.tolist(),
            }
            return train, valid
        except (ValueError, KeyError, TypeError) as error:
            raise ContractError(f"Cannot construct AutoGluon inner split: {error}") from error

    def _metric(self) -> Any:
        metric = self.context.get("metric", "rmse" if self.kind == "regression" else "macro_f1")
        mapping = {
            "rmse": "root_mean_squared_error",
            "mae": "mean_absolute_error",
            "macro_f1": "f1_macro",
            "accuracy": "accuracy",
            "log_loss": "log_loss",
            "roc_auc": "roc_auc",
        }
        if metric not in mapping:
            raise ContractError("Unsupported AutoGluon evaluation metric")
        if metric == "roc_auc" and self.kind == "multiclass":
            make_scorer = importlib.import_module("autogluon.core.metrics").make_scorer
            return make_scorer(
                name="harness_roc_auc_ovr",
                score_func=_multiclass_auc,
                optimum=1,
                greater_is_better=True,
                needs_proba=True,
            )
        return mapping[metric]

    def fit(self, X: Any, y: Any) -> "AutoGluonCandidate":
        try:
            predictor_type = importlib.import_module("autogluon.tabular").TabularPredictor
        except ImportError:
            raise ContractError("Optional pinned AutoGluon extra is not installed") from None
        label = "__harness_target__"
        while label in X.columns:
            label += "_"
        data = X.copy()
        data[label] = np.asarray(y)
        train, valid = self._split(X, y)
        eval_metric = self._metric()
        self.path_ = tempfile.mkdtemp(prefix="harness-autogluon-", dir=Path.cwd())
        problem = (
            "regression"
            if self.kind == "regression"
            else ("binary" if self.kind == "binary" else "multiclass")
        )
        self.predictor = predictor_type(
            label=label,
            problem_type=problem,
            path=self.path_,
            verbosity=0,
            eval_metric=eval_metric,
            learner_kwargs={"random_state": self.seed},
        )
        self.predictor.fit(
            data.iloc[train],
            tuning_data=data.iloc[valid],
            time_limit=self.time_limit,
            hyperparameters=self.hyperparameters,
            num_cpus=self.threads,
            num_gpus=0,
            num_bag_folds=0,
            num_stack_levels=0,
            auto_stack=False,
            dynamic_stacking=False,
            fit_weighted_ensemble=self.comparison,
            ag_args_fit={"num_cpus": self.threads, "num_gpus": 0},
            calibrate=False,
            calibrate_decision_threshold=False,
        )
        selected = self.predictor.model_best
        refitted = self.predictor.refit_full(
            model="best",
            set_best_to_refit_full=True,
            num_cpus=self.threads,
            num_gpus=0,
            fit_strategy="sequential",
        )
        if selected not in refitted or self.predictor.model_best != refitted[selected]:
            raise ContractError("AutoGluon did not activate a model refit on all supplied rows")
        if self.kind != "regression":
            self.classes_ = np.asarray(self.predictor.class_labels)
        return self

    def predict(self, X: Any) -> Any:
        return np.asarray(self.predictor.predict(X)).reshape(-1)

    def predict_proba(self, X: Any) -> Any:
        probability = self.predictor.predict_proba(X, as_multiclass=True)
        return np.asarray(probability.loc[:, self.classes_.tolist()])
