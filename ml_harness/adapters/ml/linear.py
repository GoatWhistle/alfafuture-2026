"""Cheap comparison-only linear learner with train-fitted preprocessing."""

from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from sklearn.compose import ColumnTransformer  # type: ignore[import-untyped]
from sklearn.impute import SimpleImputer  # type: ignore[import-untyped]
from sklearn.linear_model import LogisticRegression, Ridge  # type: ignore[import-untyped]
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # type: ignore[import-untyped]

from ml_harness.domain.models import ContractError


class LinearCandidate:
    def __init__(self, kind: str, parameters: dict[str, Any], threads: int, seed: int):
        if set(parameters) - {"backend"}:
            raise ContractError("Linear comparison accepts no model overrides")
        self.kind, self.seed = kind, seed
        self.classes_: Any = np.array([])
        self.model: Any = None
        self.categorical: list[str] = []

    def fit(self, X: Any, y: Any) -> "LinearCandidate":
        numeric = [c for c in X if pd.api.types.is_numeric_dtype(X[c])]
        categorical = [c for c in X if c not in numeric]
        self.categorical = categorical
        X = self._prepare(X)
        preprocessing = ColumnTransformer(
            [
                (
                    "numeric",
                    Pipeline(
                        [
                            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                            ("scale", StandardScaler()),
                        ]
                    ),
                    numeric,
                ),
                (
                    "category",
                    Pipeline(
                        [
                            (
                                "impute",
                                SimpleImputer(strategy="most_frequent", keep_empty_features=True),
                            ),
                            ("encode", OneHotEncoder(handle_unknown="ignore")),
                        ]
                    ),
                    categorical,
                ),
            ]
        )
        learner = (
            Ridge(solver="lsqr")
            if self.kind == "regression"
            else LogisticRegression(max_iter=300, random_state=self.seed, n_jobs=1)
        )
        self.model = Pipeline([("preprocess", preprocessing), ("model", learner)])
        self.model.fit(X, y)
        if self.kind != "regression":
            self.classes_ = self.model.classes_
        return self

    def _prepare(self, X: Any) -> Any:
        frame = X.copy()
        for column in self.categorical:
            frame[column] = frame[column].map(
                lambda value: str(value) if pd.notna(value) else np.nan
            )
        return frame

    def predict(self, X: Any) -> Any:
        return self.model.predict(self._prepare(X))

    def predict_proba(self, X: Any) -> Any:
        return self.model.predict_proba(self._prepare(X))
