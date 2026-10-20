"""Train-fitted feature recipes with row/schema checks, inside the existing worker."""

from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from ml_harness.adapters.ml.candidates import build_model
from ml_harness.domain.models import ContractError


class FeatureRecipe:
    def __init__(
        self,
        kind: str,
        options: dict[str, Any],
        threads: int,
        seed: int,
        transforms: list[dict[str, Any]],
        python_source: str,
    ):
        self.model = build_model(kind, options, threads, seed)
        self.backend = options["backend"]
        if hasattr(self.model, "set_feature_pipeline"):
            self.model.set_feature_pipeline(transforms, python_source)
        self.transforms, self.python_source = transforms, python_source
        self.frequencies: dict[str, Any] = {}
        self.custom: Any = None
        self.columns: list[str] | None = None

    def configure_validation(self, context: dict[str, Any]) -> None:
        configure = getattr(self.model, "configure_validation", None)
        if configure:
            configure(context)

    def _transform(self, X: Any, fitting: bool = False) -> Any:
        frame = X.copy(deep=True)
        if fitting and self.python_source:
            namespace: dict[str, Any] = {}
            exec(compile(self.python_source, "<feature_transform>", "exec"), namespace)
            self.custom = namespace["build_transformer"]()
        for operation in self.transforms:
            columns = operation["columns"]
            if operation["operation"] == "drop":
                frame = frame.drop(columns=columns)
            elif operation["operation"] == "signed_log1p":
                for column in columns:
                    values = pd.to_numeric(frame[column], errors="raise")
                    frame[column] = np.sign(values) * np.log1p(np.abs(values))
            else:
                for column in columns:
                    values = frame[column].fillna("__missing__").astype(str)
                    if fitting:
                        self.frequencies[column] = values.value_counts(normalize=True)
                    frame[column + "__frequency"] = values.map(self.frequencies[column]).fillna(0)
        if self.custom is not None:
            if fitting:
                self.custom.fit(frame.copy(deep=True))  # Targets deliberately absent.
            frame = self.custom.transform(frame)
        if (
            not isinstance(frame, pd.DataFrame)
            or not frame.index.equals(X.index)
            or not frame.columns.is_unique
            or len(frame.columns) == 0
            or not all(isinstance(c, str) for c in frame.columns)
        ):
            raise ContractError(
                "Transform must preserve row index/order and unique feature columns"
            )
        if fitting:
            self.columns = list(frame.columns)
        elif list(frame.columns) != self.columns:
            raise ContractError("Transform changed its fitted feature schema")
        return frame

    def fit(self, X: Any, y: Any) -> "FeatureRecipe":
        if hasattr(self.model, "set_feature_pipeline"):
            try:
                self.model.fit(X, y)
            finally:
                self.tuning_report_ = self.model.tuning_report_
            self.refit_parameters_ = self.model.refit_parameters_
        else:
            self.model.fit(self._transform(X, fitting=True), y)
        self.classes_ = self.model.classes_
        return self

    def fit_inner(self, X: Any, y: Any, validation: Any) -> "FeatureRecipe":
        valid_X, valid_y = validation
        train = self._transform(X, fitting=True)
        self.model.fit(train, y, validation=(self._transform(valid_X), valid_y))
        self.classes_ = self.model.classes_
        self.selected_iterations_ = self.model.selected_iterations_
        return self

    def predict(self, X: Any) -> Any:
        return self.model.predict(
            X if hasattr(self.model, "set_feature_pipeline") else self._transform(X)
        )

    def predict_proba(self, X: Any) -> Any:
        return self.model.predict_proba(
            X if hasattr(self.model, "set_feature_pipeline") else self._transform(X)
        )
