"""A tuned recipe fits fresh train-only transformations and freezes its final model."""

import numpy as np
import pandas as pd
import pytest

from ml_harness.adapters.ml.feature_recipe import FeatureRecipe


@pytest.mark.parametrize("backend", ["catboost", "lightgbm"])
@pytest.mark.parametrize("strategy", ["group", "time"])
def test_tuned_feature_pipeline_keeps_split_metadata_and_freezes_refit(backend, strategy):
    frame = pd.DataFrame({"x": range(100), "group": np.repeat(range(10), 10), "time": range(100)})
    target = pd.Series(np.sin(np.arange(100)))
    source = """
class Transformer:
    def fit(self, X):
        self.rows = set(X.index)
    def transform(self, X):
        return X.assign(useful=X.x ** 2)
def build_transformer():
    return Transformer()
"""
    options = {
        "backend": backend,
        "iterations" if backend == "catboost" else "n_estimators": 8,
        "tuning": {"trials": 2, "time_limit": 10},
    }
    transforms = [{"operation": "drop", "columns": ["group", "time"]}]
    model = FeatureRecipe("regression", options, 1, 42, transforms, source)
    model.configure_validation({"kind": strategy, "group_column": "group", "time_column": "time"})
    model.fit(frame, target)
    final_pipeline = model.model.model
    assert final_pipeline.custom.rows == set(frame.index)
    assert "useful" in final_pipeline.columns
    assert "group" not in final_pipeline.columns
    selected = model.tuning_report_["best"]["refit_params"]
    assert selected["iterations" if backend == "catboost" else "n_estimators"] >= 1
    refit = FeatureRecipe("regression", model.refit_parameters_, 1, 42, transforms, source)
    refit.fit(frame, target)
    np.testing.assert_allclose(refit.predict(frame), model.predict(frame))


def test_transform_fit_never_sees_inner_validation(monkeypatch):
    observed = []
    original = FeatureRecipe._transform

    def record(self, frame, fitting=False):
        if fitting:
            observed.append(set(frame.index))
        return original(self, frame, fitting)

    monkeypatch.setattr(FeatureRecipe, "_transform", record)
    frame = pd.DataFrame({"x": range(50), "category": ["a", "b"] * 25})
    model = FeatureRecipe(
        "regression",
        {"backend": "catboost", "iterations": 3, "tuning": {"trials": 2}},
        1,
        42,
        [{"operation": "frequency", "columns": ["category"]}],
        "",
    )
    model.fit(frame, pd.Series(range(50)))
    split = model.tuning_report_["inner_split"]
    assert observed[:2] == [set(split["train_indices"])] * 2
    assert not observed[0] & set(split["validation_indices"])
    assert observed[-1] == set(frame.index)


def test_study_resume_requires_same_pipeline_and_metric(tmp_path):
    from ml_harness.adapters.ml.tuning import TunedCandidate

    frame = pd.DataFrame({"x": range(40)})
    target = pd.Series(range(40))
    options = {
        "backend": "catboost",
        "iterations": 3,
        "tuning": {
            "trials": 1,
            "storage": f"sqlite:///{tmp_path / 'study.db'}",
            "study_name": "resume",
        },
    }
    first = TunedCandidate("regression", options, 1, 42)
    first.fit(frame, target)
    repeat = TunedCandidate("regression", options, 1, 42)
    repeat.fit(frame, target)
    assert len(repeat.tuning_report_["trials"]) == 2
    changed = TunedCandidate("regression", options, 1, 42)
    changed.configure_validation({"metric": "mae"})
    changed.fit(frame, target)
    assert len(changed.tuning_report_["trials"]) == 1
    assert first.tuning_report_["study_fingerprint"] != changed.tuning_report_["study_fingerprint"]
    transformed = FeatureRecipe(
        "regression",
        options,
        1,
        42,
        [{"operation": "signed_log1p", "columns": ["x"]}],
        "",
    )
    transformed.fit(frame, target)
    assert len(transformed.tuning_report_["trials"]) == 1
    assert (
        transformed.tuning_report_["study_fingerprint"] != first.tuning_report_["study_fingerprint"]
    )


def test_materialized_tuned_feature_pipeline_in_real_worker(tmp_path):
    from test_prepare_task import task_fixture

    from ml_harness.adapters.data.task import CsvTaskData
    from ml_harness.adapters.execution.runner import SubprocessRunner
    from ml_harness.adapters.llm.proposals import materialize
    from ml_harness.application.ports.execution import RunRequest
    from ml_harness.domain.models import ExperimentStatus
    from ml_harness.domain.proposals import ResolvedExperiment

    task = CsvTaskData().prepare(str(task_fixture(tmp_path, role="time")), 42)
    recipe = ResolvedExperiment(
        "regression",
        "catboost",
        {"iterations": 3},
        ({"operation": "drop", "columns": [task.spec.time_column]},),
        "",
        {"trials": 2, "time_limit": 5},
    )
    candidate = tmp_path / "candidate.py"
    candidate.write_text(materialize(recipe))
    result = SubprocessRunner().run(
        RunRequest(task, str(tmp_path / "worker"), candidate_path=str(candidate), threads=1),
        15,
    )
    assert result.status == ExperimentStatus.SUCCEEDED, result.detail
    assert result.output["refit_parameters"]["tuning"] == {"enabled": False}
    assert result.output["tuning_report"]["inner_split"]["kind"] == "time"


@pytest.mark.parametrize("backend", ["catboost", "lightgbm"])
def test_classification_tuning_uses_shared_balancing_catalog(backend):
    from ml_harness.adapters.ml.tuning import TunedCandidate
    from ml_harness.domain.proposals import CLASSIFICATION_CHOICES

    frame = pd.DataFrame({"x": range(60)})
    target = pd.Series(["a", "b"] * 30)
    model = TunedCandidate(
        "binary",
        {
            "backend": backend,
            "iterations" if backend == "catboost" else "n_estimators": 3,
            "tuning": {"trials": 2},
        },
        1,
        42,
    ).fit(frame, target)
    for trial in model.tuning_report_["trials"]:
        for name, choices in CLASSIFICATION_CHOICES[backend].items():
            assert name in trial["params"] and trial["params"][name] in choices
    for name in CLASSIFICATION_CHOICES[backend]:
        assert name in model.refit_parameters_
