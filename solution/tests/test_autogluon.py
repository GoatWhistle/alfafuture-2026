"""AutoGluon на фолдах схемы cv (пропускается, если autogluon не установлен)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("autogluon.tabular")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import models  # noqa: E402
import solution  # noqa: E402
from test_independence import panel  # noqa: E402


@pytest.mark.parametrize("cv", [{"kind": "kfold"}, {"kind": "group", "keys": ["obj"]},
                                {"kind": "temporal", "period": "per"}], ids=lambda c: c["kind"])
def test_autogluon_uses_scheme_folds(cv, monkeypatch):
    df = panel(n_obj=100)
    test = panel(n_obj=10, seed=1).drop(columns="y")
    prof = {"train": df, "test": test, "problem_type": "regression", "metric": "rmse", "ag_metric": "root_mean_squared_error",
            "features": ["x", "x2", "per"], "log_target": False, "output_proba": False, "cv": cv,
            "target_specs": [], "model_ops": {}}
    monkeypatch.setattr(models, "N_CPUS", 2)
    cand = solution.run_autogluon(prof, "y", 300)
    folds = models.outer_folds(prof, df, df["y"].to_numpy())
    covered = np.unique(np.concatenate([va for _, va in folds]))
    if cv["kind"] == "temporal":
        # Holdout-fit на прошлом: оценка — последний период.
        covered = folds[-1][1]
    assert np.array_equal(np.sort(cand["rows"]), df.index[covered].to_numpy())
    assert np.isfinite(cand["score"]) and len(cand["test"]) == len(test)
    kq = models.quick_oof(prof, "y")
    out = solution.align_candidates([cand, {"name": "quick_kfold", **kq}])
    assert len(out) == 2 and len(out[0]["rows"]) == len(covered)


def test_autogluon_classification_strings(monkeypatch):
    df = panel(n_obj=100)
    df["y"] = np.where(df["x"] + 0.3 * np.random.RandomState(0).randn(len(df)) > 0, "yes", "no")
    test = panel(n_obj=10, seed=1).drop(columns="y")
    prof = {"train": df, "test": test, "problem_type": "binary", "metric": "log_loss", "ag_metric": "log_loss",
            "features": ["x", "x2", "per"], "log_target": False, "output_proba": True, "cv": {"kind": "kfold"},
            "target_specs": [], "model_ops": {}}
    monkeypatch.setattr(models, "N_CPUS", 2)
    cand = solution.run_autogluon(prof, "y", 300)
    assert len(cand["rows"]) == len(df) and np.isfinite(cand["score"])
    assert np.corrcoef(cand["test_pred"], test["x"])[0, 1] > 0.5  # вероятность "yes"
    kq = models.quick_oof(prof, "y")
    out = solution.align_candidates([cand, {"name": "quick_kfold", **kq}])
    assert out[0]["classes"] == out[1]["classes"]


@pytest.fixture(autouse=True)
def _cleanup_ag():
    yield
    models.cleanup_autogluon()


def test_autogluon_temporal_gets_past_only_target_features(monkeypatch):
    from test_independence import SPECS

    df = panel(n_obj=100)
    test = panel(n_obj=10, seed=1).drop(columns="y")
    test["per"] = 5.0
    prof = {"train": df, "test": test, "problem_type": "regression", "metric": "rmse",
            "ag_metric": "root_mean_squared_error", "features": ["x", "x2", "per"], "log_target": False,
            "output_proba": False, "cv": {"kind": "temporal", "period": "per"}, "target_specs": [SPECS[0]],
            "model_ops": {}}
    monkeypatch.setattr(models, "N_CPUS", 2)
    fitted = []
    real_fit = solution.fit_autogluon

    def spy(*a, **k):
        fitted.append(k.get("extra"))
        return real_fit(*a, **k)

    monkeypatch.setattr(solution, "fit_autogluon", spy)
    cand = solution.run_autogluon(prof, "y", 700)
    assert all(e is not None and e.shape[1] > 0 for e in fitted)
    # Объектный уровень таргета из прошлого делает AutoGluon заметно точнее разброса таргета.
    assert cand["score"] < 0.5 * df["y"].std()
