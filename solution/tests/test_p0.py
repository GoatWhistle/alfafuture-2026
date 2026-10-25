"""Проверки P0 (issue #2): метрика, формат вывода, сохранение результата.

Запуск из solution/:
uvx --python 3.12 --with numpy==1.26.4 --with pandas==2.2.3 --with scikit-learn==1.5.2 \
    --with lightgbm==4.6.0 --with scipy==1.12.0 pytest tests -q
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.pop("OPENAI_API_KEY", None)

import models  # noqa: E402
import task_profile  # noqa: E402


def make_task(tmp: Path, train: pd.DataFrame, test: pd.DataFrame, sample: pd.DataFrame | None, text: str) -> Path:
    train.to_csv(tmp / "train.csv", index=False)
    test.to_csv(tmp / "test.csv", index=False)
    if sample is not None:
        sample.to_csv(tmp / "sample_submission.csv", index=False)
    (tmp / "task.md").write_text(text, encoding="utf-8")
    return tmp / "task.md"


def binary_frames(n=400, labels=(0, 1), seed=0):
    rng = np.random.RandomState(seed)
    x = rng.randn(n + 100)
    y = np.where(x + 0.5 * rng.randn(n + 100) > 0, labels[1], labels[0])
    train = pd.DataFrame({"id": np.arange(n), "x": x[:n], "z": rng.randn(n), "target": y[:n]})
    test = pd.DataFrame({"id": np.arange(n, n + 100), "x": x[n:], "z": rng.randn(100)})
    return train, test


# --- метрика ---------------------------------------------------------------

def test_rmsle_example():
    assert models.score("rmsle", [9, 99], [0, 9]) == pytest.approx(2.302585, abs=1e-6)


def test_rmse_stays_separate():
    assert models.score("rmse", [9, 99], [0, 9]) == pytest.approx(np.sqrt((81 + 8100) / 2))


def test_positive_label_rules():
    assert models.positive_label([0, 1]) == 1
    assert models.positive_label([True, False]) is True
    assert models.positive_label(["no", "yes"]) == "yes"
    assert models.positive_label(["good", "bad"]) == "bad"
    assert models.positive_label(["normal", "xyz"]) == "xyz"
    assert models.positive_label(["b", "a"]) == "b"


@pytest.mark.parametrize("labels", [(0, 1), ("no", "yes")])
def test_binary_f1_auc_numeric_and_string(labels):
    neg, pos = labels
    y = np.array([neg, pos, pos, neg, pos], dtype=object)
    pred = np.array([neg, pos, neg, neg, pos], dtype=object)
    p_pos = np.array([0.1, 0.9, 0.4, 0.3, 0.8])
    f1 = models.score("f1", y, pred)
    assert f1 == pytest.approx(0.8)
    order = [neg, pos]
    proba = np.column_stack([1 - p_pos, p_pos])
    auc = models.score("roc_auc", y, pred, proba=proba, labels=order)
    # Перестановка колонок с согласованным порядком классов не меняет результат.
    auc_perm = models.score("roc_auc", y, pred, proba=proba[:, ::-1], labels=order[::-1])
    assert auc == pytest.approx(1.0) and auc_perm == pytest.approx(auc)


def test_unknown_metric_raises():
    with pytest.raises(ValueError):
        models.score("weird", [1], [1])


def test_unscored_never_wins():
    assert not models.is_better("rmse", float("nan"), 1.0)
    assert not models.is_better("rmse", None, 1.0)
    assert not models.is_better("rmse", 0.5, float("nan"))
    assert models.is_better("rmse", 0.5, None)


# --- профиль и формат вывода ---------------------------------------------

def test_binary_logloss_zero_sample_requires_proba(tmp_path):
    train, test = binary_frames()
    sample = pd.DataFrame({"id": test["id"], "target": 0})
    task = make_task(tmp_path, train, test, sample, "Бинарная классификация. Метрика: log loss.")
    prof = task_profile.build_profile(task)
    assert prof["problem_type"] == "binary"
    assert prof["metric"] == "log_loss"
    assert prof["output_proba"] is True
    assert prof["pos_label"] == 1


@pytest.mark.parametrize("names", [["A", "B", "C"], ["target_A", "target_B", "target_C"], ["p_A", "p_B", "p_C"]])
def test_multiclass_proba_columns_without_dictionary(tmp_path, names):
    rng = np.random.RandomState(1)
    n = 300
    train = pd.DataFrame({"id": range(n), "x": rng.randn(n), "target": rng.choice(["A", "B", "C"], n)})
    test = pd.DataFrame({"id": range(n, n + 50), "x": rng.randn(50)})
    sample = pd.DataFrame({"id": test["id"], **{c: 0.0 for c in names[::-1]}})
    task = make_task(tmp_path, train, test, sample, "Предскажите вероятности классов. Метрика logloss.")
    prof = task_profile.build_profile(task)
    assert prof["target_cols"] == ["target"]
    assert prof["problem_type"] == "multiclass"
    assert prof["output_proba"] is True
    mapping = dict(prof["proba_cols"]["target"])
    assert {mapping[c] for c in names} == {"A", "B", "C"}
    assert mapping[names[0]] == "A"

    # Модель с другим порядком классов: колонки выравниваются по классам.
    classes = ["C", "A", "B"]
    proba = rng.dirichlet(np.ones(3), size=50)
    vals = models.output_from_proba(prof, "target", proba, classes)
    col_a = [c for c, _ in prof["proba_cols"]["target"]].index(names[0])
    assert np.allclose(vals[:, col_a], proba[:, 1])

    current = {"target": vals}
    assert models.write_submission(prof, current)
    out = pd.read_csv(tmp_path / "submission.csv")
    assert list(out.columns) == list(sample.columns)
    assert np.allclose(out[list(sample.columns[1:])].sum(axis=1), 1.0)

    const = {"target": models.constant_predictions(prof, "target")}
    assert models.write_submission(prof, const)


def test_explicit_regression_with_few_integers(tmp_path):
    rng = np.random.RandomState(2)
    n = 300
    train = pd.DataFrame({"id": range(n), "x": rng.randn(n), "target": rng.randint(1, 6, n)})
    test = pd.DataFrame({"id": range(n, n + 30), "x": rng.randn(30)})
    sample = pd.DataFrame({"id": test["id"], "target": 0})
    task = make_task(tmp_path, train, test, sample, "Постройте регрессионную модель рейтинга. Метрика MAE.")
    prof = task_profile.build_profile(task)
    assert prof["problem_type"] == "regression"
    assert prof["metric"] == "mae"


def test_multi_target_profiles_are_separate(tmp_path):
    rng = np.random.RandomState(3)
    n = 300
    train = pd.DataFrame({
        "id": range(n), "x": rng.randn(n),
        "price": rng.rand(n) * 100, "flag": rng.choice(["yes", "no"], n),
    })
    test = pd.DataFrame({"id": range(n, n + 30), "x": rng.randn(30)})
    sample = pd.DataFrame({"id": test["id"], "price": 0.0, "flag": "no"})
    task = make_task(tmp_path, train, test, sample, "Две цели: цена и флаг.")
    prof = task_profile.build_profile(task)
    pt = prof["per_target"]
    assert pt["price"]["problem_type"] == "regression"
    assert pt["flag"]["problem_type"] == "binary"
    assert pt["flag"]["pos_label"] == "yes"
    assert prof["features"] == ["x"]


# --- запись сабмита -------------------------------------------------------

def _binary_prof(tmp_path, proba=False):
    train, test = binary_frames()
    sample = pd.DataFrame({"id": test["id"], "target": 0})
    text = "Метрика ROC AUC." if proba else "Метрика accuracy."
    return task_profile.build_profile(make_task(tmp_path, train, test, sample, text))


def test_invalid_write_keeps_previous_file(tmp_path):
    prof = _binary_prof(tmp_path, proba=True)
    n = len(prof["test"])
    good = np.linspace(0.1, 0.9, n)
    current = {"target": good.copy()}
    assert models.write_submission(prof, current)
    before = (tmp_path / "submission.csv").read_bytes()

    for bad in (np.full(n, np.nan), np.full(n, 1.5), np.linspace(0, 1, n - 1)):
        current["target"] = bad
        assert not models.write_submission(prof, current)
        assert (tmp_path / "submission.csv").read_bytes() == before
        assert np.allclose(current["target"], good)
    assert not list(tmp_path.glob(".submission.*"))


def test_interrupted_write_keeps_previous_file(tmp_path, monkeypatch):
    prof = _binary_prof(tmp_path)
    n = len(prof["test"])
    current = {"target": np.zeros(n, dtype=object) + 1}
    assert models.write_submission(prof, current)
    before = (tmp_path / "submission.csv").read_bytes()

    def boom(self, path, *a, **k):
        Path(path).write_text("id,tar")
        raise KeyboardInterrupt

    monkeypatch.setattr(pd.DataFrame, "to_csv", boom)
    current["target"] = np.zeros(n, dtype=object)
    with pytest.raises(KeyboardInterrupt):
        models.write_submission(prof, current)
    assert (tmp_path / "submission.csv").read_bytes() == before
    assert not list(tmp_path.glob(".submission.*"))


# --- сохранение победителя -------------------------------------------------

def test_losing_quick_with_new_features_keeps_original(tmp_path, monkeypatch):
    import solution

    prof = _binary_prof(tmp_path)
    n = len(prof["test"])
    q1 = {"val_score": 0.9, "test_pred": np.ones(n, dtype=object), "proba": None}
    qb = {"val_score": 0.85, "test_pred": np.ones(n, dtype=object), "proba": None}
    q2 = {"val_score": 0.7, "test_pred": np.zeros(n, dtype=object), "proba": None}
    calls = iter([q1, qb, q2])
    monkeypatch.setattr(solution, "quick_model", lambda p, t, **kw: next(calls))
    monkeypatch.setattr(solution, "run_autoeda", lambda p, t, b: {"period": None, "label_copy": False,
                                                                   "accepted": [{"op": "drop"}]})
    monkeypatch.setattr(solution, "fit_autogluon", lambda *a: (_ for _ in ()).throw(RuntimeError("off")))
    monkeypatch.setattr(solution, "quick_oof", lambda p, t, **kw: None)

    current = {"target": models.constant_predictions(prof, "target")}
    solution.solve_target(prof, "target", current, budget=5000)
    assert (current["target"] == 1).all()
    out = pd.read_csv(tmp_path / "submission.csv")
    assert (out["target"] == 1).all()


def test_numeric_labels_in_object_arrays():
    y = np.array([1, 2, 3, 1, 2, 3], dtype=object)
    p = np.array([1, 2, 2, 1, 2, 3], dtype=object)
    assert models.score("f1_macro", y, p) == pytest.approx(models.score("f1_macro", y.astype(int), p.astype(int)))
    assert models.score("accuracy", y, p) == pytest.approx(5 / 6)
