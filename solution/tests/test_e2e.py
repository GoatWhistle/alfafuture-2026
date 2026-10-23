"""Сквозной запуск solution.py на синтетике: quick LightGBM + k-fold, без AutoEDA/AutoGluon/LLM."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SOLUTION = Path(__file__).resolve().parents[1] / "solution.py"
ENV = {
    "AGENT_TIME_LIMIT_SEC": "2000", "AGENT_SAFETY_SEC": "0", "AGENT_AUTOEDA_SEC": "60",
    "AGENT_PREDICT_RESERVE_SEC": "600", "AGENT_KFOLD_RESERVE_SEC": "600",
}


def run(tmp: Path, train, test, sample, text) -> pd.DataFrame:
    train.to_csv(tmp / "train.csv", index=False)
    test.to_csv(tmp / "test.csv", index=False)
    sample.to_csv(tmp / "sample_submission.csv", index=False)
    (tmp / "task.md").write_text(text, encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"} | ENV
    res = subprocess.run([sys.executable, str(SOLUTION), str(tmp / "task.md")], cwd=tmp, env=env,
                         capture_output=True, text=True, timeout=600)
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-3000:]
    (tmp / "log.txt").write_text(res.stdout)
    out = pd.read_csv(tmp / "submission.csv")
    assert list(out.columns) == list(sample.columns)
    assert out["id"].tolist() == test["id"].tolist()
    return out


def frames(n, y_fn, seed=0):
    rng = np.random.RandomState(seed)
    x = rng.randn(n + 200, 3)
    y = y_fn(x, rng)
    train = pd.DataFrame({"id": np.arange(n), "a": x[:n, 0], "b": x[:n, 1], "c": x[:n, 2]})
    test = pd.DataFrame({"id": np.arange(n, n + 200), "a": x[n:, 0], "b": x[n:, 1], "c": x[n:, 2]})
    return train, test, y[:n]


def test_binary_logloss_strings(tmp_path):
    train, test, y = frames(1500, lambda x, r: np.where(x[:, 0] + r.randn(len(x)) > 0, "yes", "no"))
    train["target"] = y
    sample = pd.DataFrame({"id": test["id"], "target": 0})
    out = run(tmp_path, train, test, sample, "Классификация. Метрика log loss.")
    p = out["target"].to_numpy(dtype=float)
    assert np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()
    # Вероятность класса "yes" растёт с признаком a.
    assert np.corrcoef(p, test["a"])[0, 1] > 0.5


def test_binary_logloss_numeric_uses_final_folds(tmp_path):
    train, test, y = frames(1500, lambda x, r: (x[:, 0] + r.randn(len(x)) > 0).astype(int))
    train["target"] = y
    sample = pd.DataFrame({"id": test["id"], "target": 0})
    out = run(tmp_path, train, test, sample, "Классификация. Метрика log loss.")
    log = (tmp_path / "log.txt").read_text()
    assert "quick_oof[target]: log_loss=" in log
    assert "Финальный источник: quick_kfold" in log
    assert "quick_oof упал" not in log
    assert np.corrcoef(out["target"], test["a"])[0, 1] > 0.5


def test_numeric_multiclass_f1_uses_final_folds(tmp_path):
    train, test, y = frames(1500, lambda x, r: np.array([2, 5, 9])[np.digitize(x[:, 1], [-0.5, 0.5])])
    train["target"] = y
    sample = pd.DataFrame({"id": test["id"], "target": 2})
    out = run(tmp_path, train, test, sample, "Классификация. Метрика Macro F1.")
    log = (tmp_path / "log.txt").read_text()
    assert "quick_oof[target]: f1_macro=" in log
    assert "quick_oof упал" not in log and "Финальный выбор упал" not in log
    assert set(out["target"]) <= {2, 5, 9}


def test_binary_f1_string_labels(tmp_path):
    train, test, y = frames(1500, lambda x, r: np.where(x[:, 0] + 0.3 * r.randn(len(x)) > 1, "fraud", "ok"))
    train["target"] = y
    sample = pd.DataFrame({"id": test["id"], "target": "ok"})
    out = run(tmp_path, train, test, sample, "Классификация, метрика F1.")
    assert set(out["target"]) <= {"fraud", "ok"}
    assert "quick_oof[target]: f1=" in (tmp_path / "log.txt").read_text()


def test_multiclass_proba_columns(tmp_path):
    train, test, y = frames(1500, lambda x, r: np.array(["A", "B", "C"])[np.digitize(x[:, 1], [-0.5, 0.5])])
    train["target"] = y
    sample = pd.DataFrame({"id": test["id"], "C": 0.0, "A": 0.0, "B": 0.0})
    out = run(tmp_path, train, test, sample, "Верните вероятности классов, метрика logloss.")
    m = out[["C", "A", "B"]].to_numpy()
    assert np.allclose(m.sum(axis=1), 1, atol=1e-6) and (m >= 0).all()
    hi = test["b"].to_numpy() > 1
    assert (out.loc[hi, "C"] > 0.5).mean() > 0.8


def test_rmsle_regression(tmp_path):
    train, test, y = frames(1500, lambda x, r: np.exp(1 + x[:, 2] + 0.1 * r.randn(len(x))))
    train["target"] = y
    sample = pd.DataFrame({"id": test["id"], "target": 0.0})
    out = run(tmp_path, train, test, sample, "Регрессия. Метрика RMSLE.")
    assert (out["target"] >= 0).all()
    log = (tmp_path / "log.txt").read_text()
    assert "metric=rmsle" in log


def test_multi_target(tmp_path):
    train, test, y = frames(1500, lambda x, r: x[:, 0] * 10 + r.randn(len(x)))
    train["price"] = y
    train["flag"] = np.where(train["b"] > 0, "yes", "no")
    sample = pd.DataFrame({"id": test["id"], "price": 0.0, "flag": "no"})
    out = run(tmp_path, train, test, sample, "Две цели: цена (регрессия) и флаг.")
    assert set(out["flag"]) <= {"yes", "no"}
    assert np.corrcoef(out["price"], test["a"])[0, 1] > 0.8
    assert ((out["flag"] == "yes") == (test["b"] > 0)).mean() > 0.9


def test_panel_with_autoeda(tmp_path):
    """Панель объект × период, test — следующий период: AutoEDA выбирает схему, target-спеки идут по фолдам."""
    rng = np.random.RandomState(5)
    n_obj, n_per = 400, 6
    obj = np.repeat(np.arange(n_obj), n_per)
    per = np.tile(np.arange(n_per), n_obj)
    level = rng.randn(n_obj)[obj] * 5
    x = rng.randn(len(obj))
    y = level + 0.3 * per + x + 0.3 * rng.randn(len(obj))
    df = pd.DataFrame({"id": np.arange(len(obj)), "obj": obj, "per": per, "x": x, "noise": rng.randn(len(obj)),
                       "target": y})
    train = df[df["per"] < 5].reset_index(drop=True)
    test = df[df["per"] == 5].drop(columns="target").reset_index(drop=True)
    sample = pd.DataFrame({"id": test["id"], "target": 0.0})
    env_backup = dict(ENV)
    ENV.update(AGENT_AUTOEDA_SEC="150", AGENT_TIME_LIMIT_SEC="3000")
    try:
        out = run(tmp_path, train, test, sample, "Регрессия, метрика RMSE. per — номер года, obj — объект.")
    finally:
        ENV.clear()
        ENV.update(env_backup)
    log = (tmp_path / "log.txt").read_text()
    assert "AutoEDA: схема валидации" in log
    truth = df[df["per"] == 5]["target"].to_numpy()
    rmse = float(np.sqrt(np.mean((out["target"].to_numpy() - truth) ** 2)))
    assert rmse < 2.0, rmse
