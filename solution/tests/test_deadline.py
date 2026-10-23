"""Проверки P0 (issue #2): глобальный дедлайн и зависшие этапы."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.pop("OPENAI_API_KEY", None)

import models  # noqa: E402
import solution  # noqa: E402
import task_profile  # noqa: E402
from test_p0 import _binary_prof, binary_frames, make_task  # noqa: E402

RUNNER = Path(__file__).with_name("hang_runner.py")


@pytest.mark.parametrize("stage", ["fit", "autogluon"])
def test_watchdog_stops_hung_stage_before_limit(tmp_path, stage):
    train, test = binary_frames(n=800)
    sample = pd.DataFrame({"id": test["id"], "target": 0})
    task = make_task(tmp_path, train, test, sample, "Метрика accuracy.")
    limit, margin = 60, 10
    env = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"} | {
        "AGENT_TIME_LIMIT_SEC": str(limit), "AGENT_HARD_MARGIN_SEC": str(margin), "AGENT_SAFETY_SEC": "0",
        "AGENT_PREDICT_RESERVE_SEC": "0", "AGENT_KFOLD_RESERVE_SEC": "0", "AGENT_AUTOEDA_SEC": "60",
    }
    if stage == "autogluon":
        env["AGENT_TIME_LIMIT_SEC"] = "400"  # AutoGluon запускается только при бюджете ≥ 300s
        limit = 400
        env["AGENT_HARD_MARGIN_SEC"] = str(limit - 45)
    pidfile = tmp_path / "child.pid"
    t0 = time.time()
    res = subprocess.run([sys.executable, str(RUNNER), str(task), stage, str(pidfile)], cwd=tmp_path, env=env,
                         capture_output=True, text=True, timeout=limit + 60)
    took = time.time() - t0
    assert res.returncode == 0, res.stdout[-2000:] + res.stderr[-2000:]
    assert "Жёсткий дедлайн" in res.stdout
    assert took < 60, took
    child = int(pidfile.read_text())
    assert not psutil.pid_exists(child) or psutil.Process(child).status() == psutil.STATUS_ZOMBIE
    out = pd.read_csv(tmp_path / "submission.csv")
    assert out["id"].tolist() == test["id"].tolist() and set(out["target"]) <= {0, 1}
    assert not list(tmp_path.glob(".submission.*"))


def test_hung_autoeda_is_abandoned_and_profile_untouched(tmp_path, monkeypatch):
    prof = _binary_prof(tmp_path)
    features = list(prof["features"])

    def hang(work, target, budget):
        work["features"] = ["broken"]
        work["target_specs"] = ["broken"]
        time.sleep(1000)  # сигнал прерывает и sleep

    monkeypatch.setattr(solution, "run_autoeda", hang)
    monkeypatch.setattr(solution, "AUTOEDA_SEC", 2)
    monkeypatch.setattr(solution, "MIN_AUTOEDA_SEC", 1)
    monkeypatch.setattr(solution, "AUTOEDA_SLACK_SEC", 1)
    monkeypatch.setattr(solution, "fit_autogluon", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("off")))
    current = {"target": models.constant_predictions(prof, "target")}
    t0 = time.time()
    solution.solve_target(prof, "target", current, budget=5000)
    assert time.time() - t0 < 60
    assert prof["features"] == features and "target_specs" not in prof


def test_hung_llm_returns_none(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setattr(task_profile, "_llm_json_call", lambda *a: time.sleep(1000))
    t0 = time.time()
    assert task_profile._llm_json("prompt", "test", timeout=1, retries=0) is None
    assert time.time() - t0 < 20


def test_lgb_time_guard_stops_training():
    import lightgbm as lgb

    X = np.random.RandomState(0).randn(5000, 20)
    y = X[:, 0]
    m = lgb.LGBMRegressor(n_estimators=100000, learning_rate=0.001, verbosity=-1)
    t0 = time.time()
    m.fit(X, y, callbacks=[models.time_guard(time.time() + 2)])
    assert time.time() - t0 < 10
    assert m.predict(X[:5]).shape == (5,)


def test_multi_target_budget_uses_actual_time(tmp_path, monkeypatch):
    rng = np.random.RandomState(3)
    n = 300
    train = pd.DataFrame({"id": range(n), "x": rng.randn(n), "a": rng.rand(n), "b": rng.rand(n)})
    test = pd.DataFrame({"id": range(n, n + 30), "x": rng.randn(30)})
    sample = pd.DataFrame({"id": test["id"], "a": 0.0, "b": 0.0})
    task = make_task(tmp_path, train, test, sample, "Регрессия, две цели.")
    budgets = []

    def fake_solve(prof, t, current, budget):
        budgets.append(budget)
        time.sleep(3)

    monkeypatch.setattr(solution, "solve_target", fake_solve)
    monkeypatch.setattr(solution, "start_watchdog", lambda *a: None)
    monkeypatch.setattr(sys, "argv", ["solution.py", str(task)])
    solution.main()
    # Первый таргет — половина остатка; второй — весь фактический остаток после первого (минус ~3s).
    assert budgets[1] <= 2 * budgets[0] - 2.5
