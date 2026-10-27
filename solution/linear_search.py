"""Bounded family-aware search: Ridge with raw and raw + signed-log features.

Every fitted transformer belongs to a recipe and learns only from a fold's train.
The winning recipe is refitted on all labeled rows; tree feature selection does
not determine whether its transformations are useful to the linear family.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

import models
from agent_utils import StageTimeout, log, time_limit

MAX_ROWS = 100_000
MAX_COLUMNS = 200
MIN_SECONDS = 2
MIN_REL_GAIN = 0.002


def recipes(prof: dict, target: str) -> tuple[list[dict], str]:
    if prof["problem_type"] != "regression":
        return [], "поддерживается только регрессия"
    train = models.labeled(prof, target)
    if not 20 <= len(train) <= MAX_ROWS:
        return [], f"число размеченных строк вне [20, {MAX_ROWS}]"
    features = prof.get("base_features") or prof["features"]
    numeric = [c for c in features if c in train and c in prof["test"]
               and pd.api.types.is_numeric_dtype(train[c]) and train[c].dtype != bool]
    if not numeric or len(numeric) > MAX_COLUMNS:
        return [], f"число числовых признаков вне [1, {MAX_COLUMNS}]"
    return [dict(family="ridge", transform=transform, columns=list(numeric), alpha=1.0)
            for transform in ("raw", "raw_log")], ""


def recipe_key(recipe: dict) -> str:
    return json.dumps(recipe, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def matrix(frame: pd.DataFrame, recipe: dict) -> np.ndarray:
    X = frame[recipe["columns"]].to_numpy(dtype=float, na_value=np.nan, copy=True)
    X[~np.isfinite(X)] = np.nan
    if recipe["transform"] == "raw_log":
        X = np.column_stack([X, np.sign(X) * np.log1p(np.abs(X))])
    elif recipe["transform"] != "raw":
        raise ValueError("unsupported Ridge transform")
    return X


def fit_predict(recipe: dict, train: pd.DataFrame, y, test: pd.DataFrame, log_target: bool):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits

    pipeline = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                             StandardScaler(), Ridge(alpha=recipe["alpha"], solver="lsqr",
                                                     max_iter=1000, tol=1e-6))
    target = np.asarray(y, dtype=float)
    if log_target:
        target = np.log1p(np.clip(target, 0, None))
    with threadpool_limits(limits=models.N_CPUS):
        pipeline.fit(matrix(train, recipe), target)
        pred = pipeline.predict(matrix(test, recipe))
    if log_target:
        pred = np.clip(np.expm1(pred), 0, None)
    if not np.isfinite(pred).all():
        raise ValueError("non-finite Ridge predictions")
    return pred, pipeline


def search(prof: dict, target: str, budget: float, register=None) -> dict | None:
    proposed, reason = recipes(prof, target)
    if not proposed or budget < MIN_SECONDS:
        reason = reason or f"недостаточно времени ({budget:.1f}s)"
        log(f"Ridge: пропуск — {reason}")
        if register:
            register("ridge_skip", reason, None, False)
        return None
    started = time.monotonic()
    try:
        # A timed-out search returns no candidate, preserving the running winner.
        with time_limit(budget):
            return _search(prof, target, proposed, started, register)
    except StageTimeout:
        reason = f"Ridge: превышен бюджет {budget:.1f}s"
    except Exception as exc:  # noqa: BLE001 — optional family must not break the cascade
        reason = f"Ridge: {type(exc).__name__}: {exc}"
    log(reason)
    if register:
        register("ridge_error", reason, None, False)
    return None


def _search(prof, target, proposed, started, register):
    train = models.labeled(prof, target)
    y = train[target].to_numpy(float)
    folds = models.outer_folds(prof, train, y)
    if not folds:
        raise ValueError("no validation folds")
    # This cache is scoped to this target and its folds; its identity includes family.
    evaluated = {}
    for recipe in proposed:
        key = recipe_key(recipe)
        if key in evaluated:
            continue
        t0 = time.monotonic()
        oof = np.full(len(train), np.nan)
        for fit, val in folds:
            oof[val], _ = fit_predict(recipe, train.iloc[fit], y[fit], train.iloc[val], prof["log_target"])
        covered = np.isfinite(oof)
        score = models.score(prof["metric"], y[covered], oof[covered])
        if not np.isfinite(score):
            raise ValueError("non-finite Ridge score")
        evaluated[key] = dict(name=f"ridge_{recipe['transform']}", family="ridge", recipe=recipe,
                              rows=train.index.to_numpy()[covered], y=y[covered], oof=oof[covered],
                              classes=None, score=score)
        log(f"Ridge {recipe['transform']}: {prof['metric']}={score:.6f}, "
            f"{len(recipe['columns'])} числовых признаков, {time.monotonic()-t0:.2f}s")
    raw, transformed = [evaluated[recipe_key(r)] for r in proposed]
    metric = prof["metric"]
    direction = 1 if models.HIGHER_BETTER[metric] else -1
    gain = direction * (transformed["score"] - raw["score"]) / max(abs(raw["score"]), 1e-9)
    useful = np.isfinite(gain) and gain > MIN_REL_GAIN
    winner = transformed if useful else raw
    if register:
        register("ridge_transform", f"raw → raw+signed_log1p, прирост {gain:+.2%}; "
                 f"выбрано {winner['name']}; recipe={recipe_key(winner['recipe'])}", winner["score"], useful)
    prediction, _ = fit_predict(winner["recipe"], train, y, prof["test"], prof["log_target"])
    return {**winner, "test": prediction, "test_pred": prediction,
            "seconds": time.monotonic() - started, "raw_score": raw["score"], "log_score": transformed["score"]}
