"""Модели и запись сабмита: быстрый LightGBM, AutoGluon, постпроцессинг под метрику."""

from __future__ import annotations

import os
import tempfile
import time

import numpy as np
import pandas as pd

from agent_utils import available_cpus, deadline, log
from eda_probe import row_keys

N_CPUS = available_cpus()

HIGHER_BETTER = {
    "rmse": False, "rmsle": False, "mae": False, "mape": False, "log_loss": False,
    "r2": True, "f1": True, "f1_macro": True, "f1_weighted": True,
    "roc_auc": True, "accuracy": True, "balanced_accuracy": True,
}


POSITIVE_TOKENS = ("1", "1.0", "true", "yes", "y", "да", "positive", "pos", "fraud", "default", "churn",
                   "spam", "malignant", "bad", "fail", "failure", "anomaly", "attack", "abnormal")
NEGATIVE_TOKENS = ("0", "0.0", "false", "no", "n", "нет", "negative", "neg", "normal", "benign", "good",
                   "ok", "legit", "legitimate", "pass")


def positive_label(labels):
    """Положительный класс бинарной задачи — одно правило для метрик, весов и вывода.

    Числовые/булевы метки -> максимум (1, True); строки -> словарь
    положительных/отрицательных токенов; иначе последняя метка по str-сортировке.
    """
    vals = list(pd.unique(pd.Series(list(labels), dtype=object).dropna()))
    if not vals:
        return None
    if all(isinstance(v, (bool, np.bool_, int, float, np.integer, np.floating)) for v in vals):
        return max(vals)
    low = {str(v).strip().lower(): v for v in vals}
    for tok in POSITIVE_TOKENS:
        if tok in low:
            return low[tok]
    if len(vals) == 2:
        for tok in NEGATIVE_TOKENS:
            if tok in low:
                return next(v for v in vals if v is not low[tok])
    return sorted(vals, key=str)[-1]


def _positive_column(proba, labels, y_true):
    """Вероятность положительного класса и сам класс (proba-матрица требует labels)."""
    proba = np.asarray(proba, dtype=float)
    pool = list(labels) if labels is not None else list(pd.unique(np.asarray(y_true, dtype=object)))
    pos = positive_label(pool)
    if proba.ndim == 1:
        return proba, pos
    if labels is None:
        raise ValueError("матрица вероятностей без порядка классов")
    return proba[:, list(labels).index(pos)], pos


def score(metric: str, y_true, y_pred, proba=None, labels=None) -> float:
    """Единый расчёт метрики. y_pred — метки/значения, proba — матрица вероятностей.

    labels — порядок классов в колонках proba. Положительный класс бинарных
    метрик — positive_label(labels). Неизвестная метрика — ValueError.
    """
    from sklearn import metrics as sm

    if metric == "rmse":
        return float(np.sqrt(sm.mean_squared_error(y_true, y_pred)))
    if metric == "rmsle":
        yt = np.log1p(np.clip(np.asarray(y_true, dtype=float), 0, None))
        yp = np.log1p(np.clip(np.asarray(y_pred, dtype=float), 0, None))
        return float(np.sqrt(np.mean((yt - yp) ** 2)))
    if metric == "mae":
        return float(sm.mean_absolute_error(y_true, y_pred))
    if metric == "mape":
        return float(sm.mean_absolute_percentage_error(y_true, y_pred))
    if metric == "r2":
        return float(sm.r2_score(y_true, y_pred))
    if metric == "f1_macro":
        return float(sm.f1_score(y_true, y_pred, average="macro"))
    if metric == "f1_weighted":
        return float(sm.f1_score(y_true, y_pred, average="weighted"))
    if metric == "f1":
        pool = list(labels) if labels is not None else list(pd.unique(np.asarray(y_true, dtype=object)))
        pos = positive_label(pool)
        yt = np.asarray(y_true, dtype=object) == pos
        yp = np.asarray(y_pred, dtype=object) == pos
        return float(sm.f1_score(yt, yp, zero_division=0))
    if metric == "roc_auc":
        p, pos = _positive_column(proba, labels, y_true)
        return float(sm.roc_auc_score(np.asarray(y_true, dtype=object) == pos, p))
    if metric == "log_loss":
        return float(sm.log_loss(y_true, proba, labels=labels))
    if metric == "balanced_accuracy":
        return float(sm.balanced_accuracy_score(y_true, y_pred))
    if metric == "accuracy":
        return float(sm.accuracy_score(y_true, y_pred))
    raise ValueError(f"неизвестная метрика {metric!r}")


def _nan(x) -> bool:
    return x is None or (isinstance(x, float) and np.isnan(x))


def is_better(metric: str, new: float | None, old: float | None) -> bool:
    """new строго лучше old. Неоценённый new (None/NaN) не лучше ничего; old=None — нет опоры."""
    if _nan(new):
        return False
    if old is None:
        return True
    if _nan(old):
        return False
    return new > old if HIGHER_BETTER[metric] else new < old


# ---------------------------------------------------------------------------
# Формат вывода и запись сабмита.
# ---------------------------------------------------------------------------

def _class_index(classes) -> dict:
    """Позиция класса по значению и по строке (метки AG и sample могут различаться типом)."""
    pos = {}
    for i, c in enumerate(classes):
        pos.setdefault(c, i)
        pos.setdefault(str(c), i)
    return pos


def output_from_proba(prof: dict, target: str, proba, classes) -> np.ndarray:
    """Значения сабмита из вероятностей (колонки в порядке classes).

    Несколько колонок вероятностей -> матрица в порядке prof["proba_cols"][target];
    одна колонка вероятности -> вероятность positive_label; иначе метки argmax.
    """
    proba = np.asarray(proba, dtype=float)
    classes = list(classes)
    cols = (prof.get("proba_cols") or {}).get(target)
    if cols:
        pos = _class_index(classes)
        missing = [cls for _, cls in cols if cls not in pos and str(cls) not in pos]
        if missing:
            log(f"output_from_proba: у модели нет классов {missing}, их вероятность 0")
        out = np.column_stack([
            proba[:, pos[cls] if cls in pos else pos[str(cls)]] if cls not in missing else np.zeros(len(proba))
            for _, cls in cols
        ])
        out = np.clip(out, 0, 1)
        s = out.sum(axis=1, keepdims=True)
        return np.where(s > 0, out / np.where(s > 0, s, 1), 1.0 / out.shape[1])
    if prof["output_proba"]:
        return np.clip(proba[:, classes.index(positive_label(classes))], 0, 1)
    return np.array(classes, dtype=object)[np.argmax(proba, axis=1)]


def _expand(prof: dict, values_by_target: dict) -> pd.DataFrame:
    test, id_col = prof["test"], prof["id_col"]
    out = pd.DataFrame({id_col: test[id_col].values})
    proba_cols = prof.get("proba_cols") or {}
    for tgt, vals in values_by_target.items():
        vals = np.asarray(vals)
        if tgt in proba_cols:
            for j, (col, _) in enumerate(proba_cols[tgt]):
                out[col] = vals[:, j].astype(float)
        else:
            out[tgt] = vals
    return out


def expected_columns(prof: dict) -> list[str]:
    if prof["sample"] is not None:
        return list(prof["sample"].columns)
    cols = [prof["id_col"]]
    for t in prof["target_cols"]:
        cols += [c for c, _ in (prof.get("proba_cols") or {}).get(t, [])] or [t]
    return cols


def validate_submission(prof: dict, out: pd.DataFrame) -> str | None:
    """Проверка контракта: колонки, строки и их порядок, конечность и диапазон вероятностей."""
    cols = expected_columns(prof)
    if list(out.columns) != cols:
        return f"колонки {list(out.columns)} != {cols}"
    test, id_col = prof["test"], prof["id_col"]
    if len(out) != len(test):
        return f"строк {len(out)} != {len(test)}"
    if not np.array_equal(out[id_col].to_numpy(), test[id_col].to_numpy()):
        return "id не совпадают с test по значениям или порядку"
    proba_cols = prof.get("proba_cols") or {}
    for t in prof["target_cols"]:
        group = [c for c, _ in proba_cols.get(t, [])]
        names = group or [t]
        is_proba = bool(group) or prof.get("per_target", {}).get(t, prof).get("output_proba", False)
        for c in names:
            col = out[c]
            if col.isna().any():
                return f"пропуски в {c}"
            if is_proba or pd.api.types.is_numeric_dtype(col):
                v = pd.to_numeric(col, errors="coerce").to_numpy(dtype=float)
                if not np.isfinite(v).all():
                    return f"нечисловые или бесконечные значения в {c}"
                if is_proba and ((v < 0).any() or (v > 1).any()):
                    return f"вероятности вне [0, 1] в {c}"
        if len(group) > 1:
            sums = out[group].to_numpy(dtype=float).sum(axis=1)
            if not np.allclose(sums, 1.0, atol=1e-6):
                return f"сумма вероятностей {t} отличается от 1 (max |d|={np.abs(sums - 1).max():.2e})"
    return None


def write_submission(prof: dict, values_by_target: dict[str, np.ndarray]) -> bool:
    """Пишет submission.csv рядом с task.md строго в формате sample_submission.

    Файл заменяется атомарно и только после проверки контракта; при
    отказе остаётся предыдущий валидный файл, а values_by_target
    откатывается к записанным в нём значениям. Пока валидного файла нет,
    пропуски и бесконечности заменяются страховочным значением.
    """
    workdir = prof["workdir"]
    last = prof.get("_last_written")
    err = None
    try:
        out = _expand(prof, values_by_target)
        out = out[[c for c in expected_columns(prof) if c in out.columns]]
        if last is None:
            out = _fill_bad(prof, out)
        err = validate_submission(prof, out)
    except Exception as exc:  # noqa: BLE001
        err = f"{type(exc).__name__}: {exc}"
    if err is not None:
        log(f"submission.csv НЕ обновлён: {err}")
        if last is not None:
            for t, v in last.items():
                values_by_target[t] = v.copy() if isinstance(v, np.ndarray) else v
        return False

    path = workdir / "submission.csv"
    tmp = workdir / f".submission.{os.getpid()}.tmp"
    try:
        out.to_csv(tmp, index=False)
        with open(tmp, "rb") as f:
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    prof["_last_written"] = {t: np.array(v, copy=True) for t, v in values_by_target.items()}
    log(f"submission.csv записан: {path} shape={out.shape}")
    return True


def _fill_bad(prof: dict, out: pd.DataFrame) -> pd.DataFrame:
    """Страховка первой записи: пропуски и бесконечности -> страховочное значение."""
    out = out.copy()
    proba_cols = prof.get("proba_cols") or {}
    for tgt in prof["target_cols"]:
        if tgt in proba_cols or tgt not in out.columns:
            continue
        col = out[tgt]
        if pd.api.types.is_numeric_dtype(col):
            bad = ~np.isfinite(col.astype(float))
            if bad.any():
                out.loc[bad, tgt] = _fallback_value(prof, tgt)
        elif col.isna().any():
            out[tgt] = col.fillna(_fallback_value(prof, tgt))
    return out


def _fallback_value(prof: dict, target: str):
    tp = prof.get("per_target", {}).get(target, prof)
    y = prof["train"][target].dropna()
    if tp["problem_type"] == "regression":
        return float(y.median())
    if tp["output_proba"]:
        pos = positive_label(y.unique())
        return float((y == pos).mean())
    return y.mode().iloc[0]


def constant_predictions(prof: dict, target: str) -> np.ndarray:
    n = len(prof["test"])
    cols = (prof.get("proba_cols") or {}).get(target)
    if cols:
        y = prof["train"][target].dropna()
        freq = y.astype(str).value_counts(normalize=True)
        prior = np.array([freq.get(str(cls), 0.0) for _, cls in cols]) + 1e-6
        return np.tile(prior / prior.sum(), (n, 1))
    tp = prof.get("per_target", {}).get(target, prof)
    return np.full(n, _fallback_value(prof, target), dtype=object if tp["problem_type"] != "regression" else float)


# ---------------------------------------------------------------------------
# Схема независимой валидации.
# ---------------------------------------------------------------------------

N_FOLDS = 5
SEG_MIN_ROWS = 300


def labeled(prof: dict, target: str) -> pd.DataFrame:
    train = prof["train"]
    return train.loc[train[target].notna()]


def outer_folds(prof: dict, lab: pd.DataFrame, y, n_splits: int = N_FOLDS) -> list[tuple[np.ndarray, np.ndarray]]:
    """Внешние фолды (позиции в lab) по prof["cv"] — та же схема, что у валидатора AutoEDA.

    temporal: train — периоды раньше валидационного (последний, при ≥4
    периодах ещё предпоследний); group: GroupKFold по ключу объекта;
    иначе (Stratified)KFold. Детерминированы: все модели видят одни фолды.
    """
    from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold

    cv = prof.get("cv") or {"kind": "kfold"}
    n = len(lab)
    if cv["kind"] == "temporal" and cv.get("period") in lab.columns:
        t = pd.to_numeric(lab[cv["period"]], errors="coerce").to_numpy(float)
        periods = np.unique(t[~np.isnan(t)])
        folds = []
        for last in (periods[-2:] if len(periods) >= 4 else periods[-1:]):
            tr, va = np.flatnonzero(t < last), np.flatnonzero(t == last)
            if len(tr) >= 200 and len(va) >= 50:
                folds.append((tr, va))
        if folds:
            return folds
        log("outer_folds: temporal-фолды не собрались, беру KFold")
    if cv["kind"] == "group" and set(cv.get("keys") or []) <= set(lab.columns):
        g = row_keys(lab, cv["keys"])
        ng = int(pd.Series(g).nunique())
        if ng >= 2:
            return list(GroupKFold(min(n_splits, ng)).split(np.arange(n), groups=g))
        log("outer_folds: групп меньше двух, беру KFold")
    ys = pd.Series(np.asarray(y, dtype=object))
    if prof["problem_type"] != "regression" and ys.value_counts().min() >= n_splits:
        return list(StratifiedKFold(n_splits, shuffle=True, random_state=7).split(np.arange(n), ys))
    return list(KFold(n_splits, shuffle=True, random_state=7).split(np.arange(n)))


def fold_ids(n: int, folds) -> np.ndarray:
    """Номер валидационного фолда каждой строки (-1 — строка ни в одном val)."""
    out = np.full(n, -1, dtype=np.int64)
    for i, (_, va) in enumerate(folds):
        out[va] = i
    return out


def es_split(prof: dict, lab: pd.DataFrame, tr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Строки tr для обучения и для ранней остановки: оцениваемый фолд её не видит.

    temporal — последний период tr; group — 10% групп; иначе — 10% строк.
    """
    cv = prof.get("cv") or {}
    sub = lab.iloc[tr]
    rng = np.random.RandomState(3)
    if cv.get("kind") == "temporal" and cv.get("period") in sub.columns:
        t = pd.to_numeric(sub[cv["period"]], errors="coerce").to_numpy(float)
        if np.isfinite(t).any():
            es = t == np.nanmax(t)
            if 50 <= es.sum() <= 0.5 * len(t):
                return tr[~es], tr[es]
    if cv.get("kind") == "group" and set(cv.get("keys") or []) <= set(sub.columns):
        g = row_keys(sub, cv["keys"])
        ug = np.unique(g)
        if len(ug) >= 5:
            es = np.isin(g, rng.choice(ug, max(1, len(ug) // 10), replace=False))
            if 0 < es.sum() < len(tr):
                return tr[~es], tr[es]
    es = rng.rand(len(tr)) < 0.1
    return tr[~es], tr[es]


def target_features(prof: dict, target: str, fit: pd.DataFrame, app: pd.DataFrame):
    """Признаки target-спек AutoEDA: fit — cross-fit по схеме cv, app — по всему fit.

    Строятся заново для каждого разбиения, поэтому таргет строк app на
    признаки не влияет. Возвращает (fit_df, app_df) с RangeIndex или (None, None).
    """
    import fe_dsl

    specs = prof.get("target_specs") or []
    if not specs:
        return None, None
    classes = None
    if prof["problem_type"] != "regression":
        classes = sorted(fit[target].astype(str).unique().tolist())
    fparts, aparts = [], []
    for spec in fe_dsl.unique_specs(specs):
        res = fe_dsl.safe_apply(spec, fit, fit[target].to_numpy(), app, classes, prof.get("cv"))
        if res is not None:
            fparts.append(res[0])
            aparts.append(res[1])
    if not fparts:
        return None, None
    f = pd.concat(fparts, axis=1, verify_integrity=True)
    a = pd.concat(aparts, axis=1, verify_integrity=True)
    return f, a.reindex(columns=f.columns)


def with_target_features(prof: dict, target: str, fit_df: pd.DataFrame, Xfit: pd.DataFrame,
                         app_df: pd.DataFrame, Xapp: pd.DataFrame):
    """Матрицы с target-признаками (RangeIndex); без target-спек — как есть."""
    F, A = target_features(prof, target, fit_df, app_df)
    Xfit, Xapp = Xfit.reset_index(drop=True), Xapp.reset_index(drop=True)
    if F is None:
        return Xfit, Xapp
    return (pd.concat([Xfit, F], axis=1, verify_integrity=True),
            pd.concat([Xapp, A], axis=1, verify_integrity=True))


# ---------------------------------------------------------------------------
# Быстрый LightGBM — страховочная модель и ранний сабмит.
# ---------------------------------------------------------------------------

def _prep_matrices(prof: dict, target: str):
    train, test, features = prof["train"], prof["test"], prof["features"]
    mask = train[target].notna()
    X = train.loc[mask, features].copy()
    y = train.loc[mask, target]
    Xte = test[features].copy()
    for c in features:
        if X[c].dtype == object or str(X[c].dtype) == "category" or X[c].dtype == bool:
            cats = pd.api.types.union_categoricals(
                [X[c].astype(str).astype("category"), Xte[c].astype(str).astype("category")]
            ).categories
            X[c] = pd.Categorical(X[c].astype(str), categories=cats)
            Xte[c] = pd.Categorical(Xte[c].astype(str), categories=cats)
    return X, y, Xte


def _lgb_model(prof: dict, n_estimators: int, seed: int, objective: dict | None = None):
    import lightgbm as lgb

    common = dict(
        n_estimators=n_estimators, learning_rate=0.05, num_leaves=127,
        colsample_bytree=0.8, subsample=0.8, subsample_freq=1,
        n_jobs=N_CPUS, random_state=seed, verbosity=-1,
    )
    if prof["problem_type"] == "regression":
        return lgb.LGBMRegressor(**(objective or {"objective": "regression"}), **common)
    cw = "balanced" if prof["metric"] in ("f1_macro", "balanced_accuracy", "f1") else None
    return lgb.LGBMClassifier(class_weight=cw, **common)


def time_guard(deadline_ts: float):
    """Callback LightGBM: остановка обучения по абсолютному дедлайну (модель на текущей итерации)."""
    import lightgbm as lgb

    def cb(env):
        if time.time() > deadline_ts:
            raise lgb.callback.EarlyStopException(env.iteration, env.evaluation_result_list)

    cb.order = 40
    return cb


def _pred(m, X, classes):
    if classes is None:
        return m.predict(X)
    out = np.zeros((len(X), len(classes)))
    pos = {c: i for i, c in enumerate(classes)}
    out[:, [pos[c] for c in m.classes_]] = m.predict_proba(X)
    return out


def _fit_segmented(make, Xf, yf, Xes, yes, cf=None, ces=None, classes=None, n_iter=None, deadline_ts=None):
    """Общая модель + модели сегментов (коды cf). n_iter задан — без ранней остановки (финальный fit).

    deadline_ts — абсолютный дедлайн: обучение обрывается на текущей итерации.
    """
    import lightgbm as lgb

    yf, yes = np.asarray(yf), (None if yes is None else np.asarray(yes))
    guard = time_guard(deadline() if deadline_ts is None else deadline_ts)

    def fit(X, y, Xe, ye):
        m = make()
        if n_iter is not None:
            m.set_params(n_estimators=max(int(n_iter), 20))
            m.fit(X, y, callbacks=[guard])
        else:
            m.fit(X, y, eval_set=[(Xe, ye)],
                  callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0), guard])
        return m

    glob = fit(Xf, yf, Xes, yes)
    segs = {}
    if cf is not None:
        for code in np.unique(cf):
            rows = cf == code
            if rows.sum() < SEG_MIN_ROWS or (classes is not None and pd.Series(yf[rows]).nunique() < 2):
                continue
            er = None if ces is None else ces == code
            Xe, ye = (Xes[er], yes[er]) if er is not None and er.sum() > 0 else (Xes, yes)
            segs[code] = fit(Xf[rows], yf[rows], Xe, ye)
    return glob, segs


def _predict_segmented(glob, segs, X, codes, classes):
    p = _pred(glob, X, classes)
    if codes is not None:
        for code, m in segs.items():
            r = codes == code
            if r.any():
                p[r] = _pred(m, X[r], classes)
    return p


def quick_model(prof: dict, target: str, max_rows: int = 300_000, use_cv: bool = False,
                deadline_ts: float | None = None) -> dict:
    """LightGBM на одном разбиении: быстрый ориентир качества + ранние предсказания.

    use_cv=False — случайные 15% (до AutoEDA схема ещё неизвестна);
    use_cv=True — валидационный фолд схемы prof["cv"], target-признаки
    строятся по train фолда. Ранняя остановка — на отдельных строках train.
    """
    from sklearn.model_selection import train_test_split

    metric, ptype = prof["metric"], prof["problem_type"]
    X, y, Xte = _prep_matrices(prof, target)
    lab = labeled(prof, target)
    if len(X) > max_rows:
        idx = np.sort(np.random.RandomState(0).choice(len(X), max_rows, replace=False))
        X, y, lab = X.iloc[idx], y.iloc[idx], lab.iloc[idx]
    y_orig = y.to_numpy()
    if prof["log_target"]:
        y = np.log1p(y.clip(lower=0))

    pos = np.arange(len(X))
    if use_cv:
        folds = outer_folds(prof, lab, y_orig)
        tr, va = folds[-1] if (prof.get("cv") or {}).get("kind") == "temporal" else folds[0]
    else:
        strat = y_orig if ptype != "regression" and pd.Series(y_orig).value_counts().min() >= 2 else None
        tr, va = train_test_split(pos, test_size=0.15, random_state=42, stratify=strat)
    fit_i, es_i = es_split(prof, lab, np.sort(tr))
    app = np.concatenate([es_i, va])
    Xf, Xa = with_target_features(prof, target, lab.iloc[fit_i], X.iloc[fit_i],
                                  pd.concat([lab.iloc[app], prof["test"]]), pd.concat([X.iloc[app], Xte]))
    Xes, Xval, Xt = Xa.iloc[:len(es_i)], Xa.iloc[len(es_i):len(app)], Xa.iloc[len(app):]
    yv = y.iloc[va].to_numpy()
    classes = None if ptype == "regression" else sorted(pd.unique(y_orig).tolist(), key=str)
    glob, _ = _fit_segmented(lambda: _lgb_model(prof, 3000, 42), Xf, y.iloc[fit_i], Xes, y.iloc[es_i],
                             classes=classes, deadline_ts=deadline_ts)

    result = {"name": "quick_lgbm", "rows": lab.index.to_numpy()[va]}
    if ptype == "regression":
        val_pred, test_pred = glob.predict(Xval), glob.predict(Xt)
        if prof["log_target"]:
            val_pred, test_pred = np.expm1(val_pred), np.clip(np.expm1(test_pred), 0, None)
            yv = np.expm1(yv)
        val_score = score(metric, yv, val_pred)
        proba_te = None
    else:
        val_proba = _pred(glob, Xval, classes)
        val_pred = np.array(classes, dtype=object)[np.argmax(val_proba, axis=1)]
        val_score = score(metric, yv, val_pred, proba=val_proba, labels=classes)
        proba_te = _pred(glob, Xt, classes)
        test_pred = output_from_proba(prof, target, proba_te, classes)
        result.update(val_proba=val_proba, y_val=yv, classes=classes)

    log(f"quick_lgbm[{target}]: val {metric}={val_score:.5f}, best_iter={glob.best_iteration_}"
        + (" (схема cv)" if use_cv else ""))
    result.update(val_score=val_score, test_pred=test_pred, proba=proba_te,
                  importances=dict(zip(list(Xf.columns), glob.feature_importances_.tolist())))
    return result


def _segment_codes(prof: dict, target: str):
    """Коды сегментов train (размеченные строки) и test по принятой segment_model, иначе None."""
    import fe_dsl

    seg = (prof.get("model_ops") or {}).get("segment_model")
    if seg is None:
        return None, None, None
    lab = labeled(prof, target)
    ctr, cte = fe_dsl.segment_codes(seg, lab, prof["test"])
    return seg, ctr, cte


def quick_oof(prof: dict, target: str, max_rows: int = 400_000, deadline_ts: float | None = None) -> dict | None:
    """K-fold LightGBM по схеме prof["cv"]: OOF на валидационных строках + предсказания test.

    На каждом фолде target-признаки строятся по train фолда, ранняя
    остановка — на части train фолда. Test предсказывает модель,
    переобученная на всём размеченном train (число деревьев — медиана по
    фолдам). Учитывает модельные операции AutoEDA (сегменты, функция
    потерь). Возвращает {"rows" — индексы train с OOF, "y", "oof", "test",
    "classes", "score"} или None, когда датасет слишком велик или ни один
    фолд не успел до deadline_ts (оставшиеся фолды пропускаются, финальный
    fit обрывается по дедлайну).
    """
    dl = deadline() if deadline_ts is None else deadline_ts
    metric, ptype = prof["metric"], prof["problem_type"]
    X, y, Xte = _prep_matrices(prof, target)
    if len(X) > max_rows:
        log(f"quick_oof: {len(X)} строк > {max_rows}, пропускаю бленд")
        return None
    lab = labeled(prof, target)
    y_orig = y.to_numpy()
    if prof["log_target"]:
        y = np.log1p(y.clip(lower=0))
    seg, ctr, cte = _segment_codes(prof, target)
    if seg is not None:
        log(f"quick_oof: отдельные модели по сегментам {seg['cols'][0]}:{seg['params']['rule']}")
    obj = None
    if ptype == "regression":
        o = (prof.get("model_ops") or {}).get("objective")
        if o is not None:
            p = o["params"]
            obj = {"objective": p["name"]}
            if "alpha" in p:
                obj["alpha" if p["name"] == "huber" else "fair_c"] = p["alpha"]
            log(f"quick_oof: функция потерь {obj}")
    classes = None if ptype == "regression" else sorted(pd.unique(y_orig).tolist(), key=str)
    make = lambda: _lgb_model(prof, 1500, 7, obj)  # noqa: E731

    folds = outer_folds(prof, lab, y_orig)
    oof = np.full(len(X), np.nan) if classes is None else np.full((len(X), len(classes)), np.nan)
    iters = []
    for i, (tr, va) in enumerate(folds):
        if time.time() > dl:
            log(f"quick_oof: дедлайн — оценено {i}/{len(folds)} фолдов")
            break
        fit_i, es_i = es_split(prof, lab, tr)
        app = np.concatenate([es_i, va])
        Xf, Xa = with_target_features(prof, target, lab.iloc[fit_i], X.iloc[fit_i], lab.iloc[app], X.iloc[app])
        cf = None if ctr is None else ctr[fit_i]
        glob, segs = _fit_segmented(make, Xf, y.iloc[fit_i], Xa.iloc[:len(es_i)], y.iloc[es_i],
                                    cf, None if ctr is None else ctr[es_i], classes, deadline_ts=dl)
        oof[va] = _predict_segmented(glob, segs, Xa.iloc[len(es_i):], None if ctr is None else ctr[va], classes)
        iters.append(glob.best_iteration_ or 1500)

    if not iters:
        return None
    n_iter = int(np.median(iters) * 1.1)
    Xall, Xt = with_target_features(prof, target, lab, X, prof["test"], Xte)
    glob, segs = _fit_segmented(make, Xall, y, None, None, ctr, None, classes, n_iter=n_iter,
                                deadline_ts=max(dl, time.time() + 60))
    te = _predict_segmented(glob, segs, Xt, cte, classes)

    cov = np.isfinite(oof if oof.ndim == 1 else oof[:, 0])
    oof, y_cov = oof[cov], y_orig[cov]
    if classes is None:
        if prof["log_target"]:
            oof = np.clip(np.expm1(oof), 0, None)
            te = np.clip(np.expm1(te), 0, None)
        s = score(metric, y_cov, oof)
    else:
        pred = np.array(classes, dtype=object)[np.argmax(oof, axis=1)]
        s = score(metric, y_cov, pred, proba=oof, labels=classes)
    log(f"quick_oof[{target}]: {metric}={s:.5f} на {int(cov.sum())} строках ({len(folds)} фолд.), "
        f"финальный fit {n_iter} деревьев")
    return {"rows": lab.index.to_numpy()[cov], "y": y_cov, "oof": oof, "test": te, "classes": classes, "score": s}


def align_classes(mat: np.ndarray, src: list, dst: list) -> np.ndarray:
    """Переставляет колонки матрицы вероятностей из порядка src в порядок dst (сверка и по str)."""
    pos = _class_index(src)
    missing = [c for c in dst if c not in pos and str(c) not in pos]
    if missing or len(src) != len(dst):
        raise ValueError(f"классы не совпадают: {list(src)} vs {list(dst)}")
    return mat[:, [pos[c] if c in pos else pos[str(c)] for c in dst]]


def tune_blend(metric: str, y, a, b, classes: list | None = None):
    """Смесь alpha*a + (1-alpha)*b; возвращает (alpha, лучший скор)."""
    best_alpha, best = 1.0, None
    for alpha in np.linspace(0.0, 1.0, 11):
        mix = alpha * a + (1 - alpha) * b
        if classes is not None:
            pred = np.array(classes, dtype=object)[np.argmax(mix, axis=1)]
            s = score(metric, y, pred, proba=mix, labels=classes)
        else:
            s = score(metric, y, mix)
        if best is None or is_better(metric, s, best):
            best, best_alpha = s, float(alpha)
    return best_alpha, best


def copy_duplicate_labels(prof: dict, target: str, y_oof, oof_pred, test_pred, metric: str, rows=None):
    """Копирование меток точных дублей признаков (классификация).

    Правило: строка получает метку, если все остальные строки train с тем же
    ключом признаков (≥1) несут одну и ту же метку. На train проверяется
    leave-one-out тем же правилом: копия должна быть не хуже модели (OOF) и
    по точности на затронутых строках, и по метрике в целом.
    rows — индексы train, которым соответствуют y_oof/oof_pred (по умолчанию
    все размеченные). Возвращает (test_pred | None, base, new, n_test_copied).
    """
    train, test = prof["train"], prof["test"]
    feats = list(prof.get("base_features") or prof["features"])
    lab = train.loc[train[target].notna()] if rows is None else train.loc[rows]
    y = np.asarray(y_oof, dtype=object)
    ktr = row_keys(lab, feats)
    kte = row_keys(test, feats)
    labels = lab[target].astype(str).to_numpy()

    df = pd.DataFrame({"k": ktr, "y": labels})
    cnt = df.groupby("k")["y"].transform("size").to_numpy()
    # LOO: метки остальных строк группы единогласны <=> по группе не больше 2
    # меток и (одна метка у всех) или (своя метка уникальна, остальные совпадают).
    per = df.groupby(["k", "y"]).size().rename("n_ky").reset_index()
    nlab = per.groupby("k")["y"].transform("size")
    per = per.assign(nlab=nlab.to_numpy())
    df = df.merge(per, on=["k", "y"], how="left")
    others = cnt - df["n_ky"].to_numpy()
    nlab_row = df["nlab"].to_numpy()
    # Остальные несут свою метку (nlab==1) или одну чужую (nlab==2 и я — единственный носитель своей).
    same = (nlab_row == 1) & (cnt >= 2)
    flip = (nlab_row == 2) & (df["n_ky"].to_numpy() == 1) & (others >= 1)
    affected = same | flip
    if affected.sum() < 30:
        return None, None, None, 0

    # Метка «остальных» для flip-строк — вторая метка группы.
    other_label = labels.copy()
    if flip.any():
        two = per[per["nlab"] == 2]
        first = two.groupby("k")["y"].agg(list)
        for i in np.flatnonzero(flip):
            a, b = first[ktr[i]]
            other_label[i] = b if labels[i] == a else a

    base_pred = np.asarray(oof_pred, dtype=object).copy()
    loo_pred = base_pred.copy()
    loo_pred[affected] = other_label[affected]
    # y в OOF и метки в train могут различаться типом (int vs str) — сравниваем строками.
    y_s = y.astype(str)
    copy_acc = float((loo_pred[affected].astype(str) == y_s[affected]).mean())
    model_acc = float((base_pred[affected].astype(str) == y_s[affected]).mean())
    to_orig = {str(v): v for v in pd.unique(y)}
    base = score(metric, y_s, base_pred.astype(str))
    new = score(metric, y_s, loo_pred.astype(str))
    log(f"label_copy LOO: строк={int(affected.sum())}, acc копии={copy_acc:.4f}, acc модели={model_acc:.4f}, "
        f"{metric}: {base:.5f} -> {new:.5f}")
    if copy_acc < model_acc or not (is_better(metric, new, base) or new == base):
        return None, base, new, 0

    pure = df[df["nlab"] == 1].drop_duplicates("k").set_index("k")["y"]
    hit = pd.Series(kte).map(pure)
    mask = hit.notna().to_numpy()
    out = np.asarray(test_pred, dtype=object).copy()
    out[mask] = [to_orig.get(v, v) for v in hit[mask]]
    return out, base, new, int(mask.sum())


# ---------------------------------------------------------------------------
# AutoGluon — основной движок.
# ---------------------------------------------------------------------------

FOLD_COL = "__cv_fold__"
AG_PATHS: list[str] = []  # каталоги моделей AutoGluon этого запуска — удаляются после таргета и при выходе


def cleanup_autogluon() -> None:
    import shutil

    while AG_PATHS:
        shutil.rmtree(AG_PATHS.pop(), ignore_errors=True)


def fit_autogluon(prof: dict, target: str, time_budget: float, rows=None, folds=None):
    """AutoGluon на признаках prof["features"] (без target-признаков AutoEDA).

    rows — индексы train для обучения (temporal holdout-fit), иначе весь
    размеченный train. folds — номера фолдов строк: бэггинг AutoGluon
    идёт по тем же фолдам, что и остальные модели (параметр groups).
    """
    from autogluon.tabular import TabularPredictor

    features = prof["features"]
    lab = labeled(prof, target)
    if rows is not None:
        lab = lab.loc[rows]
    df = lab[features + [target]].copy()
    if prof["log_target"]:
        df[target] = np.log1p(df[target].clip(lower=0))
    extra: dict = {}
    if folds is not None and rows is None and len(np.unique(folds)) >= 2 and (np.asarray(folds) >= 0).all():
        df[FOLD_COL] = np.asarray(folds)
        extra["groups"] = FOLD_COL

    n_cells = len(df) * max(len(features), 1)
    if time_budget > 2700 and n_cells <= 60_000_000:
        presets = "best_quality"
    elif time_budget > 1200:
        presets = "high_quality"
    else:
        presets = "good_quality"

    path = tempfile.mkdtemp(prefix="ag_", dir="/tmp")
    AG_PATHS.append(path)
    # Macro-метрики требуют баланса классов, иначе редкие классы тонут
    # (на UNSW-NB15 AG без взвешивания проигрывает балансированному LightGBM).
    if prof["problem_type"] != "regression" and prof["metric"] in (
        "f1_macro", "balanced_accuracy", "f1",
    ):
        extra["sample_weight"] = "balance_weight"
    predictor = TabularPredictor(
        label=target,
        problem_type=prof["problem_type"],
        eval_metric=prof["ag_metric"],
        path=path,
        verbosity=2,
        **extra,
    )
    log(f"AutoGluon[{target}]: presets={presets}, time_limit={time_budget:.0f}s, rows={len(df)}"
        + (f", фолды схемы cv ({len(np.unique(folds))})" if "groups" in extra else ""))
    predictor.fit(
        train_data=df,
        time_limit=max(int(time_budget), 120),
        presets=presets,
        num_cpus=N_CPUS,
        num_gpus=0,
        # KNN с бэггингом пишет копии всего трейна на диск и почти не помогает на табличках.
        excluded_model_types=["KNN"],
    )
    try:
        lb = predictor.leaderboard(silent=True)
        log("AutoGluon leaderboard (top5):\n" + lb.head(5).to_string())
    except Exception:  # noqa: BLE001
        pass
    return predictor


def ag_predict(prof: dict, predictor, target: str, frame: pd.DataFrame | None = None):
    """Предсказания AutoGluon (по умолчанию на test).

    Регрессия: (значения, None, None); классификация: (вектор для сабмита,
    матрица вероятностей, классы).
    """
    X = (prof["test"] if frame is None else frame)[prof["features"]]
    if prof["problem_type"] == "regression":
        pred = predictor.predict(X).to_numpy()
        if prof["log_target"]:
            pred = np.clip(np.expm1(pred), 0, None)
        return pred, None, None
    proba = predictor.predict_proba(X)
    classes = list(proba.columns)
    return output_from_proba(prof, target, proba.to_numpy(), classes), proba.to_numpy(), classes


def ag_oof(predictor, prof: dict, target: str):
    """OOF бэггинга: (rows — индексы train, y_true, oof_pred, oof_proba, classes) или None."""
    try:
        train = prof["train"]
        if prof["problem_type"] == "regression":
            oof = predictor.predict_oof()
            rows = oof.index.to_numpy()
            vals = oof.to_numpy()
            if prof["log_target"]:
                vals = np.clip(np.expm1(vals), 0, None)
            return rows, train.loc[rows, target].to_numpy(), vals, None, None
        proba = predictor.predict_proba_oof()
        rows = proba.index.to_numpy()
        classes = list(proba.columns)
        arr = proba.to_numpy()
        pred = np.array(classes, dtype=object)[np.argmax(arr, axis=1)]
        return rows, train.loc[rows, target].to_numpy(), pred, arr, classes
    except Exception as exc:  # noqa: BLE001
        log(f"OOF AutoGluon недоступен: {exc}")
        return None


# ---------------------------------------------------------------------------
# AutoResearch-постпроцессинг.
# ---------------------------------------------------------------------------

def _confusion_metric(metric: str, cm: np.ndarray, positive_idx: int) -> float:
    """Метрика из confusion matrix (cm[i, j]: истинный i, предсказан j)."""
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    if metric == "f1_macro":
        return float(f1.mean())
    if metric == "f1_weighted":
        return float((f1 * cm.sum(axis=1) / max(cm.sum(), 1)).sum())
    if metric == "balanced_accuracy":
        return float((tp / np.maximum(cm.sum(axis=1), 1)).mean())
    if metric == "accuracy":
        return float(tp.sum() / max(cm.sum(), 1))
    return float(f1[positive_idx])  # f1 (бинарная)


def tune_class_weights(
    y_true, proba, classes, metric: str, min_gain: float = 3e-3
) -> tuple[np.ndarray | None, float, float]:
    """Координатный подбор классовых множителей по OOF для macro/balanced метрик.

    Скоринг — по confusion matrix через bincount: на два порядка быстрее
    вызова sklearn на каждой точке сетки. Возвращает (weights|None,
    базовый скор, лучший скор).
    """
    classes_arr = np.array(classes, dtype=object)
    k = len(classes)
    pos = {c: i for i, c in enumerate(classes)}
    yi = np.fromiter((pos[v] for v in y_true), dtype=np.int64, count=len(y_true))
    positive_idx = classes.index(positive_label(classes))
    fast = metric in ("f1_macro", "f1_weighted", "balanced_accuracy", "accuracy", "f1")

    def _score(w):
        pred_idx = np.argmax(proba * w, axis=1)
        if fast:
            cm = np.bincount(yi * k + pred_idx, minlength=k * k).reshape(k, k)
            return _confusion_metric(metric, cm, positive_idx)
        return score(metric, y_true, classes_arr[pred_idx])

    w = np.ones(k)
    base = _score(w)
    best = base
    grid = [0.5, 0.67, 0.8, 0.9, 1.0, 1.12, 1.25, 1.5, 2.0]
    for _ in range(3):
        improved = False
        for j in range(k):
            cur = w[j]
            for g in grid:
                w[j] = cur * g
                s = _score(w)
                if s > best + 1e-6:
                    best, cur = s, w[j]
                    improved = True
                else:
                    w[j] = cur
            w[j] = cur
        if not improved:
            break
    # Принимаем только осмысленный прирост: мелкие дельты — подгонка под шум OOF.
    if best > base + min_gain:
        return w, base, best
    return None, base, base


def tune_class_weights_stable(
    y_true, proba, classes, metric: str, n_boot: int = 5, max_rows: int = 30_000
):
    """Устойчивый подбор весов: усреднение по бутстреп-подвыборкам OOF.

    Подвыборки ограничены max_rows строк (координатный спуск по полному OOF
    на больших датасетах занимает десятки минут), между итерациями
    проверяется дедлайн. Возвращает (weights|None, базовый скор,
    скор усреднённых весов на полном OOF).
    """
    from agent_utils import time_left

    y_true = np.asarray(y_true)
    rng = np.random.RandomState(0)
    collected = []
    for _ in range(n_boot):
        idx = rng.choice(len(y_true), min(len(y_true), max_rows), replace=True)
        w, _, _ = tune_class_weights(y_true[idx], proba[idx], classes, metric, min_gain=0.0)
        collected.append(w if w is not None else np.ones(len(classes)))
        if time_left() < 180 and collected:
            log(f"tune_class_weights_stable: мало времени, останавливаюсь на {len(collected)} бутстрепах")
            break
    w_avg = np.exp(np.mean(np.log(np.clip(np.vstack(collected), 1e-3, None)), axis=0))

    classes_arr = np.array(classes, dtype=object)
    base_pred = classes_arr[np.argmax(proba, axis=1)]
    base = score(metric, y_true, base_pred)
    new_pred = classes_arr[np.argmax(proba * w_avg, axis=1)]
    new = score(metric, y_true, new_pred)
    if new > base + 3e-3:
        return w_avg, base, new
    if time_left() < 240:
        return None, base, base
    # Усреднение не помогло — пробуем обычный подбор на подвыборке полного OOF.
    sub = rng.choice(len(y_true), min(len(y_true), 2 * max_rows), replace=False)
    w, _, _ = tune_class_weights(y_true[sub], proba[sub], classes, metric, min_gain=0.0)
    if w is None:
        return None, base, base
    final_pred = classes_arr[np.argmax(proba * w, axis=1)]
    final = score(metric, y_true, final_pred)
    if final > base + 3e-3:
        return w, base, final
    return None, base, base


def clip_to_train_range(y_true, oof_pred, test_pred, metric: str, y_range=None):
    """Клиппинг регрессии к диапазону трейна — принимается только при улучшении OOF.

    y_range — (min, max) таргета во всём размеченном train; по умолчанию — по y_true.
    """
    lo, hi = y_range if y_range is not None else (float(np.min(y_true)), float(np.max(y_true)))
    base = score(metric, y_true, oof_pred)
    clipped_oof = np.clip(oof_pred, lo, hi)
    new = score(metric, y_true, clipped_oof)
    if is_better(metric, new, base) and new != base:
        return np.clip(test_pred, lo, hi), base, new
    return test_pred, base, base
