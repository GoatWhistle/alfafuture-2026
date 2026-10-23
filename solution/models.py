"""Модели и запись сабмита: быстрый LightGBM, AutoGluon, постпроцессинг под метрику."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd

from agent_utils import log
from eda_probe import row_keys

N_CPUS = os.cpu_count() or 8

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


def quick_model(prof: dict, target: str, max_rows: int = 300_000) -> dict:
    """LightGBM на одном сплите: быстрый ориентир качества + ранние предсказания."""
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split

    metric, ptype = prof["metric"], prof["problem_type"]
    X, y, Xte = _prep_matrices(prof, target)
    if len(X) > max_rows:
        idx = np.random.RandomState(0).choice(len(X), max_rows, replace=False)
        X, y = X.iloc[idx], y.iloc[idx]

    if prof["log_target"]:
        y = np.log1p(y.clip(lower=0))

    stratify = y if ptype != "regression" else None
    Xtr, Xval, ytr, yval = train_test_split(X, y, test_size=0.15, random_state=42, stratify=stratify)

    common = dict(
        n_estimators=3000, learning_rate=0.05, num_leaves=127,
        colsample_bytree=0.8, subsample=0.8, subsample_freq=1,
        n_jobs=N_CPUS, random_state=42, verbosity=-1,
    )
    if ptype == "regression":
        model = lgb.LGBMRegressor(objective="regression", **common)
        eval_metric = "l1" if metric == "mae" else "rmse"
    else:
        cw = "balanced" if metric in ("f1_macro", "balanced_accuracy", "f1") else None
        model = lgb.LGBMClassifier(class_weight=cw, **common)
        eval_metric = None
    model.fit(
        Xtr, ytr, eval_set=[(Xval, yval)], eval_metric=eval_metric,
        callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(0)],
    )

    if ptype == "regression":
        val_pred = model.predict(Xval)
        test_pred = model.predict(Xte)
        if prof["log_target"]:
            yval, val_pred = np.expm1(yval), np.expm1(val_pred)
            test_pred = np.expm1(test_pred)
            test_pred = np.clip(test_pred, 0, None)
        val_score = score(metric, yval, val_pred)
        proba_te = None
    else:
        val_proba = model.predict_proba(Xval)
        val_pred = model.classes_[np.argmax(val_proba, axis=1)]
        val_score = score(metric, yval, val_pred, proba=val_proba, labels=list(model.classes_))
        proba_te = model.predict_proba(Xte)
        test_pred = output_from_proba(prof, target, proba_te, list(model.classes_))

    log(f"quick_lgbm[{target}]: val {metric}={val_score:.5f}, best_iter={model.best_iteration_}")
    result = {
        "name": "quick_lgbm", "val_score": val_score, "test_pred": test_pred, "proba": proba_te,
        "importances": dict(zip(list(X.columns), model.feature_importances_.tolist())),
    }
    if ptype != "regression":
        result.update(val_proba=val_proba, y_val=np.asarray(yval), classes=list(model.classes_))
    return result


SEG_MIN_ROWS = 300


def _segment_codes(prof: dict, target: str):
    """Коды сегментов train (размеченные строки) и test по принятой segment_model, иначе None."""
    import fe_dsl

    seg = (prof.get("model_ops") or {}).get("segment_model")
    if seg is None:
        return None, None, None
    train = prof["train"]
    lab = train.loc[train[target].notna()]
    ctr, cte = fe_dsl.segment_codes(seg, lab, prof["test"])
    return seg, ctr, cte


def _fit_fold(make, X, y, tr_i, va_i, Xte, ctr, cte, classes=None):
    """Модель фолда (по сегментам, если заданы коды): предсказания val и test.

    Регрессия — векторы, классификация — вероятности в порядке classes.
    Сегмент с малым train предсказывает общая модель.
    """
    import lightgbm as lgb

    def fit(rows_tr, code=None):
        m = make()
        va_rows = va_i if code is None else va_i[ctr[va_i] == code]
        if len(va_rows) == 0:
            va_rows = va_i
        m.fit(X.iloc[rows_tr], y.iloc[rows_tr], eval_set=[(X.iloc[va_rows], y.iloc[va_rows])],
              callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
        return m

    def pred(m, Xp):
        if classes is None:
            return m.predict(Xp)
        out = np.zeros((len(Xp), len(classes)))
        pos = {c: i for i, c in enumerate(classes)}
        out[:, [pos[c] for c in m.classes_]] = m.predict_proba(Xp)
        return out

    glob = fit(tr_i)
    p_va, p_te = pred(glob, X.iloc[va_i]), pred(glob, Xte)
    if ctr is None:
        return p_va, p_te
    for code in np.unique(ctr[tr_i]):
        rows = tr_i[ctr[tr_i] == code]
        if len(rows) < SEG_MIN_ROWS or (classes is not None and y.iloc[rows].nunique() < 2):
            continue
        m = fit(rows, code)
        mv, mt = ctr[va_i] == code, cte == code
        if mv.any():
            p_va[mv] = pred(m, X.iloc[va_i[mv]])
        if mt.any():
            p_te[mt] = pred(m, Xte[mt])
    return p_va, p_te


def quick_oof(prof: dict, target: str, n_splits: int = 5, max_rows: int = 400_000) -> dict | None:
    """K-fold LightGBM: OOF на всём train (выравнен с OOF AutoGluon) + предсказания test.

    Учитывает модельные операции AutoEDA (prof["model_ops"]): отдельные модели
    по сегментам и функцию потерь регрессии. Возвращает None, когда датасет
    слишком велик для честного полнорядного OOF.
    """
    import lightgbm as lgb
    from sklearn.model_selection import KFold, StratifiedKFold

    metric, ptype = prof["metric"], prof["problem_type"]
    X, y, Xte = _prep_matrices(prof, target)
    if len(X) > max_rows:
        log(f"quick_oof: {len(X)} строк > {max_rows}, пропускаю бленд")
        return None
    y_orig = y.to_numpy()
    if prof["log_target"]:
        y = np.log1p(y.clip(lower=0))
    seg, ctr, cte = _segment_codes(prof, target)
    if seg is not None:
        log(f"quick_oof: отдельные модели по сегментам {seg['cols'][0]}:{seg['params']['rule']}")

    common = dict(
        n_estimators=1500, learning_rate=0.05, num_leaves=127,
        colsample_bytree=0.8, subsample=0.8, subsample_freq=1,
        n_jobs=N_CPUS, random_state=7, verbosity=-1,
    )
    if ptype == "regression":
        obj = {"objective": "regression"}
        o = (prof.get("model_ops") or {}).get("objective")
        if o is not None:
            p = o["params"]
            obj = {"objective": p["name"]}
            if "alpha" in p:
                obj["alpha" if p["name"] == "huber" else "fair_c"] = p["alpha"]
            log(f"quick_oof: функция потерь {obj}")
        oof = np.zeros(len(X))
        te = np.zeros(len(Xte))
        for tr_i, va_i in KFold(n_splits, shuffle=True, random_state=7).split(X):
            pv, pt = _fit_fold(lambda: lgb.LGBMRegressor(**obj, **common), X, y, tr_i, va_i, Xte, ctr, cte)
            oof[va_i] = pv
            te += pt / n_splits
        if prof["log_target"]:
            oof = np.clip(np.expm1(oof), 0, None)
            te = np.clip(np.expm1(te), 0, None)
        s = score(metric, y_orig, oof)
        log(f"quick_oof[{target}]: {metric}={s:.5f}")
        return {"y": y_orig, "oof": oof, "test": te, "classes": None, "score": s}

    classes = sorted(pd.Series(y_orig).dropna().unique().tolist())
    oof = np.zeros((len(X), len(classes)))
    te = np.zeros((len(Xte), len(classes)))
    cw = "balanced" if metric in ("f1_macro", "balanced_accuracy", "f1") else None
    for tr_i, va_i in StratifiedKFold(n_splits, shuffle=True, random_state=7).split(X, y):
        pv, pt = _fit_fold(lambda: lgb.LGBMClassifier(class_weight=cw, **common), X, y, tr_i, va_i, Xte,
                           ctr, cte, classes)
        oof[va_i] = pv
        te += pt / n_splits
    pred = np.array(classes, dtype=object)[np.argmax(oof, axis=1)]
    s = score(metric, y_orig, pred, proba=oof, labels=classes)
    log(f"quick_oof[{target}]: {metric}={s:.5f}")
    return {"y": y_orig, "oof": oof, "test": te, "classes": classes, "score": s}


def align_classes(mat: np.ndarray, src: list, dst: list) -> np.ndarray:
    """Переставляет колонки матрицы вероятностей из порядка src в порядок dst."""
    pos = {c: i for i, c in enumerate(src)}
    return mat[:, [pos[c] for c in dst]]


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


def copy_duplicate_labels(prof: dict, target: str, y_oof, oof_pred, test_pred, metric: str):
    """Копирование меток точных дублей признаков (классификация).

    Правило: строка получает метку, если все остальные строки train с тем же
    ключом признаков (≥1) несут одну и ту же метку. На train проверяется
    leave-one-out тем же правилом: копия должна быть не хуже модели (OOF) и
    по точности на затронутых строках, и по метрике в целом.
    Возвращает (test_pred | None, base, new, n_test_copied).
    """
    train, test = prof["train"], prof["test"]
    feats = list(prof.get("base_features") or prof["features"])
    lab = train.loc[train[target].notna()]
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

def fit_autogluon(prof: dict, target: str, time_budget: float):
    from autogluon.tabular import TabularPredictor

    train, features = prof["train"], prof["features"]
    df = train.loc[train[target].notna(), features + [target]].copy()
    if prof["log_target"]:
        df[target] = np.log1p(df[target].clip(lower=0))

    n_cells = len(df) * max(len(features), 1)
    if time_budget > 2700 and n_cells <= 60_000_000:
        presets = "best_quality"
    elif time_budget > 1200:
        presets = "high_quality"
    else:
        presets = "good_quality"

    path = tempfile.mkdtemp(prefix="ag_", dir="/tmp")
    extra: dict = {}
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
    log(f"AutoGluon[{target}]: presets={presets}, time_limit={time_budget:.0f}s, rows={len(df)}")
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


def ag_predict(prof: dict, predictor, target: str) -> tuple[np.ndarray, np.ndarray | None]:
    """Предсказания AutoGluon на test: (вектор для сабмита, матрица вероятностей)."""
    test = prof["test"][prof["features"]]
    if prof["problem_type"] == "regression":
        pred = predictor.predict(test).to_numpy()
        if prof["log_target"]:
            pred = np.clip(np.expm1(pred), 0, None)
        return pred, None
    proba = predictor.predict_proba(test)
    return output_from_proba(prof, target, proba.to_numpy(), list(proba.columns)), proba.to_numpy()


def ag_oof(predictor, prof: dict, target: str):
    """OOF-предсказания (есть при бэггинге). Возвращает (y_true, oof_pred, oof_proba, classes) или None."""
    try:
        train = prof["train"]
        y_true = train.loc[train[target].notna(), target].reset_index(drop=True)
        if prof["problem_type"] == "regression":
            for name in ("predict_oof", "get_oof_pred"):
                fn = getattr(predictor, name, None)
                if fn is not None:
                    oof = np.asarray(fn())
                    if prof["log_target"]:
                        oof = np.clip(np.expm1(oof), 0, None)
                    return y_true.to_numpy(), oof, None, None
            return None
        for name in ("predict_proba_oof", "get_oof_pred_proba"):
            fn = getattr(predictor, name, None)
            if fn is not None:
                proba = fn()
                classes = list(proba.columns)
                arr = proba.to_numpy()
                pred = np.array(classes, dtype=object)[np.argmax(arr, axis=1)]
                return y_true.to_numpy(), pred, arr, classes
        return None
    except Exception as exc:  # noqa: BLE001
        log(f"OOF недоступен: {exc}")
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


def clip_to_train_range(y_true, oof_pred, test_pred, metric: str, eval_mask=None):
    """Клиппинг регрессии к диапазону трейна — принимается только при улучшении OOF.

    eval_mask ограничивает оценку строками (например, последним периодом train).
    """
    lo, hi = float(np.min(y_true)), float(np.max(y_true))
    if eval_mask is not None:
        y_true, oof_pred = np.asarray(y_true)[eval_mask], np.asarray(oof_pred)[eval_mask]
    base = score(metric, y_true, oof_pred)
    clipped_oof = np.clip(oof_pred, lo, hi)
    new = score(metric, y_true, clipped_oof)
    if is_better(metric, new, base) and new != base:
        return np.clip(test_pred, lo, hi), base, new
    return test_pred, base, base
