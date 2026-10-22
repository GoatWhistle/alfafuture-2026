"""EDA-примитивы: ключи строк, временные колонки, квази-ID объекта, дубли.

Используются детекторами AutoEDA (eda_detectors) и DSL признаков (fe_dsl).
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

MAX_ROWS = 50_000          # общая выборка для статистик колонок
QID_MAX_ROWS = 200_000     # квази-ID ломается от сабсемплинга: группы дробятся
DUP_MAX_ROWS = 300_000
QID_MAX_COMBOS = 1000     # одна оценка ~10 мс на 50k строк
QID_BEAM = 5
QID_EXTEND_TOP = 60
QID_MIN_COVERAGE = 0.3    # колонка, чьи значения в test новые, ключом быть не может


def normalize_keys(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Строковое представление колонок, одинаковое для train и test.

    Числа приводятся к float: иначе int-колонка train ("1942") и float-колонка
    test с пропусками ("1942.0") дают разные ключи.
    """
    out = {}
    for c in cols:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s) and s.dtype != bool:
            out[c] = s.astype(float).astype(str)
        else:
            out[c] = s.astype(str)
    return pd.DataFrame(out, index=df.index)


def row_keys(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """Хэш строки по нормализованным значениям колонок (точные дубли, группы)."""
    return pd.util.hash_pandas_object(normalize_keys(df, cols), index=False).to_numpy()


def _sample(df: pd.DataFrame, n: int, seed: int = 0) -> pd.DataFrame:
    return df if len(df) <= n else df.sample(n, random_state=seed)


def _is_cat(s: pd.Series) -> bool:
    return s.dtype == object or str(s.dtype) in ("category", "bool")


def _temporal_candidates(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> list[dict]:
    """Числовые колонки-«периоды»: значения test не пересекаются с train и лежат правее."""
    out = []
    for c in features:
        if _is_cat(train[c]) or not pd.api.types.is_numeric_dtype(train[c]):
            continue
        tr, te = train[c].dropna(), test[c].dropna()
        if tr.empty or te.empty:
            continue
        if te.min() > tr.max() and not set(te.unique()) & set(tr.unique()):
            last = tr.max()
            out.append({
                "column": c,
                "train_range": [float(tr.min()), float(last)],
                "test_values": sorted(float(v) for v in te.unique()[:5]),
                "last_period_rows": int((train[c] == last).sum()),
            })
    return out


def target_matrix(y: pd.Series, problem_type: str, max_classes: int = 20) -> np.ndarray | None:
    """Таргет как матрица n×K: регрессия — один столбец, классификация — one-hot."""
    if problem_type == "regression":
        return pd.to_numeric(y, errors="coerce").to_numpy(float).reshape(-1, 1)
    ys = y.astype(str).to_numpy()
    classes = pd.unique(ys)
    if len(classes) > max_classes:
        return None
    return np.column_stack([(ys == c).astype(float) for c in classes])


def _quasi_id(train: pd.DataFrame, test: pd.DataFrame, target: str, features: list[str],
              exclude: set[str], deadline: float, problem_type: str = "regression") -> dict | None:
    """Жадный beam-поиск ключа группы (1–4 колонки), повторяющегося в train и test.

    Скор = покрытие_test × (1 − std_внутри/std_глобальный). Внутригрупповой std —
    пулированный по группам размера ≥2 (с поправкой на число групп); для
    классификации — по one-hot таргету (сумма дисперсий индикаторов классов).
    """
    train = train.loc[train[target].notna()]
    Y = target_matrix(train[target], problem_type)
    if Y is None:
        return None
    ok = ~np.isnan(Y).any(axis=1)
    train, Y = train.loc[ok], Y[ok]
    n_tr = len(train)
    g_std = float(np.sqrt(Y.var(axis=0, ddof=1).sum())) if n_tr > 1 else 0.0
    if n_tr < 100 or g_std == 0:
        return None
    ycols = [f"y{i}" for i in range(Y.shape[1])]

    cands = []
    for c in features:
        if c in exclude:
            continue
        nu = train[c].nunique()
        if _is_cat(train[c]):
            if nu >= 2:
                cands.append(c)
        elif 20 <= nu <= 0.5 * n_tr:
            cands.append(c)
    if not cands:
        return None

    codes = {}
    for c in cands:
        both = pd.concat([train[c], test[c]], ignore_index=True)
        if not _is_cat(both):
            both = both.astype(float)
        codes[c] = pd.factorize(both)[0].astype(np.int64) + 1  # NaN -> 0

    def evaluate(cols: tuple) -> dict:
        k = codes[cols[0]]
        for c in cols[1:]:
            k = pd.factorize(k * (codes[c].max() + 1) + codes[c])[0].astype(np.int64)
        ktr, kte = k[:n_tr], k[n_tr:]
        coverage = float(np.isin(kte, ktr).mean())
        df = pd.DataFrame(Y, columns=ycols)
        df["k"] = ktr
        grp = df.groupby("k")
        cnt = grp[ycols[0]].transform("size").to_numpy()
        rep = cnt >= 2
        n_groups = int(pd.Series(ktr[rep]).nunique())
        dof = int(rep.sum()) - n_groups
        res = {"key": list(cols), "coverage": round(coverage, 3), "rep_share": round(float(rep.mean()), 3)}
        if dof < 30 or rep.mean() < 0.02:
            res.update(score=0.0, std_within=None)
            return res
        dev = (Y - grp[ycols].transform("mean").to_numpy())[rep]
        within = float(np.sqrt((dev ** 2).sum() / dof))
        res.update(std_within=round(within, 4), score=round(coverage * max(0.0, 1 - within / g_std), 4))
        return res

    evaluated: dict[tuple, dict] = {}

    def run(cols: tuple) -> None:
        key = tuple(sorted(cols))
        if key not in evaluated and len(evaluated) < QID_MAX_COMBOS and time.time() < deadline:
            evaluated[key] = evaluate(key)

    for c in cands:
        run((c,))
    # Расширяем по колонкам, живущим в test: одиночный скор у них бывает низким
    # (year_built сам по себе почти не сжимает std), но в паре они дают ключ.
    singles = sorted(
        (v for v in evaluated.values() if v["coverage"] >= QID_MIN_COVERAGE), key=lambda r: -r["score"]
    )
    ext = [r["key"][0] for r in singles[:QID_EXTEND_TOP]]
    beam = [tuple(r["key"]) for r in singles[:QID_BEAM]]
    for _ in range(3):
        for b in beam:
            for c in ext:
                if c not in b:
                    run(b + (c,))
        level = [v for k, v in evaluated.items() if len(k) == len(beam[0]) + 1] if beam else []
        if not level:
            break
        beam = [tuple(r["key"]) for r in sorted(level, key=lambda r: -r["score"])[:QID_BEAM]]

    ranked = sorted(evaluated.values(), key=lambda r: -r["score"])
    best = ranked[0]
    if best["score"] <= 0:
        return {"best": None, "evaluated": len(evaluated)}

    # LOO-RMSE группового среднего для лучшего ключа (строки в группах ≥2).
    k = codes[best["key"][0]]
    for c in best["key"][1:]:
        k = pd.factorize(k * (codes[c].max() + 1) + codes[c])[0].astype(np.int64)
    df = pd.DataFrame(Y, columns=ycols)
    df["k"] = k[:n_tr]
    s = df.groupby("k")[ycols].transform("sum").to_numpy()
    n = df.groupby("k")[ycols[0]].transform("size").to_numpy()
    rep = n >= 2
    loo = (s[rep] - Y[rep]) / (n[rep] - 1)[:, None]
    best = dict(best)
    best["loo_rmse"] = round(float(np.sqrt(np.mean(((Y[rep] - loo) ** 2).sum(axis=1)))), 4)
    best["std_global"] = round(g_std, 4)
    best["compression"] = round(g_std / best["std_within"], 2) if best["std_within"] else None
    return {
        "best": best,
        "alternatives": [{kk: r[kk] for kk in ("key", "coverage", "std_within", "score")} for r in ranked[1:4]],
        "evaluated": len(evaluated),
    }


def _duplicates(train: pd.DataFrame, test: pd.DataFrame, target: str, features: list[str]) -> dict:
    """Точные дубли признаков: в train (в т.ч. с конфликтом меток) и test-строки, найденные в train."""
    tr = _sample(train.loc[train[target].notna()], DUP_MAX_ROWS)
    te = _sample(test, DUP_MAX_ROWS)
    ktr, kte = row_keys(tr, features), row_keys(te, features)
    df = pd.DataFrame({"k": ktr, "y": tr[target].astype(str).to_numpy()})
    cnt = df.groupby("k")["y"].transform("size")
    nlab = df.groupby("k")["y"].transform("nunique")
    pure = set(df.loc[nlab == 1, "k"].unique())
    in_train = np.isin(kte, ktr)
    return {
        "train_dup_share": round(float((cnt >= 2).mean()), 4),
        "train_conflict_share": round(float((nlab > 1).mean()), 4),
        "test_in_train_share": round(float(in_train.mean()), 4),
        "test_in_train_pure_share": round(float(pd.Series(kte).isin(pure).mean()), 4),
    }
