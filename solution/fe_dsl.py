"""DSL гипотез feature engineering: декларативные спеки и их исполнитель.

Спека — dict {"op", "cols", "params", "source", "prior"}. Детекторы и LLM
выдают только спеки, код признаков живёт здесь. Исполнение всегда идёт по
схеме fit -> apply: статистики (в т.ч. по таргету) берутся только из
fit-строк; fit-строки получают cross-fitted значения (своя строка в
статистику не попадает), apply-строки — значения по всему fit. Одна и та же
функция строит признаки и для валидационного фолда (fit = train фолда,
apply = val), и для финала (fit = весь train, apply = test).
"""

from __future__ import annotations

import hashlib
import re

import numpy as np
import pandas as pd

from agent_utils import log
from eda_probe import row_keys

TARGET = "__target__"
MAX_CLASSES = 20
CROSSFIT_SPLITS = 5

# op -> (мин. число cols, макс. число cols, нужен ли таргет)
OPS = {
    "group_stat": (0, 1, False),   # cols=[src] или [] для таргета; params.keys
    "group_dev": (1, 1, False),
    "lag": (0, 1, False),          # cols=[src] или [] для таргета; params.keys, params.period
    "log": (1, 1, False),
    "rank": (1, 1, False),
    "ratio": (2, 2, False),
    "diff": (2, 2, False),
    "prod": (2, 2, False),
    "agg_cols": (2, 64, False),
    "count_enc": (1, 1, False),
    "missing_ind": (1, 1, False),
    "bin": (1, 1, False),
    "interaction": (2, 2, False),
    "knn_target": (1, 20, True),
    "drop": (1, 1, False),
}
TARGET_AGGS = ("mean", "median", "std", "count")
LAG_AGGS = ("last", "mean", "count", "gap")
COL_AGGS = ("mean", "std", "min", "max", "sum", "range")


# ---------------------------------------------------------------------------
# Валидация и имена.
# ---------------------------------------------------------------------------

def _str_list(x) -> list[str] | None:
    if isinstance(x, str):
        return [x]
    if isinstance(x, (list, tuple)) and all(isinstance(v, str) for v in x):
        return list(dict.fromkeys(x))
    return None


def validate(spec, columns, numeric: set[str] | None = None) -> dict | None:
    """Нормализованная спека или None (неизвестная операция, чужие колонки, кривые параметры)."""
    if not isinstance(spec, dict) or spec.get("op") not in OPS:
        return None
    op = spec["op"]
    columns = set(columns)
    cols = _str_list(spec.get("cols") or [])
    raw = spec.get("params") if isinstance(spec.get("params"), dict) else {}
    if cols is None:
        return None
    # Таргет в cols допустим как источник group_stat/lag.
    src_target = TARGET in cols or raw.get("src") == "target"
    cols = [c for c in cols if c != TARGET]
    lo, hi, _ = OPS[op]
    if not lo <= len(cols) <= hi or not set(cols) <= columns:
        return None
    if numeric is not None and op in ("log", "rank", "ratio", "diff", "prod", "agg_cols", "bin", "group_dev", "knn_target"):
        if not set(cols) <= numeric:
            return None
    params: dict = {}
    if op in ("group_stat", "group_dev", "lag"):
        keys = _str_list(raw.get("keys"))
        if not keys or not 1 <= len(keys) <= 5 or not set(keys) <= columns:
            return None
        params["keys"] = keys
        if op == "group_stat" and not cols and not src_target:
            src_target = True
        bins = raw.get("bin")
        if isinstance(bins, (int, float)) and 2 <= int(bins) <= 100:
            params["bin"] = int(bins)
    if op == "group_stat":
        aggs = [a for a in (_str_list(raw.get("aggs")) or []) if a in TARGET_AGGS] or ["mean", "count"]
        params["aggs"] = aggs
    if op == "lag":
        period = raw.get("period")
        if not isinstance(period, str) or period not in columns or period in params["keys"]:
            return None
        if numeric is not None and period not in numeric:
            return None
        params["period"] = period
        if not cols:
            src_target = True
        params["aggs"] = [a for a in (_str_list(raw.get("aggs")) or []) if a in LAG_AGGS] or list(LAG_AGGS)
    if op in ("group_stat", "lag") and src_target and cols:
        return None
    if op == "agg_cols":
        params["aggs"] = [a for a in (_str_list(raw.get("aggs")) or []) if a in COL_AGGS] or ["mean", "std", "range"]
    if op in ("bin", "interaction"):
        q = raw.get("bins", raw.get("bin", 10))
        params["bins"] = int(q) if isinstance(q, (int, float)) and 2 <= int(q) <= 100 else 10
    if op == "knn_target":
        k = raw.get("k", 10)
        params["k"] = int(k) if isinstance(k, (int, float)) and 1 <= int(k) <= 50 else 10
    if src_target and op in ("group_stat", "lag"):
        params["src"] = "target"
    prior = spec.get("prior")
    return {
        "op": op, "cols": cols, "params": params,
        "source": str(spec.get("source") or "unknown")[:80],
        "prior": float(prior) if isinstance(prior, (int, float)) and np.isfinite(prior) else 0.0,
        "why": str(spec.get("why") or "")[:300],
    }


def uses_target(spec: dict) -> bool:
    return spec["params"].get("src") == "target" or spec["op"] == "knn_target"


def spec_key(spec: dict) -> str:
    """Каноническая строка спеки (дедупликация, кэш)."""
    p = spec["params"]
    return f"{spec['op']}|{','.join(spec['cols'])}|" + "|".join(f"{k}={p[k]}" for k in sorted(p))


def describe(spec: dict) -> str:
    p = {k: v for k, v in spec["params"].items()}
    return f"{spec['op']}({', '.join(spec['cols']) or 'target'}{', ' if p else ''}{p if p else ''})"


def _name(spec: dict, suffix: str) -> str:
    parts = [spec["op"], *spec["cols"]]
    if "keys" in spec["params"]:
        parts += ["by", *spec["params"]["keys"]]
    if spec["params"].get("src") == "target":
        parts.insert(1, "y")
    base = "__".join(re.sub(r"[^0-9A-Za-z_]", "_", str(x)) for x in parts)
    if len(base) > 90:
        base = base[:70] + "_" + hashlib.md5(base.encode()).hexdigest()[:8]
    return f"fe_{base}__{suffix}" if suffix else f"fe_{base}"


# ---------------------------------------------------------------------------
# Ключи групп.
# ---------------------------------------------------------------------------

def _is_num(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and s.dtype != bool


def _edges(s: pd.Series, q: int) -> np.ndarray:
    v = pd.to_numeric(s, errors="coerce").dropna().to_numpy(float)
    if len(v) == 0:
        return np.array([])
    return np.unique(np.quantile(v, np.linspace(0, 1, q + 1)[1:-1]))


def _binned(s: pd.Series, edges: np.ndarray) -> pd.Series:
    v = pd.to_numeric(s, errors="coerce").to_numpy(float)
    out = np.digitize(v, edges).astype(float)
    out[np.isnan(v)] = -1
    return pd.Series(out, index=s.index)


def _key_frames(fit: pd.DataFrame, app: pd.DataFrame, keys: list[str], bins: int | None):
    """Колонки ключа для fit и apply; числовые с bin — по квантилям fit."""
    if not bins:
        return fit[keys], app[keys]
    kf, ka = {}, {}
    for c in keys:
        if _is_num(fit[c]) and fit[c].nunique() > bins:
            e = _edges(fit[c], bins)
            kf[c], ka[c] = _binned(fit[c], e), _binned(app[c], e)
        else:
            kf[c], ka[c] = fit[c], app[c]
    return pd.DataFrame(kf, index=fit.index), pd.DataFrame(ka, index=app.index)


def _keys(fit, app, keys, bins=None):
    kf, ka = _key_frames(fit, app, keys, bins)
    return row_keys(kf, keys), row_keys(ka, keys)


# ---------------------------------------------------------------------------
# Статистики таргета (cross-fit).
# ---------------------------------------------------------------------------

def _y_matrix(y: np.ndarray, classes: list | None) -> np.ndarray:
    if classes is None:
        return np.asarray(y, dtype=float).reshape(-1, 1)
    ys = pd.Series(y).astype(str).to_numpy()
    return np.column_stack([(ys == str(c)).astype(float) for c in classes])


def _group_table(k_src: np.ndarray, Y: np.ndarray, aggs: list[str], is_cls: bool) -> pd.DataFrame:
    df = pd.DataFrame(Y, columns=[f"c{i}" for i in range(Y.shape[1])])
    df["k"] = k_src
    g = df.groupby("k")
    cols = {}
    if is_cls:
        m = g.mean()
        for i in range(Y.shape[1]):
            cols[f"p{i}"] = m[f"c{i}"]
    else:
        if "mean" in aggs:
            cols["mean"] = g["c0"].mean()
        if "median" in aggs:
            cols["median"] = g["c0"].median()
        if "std" in aggs:
            cols["std"] = g["c0"].std()
    cols["count"] = g.size().astype(float)
    return pd.DataFrame(cols)


def _lookup(table: pd.DataFrame, k_dst: np.ndarray, cnt_scale: float = 1.0) -> pd.DataFrame:
    out = table.reindex(k_dst).reset_index(drop=True)
    out["count"] = out["count"].fillna(0) * cnt_scale
    return out


def _crossfit(n_fit: int, fn, seed: int = 11) -> pd.DataFrame:
    """fn(src_idx, dst_idx) -> DataFrame для dst строк; собирает OOF-значения для всех fit-строк."""
    from sklearn.model_selection import KFold

    parts, idxs = [], []
    for tr_i, va_i in KFold(CROSSFIT_SPLITS, shuffle=True, random_state=seed).split(np.arange(n_fit)):
        parts.append(fn(tr_i, va_i))
        idxs.append(va_i)
    out = pd.concat(parts, ignore_index=True)
    order = np.empty(n_fit, dtype=np.int64)
    order[np.concatenate(idxs)] = np.arange(n_fit)
    return out.iloc[order].reset_index(drop=True)


def _target_group_stat(spec, fit, fit_y, app, classes):
    p = spec["params"]
    kf, ka = _keys(fit, app, p["keys"], p.get("bin"))
    Y = _y_matrix(fit_y, classes)
    is_cls = classes is not None
    aggs = p["aggs"]
    scale = CROSSFIT_SPLITS / (CROSSFIT_SPLITS - 1)

    def fn(src, dst):
        return _lookup(_group_table(kf[src], Y[src], aggs, is_cls), kf[dst], scale)

    f = _crossfit(len(fit), fn)
    a = _lookup(_group_table(kf, Y, aggs, is_cls), ka)
    names = {c: _name(spec, c if not c.startswith("p") else f"p_{classes[int(c[1:])]}") for c in f.columns}
    if "count" not in aggs and not is_cls:
        f, a = f.drop(columns="count"), a.drop(columns="count")
    return f.rename(columns=names), a.rename(columns=names)


def _col_group_stat(spec, fit, app):
    """Статистика признака по группе — без таргета, по fit+apply вместе."""
    p = spec["params"]
    src = spec["cols"][0]
    both = pd.concat([fit, app], ignore_index=True)
    kf, _ = _keys(both, both.iloc[:0], p["keys"], p.get("bin"))
    v = pd.to_numeric(both[src], errors="coerce") if _is_num(both[src]) else both[src].astype(str)
    g = pd.DataFrame({"k": kf, "v": v}).groupby("k")["v"]
    out = {}
    for agg in p["aggs"]:
        if agg == "count":
            out[_name(spec, "count")] = g.transform("size").astype(float)
        elif _is_num(both[src]):
            out[_name(spec, agg)] = g.transform(agg)
        elif agg == "mean":
            out[_name(spec, "nunique")] = g.transform("nunique").astype(float)
    df = pd.DataFrame(out)
    return df.iloc[: len(fit)].reset_index(drop=True), df.iloc[len(fit):].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Лаги: история той же сущности в прошлых периодах.
# ---------------------------------------------------------------------------

def _lag_table(k: np.ndarray, period: np.ndarray, v: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame({"k": k.astype(np.int64), "t": period, "v": v}).dropna(subset=["t", "v"])
    per = df.groupby(["k", "t"])["v"].agg(["sum", "size"]).reset_index().sort_values(["k", "t"])
    per["last"] = per["sum"] / per["size"]
    g = per.groupby("k")
    per["cum_sum"] = g["sum"].cumsum()
    per["cum_n"] = g["size"].cumsum()
    per["mean"] = per["cum_sum"] / per["cum_n"]
    per["t_last"] = per["t"]
    return per[["k", "t", "last", "mean", "cum_n", "t_last"]]


def _lag_lookup(table: pd.DataFrame, k: np.ndarray, period: np.ndarray) -> pd.DataFrame:
    """Для каждой строки (k, t) — накопленные значения по периодам строго < t."""
    q = pd.DataFrame({"k": k.astype(np.int64), "t": period.astype(float), "_i": np.arange(len(k))})
    has_t = q["t"].notna()
    res = pd.DataFrame(index=np.arange(len(k)), columns=["last", "mean", "cum_n", "t_last"], dtype=float)
    if has_t.any() and len(table):
        qq = q[has_t].sort_values("t")
        tt = table.astype({"t": float}).sort_values("t")
        m = pd.merge_asof(qq, tt, on="t", by="k", allow_exact_matches=False, direction="backward")
        res.loc[m["_i"].to_numpy(), ["last", "mean", "cum_n", "t_last"]] = m[["last", "mean", "cum_n", "t_last"]].to_numpy()
    res["gap"] = period.astype(float) - res["t_last"].astype(float)
    res["count"] = res["cum_n"].astype(float).fillna(0)
    return res


def _lag(spec, fit, fit_y, app, classes):
    p = spec["params"]
    period = p["period"]
    kf, ka = _keys(fit, app, p["keys"])
    tf = pd.to_numeric(fit[period], errors="coerce").to_numpy(float)
    ta = pd.to_numeric(app[period], errors="coerce").to_numpy(float)
    if p.get("src") == "target":
        if classes is not None:
            # Классификация: история доли самого частого класса в сущности не информативна —
            # берём индикатор каждого класса (ограничено MAX_CLASSES).
            Y = _y_matrix(fit_y, classes)
        else:
            Y = np.asarray(fit_y, dtype=float).reshape(-1, 1)
        k_src, t_src = kf, tf
    else:
        src = spec["cols"][0]
        both_v = pd.concat([fit[src], app[src]], ignore_index=True)
        Y = pd.to_numeric(both_v, errors="coerce").to_numpy(float).reshape(-1, 1)
        k_src, t_src = np.concatenate([kf, ka]), np.concatenate([tf, ta])
    outs_f, outs_a = {}, {}
    for j in range(Y.shape[1]):
        tab = _lag_table(k_src, t_src, Y[:, j])
        lf, la = _lag_lookup(tab, kf, tf), _lag_lookup(tab, ka, ta)
        tag = "" if Y.shape[1] == 1 else f"p_{classes[j]}_"
        for agg in p["aggs"]:
            if agg in ("count", "gap") and j > 0:
                continue
            col = {"last": "last", "mean": "mean", "count": "count", "gap": "gap"}[agg]
            outs_f[_name(spec, tag + agg)] = lf[col].astype(float).to_numpy()
            outs_a[_name(spec, tag + agg)] = la[col].astype(float).to_numpy()
    return pd.DataFrame(outs_f), pd.DataFrame(outs_a)


# ---------------------------------------------------------------------------
# Соседи.
# ---------------------------------------------------------------------------

def _knn_target(spec, fit, fit_y, app, classes):
    from sklearn.neighbors import NearestNeighbors

    cols, k = spec["cols"], spec["params"]["k"]
    Xf = fit[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    Xa = app[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    med = np.nanmedian(Xf, axis=0)
    med = np.where(np.isnan(med), 0, med)
    sd = np.nanstd(Xf, axis=0)
    sd = np.where((sd > 0) & np.isfinite(sd), sd, 1)
    Xf = np.where(np.isnan(Xf), med, Xf)
    Xa = np.where(np.isnan(Xa), med, Xa)
    Xf, Xa = (Xf - med) / sd, (Xa - med) / sd
    Y = _y_matrix(fit_y, classes)

    def fn_frame(src, Xq):
        nn = NearestNeighbors(n_neighbors=min(k, len(src))).fit(Xf[src])
        dist, ind = nn.kneighbors(Xq)
        vals = Y[src][ind]  # (n, k, K)
        out = {}
        if classes is None:
            out["mean"] = vals[:, :, 0].mean(axis=1)
        else:
            m = vals.mean(axis=1)
            for j, c in enumerate(classes):
                out[f"p_{c}"] = m[:, j]
        out["dist1"] = dist[:, 0]
        return pd.DataFrame(out)

    f = _crossfit(len(fit), lambda s, d: fn_frame(s, Xf[d]))
    a = fn_frame(np.arange(len(fit)), Xa)
    names = {c: _name(spec, c) for c in f.columns}
    return f.rename(columns=names), a.rename(columns=names)


# ---------------------------------------------------------------------------
# Безтаргетные преобразования (по fit+apply вместе).
# ---------------------------------------------------------------------------

def _num(df: pd.DataFrame, c: str) -> pd.Series:
    return pd.to_numeric(df[c], errors="coerce").astype(float)


def _unsupervised(spec, both: pd.DataFrame) -> pd.DataFrame:
    op, cols, p = spec["op"], spec["cols"], spec["params"]
    if op == "log":
        x = _num(both, cols[0])
        return pd.DataFrame({_name(spec, ""): np.sign(x) * np.log1p(x.abs())})
    if op == "rank":
        return pd.DataFrame({_name(spec, ""): _num(both, cols[0]).rank(pct=True)})
    if op in ("ratio", "diff", "prod"):
        a, b = _num(both, cols[0]), _num(both, cols[1])
        v = a / b.replace(0, np.nan) if op == "ratio" else (a - b if op == "diff" else a * b)
        return pd.DataFrame({_name(spec, ""): v.replace([np.inf, -np.inf], np.nan)})
    if op == "agg_cols":
        m = both[cols].apply(pd.to_numeric, errors="coerce")
        out = {}
        for agg in p["aggs"]:
            if agg == "range":
                out[_name(spec, agg)] = m.max(axis=1) - m.min(axis=1)
            else:
                out[_name(spec, agg)] = getattr(m, agg)(axis=1)
        return pd.DataFrame(out)
    if op == "count_enc":
        s = both[cols[0]].astype(str)
        return pd.DataFrame({_name(spec, ""): s.map(s.value_counts()).astype(float)})
    if op == "missing_ind":
        return pd.DataFrame({_name(spec, ""): both[cols[0]].isna().astype(float)})
    if op == "bin":
        return pd.DataFrame({_name(spec, ""): _binned(both[cols[0]], _edges(both[cols[0]], p["bins"]))})
    if op == "interaction":
        kf, _ = _key_frames(both, both.iloc[:0], cols, p["bins"])
        code = kf[cols[0]].astype(str) + "|" + kf[cols[1]].astype(str)
        return pd.DataFrame({
            _name(spec, "code"): code.astype(object),
            _name(spec, "count"): code.map(code.value_counts()).astype(float),
        })
    if op == "group_dev":
        kf, _ = _keys(both, both.iloc[:0], p["keys"], p.get("bin"))
        x = _num(both, cols[0])
        return pd.DataFrame({_name(spec, ""): x - x.groupby(kf).transform("mean").to_numpy()})
    raise ValueError(op)


# ---------------------------------------------------------------------------
# Исполнитель.
# ---------------------------------------------------------------------------

def apply(spec: dict, fit: pd.DataFrame, fit_y, app: pd.DataFrame, classes: list | None = None):
    """Признаки спеки: (fit_df, app_df) с RangeIndex по строкам fit/app или None.

    classes — список классов (классификация) или None (регрессия).
    """
    op = spec["op"]
    if op == "drop":
        return None
    fit = fit.reset_index(drop=True)
    app = app.reset_index(drop=True)
    fit_y = np.asarray(fit_y)
    if classes is not None and len(classes) > MAX_CLASSES and uses_target(spec):
        return None
    if op == "group_stat":
        if spec["params"].get("src") == "target":
            return _target_group_stat(spec, fit, fit_y, app, classes)
        return _col_group_stat(spec, fit, app)
    if op == "lag":
        return _lag(spec, fit, fit_y, app, classes)
    if op == "knn_target":
        return _knn_target(spec, fit, fit_y, app, classes)
    both = pd.concat([fit, app], ignore_index=True)
    out = _unsupervised(spec, both)
    return out.iloc[: len(fit)].reset_index(drop=True), out.iloc[len(fit):].reset_index(drop=True)


def safe_apply(spec, fit, fit_y, app, classes=None):
    """apply без исключений; пустой/константный результат -> None."""
    try:
        res = apply(spec, fit, fit_y, app, classes)
    except Exception as exc:  # noqa: BLE001 — кривая спека не роняет агента
        log(f"fe_dsl: спека {describe(spec)} упала: {type(exc).__name__}: {exc}")
        return None
    if res is None:
        return None
    f, a = res
    keep = [c for c in f.columns if f[c].nunique(dropna=True) > 1]
    if not keep:
        return None
    return f[keep], a[keep]


def materialize(prof: dict, spec: dict, target: str, classes: list | None = None):
    """Финальные признаки спеки: fit = размеченный train, apply = test (+ неразмеченный train).

    Возвращает (tr_df с индексом train, te_df с индексом test) или None.
    """
    train, test = prof["train"], prof["test"]
    lab = train[target].notna().to_numpy()
    fit = train.loc[lab]
    app = pd.concat([train.loc[~lab], test], ignore_index=True)
    res = safe_apply(spec, fit, fit[target].to_numpy(), app, classes)
    if res is None:
        return None
    f, a = res
    n_unl = int((~lab).sum())
    tr = pd.concat([f.set_axis(train.index[lab]), a.iloc[:n_unl].set_axis(train.index[~lab])]).reindex(train.index)
    te = a.iloc[n_unl:].set_axis(test.index)
    return tr, te
