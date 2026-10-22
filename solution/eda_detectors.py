"""Детекторы зависимостей: каждый смотрит на данные и выдаёт спеки гипотез FE.

Детектор — функция detect(ctx) -> list[spec]. prior — непрерывная оценка
пользы (доля объяснённой дисперсии остатков/таргета, покрытие × сжатие), без
жёстких порогов: решение принимает валидатор (fe_search), prior задаёт лишь
порядок оценки под бюджетом. Детекторы могут дописывать в ctx схему
валидации (period, entity) и флаги постпроцессинга (label_copy).
"""

from __future__ import annotations

import re
import time
import traceback
from itertools import combinations

import numpy as np
import pandas as pd

import fe_dsl
from agent_utils import log
from eda_probe import _duplicates, _is_cat, _quasi_id, _sample, _temporal_candidates, row_keys

DETECTOR_SEC = 60
TOP_RESID_COLS = 12
PAIR_TOP = 8


# ---------------------------------------------------------------------------
# Общие оценки.
# ---------------------------------------------------------------------------

def _codes(s: pd.Series, q: int = 10, top: int = 30) -> np.ndarray:
    """Дискретизация колонки: квантильные бины (числа) или топ-уровни (категории); NaN — свой код."""
    if not _is_cat(s) and pd.api.types.is_numeric_dtype(s):
        v = s.astype(float)
        if v.nunique() > q:
            r = v.rank(method="dense")
            c = pd.qcut(r, q, labels=False, duplicates="drop")
        else:
            c = pd.Series(pd.factorize(v)[0], index=s.index).where(v.notna())
        return c.fillna(-1).to_numpy(np.int64) + 1
    st = s.astype(str).where(s.notna(), "__nan__")
    keep = set(st.value_counts().index[:top])
    return pd.factorize(st.where(st.isin(keep), "__other__"))[0].astype(np.int64)


def omega2(r: np.ndarray, codes: np.ndarray) -> float:
    """Доля дисперсии r, объяснённая группами (ω², ≈0 при отсутствии связи)."""
    ok = np.isfinite(r)
    r, codes = r[ok], codes[ok]
    n = len(r)
    if n < 50:
        return 0.0
    _, inv = np.unique(codes, return_inverse=True)
    g = inv.max() + 1
    if g < 2:
        return 0.0
    cnt = np.bincount(inv, minlength=g)
    s = np.bincount(inv, weights=r, minlength=g)
    mean = r.mean()
    ssb = float(((s / np.maximum(cnt, 1) - mean) ** 2 * cnt).sum())
    sst = float(((r - mean) ** 2).sum())
    if sst <= 0 or n <= g:
        return 0.0
    msw = (sst - ssb) / (n - g)
    return max(0.0, (ssb - (g - 1) * msw) / (sst + msw))


def _resid_prior(ctx: dict, spec: dict, base_cols: list[str] | None = None) -> float:
    """Прирост ω² остатков от признаков спеки над её входными колонками (выборка ≤20k)."""
    df = ctx["df"]
    idx = np.arange(len(df))
    if len(idx) > 20_000:
        idx = np.random.RandomState(1).choice(len(df), 20_000, replace=False)
    sub = df.iloc[idx].reset_index(drop=True)
    r = ctx["resid"][idx]
    res = fe_dsl.safe_apply(spec, sub, ctx["y"][idx], sub.iloc[:0], ctx["classes"])
    if res is None:
        return 0.0
    new = max((omega2(r, _codes(res[0][c])) for c in res[0].columns), default=0.0)
    cols = base_cols if base_cols is not None else spec["cols"]
    old = max((omega2(r, _codes(sub[c])) for c in cols), default=0.0)
    return max(0.0, new - old)


def _spec(op, cols, params=None, source="", prior=0.0, why=""):
    return {"op": op, "cols": list(cols), "params": params or {}, "source": source, "prior": float(prior), "why": why}


def _numeric(ctx: dict) -> list[str]:
    df = ctx["df"]
    return [c for c in ctx["features"] if not _is_cat(df[c]) and pd.api.types.is_numeric_dtype(df[c])]


def _top_important(ctx: dict, cols: list[str], n: int) -> list[str]:
    imp = ctx["importances"]
    return sorted(cols, key=lambda c: -imp.get(c, 0))[:n]


# ---------------------------------------------------------------------------
# 1. Сущности: квази-ID объекта.
# ---------------------------------------------------------------------------

def detect_entities(ctx: dict) -> list[dict]:
    prof, target = ctx["prof"], ctx["target"]
    temporal = {t["column"] for t in ctx["facts"].get("temporal_candidates", [])}
    n = 200_000 if ctx["ptype"] == "regression" else 50_000
    q = _quasi_id(_sample(prof["train"], n), _sample(prof["test"], n), target, ctx["features"],
                  temporal, time.time() + DETECTOR_SEC, ctx["ptype"])
    ctx["facts"]["quasi_id"] = q
    if not q or not q.get("best"):
        return []
    best = q["best"]
    alts = [r for r in [best] + list(q.get("alternatives") or [])[:2] if r.get("score", 0) > 0]
    ctx["entity"] = {"key": best["key"], "coverage": best["coverage"], "score": best["score"],
                     "candidates": [r["key"] for r in alts]}
    out = []
    for r in alts:
        out.append(_spec("group_stat", [], {"keys": r["key"], "src": "target", "aggs": list(fe_dsl.TARGET_AGGS)},
                         "entities", r["score"], f"ключ покрывает {r['coverage']:.0%} test"))
    num = [c for c in _top_important(ctx, _numeric(ctx), 6) if c not in best["key"]][:3]
    for c in num:
        out.append(_spec("group_dev", [c], {"keys": best["key"]}, "entities", best["score"] * 0.2,
                         f"{c} относительно среднего по объекту"))
    return out


# ---------------------------------------------------------------------------
# 2. Время: период + история сущности.
# ---------------------------------------------------------------------------

def _lag_prior(ctx: dict, spec: dict) -> float:
    """Покрытие историей × доля дисперсии остатков, объяснённая последним значением."""
    df = ctx["df"]
    res = fe_dsl.safe_apply(fe_dsl.validate(spec, df.columns), df, ctx["y"], df.iloc[:0], ctx["classes"])
    if res is None:
        return 0.0
    f = res[0]
    col = next((c for c in f.columns if c.endswith("__last")), f.columns[0])
    v = f[col].to_numpy(float)
    ok = np.isfinite(v)
    if ok.sum() <= 50:
        return 0.0
    return float(ok.mean()) * omega2(ctx["resid"][ok], _codes(pd.Series(v[ok])))


def detect_temporal(ctx: dict) -> list[dict]:
    """Период задаёт схему валидации; для каждого кандидата ключа объекта — история таргета.

    Ключ с колонками, меняющимися от периода к периоду (погода года), теряет
    историю — поэтому лаги строятся по всем кандидатам, prior рассудит.
    """
    cands = ctx["facts"].get("temporal_candidates") or []
    if not cands:
        return []
    period = cands[0]["column"]
    ctx["period"] = period
    if not ctx["entity"]:
        return []
    out = []
    for key in ctx["entity"]["candidates"]:
        keys = [k for k in key if k != period]
        if not keys:
            continue
        spec = _spec("lag", [], {"keys": keys, "period": period, "src": "target", "aggs": list(fe_dsl.LAG_AGGS)},
                     "temporal", 0.0, f"таргет объекта {keys} в прошлых периодах {period}")
        spec["prior"] = _lag_prior(ctx, spec)
        out.append(spec)
    if not out:
        return []
    best = max(out, key=lambda s: s["prior"])
    keys = best["params"]["keys"]
    for c in _top_important(ctx, [c for c in _numeric(ctx) if c not in keys and c != period], 2):
        out.append(_spec("lag", [c], {"keys": keys, "period": period, "aggs": ["last"]}, "temporal",
                         best["prior"] * 0.2, f"{c} объекта в прошлом периоде"))
    return out


# ---------------------------------------------------------------------------
# 3. Остатки: где модель систематически ошибается.
# ---------------------------------------------------------------------------

def detect_residuals(ctx: dict) -> list[dict]:
    df, r = ctx["df"], ctx["resid"]
    t_end = time.time() + DETECTOR_SEC
    cols = [c for c in ctx["features"] if c != ctx["period"]]
    single = {}
    codes = {}
    for c in cols:
        if time.time() > t_end:
            break
        codes[c] = _codes(df[c])
        single[c] = omega2(r, codes[c])
    top = sorted(single, key=lambda c: -single[c])[:TOP_RESID_COLS]
    out = []
    for c in top:
        if single[c] <= 0:
            continue
        params = {"keys": [c], "src": "target", "aggs": ["mean", "std", "count"]}
        if not _is_cat(df[c]) and df[c].nunique() > 50:
            params["bin"] = 20
        out.append(_spec("group_stat", [], params, "residuals", single[c], f"остаток модели зависит от {c}"))
    # Пары: прирост ω² пары над лучшей из одиночных колонок.
    pair_cols = top[:PAIR_TOP]
    for a, b in combinations(pair_cols, 2):
        if time.time() > t_end:
            break
        ca, cb = _codes(df[a], q=5, top=10), _codes(df[b], q=5, top=10)
        gain = omega2(r, ca * (cb.max() + 1) + cb) - max(single[a], single[b])
        if gain > 0:
            out.append(_spec("group_stat", [], {"keys": [a, b], "src": "target", "aggs": ["mean", "count"], "bin": 5},
                             "residuals", gain, f"взаимодействие {a}×{b} в остатках"))
            out.append(_spec("interaction", [a, b], {"bins": 5}, "residuals", gain * 0.5, f"код пары {a}×{b}"))
    if ctx["classes"]:
        out += _confused_pairs(ctx, codes, t_end)
    return out


def _confused_pairs(ctx: dict, codes: dict, t_end: float) -> list[dict]:
    """Классификация: срезы, где модель путает конкретную пару классов."""
    oof, yi, classes = ctx["oof"], ctx["y_idx"], ctx["classes"]
    k = len(classes)
    pred = np.argmax(oof, axis=1)
    cm = np.bincount(yi * k + pred, minlength=k * k).reshape(k, k).astype(float)
    sym = cm + cm.T
    np.fill_diagonal(sym, 0)
    pairs = []
    for _ in range(2):
        i, j = np.unravel_index(np.argmax(sym), sym.shape)
        if sym[i, j] <= 0:
            break
        pairs.append((i, j))
        sym[i, j] = sym[j, i] = 0
    out = []
    for i, j in pairs:
        rows = (yi == i) | (yi == j)
        if rows.sum() < 100:
            continue
        pa, pb = oof[rows, i], oof[rows, j]
        r = (yi[rows] == i).astype(float) - pa / np.maximum(pa + pb, 1e-9)
        scored = []
        for c, cd in codes.items():
            if time.time() > t_end:
                break
            scored.append((omega2(r, cd[rows]), c))
        for w, c in sorted(scored, reverse=True)[:3]:
            if w <= 0:
                continue
            params = {"keys": [c], "src": "target", "aggs": ["mean", "count"]}
            if not _is_cat(ctx["df"][c]) and ctx["df"][c].nunique() > 50:
                params["bin"] = 20
            share = rows.mean()
            out.append(_spec("group_stat", [], params, f"residuals:{classes[i]}/{classes[j]}", w * share,
                             f"{c} разделяет путаемые классы {classes[i]} и {classes[j]}"))
    return out


# ---------------------------------------------------------------------------
# 4. Преобразования: монотонные и арифметика пар топ-признаков.
# ---------------------------------------------------------------------------

def detect_transforms(ctx: dict) -> list[dict]:
    df = ctx["df"]
    num = _numeric(ctx)
    out = []
    t_end = time.time() + DETECTOR_SEC
    for c in num:
        x = df[c].astype(float)
        try:
            skew = float(x.skew())
        except (TypeError, ValueError):
            continue
        if abs(skew) > 2 and x.nunique() > 10:
            # Деревьям монотонные преобразования не нужны; польза — для линейных/NN в ансамбле.
            out.append(_spec("log", [c], {}, "transforms", 1e-4 * min(abs(skew), 10), f"тяжёлый хвост skew={skew:.1f}"))
    top = [c for c in _top_important(ctx, num, 6) if df[c].nunique() > 10]
    for a, b in combinations(top, 2):
        for op in ("ratio", "diff", "prod"):
            if time.time() > t_end:
                break
            s = _spec(op, [a, b], {}, "transforms")
            s["prior"] = _resid_prior(ctx, s)
            if s["prior"] > 0:
                out.append(s)
    return out


# ---------------------------------------------------------------------------
# 5. Однородные группы колонок (имена и единицы словаря).
# ---------------------------------------------------------------------------

def _dict_units(ctx: dict) -> dict:
    dd = ctx["prof"].get("data_dictionary")
    if dd is None:
        return {}
    low = {c.lower(): c for c in dd.columns}
    if "feature" not in low or "unit" not in low:
        return {}
    return dict(zip(dd[low["feature"]].astype(str), dd[low["unit"]].astype(str)))


def homogeneous_groups(columns: list[str], units: dict | None = None) -> list[dict]:
    """Группы колонок, различающихся одним токеном имени: january_{min,avg,max}_temp_c и т.п."""
    units = units or {}
    toks = {c: re.split(r"[_\W]+", c.lower()) for c in columns}
    groups: dict[tuple, list[tuple[str, str]]] = {}
    for c, t in toks.items():
        for i in range(len(t)):
            sig = (len(t), i, tuple(t[:i]), tuple(t[i + 1:]))
            groups.setdefault(sig, []).append((t[i], c))
    out, seen = [], set()
    for (_, i, pre, suf), members in groups.items():
        cols = [c for _, c in members]
        if len(cols) < 2 or tuple(sorted(cols)) in seen:
            continue
        if units and len({units.get(c) for c in cols}) > 1:
            continue
        seen.add(tuple(sorted(cols)))
        out.append({"cols": cols, "vary": {v: c for v, c in members}, "name": "_".join(pre + ("*",) + suf)})
    return out


def detect_homogeneous(ctx: dict) -> list[dict]:
    num = _numeric(ctx)
    out = []
    t_end = time.time() + DETECTOR_SEC
    for g in homogeneous_groups(num, _dict_units(ctx)):
        if time.time() > t_end:
            break
        vary, cols = g["vary"], g["cols"]
        if "min" in vary and "max" in vary:
            s = _spec("diff", [vary["max"], vary["min"]], {}, "homogeneous", why=f"амплитуда {g['name']}")
            s["prior"] = _resid_prior(ctx, s)
            out.append(s)
        if len(cols) >= 3:
            s = _spec("agg_cols", cols, {"aggs": ["mean", "std", "min", "max", "range"]}, "homogeneous",
                      why=f"агрегаты группы {g['name']} ({len(cols)} колонок)")
            s["prior"] = _resid_prior(ctx, s)
            out.append(s)
    return out


# ---------------------------------------------------------------------------
# 6–7. Пропуски и категориальные.
# ---------------------------------------------------------------------------

def detect_missingness(ctx: dict) -> list[dict]:
    df, r = ctx["df"], ctx["resid"]
    out = []
    for c in ctx["features"]:
        miss = df[c].isna()
        if 0.01 < miss.mean() < 0.99:
            w = omega2(r, miss.to_numpy(np.int64))
            if w > 0:
                out.append(_spec("missing_ind", [c], {}, "missingness", w, f"пропуск в {c} связан с ошибкой модели"))
    return sorted(out, key=lambda s: -s["prior"])[:5]


def detect_categoricals(ctx: dict) -> list[dict]:
    df, r = ctx["df"], ctx["resid"]
    out = []
    for c in ctx["features"]:
        nu = df[c].nunique()
        if not _is_cat(df[c]) or nu <= 30:
            continue
        w = omega2(r, _codes(df[c], top=200))
        out.append(_spec("count_enc", [c], {}, "categoricals", w * 0.5, f"частота уровня {c} ({nu} уровней)"))
        out.append(_spec("group_stat", [], {"keys": [c], "src": "target", "aggs": ["mean", "std", "count"]},
                         "categoricals", w, f"target encoding {c} ({nu} уровней)"))
    return out


# ---------------------------------------------------------------------------
# 8. Соседи и дубли.
# ---------------------------------------------------------------------------

def detect_neighbors(ctx: dict) -> list[dict]:
    prof, target = ctx["prof"], ctx["target"]
    dups = _duplicates(prof["train"], prof["test"], target, ctx["features"])
    ctx["facts"]["duplicates"] = dups
    if ctx["classes"]:
        ctx["facts"]["label_conflicts"] = label_conflicts(prof, target, ctx["features"])
    share = dups["test_in_train_share"]
    if ctx["classes"] and share > 0.02:
        ctx["label_copy"] = True
    num = _top_important(ctx, [c for c in _numeric(ctx) if c != ctx["period"]], 10)
    if len(num) < 2:
        return []
    return [_spec("knn_target", num, {"k": 10}, "neighbors", max(share, 1e-3),
                  f"таргет 10 ближайших соседей; {share:.1%} test — точные дубли train")]


# ---------------------------------------------------------------------------
# 9. Сдвиг train/test (adversarial validation).
# ---------------------------------------------------------------------------

def detect_shift(ctx: dict) -> list[dict]:
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedKFold

    import fe_search

    feats = [c for c in ctx["features"] if c != ctx["period"]]
    tr = _sample(ctx["df"][feats], 10_000, 2)
    te = _sample(ctx["test"][feats], 10_000, 2)
    X = pd.concat([tr, te], ignore_index=True)
    y = np.r_[np.zeros(len(tr)), np.ones(len(te))]
    X, _ = fe_search._to_lgb(X, X.iloc[:0])
    pred = np.zeros(len(X))
    imp = np.zeros(len(feats))
    for a, b in StratifiedKFold(3, shuffle=True, random_state=4).split(X, y):
        m = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, num_leaves=31, n_jobs=fe_search.models.N_CPUS,
                               verbosity=-1)
        m.fit(X.iloc[a], y[a])
        pred[b] = m.predict_proba(X.iloc[b])[:, 1]
        imp += m.booster_.feature_importance("gain")
    from sklearn.metrics import roc_auc_score

    auc = float(roc_auc_score(y, pred))
    share = imp / max(imp.sum(), 1e-9)
    top = sorted(zip(share, feats), reverse=True)[:5]
    ctx["facts"]["adversarial"] = {"auc": round(auc, 4), "top": [(c, round(float(s), 3)) for s, c in top]}
    log(f"AutoEDA shift: adversarial AUC={auc:.3f}, топ дрейфа {[c for _, c in top]}")
    if auc < 0.6:
        return []
    return [_spec("drop", [c], {}, "shift", (auc - 0.5) * float(s) * 0.5, f"дрейф train/test, доля важности {s:.2f}")
            for s, c in top if s > 0.05]


# ---------------------------------------------------------------------------
# 10. Сегменты: смешанные режимы поведения.
# ---------------------------------------------------------------------------

SEG_MIN_SHARE = 0.05
SEG_CORR_COLS = 8


def _segment_candidates(ctx: dict) -> list[tuple[str, dict, np.ndarray]]:
    """Естественные разделители: пропуск, ноль/не ноль, малокардинальные категории."""
    df, out = ctx["df"], []
    for c in ctx["features"]:
        if c == ctx["period"]:
            continue
        s = df[c]
        miss = float(s.isna().mean())
        if SEG_MIN_SHARE < miss < 1 - SEG_MIN_SHARE:
            out.append((c, {"rule": "nan"}, s.isna().to_numpy(np.int64)))
        nu = s.nunique()
        if _is_cat(s) or nu <= fe_dsl.SEGMENT_MAX_LEVELS:
            if 2 <= nu <= fe_dsl.SEGMENT_MAX_LEVELS + 1:
                spec = {"cols": [c], "params": {"rule": "level"}}
                codes, _ = fe_dsl.segment_codes(spec, df, df.iloc[:0])
                out.append((c, {"rule": "level"}, codes))
        elif pd.api.types.is_numeric_dtype(s):
            z = float((s == 0).mean())
            if SEG_MIN_SHARE < z < 1 - SEG_MIN_SHARE:
                out.append((c, {"rule": "zero"}, (s == 0).to_numpy(np.int64)))
    return out


def _agreement(a: np.ndarray, b: np.ndarray) -> float:
    """Доля строк, где два разбиения совпадают (с точностью до перенумерации сегментов)."""
    ct = pd.crosstab(a, b).to_numpy()
    return float(ct.max(axis=1).sum() / len(a))


def detect_segments(ctx: dict) -> list[dict]:
    """Сегмент, внутри которого связи признаков другие, чем в целом (смена знака
    корреляций), и остатки модели смещены: кандидат на отдельные модели."""
    df = ctx["df"]
    t_end = time.time() + DETECTOR_SEC
    num = [c for c in _top_important(ctx, _numeric(ctx), SEG_CORR_COLS + 4) if df[c].nunique() > 2]
    found = []
    for col, params, codes in _segment_candidates(ctx):
        if time.time() > t_end:
            break
        cols = [c for c in num if c != col][:SEG_CORR_COLS]
        if len(cols) < 2:
            continue
        R = df[cols].rank().to_numpy(float)
        R = np.where(np.isnan(R), np.nanmean(R, axis=0), R)
        with np.errstate(invalid="ignore", divide="ignore"):
            c_all = np.nan_to_num(np.corrcoef(R, rowvar=False))
        iu = np.triu_indices(len(cols), 1)
        shift, flips = 0.0, []
        shares = np.bincount(codes) / len(codes)
        if (shares[shares > 0] >= 1 - SEG_MIN_SHARE).any():
            continue
        for code in np.unique(codes):
            m = codes == code
            if m.sum() < 300:
                continue
            with np.errstate(invalid="ignore", divide="ignore"):
                c_seg = np.nan_to_num(np.corrcoef(R[m], rowvar=False))
            d = np.abs(c_seg - c_all)[iu]
            shift += m.mean() * float(d.mean())
            k = int(np.argmax(d))
            a, b = cols[iu[0][k]], cols[iu[1][k]]
            flips.append({"segment": int(code), "share": round(float(m.mean()), 3), "pair": [a, b],
                          "corr_all": round(float(c_all[iu][k]), 2), "corr_seg": round(float(c_seg[iu][k]), 2)})
        w = omega2(ctx["resid"], codes)
        balance = min(1.0, 4 * float(shares[shares > 0].min()))
        found.append({"col": col, **params, "corr_shift": round(shift, 4), "resid_omega2": round(w, 4),
                      "score": (shift + w) * balance, "_codes": codes,
                      "flip": max(flips, key=lambda f: abs(f["corr_all"] - f["corr_seg"]), default=None)})
    found.sort(key=lambda r: -r["score"])
    # Одинаковое разбиение разными колонками (dbytes==0 ⇔ dpkts==0) — один сегмент.
    uniq = []
    for r in found:
        if all(_agreement(r["_codes"], u["_codes"]) < 0.98 for u in uniq):
            uniq.append(r)
        else:
            next(u for u in uniq if _agreement(r["_codes"], u["_codes"]) >= 0.98).setdefault("same_as", []).append(r["col"])
    found = uniq
    ctx["facts"]["segments"] = [{k: v for k, v in r.items() if k not in ("score", "_codes")} for r in found[:8]]
    out = []
    for r in found[:4]:
        params = {"rule": r["rule"]}
        why = f"сегмент {r['col']}:{r['rule']}, сдвиг корреляций {r['corr_shift']}"
        if r["flip"]:
            f = r["flip"]
            why += f"; {f['pair'][0]}–{f['pair'][1]}: {f['corr_all']:+.2f} в целом, {f['corr_seg']:+.2f} в сегменте"
        out.append(_spec("segment_model", [r["col"]], params, "segments", r["score"], why))
        out.append(_spec("flag", [r["col"]], params, "segments", r["score"] * 0.3, "индикатор сегмента " + r["col"]))
    return out


# ---------------------------------------------------------------------------
# 11. Хвост ошибок: какие строки определяют метрику.
# ---------------------------------------------------------------------------

def detect_tail(ctx: dict) -> list[dict]:
    df, r = ctx["df"], ctx["resid"]
    out = []
    if ctx["classes"] is None:
        e = r ** 2
        order = np.sort(e)[::-1]
        total = max(float(order.sum()), 1e-12)
        n1, n5 = max(1, len(e) // 100), max(1, len(e) // 20)
        tail = e >= order[n5 - 1]
        facts = {"sse_share_top1pct": round(float(order[:n1].sum() / total), 3),
                 "sse_share_top5pct": round(float(order[:n5].sum() / total), 3)}
    else:
        # Классификация: доля ошибок по истинному классу и «хвост» самых уверенных ошибок.
        pred = np.argmax(ctx["oof"], axis=1)
        wrong = pred != ctx["y_idx"]
        per = pd.Series(wrong).groupby(pd.Series(ctx["y"])).mean().sort_values(ascending=False)
        facts = {"error_rate_by_class": {str(k): round(float(v), 3) for k, v in per.head(10).items()}}
        tail = wrong & (r >= np.quantile(r, 0.95))
    # Что объединяет хвост: колонки, где индикатор хвоста объяснён лучше всего.
    ind = tail.astype(float)
    scored = sorted(((omega2(ind, _codes(df[c])), c) for c in ctx["features"]), reverse=True)[:5]
    facts["tail_characterized_by"] = [{"col": c, "omega2": round(w, 4)} for w, c in scored if w > 0]
    ctx["facts"]["tail"] = facts
    log(f"AutoEDA tail: {facts}")
    if ctx["classes"] is None:
        share = facts["sse_share_top1pct"]
        alpha = float(np.quantile(np.abs(r), 0.9))
        for name in ("huber", "fair"):
            out.append(_spec("objective", [], {"name": name, "alpha": alpha}, "tail", share * 0.05,
                             f"1% строк дают {share:.0%} SSE — устойчивая функция потерь"))
    for w, c in scored[:2]:
        if w > 0 and (_is_cat(df[c]) or df[c].nunique() <= fe_dsl.SEGMENT_MAX_LEVELS + 1):
            out.append(_spec("segment_model", [c], {"rule": "level"}, "tail", w,
                             f"хвост ошибок сосредоточен по {c}"))
    return out


def label_conflicts(prof: dict, target: str, features: list[str]) -> dict:
    """Неоднозначность меток: доля строк каждого класса в группах одинаковых признаков с разными классами."""
    lab = _sample(prof["train"].loc[prof["train"][target].notna()], 300_000)
    k = row_keys(lab, features)
    y = lab[target].astype(str).to_numpy()
    df = pd.DataFrame({"k": k, "y": y})
    nlab = df.groupby("k")["y"].transform("nunique").to_numpy()
    conflict = nlab > 1
    per = pd.Series(conflict).groupby(pd.Series(y)).mean().sort_values(ascending=False)
    return {"conflict_rows": int(conflict.sum()),
            "by_class": {str(c): round(float(v), 3) for c, v in per.items() if v > 0}}


DETECTORS = [
    ("entities", detect_entities),
    ("temporal", detect_temporal),
    ("residuals", detect_residuals),
    ("categoricals", detect_categoricals),
    ("homogeneous", detect_homogeneous),
    ("neighbors", detect_neighbors),
    ("segments", detect_segments),
    ("tail", detect_tail),
    ("missingness", detect_missingness),
    ("transforms", detect_transforms),
    ("shift", detect_shift),
]


def run_all(ctx: dict, deadline: float) -> list[dict]:
    """Все детекторы по очереди (каждый в try), валидированные уникальные спеки."""
    prof = ctx["prof"]
    ctx["facts"]["temporal_candidates"] = _temporal_candidates(prof["train"], prof["test"], ctx["features"])
    columns = set(ctx["features"])
    numeric = set(_numeric(ctx))
    specs, seen = [], set()
    for name, fn in DETECTORS:
        if time.time() > deadline:
            log(f"AutoEDA: детекторы остановлены по времени перед {name}")
            break
        t0 = time.time()
        try:
            raw = fn(ctx)
        except Exception:  # noqa: BLE001
            log(f"детектор {name} упал:\n" + traceback.format_exc())
            continue
        n_new = 0
        for s in raw:
            v = fe_dsl.validate(s, columns, numeric)
            if v is None:
                continue
            k = fe_dsl.spec_key(v)
            if k in seen:
                continue
            seen.add(k)
            specs.append(v)
            n_new += 1
        log(f"детектор {name}: {n_new} спек за {time.time() - t0:.1f}s")
    return specs


def summary(ctx: dict, specs: list[dict], top: int = 25) -> dict:
    """Сводка фактов для LLM: каждая тема промпта берёт свои ключи."""
    f = ctx["facts"]
    ranked = sorted(specs, key=lambda s: -s["prior"])

    def brief(rows):
        return [{"spec": fe_dsl.describe(s), "source": s["source"], "prior": round(s["prior"], 4)} for s in rows]

    return {
        "problem_type": ctx["ptype"], "metric": ctx["metric"], "scheme": ctx.get("fold_scheme"),
        "period": ctx["period"], "entity": ctx["entity"],
        "duplicates": f.get("duplicates"), "adversarial": f.get("adversarial"),
        "segments": f.get("segments"), "tail": f.get("tail"), "label_conflicts": f.get("label_conflicts"),
        "perm_importance": f.get("perm_importance"),
        "top_importances": {k: round(v, 4) for k, v in sorted(ctx["importances"].items(), key=lambda kv: -kv[1])[:15]},
        "residual_specs": brief([s for s in ranked if s["source"].startswith("residuals")][:10]),
        "detector_specs": brief(ranked[:top]),
    }
