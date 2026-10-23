"""Валидатор гипотез FE: фолды под тест, этап A (каждая спека против базы),
этап B (жадный forward-отбор прошедших).

Признаки спеки на каждом фолде строятся заново по схеме fit = train фолда,
apply = val фолда (fe_dsl.apply), поэтому target-признаки на валидации ведут
себя как на test и не дают ложного прироста.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

import fe_dsl
import models
from agent_utils import log
from eda_probe import row_keys

SAMPLE_ROWS = 50_000
SEARCH_ROW_CLASSES = 200_000  # классификация: строк валидатора × классов (цена мультикласса ~K)
SEARCH_MIN_ROWS = 15_000
REL_MIN = 0.002            # минимальный относительный прирост метрики
LCB_K = 1.0                # нижняя граница: mean − k·std/√n по фолдам > 0
N_NULL = 3                 # null-спеки (перемешанные копии признаков) для калибровки порога
ENTITY_GROUP_MAX_COVERAGE = 0.3  # сущности test в основном новые -> GroupKFold


# ---------------------------------------------------------------------------
# Контекст.
# ---------------------------------------------------------------------------

def build_context(prof: dict, target: str, deadline: float) -> dict:
    """Выборка размеченного train, классы, базовый OOF (для остатков) и важности."""
    train = prof["train"]
    lab = train.loc[train[target].notna()]
    if len(lab) > SAMPLE_ROWS:
        lab = lab.sample(SAMPLE_ROWS, random_state=0)
    ptype = prof["problem_type"]
    classes = None
    if ptype != "regression":
        classes = sorted(lab[target].astype(str).unique().tolist())
    features = list(prof.get("base_features") or prof["features"])
    ctx = {
        "prof": prof, "target": target, "ptype": ptype, "metric": prof["metric"],
        "df": lab.reset_index(drop=True), "test": prof["test"], "features": features,
        "classes": classes, "deadline": deadline, "period": None, "entity": None,
        "label_copy": False, "facts": {},
    }
    ctx["y"] = ctx["df"][target].astype(str).to_numpy() if classes else ctx["df"][target].to_numpy(float)
    _base_oof(ctx)
    return ctx


def _lgb_params(ctx: dict, fast: bool = False) -> dict:
    # Валидатор детерминирован (без бэггинга и сэмплинга колонок): иначе новая
    # колонка меняет случайные подвыборки, и шум сравнимых моделей растёт.
    p = dict(
        n_estimators=300 if fast else 500, learning_rate=0.1, num_leaves=63,
        colsample_bytree=0.8 if fast else 1.0, subsample=0.8 if fast else 1.0, subsample_freq=1,
        min_child_samples=20, n_jobs=models.N_CPUS, random_state=3, verbosity=-1,
    )
    if ctx["ptype"] != "regression" and ctx["metric"] in ("f1_macro", "balanced_accuracy", "f1"):
        p["class_weight"] = "balanced"
    return p


def _to_lgb(tr: pd.DataFrame, va: pd.DataFrame):
    """Категориальные колонки -> общий category (как models._prep_matrices)."""
    tr, va = tr.copy(), va.copy()
    for c in tr.columns:
        if tr[c].dtype == object or str(tr[c].dtype) == "category" or tr[c].dtype == bool:
            cats = pd.api.types.union_categoricals(
                [tr[c].astype(str).astype("category"), va[c].astype(str).astype("category")]
            ).categories
            tr[c] = pd.Categorical(tr[c].astype(str), categories=cats)
            va[c] = pd.Categorical(va[c].astype(str), categories=cats)
    return tr, va


SEG_MIN_ROWS = 300         # сегмент меньше — предсказывает общая модель


def fit_predict(ctx: dict, Xtr, ytr, Xva, yva, fast: bool = False, objective: dict | None = None):
    """LightGBM с фиксированными параметрами: (pred | proba в порядке ctx["classes"], модель)."""
    import lightgbm as lgb

    params = _lgb_params(ctx, fast)
    cb = [lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0), models.time_guard(ctx["deadline"])]
    if ctx["ptype"] == "regression":
        obj = {"objective": "regression"}
        if objective:
            obj = {"objective": objective["name"]}
            if "alpha" in objective:
                obj["alpha" if objective["name"] == "huber" else "fair_c"] = objective["alpha"]
        m = lgb.LGBMRegressor(**obj, **params)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="l1" if ctx["metric"] == "mae" else "rmse", callbacks=cb)
        return m.predict(Xva), m
    classes = ctx["classes"]
    proba = np.zeros((len(Xva), len(classes)))
    pos = {c: i for i, c in enumerate(classes)}
    present = pd.unique(np.asarray(ytr))
    if len(present) == 1:
        proba[:, pos[present[0]]] = 1.0
        return proba, None
    m = lgb.LGBMClassifier(**params)
    # Ранняя остановка — по строкам val с классами, известными модели (сегмент может не видеть редкий класс).
    seen = np.isin(np.asarray(yva), present)
    if seen.sum() == 0:
        m.set_params(n_estimators=100)
        m.fit(Xtr, ytr, callbacks=cb[2:])
    else:
        m.fit(Xtr, ytr, eval_set=[(Xva[seen], np.asarray(yva)[seen])], callbacks=cb)
    proba[:, [pos[c] for c in m.classes_]] = m.predict_proba(Xva)
    return proba, m


def score_pred(ctx: dict, y, pred) -> float:
    metric = ctx["metric"]
    if ctx["ptype"] == "regression":
        return models.score(metric, y, pred)
    classes = ctx["classes"]
    lab = np.array(classes, dtype=object)[np.argmax(pred, axis=1)]
    try:
        return models.score(metric, y, lab, proba=pred, labels=classes)
    except ValueError as exc:
        # Целевая метрика не подменяется другой: фолд остаётся неоценённым.
        log(f"{metric} не посчитан на фолде ({exc}) — оценка NaN")
        return float("nan")


def fit_eval(ctx: dict, Xtr, ytr, Xva, yva, fast: bool = False):
    """(score, pred|proba, importances) одной модели."""
    pred, m = fit_predict(ctx, Xtr, ytr, Xva, yva, fast)
    imp = m.feature_importances_ if m is not None else np.zeros(Xtr.shape[1])
    return score_pred(ctx, yva, pred), pred, imp


def fit_predict_segmented(ctx: dict, Xtr, ytr, Xva, yva, ctr, cva, objective: dict | None = None):
    """Отдельная модель на каждый сегмент; мелкие сегменты — общей моделью."""
    ytr, yva = np.asarray(ytr), np.asarray(yva)
    out = np.zeros((len(Xva), len(ctx["classes"]))) if ctx["classes"] else np.zeros(len(Xva))
    glob = None
    for code in np.unique(cva):
        va = cva == code
        tr = ctr == code
        if tr.sum() >= SEG_MIN_ROWS:
            pred, _ = fit_predict(ctx, Xtr[tr], ytr[tr], Xva[va], yva[va], objective=objective)
        else:
            if glob is None:
                glob, _ = fit_predict(ctx, Xtr, ytr, Xva, yva, objective=objective)
            pred = glob[va]
        out[va] = pred
    return out


def _base_oof(ctx: dict, n_splits: int = 3) -> None:
    """Случайный k-fold OOF quick-модели на выборке: остатки и важности для детекторов."""
    from sklearn.model_selection import KFold, StratifiedKFold

    df, y, feats = ctx["df"], ctx["y"], ctx["features"]
    if ctx["classes"]:
        vc = pd.Series(y).value_counts()
        splitter = StratifiedKFold(n_splits, shuffle=True, random_state=5) if vc.min() >= n_splits else \
            KFold(n_splits, shuffle=True, random_state=5)
    else:
        splitter = KFold(n_splits, shuffle=True, random_state=5)
    k = len(ctx["classes"]) if ctx["classes"] else 1
    oof = np.zeros((len(df), k)) if ctx["classes"] else np.zeros(len(df))
    imp = np.zeros(len(feats))
    for tr_i, va_i in splitter.split(df, y):
        Xtr, Xva = _to_lgb(df.loc[tr_i, feats], df.loc[va_i, feats])
        _, pred, fi = fit_eval(ctx, Xtr, y[tr_i], Xva, y[va_i], fast=True)
        oof[va_i] = pred
        imp += fi
    ctx["oof"] = oof
    if ctx["classes"]:
        pos = {c: i for i, c in enumerate(ctx["classes"])}
        yi = np.array([pos[v] for v in y])
        ctx["y_idx"] = yi
        ctx["resid"] = 1.0 - oof[np.arange(len(y)), yi]
        pred = np.array(ctx["classes"], dtype=object)[np.argmax(oof, axis=1)]
        ctx["oof_score"] = models.score(ctx["metric"], y, pred, proba=oof, labels=ctx["classes"])
    else:
        ctx["resid"] = y - oof
        ctx["oof_score"] = models.score(ctx["metric"], y, oof)
    ctx["importances"] = dict(zip(feats, (imp / max(imp.sum(), 1)).tolist()))
    log(f"AutoEDA: базовый OOF на {len(df)} строках {ctx['metric']}={ctx['oof_score']:.5f}")


# ---------------------------------------------------------------------------
# Фолды под тест.
# ---------------------------------------------------------------------------

def build_folds(ctx: dict) -> None:
    """Период -> train на прошлом, val на последнем (и предпоследнем при ≥4 периодах);
    сущность с новыми объектами в test -> GroupKFold; иначе (Stratified)KFold."""
    from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold

    df, y = ctx["df"], ctx["y"]
    rows = np.arange(len(df))
    if ctx["classes"]:
        n = max(SEARCH_MIN_ROWS, SEARCH_ROW_CLASSES // len(ctx["classes"]))
        if n < len(df):
            # Стратифицированная подвыборка: редкие классы сохраняются полностью до n/K.
            per = pd.Series(np.arange(len(df))).groupby(pd.Series(y)).apply(
                lambda g: g.sample(min(len(g), max(n // len(ctx["classes"]), int(n * len(g) / len(df)))),
                                   random_state=17))
            rows = np.sort(per.to_numpy())
    sub = df.iloc[rows]
    folds, scheme, cv = [], None, {"kind": "kfold"}
    if ctx["period"]:
        t = pd.to_numeric(sub[ctx["period"]], errors="coerce").to_numpy(float)
        periods = np.unique(t[~np.isnan(t)])
        for last in (periods[::-1][:2] if len(periods) >= 4 else periods[::-1][:1]):
            tr_i, va_i = rows[t < last], rows[t == last]
            if len(tr_i) >= 500 and len(va_i) >= 200:
                folds.append((tr_i, va_i))
        if folds:
            scheme = f"temporal({ctx['period']}, {len(folds)} фолд.)"
            cv = {"kind": "temporal", "period": ctx["period"]}
    if not folds and ctx["entity"] and ctx["entity"].get("coverage", 1) < ENTITY_GROUP_MAX_COVERAGE:
        groups = row_keys(sub, ctx["entity"]["key"])
        if pd.Series(groups).nunique() >= 3:
            folds = [(rows[a], rows[b]) for a, b in GroupKFold(3).split(sub, y[rows], groups)]
            scheme = f"group({ctx['entity']['key']})"
            cv = {"kind": "group", "keys": list(ctx["entity"]["key"])}
    if not folds:
        ys = y[rows]
        if ctx["classes"] and pd.Series(ys).value_counts().min() >= 3:
            split = StratifiedKFold(3, shuffle=True, random_state=13).split(sub, ys)
            scheme = "stratified_kfold(3)"
        else:
            split = KFold(3, shuffle=True, random_state=13).split(sub)
            scheme = "kfold(3)"
        folds = [(rows[a], rows[b]) for a, b in split]
    ctx["folds"], ctx["fold_scheme"], ctx["cv"] = folds, scheme, cv
    log(f"AutoEDA: схема валидации {scheme} на {len(rows)} строках")


# ---------------------------------------------------------------------------
# Оценка набора спек.
# ---------------------------------------------------------------------------

class Evaluator:
    """Кэширует базовые матрицы фолдов и признаки каждой спеки на каждом фолде."""

    def __init__(self, ctx: dict):
        self.ctx = ctx
        self.cache: dict[str, list] = {}
        df, feats = ctx["df"], ctx["features"]
        self.base = [(df.loc[tr, feats].reset_index(drop=True), df.loc[va, feats].reset_index(drop=True))
                     for tr, va in ctx["folds"]]
        self.n_evals = 0
        self.spent = 0.0

    def fits(self, deadline: float) -> bool:
        """Влезет ли ещё одна оценка (по средней длительности) до дедлайна."""
        avg = self.spent / max(self.n_evals, 1)
        return time.time() + avg < deadline

    def _spec_feats(self, spec: dict):
        key = fe_dsl.spec_key(spec)
        if key not in self.cache:
            ctx, df = self.ctx, self.ctx["df"]
            out = []
            for tr, va in ctx["folds"]:
                res = fe_dsl.safe_apply(spec, df.loc[tr], ctx["y"][tr], df.loc[va], ctx["classes"], ctx.get("cv"))
                if res is None:
                    out = None
                    break
                out.append(res)
            self.cache[key] = out
        return self.cache[key]

    def null_feats(self, col: str, seed: int) -> list:
        """Перемешанная копия колонки на каждом фолде: признак без информации."""
        rng = np.random.RandomState(seed)
        out = []
        for Xtr, Xva in self.base:
            out.append((pd.DataFrame({f"null__{col}": rng.permutation(Xtr[col].to_numpy())}),
                        pd.DataFrame({f"null__{col}": rng.permutation(Xva[col].to_numpy())})))
        return out

    def matrices(self, i: int, specs: list[dict], extra: list | None = None):
        """Матрицы фолда i: база − drop + признаки спек + extra. None — спека не исполнилась."""
        drops = {s["cols"][0] for s in specs if s["op"] == "drop"}
        Xtr, Xva = self.base[i]
        if drops:
            Xtr, Xva = Xtr.drop(columns=list(drops & set(Xtr.columns))), Xva.drop(columns=list(drops & set(Xva.columns)))
        parts_tr, parts_va = [Xtr], [Xva]
        for s in specs:
            if s["op"] == "drop" or s["op"] in fe_dsl.MODEL_OPS:
                continue
            f = self._spec_feats(s)
            if f is None:
                return None
            parts_tr.append(f[i][0])
            parts_va.append(f[i][1])
        if extra is not None:
            parts_tr.append(extra[i][0])
            parts_va.append(extra[i][1])
        Xtr, Xva = pd.concat(parts_tr, axis=1), pd.concat(parts_va, axis=1)
        Xtr = Xtr.loc[:, ~Xtr.columns.duplicated()]
        Xva = Xva.loc[:, ~Xva.columns.duplicated()]
        return _to_lgb(Xtr, Xva)

    def run(self, specs: list[dict], extra: list | None = None) -> list[float] | None:
        """Скоры по фолдам для базы + specs: признаки, drop, segment_model, objective."""
        ctx = self.ctx
        t0 = time.time()
        seg = [s for s in specs if s["op"] == "segment_model"]
        obj = [s for s in specs if s["op"] == "objective"]
        if len(seg) > 1 or len(obj) > 1 or (obj and ctx["ptype"] != "regression"):
            return None
        objective = obj[0]["params"] if obj else None
        df = ctx["df"]
        scores = []
        for i, (tr, va) in enumerate(ctx["folds"]):
            mats = self.matrices(i, specs, extra)
            if mats is None:
                return None
            Xtr, Xva = mats
            ytr, yva = ctx["y"][tr], ctx["y"][va]
            if seg:
                ctr, cva = fe_dsl.segment_codes(seg[0], df.loc[tr], df.loc[va])
                pred = fit_predict_segmented(ctx, Xtr, ytr, Xva, yva, ctr, cva, objective)
            else:
                pred, _ = fit_predict(ctx, Xtr, ytr, Xva, yva, objective=objective)
            scores.append(score_pred(ctx, yva, pred))
        self.n_evals += 1
        self.spent += time.time() - t0
        return scores


def _gain(metric: str, new: list[float], old: list[float]) -> tuple[float, float]:
    """Относительный прирост по фолдам (положительный = лучше): (mean, lcb)."""
    sign = 1.0 if models.HIGHER_BETTER[metric] else -1.0
    rel = np.array([sign * (n - o) / max(abs(o), 1e-9) for n, o in zip(new, old)])
    mean = float(rel.mean())
    lcb = mean - LCB_K * float(rel.std(ddof=1)) / np.sqrt(len(rel)) if len(rel) > 1 else mean
    return mean, lcb


def _looks_leaky(metric: str, new: list[float], base: list[float]) -> bool:
    """Подозрительно идеальная метрика при скромной базе — утечка, а не признак."""
    n, b = float(np.mean(new)), float(np.mean(base))
    if models.HIGHER_BETTER[metric]:
        return n >= 0.995 and b < 0.95
    return n <= 0.05 * b


def interleave(specs: list[dict]) -> list[dict]:
    """Очередь этапа A: по кругу между источниками, внутри источника — по prior.

    prior разных детекторов несравнимы по шкале; без чередования бюджет
    съедает один источник с крупными prior, а сегменты и гипотезы LLM не
    доходят до проверки.
    """
    groups: dict[str, list[dict]] = {}
    for s in sorted(specs, key=lambda s: -s["prior"]):
        groups.setdefault(s["source"].split(":")[0] if not s["source"].startswith("qwen") else s["source"], []).append(s)
    order = sorted(groups, key=lambda g: -groups[g][0]["prior"])
    out = []
    while any(groups.values()):
        for g in order:
            if groups[g]:
                out.append(groups[g].pop(0))
    return out


def perm_importance(ctx: dict, ev: Evaluator, top: int = 15) -> list[dict]:
    """Перестановочная важность на первом фолде против простой корреляции с таргетом.

    Корреляция ≠ полезность: признак со слабой связью может держать модель
    через взаимодействия — такие факты идут в промпт Qwen.
    """
    tr, va = ctx["folds"][0]
    Xtr, Xva = ev.matrices(0, [])
    ytr, yva = ctx["y"][tr], ctx["y"][va]
    _, model = fit_predict(ctx, Xtr, ytr, Xva, yva)
    if model is None:
        return []

    def predict(X):
        return model.predict(X) if ctx["ptype"] == "regression" else _align(ctx, model, model.predict_proba(X))

    base = score_pred(ctx, yva, predict(Xva))
    rng = np.random.RandomState(21)
    sign = 1.0 if models.HIGHER_BETTER[ctx["metric"]] else -1.0
    out = []
    cols = sorted(ctx["features"], key=lambda c: -ctx["importances"].get(c, 0))[:top]
    for c in cols:
        if c not in Xva.columns:
            continue
        Xp = Xva.copy()
        # iloc + set_axis сохраняет dtype (category), иначе LightGBM отвергает матрицу.
        Xp[c] = Xp[c].iloc[rng.permutation(len(Xp))].set_axis(Xp.index)
        drop = sign * (base - score_pred(ctx, yva, predict(Xp))) / max(abs(base), 1e-9)
        out.append({"col": c, "perm_drop": round(float(drop), 4), "corr": round(_target_corr(ctx, c), 3)})
    return sorted(out, key=lambda r: -r["perm_drop"])


def _align(ctx: dict, model, proba: np.ndarray) -> np.ndarray:
    out = np.zeros((len(proba), len(ctx["classes"])))
    pos = {c: i for i, c in enumerate(ctx["classes"])}
    out[:, [pos[c] for c in model.classes_]] = proba
    return out


def _target_corr(ctx: dict, col: str) -> float:
    """|Spearman| признака с таргетом (классификация — максимум по индикаторам классов)."""
    x = ctx["df"][col]
    if not pd.api.types.is_numeric_dtype(x) or x.dtype == bool:
        x = pd.Series(pd.factorize(x)[0], index=x.index).where(x.notna())
    xr = x.rank().to_numpy(float)
    ok = np.isfinite(xr)
    if ok.sum() < 50 or np.nanstd(xr) == 0:
        return 0.0
    if ctx["classes"] is None:
        yr = pd.Series(ctx["y"]).rank().to_numpy(float)
        return abs(float(np.corrcoef(xr[ok], yr[ok])[0, 1]))
    best = 0.0
    for c in ctx["classes"]:
        ind = (ctx["y"] == c).astype(float)[ok]
        if ind.std() > 0:
            best = max(best, abs(float(np.corrcoef(xr[ok], ind)[0, 1])))
    return best


class Search:
    """Валидатор на нескольких раундах гипотез: калибровка, этап A на раунд, общий этап B."""

    def __init__(self, ctx: dict, register=None):
        self.ctx, self.register = ctx, register
        self.metric = ctx["metric"]
        self.ev = Evaluator(ctx)
        self.passed: list[dict] = []
        self.results: list[dict] = []
        self.t0 = time.time()

    def calibrate(self) -> None:
        """База по фолдам и порог прироста по null-спекам (уровень шума валидатора)."""
        ev, metric = self.ev, self.metric
        self.base = ev.run([])
        log(f"AutoEDA: база по фолдам {metric}={np.round(self.base, 5).tolist()} ({time.time() - self.t0:.1f}s)")
        noise = []
        top = sorted(self.ctx["features"], key=lambda c: -self.ctx["importances"].get(c, 0))
        for i, col in enumerate(top[:N_NULL]):
            sc = ev.run([], extra=ev.null_feats(col, 100 + i))
            noise.append(abs(_gain(metric, sc, self.base)[0]))
        self.rel_min = max(REL_MIN, max(noise, default=0.0))
        self.ctx["rel_min"] = self.rel_min
        log(f"AutoEDA: шум null-спек {np.round(noise, 5).tolist()} -> порог прироста {self.rel_min:.4f}")

    def stage_a(self, specs: list[dict], deadline: float) -> list[dict]:
        """Каждая спека против базы; прошедшие копятся в self.passed. Возвращает оценённые."""
        metric, rel_min, done = self.metric, self.rel_min, []
        for spec in interleave(specs):
            if not self.ev.fits(deadline):
                log(f"AutoEDA: этап A остановлен по времени, оценено {len(done)}/{len(specs)}")
                break
            try:
                scores = self.ev.run([spec])
            except Exception as exc:  # noqa: BLE001 — одна кривая спека не роняет поиск
                log(f"AutoEDA: спека {fe_dsl.describe(spec)} упала при оценке: {type(exc).__name__}: {exc}")
                continue
            if scores is None:
                continue
            mean, lcb = _gain(metric, scores, self.base)
            leaky = _looks_leaky(metric, scores, self.base)
            ok = mean > rel_min and lcb > 0 and not leaky
            spec["delta"], spec["lcb"], spec["accepted_a"] = round(mean, 5), round(lcb, 5), ok
            done.append(spec)
            if self.register:
                self.register(f"fe_A:{spec['source']}", f"{fe_dsl.describe(spec)} prior={spec['prior']:.4f}"
                              + (" [утечка?]" if leaky else ""), round(float(np.mean(scores)), 5), ok)
            if ok:
                self.passed.append(spec)
        self.results += done
        return done

    def stage_b(self, deadline: float) -> list[dict]:
        """Жадный forward-отбор прошедших; база сдвигается после каждого принятого."""
        metric, rel_min = self.metric, self.rel_min
        accepted: list[dict] = []
        cur = self.base
        for spec in sorted(self.passed, key=lambda s: -s["delta"]):
            if not self.ev.fits(deadline):
                log("AutoEDA: этап B остановлен по времени")
                break
            if spec["op"] in fe_dsl.MODEL_OPS and any(a["op"] == spec["op"] for a in accepted):
                continue
            try:
                scores = self.ev.run(accepted + [spec])
            except Exception as exc:  # noqa: BLE001
                log(f"AutoEDA: спека {fe_dsl.describe(spec)} упала на этапе B: {type(exc).__name__}: {exc}")
                continue
            if scores is None:
                continue
            mean, lcb = _gain(metric, scores, cur)
            ok = mean > rel_min / 2 and lcb > -rel_min
            spec["accepted_b"] = ok
            if self.register:
                self.register(f"fe_B:{spec['source']}", fe_dsl.describe(spec), round(float(np.mean(scores)), 5), ok)
            if ok:
                accepted.append(spec)
                cur = scores
        total, _ = _gain(metric, cur, self.base)
        log(f"AutoEDA: принято {len(accepted)} спек, прирост {total:+.2%} "
            f"({metric} {np.mean(self.base):.5f} -> {np.mean(cur):.5f}), оценок {self.ev.n_evals}, "
            f"{time.time() - self.t0:.0f}s")
        self.ctx["search_base"], self.ctx["search_final"] = float(np.mean(self.base)), float(np.mean(cur))
        return accepted

    def feedback(self, top: int = 30) -> dict:
        """Итоги этапа A для второго раунда LLM: что прошло, что нет, с дельтами."""
        rows = sorted(self.results, key=lambda s: -s.get("delta", -1))
        return {
            "metric": self.metric, "base": round(float(np.mean(self.base)), 5),
            "threshold_rel": round(self.rel_min, 4), "scheme": self.ctx.get("fold_scheme"),
            "tested": [{"spec": fe_dsl.describe(s), "source": s["source"], "delta_rel": s.get("delta"),
                        "passed": s.get("accepted_a")} for s in rows[:top]],
        }


def search(ctx: dict, specs: list[dict], deadline: float, register=None) -> list[dict]:
    """Один раунд: калибровка, этап A (70% времени), этап B."""
    s = Search(ctx, register)
    s.calibrate()
    s.stage_a(specs, time.time() + (deadline - time.time()) * 0.7)
    return s.stage_b(deadline)
