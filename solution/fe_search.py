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
        n_estimators=300 if fast else 800, learning_rate=0.1 if fast else 0.06, num_leaves=63,
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


def fit_eval(ctx: dict, Xtr, ytr, Xva, yva, fast: bool = False):
    """LightGBM с фиксированными параметрами: (score, pred|proba, importances)."""
    import lightgbm as lgb

    params = _lgb_params(ctx, fast)
    cb = [lgb.early_stopping(40 if fast else 60, verbose=False), lgb.log_evaluation(0)]
    metric = ctx["metric"]
    if ctx["ptype"] == "regression":
        m = lgb.LGBMRegressor(objective="regression", **params)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="l1" if metric == "mae" else "rmse", callbacks=cb)
        pred = m.predict(Xva)
        return models.score(metric, yva, pred), pred, m.feature_importances_
    m = lgb.LGBMClassifier(**params)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=cb)
    classes = ctx["classes"]
    proba = np.zeros((len(Xva), len(classes)))
    pos = {c: i for i, c in enumerate(classes)}
    proba[:, [pos[c] for c in m.classes_]] = m.predict_proba(Xva)
    pred = np.array(classes, dtype=object)[np.argmax(proba, axis=1)]
    try:
        s = models.score(metric, yva, pred, proba=proba, labels=classes)
    except ValueError:
        s = models.score("accuracy", yva, pred)
    return s, proba, m.feature_importances_


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
    folds, scheme = [], None
    if ctx["period"]:
        t = pd.to_numeric(sub[ctx["period"]], errors="coerce").to_numpy(float)
        periods = np.unique(t[~np.isnan(t)])
        for last in (periods[::-1][:2] if len(periods) >= 4 else periods[::-1][:1]):
            tr_i, va_i = rows[t < last], rows[t == last]
            if len(tr_i) >= 500 and len(va_i) >= 200:
                folds.append((tr_i, va_i))
        if folds:
            scheme = f"temporal({ctx['period']}, {len(folds)} фолд.)"
    if not folds and ctx["entity"] and ctx["entity"].get("coverage", 1) < ENTITY_GROUP_MAX_COVERAGE:
        groups = row_keys(sub, ctx["entity"]["key"])
        if pd.Series(groups).nunique() >= 3:
            folds = [(rows[a], rows[b]) for a, b in GroupKFold(3).split(sub, y[rows], groups)]
            scheme = f"group({ctx['entity']['key']})"
    if not folds:
        ys = y[rows]
        if ctx["classes"] and pd.Series(ys).value_counts().min() >= 3:
            split = StratifiedKFold(3, shuffle=True, random_state=13).split(sub, ys)
            scheme = "stratified_kfold(3)"
        else:
            split = KFold(3, shuffle=True, random_state=13).split(sub)
            scheme = "kfold(3)"
        folds = [(rows[a], rows[b]) for a, b in split]
    ctx["folds"], ctx["fold_scheme"] = folds, scheme
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
                res = fe_dsl.safe_apply(spec, df.loc[tr], ctx["y"][tr], df.loc[va], ctx["classes"])
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

    def run(self, specs: list[dict], extra: list | None = None) -> list[float] | None:
        """Скоры по фолдам для базы + specs (drop убирает колонку) + extra (готовые признаки по фолдам)."""
        ctx = self.ctx
        t0 = time.time()
        drops = {s["cols"][0] for s in specs if s["op"] == "drop"}
        feats_by_spec = []
        for s in specs:
            if s["op"] == "drop":
                continue
            f = self._spec_feats(s)
            if f is None:
                return None
            feats_by_spec.append(f)
        if extra is not None:
            feats_by_spec.append(extra)
        scores = []
        for i, (tr, va) in enumerate(ctx["folds"]):
            Xtr, Xva = self.base[i]
            if drops:
                Xtr, Xva = Xtr.drop(columns=list(drops & set(Xtr.columns))), Xva.drop(columns=list(drops & set(Xva.columns)))
            extra_tr = [f[i][0] for f in feats_by_spec]
            extra_va = [f[i][1] for f in feats_by_spec]
            Xtr = pd.concat([Xtr, *extra_tr], axis=1)
            Xva = pd.concat([Xva, *extra_va], axis=1)
            Xtr = Xtr.loc[:, ~Xtr.columns.duplicated()]
            Xva = Xva.loc[:, ~Xva.columns.duplicated()]
            Xtr, Xva = _to_lgb(Xtr, Xva)
            s, _, _ = fit_eval(ctx, Xtr, ctx["y"][tr], Xva, ctx["y"][va])
            scores.append(s)
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


def search(ctx: dict, specs: list[dict], deadline: float, register=None) -> list[dict]:
    """Этап A + этап B. Возвращает принятые спеки в порядке добавления."""
    metric = ctx["metric"]
    ev = Evaluator(ctx)
    t0 = time.time()
    base = ev.run([])
    log(f"AutoEDA: база по фолдам {metric}={np.round(base, 5).tolist()} ({time.time() - t0:.1f}s)")
    # Калибровка порога: «прирост» от признаков-пустышек — уровень шума валидатора.
    noise = []
    top = sorted(ctx["features"], key=lambda c: -ctx["importances"].get(c, 0))
    for i, col in enumerate(top[:N_NULL]):
        sc = ev.run([], extra=ev.null_feats(col, 100 + i))
        noise.append(abs(_gain(metric, sc, base)[0]))
    rel_min = max(REL_MIN, max(noise, default=0.0))
    ctx["rel_min"] = rel_min
    log(f"AutoEDA: шум null-спек {np.round(noise, 5).tolist()} -> порог прироста {rel_min:.4f}")
    budget_a = (deadline - time.time()) * 0.7
    deadline_a = time.time() + budget_a

    passed = []
    for spec in sorted(specs, key=lambda s: -s["prior"]):
        if not ev.fits(deadline_a):
            log(f"AutoEDA: этап A остановлен по времени, оценено {ev.n_evals - 1 - N_NULL}/{len(specs)}")
            break
        scores = ev.run([spec])
        if scores is None:
            continue
        mean, lcb = _gain(metric, scores, base)
        leaky = _looks_leaky(metric, scores, base)
        ok = mean > rel_min and lcb > 0 and not leaky
        spec["delta"], spec["lcb"] = round(mean, 5), round(lcb, 5)
        if register:
            register(f"fe_A:{spec['source']}", f"{fe_dsl.describe(spec)} prior={spec['prior']:.4f}"
                     + (" [утечка?]" if leaky else ""), round(float(np.mean(scores)), 5), ok)
        if ok:
            passed.append(spec)

    # Этап B: жадный forward-отбор; база сдвигается после каждого принятого.
    accepted: list[dict] = []
    cur = base
    for spec in sorted(passed, key=lambda s: -s["delta"]):
        if not ev.fits(deadline):
            log("AutoEDA: этап B остановлен по времени")
            break
        scores = ev.run(accepted + [spec])
        if scores is None:
            continue
        mean, lcb = _gain(metric, scores, cur)
        ok = mean > rel_min / 2 and lcb > -rel_min
        if register:
            register(f"fe_B:{spec['source']}", fe_dsl.describe(spec), round(float(np.mean(scores)), 5), ok)
        if ok:
            accepted.append(spec)
            cur = scores
    total, _ = _gain(metric, cur, base)
    log(f"AutoEDA: принято {len(accepted)} спек, прирост {total:+.2%} "
        f"({metric} {np.mean(base):.5f} -> {np.mean(cur):.5f}), оценок {ev.n_evals}, {time.time() - t0:.0f}s")
    ctx["search_base"], ctx["search_final"] = float(np.mean(base)), float(np.mean(cur))
    return accepted
