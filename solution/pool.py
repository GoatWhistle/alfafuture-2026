"""Пул кандидатов на внешних фолдах схемы cv и жадный подбор ансамбля.

Каждый кандидат — рецепт {family, params, tf, seed}: обучается на тех же
фолдах (models.outer_folds), что k-fold LightGBM и AutoGluon, с ранней
остановкой на отдельной части train фолда (models.es_split) и target-признаками,
построенными внутри фолда. Test предсказывает модель, переобученная на всём
размеченном train с числом итераций из фолдов (frozen refit). Результат —
кандидат того же формата, что quick_oof: {"rows", "y", "oof", "test", "classes",
"score"}, поэтому выбор, ансамбль и постпроцессинг общие.

Семейства: LightGBM (варианты функции потерь, преобразования таргета,
регуляризации и seed), двухчастная модель для регрессии с нулями, CatBoost,
XGBoost, ExtraTrees. Применимость рецепта определяется профилем задачи
(plan), стоимость — по факту k-fold LightGBM; рецепт, который не помещается
в бюджет, пропускается с причиной в журнале.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

import models
from agent_utils import deadline, log
from models import N_CPUS, es_split, is_better, labeled, outer_folds, score, with_target_features

BALANCED_METRICS = ("f1_macro", "balanced_accuracy", "f1")
PROBA_METRICS = ("log_loss", "roc_auc")
MAX_CACHE_CELLS = 40_000_000   # больше — матрицы фолдов строятся заново для каждого кандидата
ZERO_SHARE_MIN = 0.1           # регрессия с нулями: доля нулей ...
POSITIVE_MIN = 500             # ... и достаточно положительных строк
SKEW_MIN = 1.0                 # преобразование таргета — при заметной правой асимметрии
EXP_CAP = 50.0                 # защита expm1 от переполнения


# ---------------------------------------------------------------------------
# Данные фолдов.
# ---------------------------------------------------------------------------

class FoldData:
    """Матрицы внешних фолдов с target-признаками (общие для всех кандидатов таргета)."""

    def __init__(self, prof: dict, target: str):
        self.prof, self.target = prof, target
        X, y, Xte = models._prep_matrices(prof, target)
        self.X, self.Xte = X, Xte
        self.lab = labeled(prof, target)
        self.y_orig = y.to_numpy()
        self.regression = prof["problem_type"] == "regression"
        self.classes = None if self.regression else sorted(pd.unique(self.y_orig).tolist(), key=str)
        self.folds = outer_folds(prof, self.lab, self.y_orig)
        self.cache_ok = X.size <= MAX_CACHE_CELLS
        self._cache: dict = {}

    def fold(self, i: int):
        """(fit_i, es_i, va, Xf, Xes, Xva) фолда i; target-признаки — по train фолда."""
        if i in self._cache:
            return self._cache[i]
        tr, va = self.folds[i]
        fit_i, es_i = es_split(self.prof, self.lab, tr)
        app = np.concatenate([es_i, va])
        Xf, Xa = with_target_features(self.prof, self.target, self.lab.iloc[fit_i], self.X.iloc[fit_i],
                                      self.lab.iloc[app], self.X.iloc[app])
        out = (fit_i, es_i, va, Xf, Xa.iloc[:len(es_i)].reset_index(drop=True),
               Xa.iloc[len(es_i):].reset_index(drop=True))
        if self.cache_ok:
            self._cache[i] = out
        return out

    def full(self):
        """(Xall, Xtest) для финального fit: target-признаки по всему размеченному train."""
        if "full" in self._cache:
            return self._cache["full"]
        out = with_target_features(self.prof, self.target, self.lab, self.X, self.prof["test"], self.Xte)
        if self.cache_ok:
            self._cache["full"] = out
        return out


def _balanced_weights(yi: np.ndarray) -> np.ndarray:
    counts = np.bincount(yi)
    return (len(yi) / (np.count_nonzero(counts) * counts[yi])).astype(float)


def _cat_cols(X: pd.DataFrame) -> list[str]:
    return [c for c in X.columns if str(X[c].dtype) == "category"]


# ---------------------------------------------------------------------------
# Семейства моделей: fit(Xf, yf, Xes, yes, wf, wes, deadline) -> (model, n_iter),
# refit(X, y, w, n_iter, deadline) -> model, predict(model, X) -> pred | proba (K колонок).
# ---------------------------------------------------------------------------

class _Family:
    def __init__(self, fd: FoldData, recipe: dict):
        self.fd, self.recipe = fd, recipe
        self.metric = fd.prof["metric"]
        self.k = None if fd.regression else len(fd.classes)
        self.balanced = (not fd.regression) and self.metric in BALANCED_METRICS
        self.seed = int(recipe.get("seed", 7))
        self.params = dict(recipe.get("params") or {})

    def _proba(self, present: np.ndarray, proba: np.ndarray) -> np.ndarray:
        """Вероятности модели (по классам present, в порядке их индексов) -> K колонок."""
        out = np.zeros((len(proba), self.k))
        out[:, present] = proba
        return out


class LGBMFamily(_Family):
    """LightGBM; params — переопределения (objective, регуляризация); f1_stop — остановка по метрике."""

    def _make(self, n_estimators: int):
        import lightgbm as lgb

        p = dict(n_estimators=n_estimators, learning_rate=0.05, num_leaves=127, colsample_bytree=0.8,
                 subsample=0.8, subsample_freq=1, n_jobs=N_CPUS, random_state=self.seed, verbosity=-1)
        p.update(self.params)
        if self.fd.regression:
            p.setdefault("objective", "regression")
            return lgb.LGBMRegressor(**p)
        if self.balanced:
            p["class_weight"] = "balanced"
        return lgb.LGBMClassifier(**p)

    def _feval(self):
        """Метрика задачи как eval LightGBM (остановка по ней, а не по добавленному log loss)."""
        metric, k = self.metric, self.k

        def f1_eval(y_true, y_pred):
            yp = y_pred.reshape(len(y_true), -1) if y_pred.ndim == 1 and k and k > 2 else y_pred
            if yp.ndim == 1:
                lab = (yp > 0.5).astype(int)
            else:
                lab = np.argmax(yp, axis=1)
            from sklearn.metrics import f1_score

            avg = "macro" if metric == "f1_macro" else "binary"
            return "task_f1", float(f1_score(y_true.astype(int), lab, average=avg)), True

        return f1_eval

    def fit(self, Xf, yf, Xes, yes, wf, wes, dl):
        import lightgbm as lgb

        m = self._make(4000)
        cb = [lgb.early_stopping(100, verbose=False, first_metric_only=True), lgb.log_evaluation(0),
              models.time_guard(dl)]
        kw = {}
        if self.recipe.get("f1_stop") and not self.fd.regression:
            m.set_params(metric="None")
            kw["eval_metric"] = self._feval()
        elif self.fd.regression and self.metric == "mae":
            kw["eval_metric"] = "l1"
        m.fit(Xf, yf, eval_set=[(Xes, yes)], callbacks=cb, **kw)
        return m, int(m.best_iteration_ or m.n_estimators)

    def refit(self, X, y, w, n_iter, dl):
        m = self._make(max(int(n_iter), 20))
        m.fit(X, y, callbacks=[models.time_guard(dl)])
        return m

    def predict(self, m, X):
        if self.fd.regression:
            return m.predict(X)
        return self._proba(np.asarray(m.classes_, dtype=int), m.predict_proba(X))


class TwoPartFamily(_Family):
    """Регрессия с нулями: P(y>0) (LightGBM binary) × E[y | y>0] (LightGBM gamma на положительных)."""

    def _clf(self, n):
        import lightgbm as lgb

        return lgb.LGBMClassifier(n_estimators=n, learning_rate=0.05, num_leaves=63, colsample_bytree=0.8,
                                  subsample=0.8, subsample_freq=1, n_jobs=N_CPUS, random_state=self.seed,
                                  verbosity=-1)

    def _reg(self, n):
        import lightgbm as lgb

        return lgb.LGBMRegressor(objective="gamma", n_estimators=n, learning_rate=0.05, num_leaves=63,
                                 colsample_bytree=0.8, subsample=0.8, subsample_freq=1, n_jobs=N_CPUS,
                                 random_state=self.seed, verbosity=-1)

    def fit(self, Xf, yf, Xes, yes, wf, wes, dl):
        import lightgbm as lgb

        cb = [lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0), models.time_guard(dl)]
        pf, pe = yf > 0, yes > 0
        if pf.sum() < 50 or pe.sum() < 10 or pf.all() or (~pf).all():
            raise ValueError("мало положительных или нулевых строк в фолде")
        c = self._clf(4000)
        c.fit(Xf, pf.astype(int), eval_set=[(Xes, pe.astype(int))], callbacks=cb)
        r = self._reg(4000)
        r.fit(Xf[pf], yf[pf], eval_set=[(Xes[pe], yes[pe])], callbacks=cb)
        return (c, r), (int(c.best_iteration_ or 100), int(r.best_iteration_ or 100))

    def refit(self, X, y, w, n_iter, dl):
        pos = y > 0
        c = self._clf(max(n_iter[0], 20)).fit(X, pos.astype(int), callbacks=[models.time_guard(dl)])
        r = self._reg(max(n_iter[1], 20)).fit(X[pos], y[pos], callbacks=[models.time_guard(dl)])
        return c, r

    def predict(self, m, X):
        c, r = m
        p = c.predict_proba(X)[:, list(c.classes_).index(1)]
        return p * r.predict(X)


class DeltaFamily(LGBMFamily):
    """Регрессия по истории объекта: LightGBM учит y − опора, опора — последний доступный таргет объекта.

    Опора — признак lag(target).last, построенный внутри фолда только по
    прошлым периодам (models.target_features), поэтому метки validation/test
    в неё не попадают. Холодный старт (нет истории): опора = среднее таргета
    train фолда. Обратный переход (+ опора) — ровно один раз в predict.
    """

    def _base(self, X):
        return X[self.recipe["delta_col"]].astype(float).fillna(self._fallback).to_numpy()

    def fit(self, Xf, yf, Xes, yes, wf, wes, dl):
        self._fallback = float(np.mean(yf))
        m, it = super().fit(Xf, yf - self._base(Xf), Xes, yes - self._base(Xes), wf, wes, dl)
        m._fallback = self._fallback
        return m, it

    def refit(self, X, y, w, n_iter, dl):
        self._fallback = float(np.mean(y))
        m = super().refit(X, y - self._base(X), w, n_iter, dl)
        m._fallback = self._fallback
        return m

    def predict(self, m, X):
        self._fallback = m._fallback
        return m.predict(X) + self._base(X)


class CatBoostFamily(_Family):
    def _prep(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for c in _cat_cols(X):
            X[c] = X[c].astype(str)
        return X

    def _make(self, iters: int, use_es: bool):
        from catboost import CatBoostClassifier, CatBoostRegressor

        p = dict(iterations=iters, learning_rate=0.08, depth=6, l2_leaf_reg=3.0, thread_count=N_CPUS,
                 random_seed=self.seed, allow_writing_files=False, verbose=False)
        if use_es:
            p.update(od_type="Iter", od_wait=100, use_best_model=True)
        p.update(self.params)
        if self.fd.regression:
            loss = "MAE" if self.metric == "mae" else "RMSE"
            return CatBoostRegressor(loss_function=loss, eval_metric=loss, **p)
        loss = "Logloss" if self.k == 2 else "MultiClass"
        if self.metric == "f1_macro" and self.k > 2:
            p["eval_metric"] = "TotalF1:average=Macro"
        elif self.metric == "roc_auc" and self.k == 2:
            p["eval_metric"] = "AUC"
        if self.balanced:
            p["auto_class_weights"] = "Balanced"
        return CatBoostClassifier(loss_function=loss, **p)

    @staticmethod
    def _guard(dl):
        class Guard:
            def after_iteration(self, info):  # noqa: ARG002
                return time.time() < dl

        return Guard()

    def fit(self, Xf, yf, Xes, yes, wf, wes, dl):
        Xf, Xes = self._prep(Xf), self._prep(Xes)
        cats = _cat_cols_obj(Xf)
        m = self._make(5000, True)
        m.fit(Xf, yf, cat_features=cats, eval_set=(Xes, yes), callbacks=[self._guard(dl)])
        return m, int(m.get_best_iteration() or m.tree_count_) + 1

    def refit(self, X, y, w, n_iter, dl):
        X = self._prep(X)
        m = self._make(max(int(n_iter), 20), False)
        m.fit(X, y, cat_features=_cat_cols_obj(X), callbacks=[self._guard(dl)])
        return m

    def predict(self, m, X):
        X = self._prep(X)
        if self.fd.regression:
            return m.predict(X)
        return self._proba(np.asarray(m.classes_, dtype=int), m.predict_proba(X))


def _cat_cols_obj(X: pd.DataFrame) -> list[int]:
    return [i for i, c in enumerate(X.columns) if X[c].dtype == object]


class XGBFamily(_Family):
    def _make(self, n: int, n_classes: int | None, use_es: bool):
        import xgboost as xgb

        p = dict(n_estimators=n, learning_rate=0.05, max_depth=7, min_child_weight=1.0, subsample=0.8,
                 colsample_bytree=0.8, tree_method="hist", enable_categorical=True, max_cat_to_onehot=8,
                 n_jobs=N_CPUS, random_state=self.seed, verbosity=0)
        if use_es:
            p["early_stopping_rounds"] = 100
        p.update(self.params)
        if self.fd.regression:
            return xgb.XGBRegressor(eval_metric="mae" if self.metric == "mae" else "rmse", **p)
        return xgb.XGBClassifier(eval_metric="logloss" if n_classes == 2 else "mlogloss", **p)

    @staticmethod
    def _guard(dl):
        import xgboost as xgb

        class Guard(xgb.callback.TrainingCallback):
            def after_iteration(self, model, epoch, evals_log):  # noqa: ARG002
                return time.time() > dl

        return Guard()

    def _local(self, y):
        """XGBoost требует метки 0..n-1: классы, которых нет в train фолда, исключаются."""
        present = np.unique(y)
        return present, np.searchsorted(present, y)

    def fit(self, Xf, yf, Xes, yes, wf, wes, dl):
        if self.fd.regression:
            m = self._make(4000, None, True)
            m.set_params(callbacks=[self._guard(dl)])
            m.fit(Xf, yf, sample_weight=wf, eval_set=[(Xes, yes)], verbose=False)
            return m, int(m.best_iteration) + 1
        present, yl = self._local(yf)
        keep = np.isin(yes, present)
        m = self._make(4000, len(present), True)
        m.set_params(callbacks=[self._guard(dl)])
        m.fit(Xf, yl, sample_weight=wf, eval_set=[(Xes[keep], np.searchsorted(present, yes[keep]))],
              sample_weight_eval_set=None if wes is None else [wes[keep]], verbose=False)
        m._present = present
        return m, int(m.best_iteration) + 1

    def refit(self, X, y, w, n_iter, dl):
        if self.fd.regression:
            m = self._make(max(int(n_iter), 20), None, False)
            m.set_params(callbacks=[self._guard(dl)])
            return m.fit(X, y, sample_weight=w, verbose=False)
        present, yl = self._local(y)
        m = self._make(max(int(n_iter), 20), len(present), False)
        m.set_params(callbacks=[self._guard(dl)])
        m.fit(X, yl, sample_weight=w, verbose=False)
        m._present = present
        return m

    def predict(self, m, X):
        if self.fd.regression:
            return m.predict(X)
        proba = m.predict_proba(X)
        if proba.ndim == 1:
            proba = np.column_stack([1 - proba, proba])
        return self._proba(m._present, proba)


class ExtraTreesFamily(_Family):
    """ExtraTrees: категории -> коды, пропуски -> медиана train (без ранней остановки)."""

    def _encode(self, X: pd.DataFrame, med: pd.Series | None = None):
        X = X.copy()
        for c in _cat_cols(X):
            X[c] = X[c].cat.codes.astype(float).replace(-1, np.nan)
        X = X.apply(pd.to_numeric, errors="coerce").astype(np.float32)
        if med is None:
            med = X.median().fillna(0)
        X = X.fillna(med).replace([np.inf, -np.inf], 0)
        return X, med

    def _make(self):
        from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

        p = dict(n_estimators=250, min_samples_leaf=2, max_features=0.5, n_jobs=N_CPUS,
                 random_state=self.seed)
        p.update(self.params)
        if self.fd.regression:
            return ExtraTreesRegressor(**p)
        return ExtraTreesClassifier(class_weight="balanced" if self.balanced else None, **p)

    def fit(self, Xf, yf, Xes, yes, wf, wes, dl):
        return self.refit(Xf, yf, None, 0, dl), 0

    def refit(self, X, y, w, n_iter, dl):
        Xe, med = self._encode(X)
        m = self._make().fit(Xe, y)
        m._med = med
        return m

    def predict(self, m, X):
        Xe, _ = self._encode(X, m._med)
        if self.fd.regression:
            return m.predict(Xe)
        return self._proba(np.asarray(m.classes_, dtype=int), m.predict_proba(Xe))


class TfidfFamily(_Family):
    """Свободный текст: TF-IDF (слова 1-2 + символы 2-4) по текстовым колонкам рецепта + линейная модель.

    Словарь и IDF fit-ятся на train фолда; матрица остаётся разреженной.
    Пропуск — пустая строка; новые слова в predict игнорируются векторизатором.
    """

    def _docs(self, X: pd.DataFrame) -> pd.Series:
        cols = self.recipe["text_cols"]
        parts = [X[c].astype(object).where(X[c].notna(), "").astype(str) for c in cols]
        doc = parts[0]
        for p in parts[1:]:
            doc = doc + " \x1f " + p
        return doc.str.slice(0, TEXT_MAX_CHARS)

    def refit(self, X, y, w, n_iter, dl):
        from scipy import sparse
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression, Ridge

        docs = self._docs(X)
        vw = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=TEXT_MAX_FEATURES, sublinear_tf=True)
        vc = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=3, max_features=TEXT_MAX_FEATURES,
                             sublinear_tf=True)
        M = sparse.hstack([vw.fit_transform(docs), vc.fit_transform(docs)]).tocsr()
        if self.fd.regression:
            m = Ridge(alpha=1.0)
        else:
            m = LogisticRegression(C=4.0, max_iter=300, class_weight="balanced" if self.balanced else None)
        m.fit(M, y)
        m._vec = (vw, vc)
        return m

    def fit(self, Xf, yf, Xes, yes, wf, wes, dl):
        return self.refit(Xf, yf, None, 0, dl), 0

    def predict(self, m, X):
        from scipy import sparse

        docs = self._docs(X)
        M = sparse.hstack([v.transform(docs) for v in m._vec]).tocsr()
        if self.fd.regression:
            return m.predict(M)
        return self._proba(np.asarray(m.classes_, dtype=int), m.predict_proba(M))


TEXT_MAX_FEATURES = 100_000
TEXT_MAX_CHARS = 5_000
TEXT_MIN_TOKENS = 4      # медиана слов в значении — признак свободного текста, а не категории/ID
TEXT_MIN_UNIQUE = 0.2


def text_columns(prof: dict, max_rows: int = 5_000) -> list[str]:
    """Строковые колонки со свободным текстом: несколько слов и высокая уникальность (ID без пробелов — нет)."""
    out = []
    train = prof["train"]
    sample = train.sample(min(len(train), max_rows), random_state=0) if len(train) else train
    for c in prof["features"]:
        s = sample[c]
        if s.dtype != object:
            continue
        s = s.dropna().astype(str)
        if len(s) < 50:
            continue
        tokens = s.str.split().str.len()
        if tokens.median() >= TEXT_MIN_TOKENS and s.nunique() / len(s) >= TEXT_MIN_UNIQUE:
            out.append(c)
    return out


FAMILIES = {"tfidf": TfidfFamily, "lgbm": LGBMFamily, "delta": DeltaFamily, "twopart": TwoPartFamily, "catboost": CatBoostFamily,
            "xgboost": XGBFamily, "extratrees": ExtraTreesFamily}
# Стоимость k-fold рецепта относительно k-fold LightGBM (для допуска по бюджету).
COST = {"tfidf": 1.0, "lgbm": 1.0, "delta": 1.0, "twopart": 1.6, "catboost": 4.0, "xgboost": 1.5, "extratrees": 1.0}


# ---------------------------------------------------------------------------
# Преобразования таргета регрессии (прямое — перед обучением, обратное — ровно один раз).
# ---------------------------------------------------------------------------

def _forward(y: np.ndarray, tf: str | None) -> np.ndarray:
    if tf == "log1p":
        return np.log1p(y)
    if tf == "sqrt":
        return np.sqrt(y)
    return y


def _inverse(p: np.ndarray, tf: str | None) -> np.ndarray:
    if tf == "log1p":
        return np.expm1(np.clip(p, None, EXP_CAP))
    if tf == "sqrt":
        return np.square(np.clip(p, 0, None))
    return p


# ---------------------------------------------------------------------------
# Кандидат: k-fold + frozen refit.
# ---------------------------------------------------------------------------

def run_candidate(fd: FoldData, recipe: dict, deadline_ts: float) -> dict | None:
    """OOF рецепта на всех внешних фолдах + test от модели на всём train.

    None — не успел все фолды (частичный OOF несопоставим) или упал.
    """
    prof, metric = fd.prof, fd.prof["metric"]
    t0 = time.time()
    fam = FAMILIES[recipe["family"]](fd, recipe)
    tf = recipe.get("tf")
    if fd.regression:
        yt = fd.y_orig.astype(float)
        if prof["log_target"]:
            yt = np.log1p(np.clip(yt, 0, None))
        yt = _forward(yt, tf)
    else:
        pos = {c: i for i, c in enumerate(fd.classes)}
        yt = np.fromiter((pos[v] for v in fd.y_orig), dtype=np.int64, count=len(fd.y_orig))
    w_all = _balanced_weights(yt) if fam.balanced else None
    shape = (len(yt),) if fd.regression else (len(yt), fam.k)
    oof = np.full(shape, np.nan)
    iters = []
    for i in range(len(fd.folds)):
        if time.time() > deadline_ts:
            log(f"pool[{recipe['name']}]: дедлайн на фолде {i + 1}/{len(fd.folds)} — кандидат пропущен")
            return None
        fit_i, es_i, va, Xf, Xes, Xva = fd.fold(i)
        if not fd.regression:
            # Ранняя остановка — по классам, которые модель видит в train фолда.
            seen = np.isin(yt[es_i], np.unique(yt[fit_i]))
            if not seen.all():
                es_i, Xes = es_i[seen], Xes.loc[seen].reset_index(drop=True)
        wf = None if w_all is None else w_all[fit_i]
        wes = None if w_all is None else w_all[es_i]
        m, it = fam.fit(Xf, yt[fit_i], Xes, yt[es_i], wf, wes, deadline_ts)
        oof[va] = fam.predict(m, Xva)
        iters.append(it)
    if isinstance(iters[0], tuple):
        n_iter = tuple(int(np.median([t[j] for t in iters]) * 1.1) for j in range(len(iters[0])))
    else:
        n_iter = int(np.median(iters) * 1.1)
    Xall, Xt = fd.full()
    m = fam.refit(Xall, yt, w_all, n_iter, min(deadline(), max(deadline_ts, time.time() + 120)))
    te = fam.predict(m, Xt)

    cov = np.isfinite(oof if oof.ndim == 1 else oof[:, 0])
    oof, y_cov = oof[cov], fd.y_orig[cov]
    if fd.regression:
        oof, te = _inverse(oof, tf), _inverse(te, tf)
        if prof["log_target"]:
            oof, te = np.expm1(np.clip(oof, None, EXP_CAP)), np.expm1(np.clip(te, None, EXP_CAP))
        if tf is not None or prof["log_target"] or recipe["family"] == "twopart":
            oof, te = np.clip(oof, 0, None), np.clip(te, 0, None)
        if not (np.isfinite(oof).all() and np.isfinite(te).all()):
            log(f"pool[{recipe['name']}]: неконечные прогнозы — кандидат отклонён")
            return None
        s = score(metric, y_cov, oof)
    else:
        te = te / np.clip(te.sum(axis=1, keepdims=True), 1e-12, None)
        oof = oof / np.clip(oof.sum(axis=1, keepdims=True), 1e-12, None)
        pred = np.array(fd.classes, dtype=object)[np.argmax(oof, axis=1)]
        s = score(metric, y_cov, pred, proba=oof, labels=fd.classes)
    sec = time.time() - t0
    log(f"pool[{recipe['name']}]: {metric}={s:.5f} на {int(cov.sum())} строках, итераций {n_iter}, {sec:.0f}s")
    return {"name": recipe["name"], "rows": fd.lab.index.to_numpy()[cov], "y": y_cov, "oof": oof, "test": te,
            "classes": fd.classes, "score": s, "fit_sec": sec,
            "recipe": {**recipe, "n_iter": n_iter}}


def _skew(y: np.ndarray) -> float:
    y = y[np.isfinite(y)]
    sd = y.std()
    return float(((y - y.mean()) ** 3).mean() / sd ** 3) if sd > 0 else 0.0


def _history_column(prof: dict) -> str | None:
    """Колонка «последний таргет объекта» из принятой AutoEDA спеки lag(target) — только при temporal-схеме."""
    import fe_dsl

    if (prof.get("cv") or {}).get("kind") != "temporal":
        return None
    for spec in prof.get("target_specs") or []:
        if spec["op"] == "lag" and spec["params"].get("src") == "target" and "last" in spec["params"]["aggs"]:
            return fe_dsl._name(spec, "last")
    return None


def plan(prof: dict, target: str) -> list[dict]:
    """Рецепты по профилю задачи (порядок = приоритет). why — основание гипотезы."""
    regression = prof["problem_type"] == "regression"
    metric = prof["metric"]
    out = []
    if regression:
        y = labeled(prof, target)[target].to_numpy(float)
        nonneg = bool((y >= 0).all())
        sk = _skew(y)
        if nonneg and not prof["log_target"] and sk > SKEW_MIN:
            out.append({"name": "lgbm_log1p", "family": "lgbm", "tf": "log1p",
                        "why": f"неотрицательный таргет с асимметрией {sk:.2f}"})
            out.append({"name": "lgbm_sqrt", "family": "lgbm", "tf": "sqrt",
                        "why": f"неотрицательный таргет с асимметрией {sk:.2f}"})
        zeros = float((y == 0).mean())
        if nonneg and zeros >= ZERO_SHARE_MIN and (y > 0).sum() >= POSITIVE_MIN and not prof["log_target"]:
            out.append({"name": "lgbm_tweedie", "family": "lgbm",
                        "params": {"objective": "tweedie", "tweedie_variance_power": 1.5},
                        "why": f"доля нулей {zeros:.2f}"})
            out.append({"name": "twopart", "family": "twopart", "why": f"доля нулей {zeros:.2f}"})
        hist = _history_column(prof)
        if hist is not None and not prof["log_target"]:
            out.append({"name": "lgbm_delta", "family": "delta", "delta_col": hist,
                        "why": "повторные объекты во времени: прогноз изменения от последнего известного таргета"})
        if metric == "mae":
            out.append({"name": "lgbm_l1", "family": "lgbm", "params": {"objective": "l1"},
                        "why": "функция потерь под MAE"})
    elif metric in ("f1_macro", "f1"):
        out.append({"name": "lgbm_f1stop", "family": "lgbm", "f1_stop": True,
                    "why": "ранняя остановка по метрике задачи вместо log loss"})
    text = text_columns(prof)
    if text and not (regression is False and len(pd.unique(labeled(prof, target)[target])) > 50):
        out.insert(0, {"name": "tfidf_linear", "family": "tfidf", "text_cols": text,
                       "why": f"свободный текст в {text}: разреженный TF-IDF + линейная модель"})
    out += [
        {"name": "catboost", "family": "catboost", "why": "другое семейство бустинга (симметричные деревья)"},
        {"name": "xgboost", "family": "xgboost", "why": "другое семейство бустинга (по уровням)"},
        {"name": "lgbm_reg", "family": "lgbm",
         "params": {"learning_rate": 0.03, "num_leaves": 63, "min_child_samples": 50, "reg_lambda": 5.0,
                    "reg_alpha": 0.5, "colsample_bytree": 0.5},
         "why": "сильнее регуляризация и сэмплирование признаков"},
        {"name": "extratrees", "family": "extratrees", "why": "бэггинг случайных деревьев: иные ошибки"},
    ]
    return out


TF_FAMILIES = ("lgbm", "catboost", "xgboost", "extratrees")
EXTRA_SEEDS = 2


def run_pool(prof: dict, target: str, deadline_ts: float, lgbm_sec: float, register=None,
             base_score: float | None = None) -> list[dict]:
    """Рецепты plan() по очереди, пока прогноз стоимости (по k-fold LightGBM) помещается до deadline_ts.

    Регрессия: если преобразование таргета (log1p/sqrt) обошло k-fold LightGBM
    на исходной шкале (base_score), остальные семейства обучаются в лучшей
    шкале. Остаток бюджета — дополнительные seed лучшего рецепта (#20):
    смесь решает ансамбль, совпадающие прогнозы отбрасываются.
    """
    fd = FoldData(prof, target)
    if (fd.classes is not None and len(fd.classes) > 50) or len(fd.folds) == 0:
        return []
    metric = prof["metric"]
    out: list[dict] = []
    tf_scores = {None: base_score}

    def best_tf():
        best, bs = None, tf_scores[None]
        for tf, sc in tf_scores.items():
            if tf is not None and is_better(metric, sc, bs):
                best, bs = tf, sc
        return best

    def attempt(r: dict) -> dict | None:
        est = COST[r["family"]] * max(lgbm_sec, 1.0)
        left = deadline_ts - time.time()
        if est > left:
            reason = f"прогноз {est:.0f}s > остатка {left:.0f}s"
            log(f"pool[{r['name']}]: пропуск — {reason}")
            if register:
                register(f"pool:{r['name']}", r["why"], None, False, reason=reason, duration=0.0)
            return None
        try:
            cand = run_candidate(fd, r, deadline_ts)
        except Exception as exc:  # noqa: BLE001 — сбой рецепта не роняет пул
            log(f"pool[{r['name']}] упал: {type(exc).__name__}: {exc}")
            if register:
                register(f"pool:{r['name']}", r["why"], None, False, error=f"{type(exc).__name__}: {exc}")
            return None
        if cand is None:
            if register:
                register(f"pool:{r['name']}", r["why"], None, False, reason="дедлайн или неконечные прогнозы")
            return None
        if register:
            register(f"pool:{r['name']}", r["why"], cand["score"], True, recipe=str(cand["recipe"]),
                     duration=cand["fit_sec"])
        out.append(cand)
        return cand

    for r in plan(prof, target):
        if fd.regression and r.get("tf") is None and r["family"] in TF_FAMILIES and "objective" not in (
                r.get("params") or {}):
            tf = best_tf()
            if tf is not None:
                r = {**r, "tf": tf, "name": f"{r['name']}_{tf}", "why": r["why"] + f"; шкала {tf} лучше исходной"}
        cand = attempt(r)
        if cand is not None and r["family"] == "lgbm" and r.get("tf") and r["name"] in ("lgbm_log1p", "lgbm_sqrt"):
            tf_scores[r["tf"]] = cand["score"]
    if out:
        best = out[0]
        for c in out[1:]:
            if is_better(metric, c["score"], best["score"]):
                best = c
        rec = best["recipe"]
        for k in range(EXTRA_SEEDS):
            r = {key: v for key, v in rec.items() if key != "n_iter"}
            r.update(seed=int(rec.get("seed", 7)) + 1000 * (k + 1), name=f"{best['name']}_s{k + 1}",
                     why=f"другой seed лучшего рецепта {best['name']}")
            if attempt(r) is None:
                break
    return out


# ---------------------------------------------------------------------------
# Жадный ансамбль (Caruana) по выровненным OOF.
# ---------------------------------------------------------------------------

def _mix_score(metric: str, y, mix: np.ndarray, classes) -> float:
    if classes is None:
        return score(metric, y, mix)
    pred = np.array(classes, dtype=object)[np.argmax(mix, axis=1)]
    return score(metric, y, pred, proba=mix, labels=classes)


def greedy_ensemble(cands: list[dict], metric: str, max_steps: int = 40, rel_tol: float = 1e-5):
    """Отбор с возвращением: старт — лучший одиночный, шаг — кандидат, сильнее всего улучшающий смесь.

    cands выровнены (align_candidates): общие строки, классы, y. Точные дубли
    прогнозов отбрасываются. Возвращает (weights по cands, score смеси).
    """
    y, classes = cands[0]["y"], cands[0].get("classes")
    keep: list[int] = []
    for i, c in enumerate(cands):
        if any(np.allclose(c["oof"], cands[j]["oof"], rtol=0, atol=1e-9) for j in keep):
            log(f"ансамбль: {c['name']} совпадает с уже включённым кандидатом — пропуск")
            continue
        keep.append(i)
    scores = {i: cands[i]["score"] for i in keep}
    best_i = keep[0]
    for i in keep[1:]:
        if is_better(metric, scores[i], scores[best_i]):
            best_i = i
    counts = np.zeros(len(cands))
    counts[best_i] = 1
    total = cands[best_i]["oof"].astype(float).copy()
    cur = scores[best_i]
    for _ in range(max_steps):
        n = counts.sum()
        step_best, step_i = None, None
        for i in keep:
            s = _mix_score(metric, y, (total + cands[i]["oof"]) / (n + 1), classes)
            if step_best is None or is_better(metric, s, step_best):
                step_best, step_i = s, i
        margin = rel_tol * max(abs(cur), 1e-9)
        shifted = cur + margin if models.HIGHER_BETTER[metric] else cur - margin
        if not is_better(metric, step_best, shifted):
            break
        counts[step_i] += 1
        total += cands[step_i]["oof"]
        cur = step_best
    return counts / counts.sum(), cur
