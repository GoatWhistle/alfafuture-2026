"""AI-агент для автоматического решения табличных ML-задач.

Запуск (интерфейс платформы): python3 solution.py task.md
Результат: submission.csv рядом с task.md.

Каскад с гарантией результата: константный сабмит -> быстрый LightGBM ->
AutoEDA (детекторы зависимостей + гипотезы Qwen -> спеки DSL -> валидатор на
фолдах, похожих на тест) -> AutoGluon -> k-fold OOF -> бленд -> подбор
классовых весов -> копирование меток дублей. Каждая ступень перезаписывает submission.csv только когда даёт
валидные предсказания, поэтому файл существует с первых минут работы.
Все гипотезы принимаются только по улучшению на валидации.
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from agent_utils import (
    TOTAL_BUDGET_SEC,
    Background,
    StageTimeout,
    deadline,
    log,
    shutdown,
    start_watchdog,
    time_left,
    time_limit,
)
import eda_detectors
import fe_dsl
import fe_search
import models
import linear_search
from models import (
    ag_oof,
    ag_predict,
    align_classes,
    clip_to_train_range,
    constant_predictions,
    copy_duplicate_labels,
    fit_autogluon,
    fold_ids,
    is_better,
    labeled,
    outer_folds,
    output_from_proba,
    quick_model,
    quick_oof,
    tune_blend,
    tune_class_weights,
    score,
    tune_class_weights_stable,
    write_submission,
)
from task_profile import build_profile, llm_propose_specs, llm_refine_specs

PREDICT_RESERVE_SEC = float(os.environ.get("AGENT_PREDICT_RESERVE_SEC", 900))
KFOLD_RESERVE_SEC = float(os.environ.get("AGENT_KFOLD_RESERVE_SEC", 600))
AUTOEDA_SEC = float(os.environ.get("AGENT_AUTOEDA_SEC", 1200))
MIN_AUTOEDA_SEC = 120
LINEAR_SEC = max(0.0, min(60.0, float(os.environ.get("AGENT_LINEAR_SEC", 60))))
# Сверх бюджета AutoEDA: после этого этап бросается (зависшее построение признаков).
AUTOEDA_SLACK_SEC = 60
MACRO_METRICS = ("f1_macro", "f1_weighted", "f1", "balanced_accuracy")
# Что AutoEDA меняет в профиле (переносится из рабочей копии только при успехе).
EDA_KEYS = ("train", "test", "features", "model_ops", "target_specs", "cv")
EXPERIMENTS: list[dict] = []


def register(name: str, hypothesis: str, val_score, accepted: bool) -> None:
    EXPERIMENTS.append(
        {"name": name, "hypothesis": hypothesis, "val_score": val_score, "accepted": accepted}
    )
    log(f"эксперимент {name}: score={val_score}, accepted={accepted} — {hypothesis}")


def _tuned_or(tuned: np.ndarray | None, fallback: np.ndarray) -> np.ndarray:
    return fallback if tuned is None else tuned


def _clearly_better(metric: str, new: float, old: float, rel: float = 0.002) -> bool:
    """new лучше old с запасом rel (страховка от шума валидации)."""
    margin = rel * abs(old) if old != 0 else 1e-9
    shifted = old + margin if models.HIGHER_BETTER[metric] else old - margin
    return is_better(metric, new, shifted)


def tune_quick_weights(prof: dict, quick: dict | None, target: str, current: dict[str, np.ndarray]) -> np.ndarray | None:
    """Подбор классовых весов для quick-модели, когда она осталась финальной."""
    if (
        quick is None
        or prof["problem_type"] == "regression"
        or prof["output_proba"]
        or quick.get("proba") is None
        or prof["metric"] not in MACRO_METRICS
    ):
        return None
    weights, _, new = tune_class_weights(quick["y_val"], quick["val_proba"], quick["classes"], prof["metric"])
    register("class_weights_quick", "классовые множители для quick-модели", new, weights is not None)
    if weights is None:
        return None
    classes_arr = np.array(quick["classes"], dtype=object)
    pred = classes_arr[np.argmax(quick["proba"] * weights, axis=1)]
    current[target] = pred
    write_submission(prof, current)
    return pred


def _validate_llm(raw: list, ctx: dict, known: set[str], source: str) -> list[dict]:
    """Валидация и дедупликация сырых спек LLM; prior — медиана детекторов (середина очереди)."""
    columns = list(ctx["features"])
    numeric = set(eda_detectors._numeric(ctx))
    out = []
    for r in raw:
        if not isinstance(r, dict):
            continue
        theme = r.get("theme")
        v = fe_dsl.validate({**r, "source": f"{source}:{theme}" if theme else source, "prior": ctx["llm_prior"]},
                            columns, numeric)
        if v is None:
            log(f"LLM-спека отброшена: {r}")
            continue
        k = fe_dsl.spec_key(v)
        if k not in known:
            known.add(k)
            out.append(v)
    log(f"LLM[{source}]: предложено {len(raw)}, валидных новых {len(out)}")
    return out


def _llm_round1(prof: dict, ctx: dict, summary: dict, known: set[str], planned: list[str]) -> list[dict]:
    proposal = llm_propose_specs(prof["task_text"], summary, prof.get("data_dictionary"), list(ctx["features"]),
                                 planned)
    if not proposal:
        return []
    raw = list(proposal["specs"])
    if proposal.get("entity") and ctx["entity"] is None:
        raw.append({"op": "group_stat", "cols": [], "params": {"keys": proposal["entity"], "src": "target",
                    "aggs": list(fe_dsl.TARGET_AGGS)}, "why": "ключ объекта по мнению LLM"})
    log(f"LLM: подсказка периода {proposal.get('period')!r}, схема валидации {ctx['fold_scheme']}")
    return _validate_llm(raw, ctx, known, "qwen")


def run_autoeda(prof: dict, target: str, budget: float) -> dict:
    """Детекторы -> валидатор; Qwen: раунд 1 (темы параллельно, в фоне) -> раунд 2 по итогам.

    Возвращает {"period", "label_copy", "accepted"}: схему валидации для
    финального выбора и флаг постпроцессинга. Принятые признаки дописываются
    в prof, модельные операции — в prof["model_ops"].
    """
    t0 = time.time()
    deadline = t0 + budget
    ctx = fe_search.build_context(prof, target, deadline)
    specs = eda_detectors.run_all(ctx, t0 + 0.25 * budget)
    fe_search.build_folds(ctx)
    srch = fe_search.Search(ctx, register)
    srch.calibrate()
    try:
        ctx["facts"]["perm_importance"] = fe_search.perm_importance(ctx, srch.ev)
    except Exception:  # noqa: BLE001
        log("perm_importance упал:\n" + traceback.format_exc())
    ctx["llm_prior"] = float(np.median([s["prior"] for s in specs])) if specs else 0.0
    known = {fe_dsl.spec_key(s) for s in specs}
    summary = eda_detectors.summary(ctx, specs)
    register("autoeda_specs", f"{len(specs)} гипотез детекторов: " + "; ".join(
        f"{s['source']}:{fe_dsl.describe(s)}" for s in sorted(specs, key=lambda s: -s["prior"])[:12]), None, True)

    use_llm = time_left() > 1800
    planned = [fe_dsl.describe(s) for s in sorted(specs, key=lambda s: -s["prior"])]
    fut = Background(_llm_round1, prof, ctx, summary, known, planned) if use_llm else None
    srch.stage_a(specs, t0 + 0.5 * budget)
    if fut is not None:
        try:
            qwen = fut.result(timeout=max(5.0, t0 + 0.6 * budget - time.time()))
        except TimeoutError:
            log("LLM раунд 1 не успел — продолжаю без него")
            qwen = []
        except Exception:  # noqa: BLE001
            log("LLM раунд 1 упал:\n" + traceback.format_exc())
            qwen = []
        srch.stage_a(qwen, t0 + 0.65 * budget)
        if qwen and time.time() < t0 + 0.65 * budget:
            try:
                planned = [fe_dsl.describe(s) for s in sorted(specs + qwen, key=lambda s: -s["prior"])]
                ref = llm_refine_specs(prof["task_text"], summary, srch.feedback(), list(ctx["features"]), planned)
                qwen2 = _validate_llm(ref["specs"], ctx, known, "qwen2") if ref else []
            except Exception:  # noqa: BLE001
                log("LLM раунд 2 упал:\n" + traceback.format_exc())
                qwen2 = []
            srch.stage_a(qwen2, t0 + 0.8 * budget)
    accepted = srch.stage_b(deadline)

    added = []
    model_ops = {}
    target_specs = []
    materialized = set()
    for spec in fe_dsl.unique_specs(accepted):
        if spec["op"] == "drop":
            col = spec["cols"][0]
            prof["features"] = [c for c in prof["features"] if c != col]
            added.append(f"-{col}")
            continue
        if spec["op"] in fe_dsl.MODEL_OPS:
            model_ops[spec["op"]] = spec
            added.append(fe_dsl.describe(spec))
            if spec["op"] != "segment_model":
                continue
            # Режим сегмента — ещё и признак для AutoGluon.
            spec = fe_dsl.validate({**spec, "op": "flag"}, prof["train"].columns)
            if spec is None:
                continue
        if fe_dsl.uses_target(spec):
            # Признаки из таргета строятся внутри каждого внешнего фолда (models.target_features),
            # а не один раз на всём train: иначе таргет валидации протекает в признаки обучения.
            target_specs.append(spec)
            added.append(fe_dsl.describe(spec) + " [по фолдам]")
            continue
        key = fe_dsl.spec_key(spec)
        if key in materialized:
            # segment_model тоже добавляет flag; отдельная такая же flag-спека
            # не должна повторно добавлять этот индикатор.
            continue
        res = fe_dsl.materialize(prof, spec, target, ctx["classes"])
        if res is None:
            continue
        tr_new, te_new = res
        new_cols = list(tr_new.columns)
        prof["train"] = pd.concat([prof["train"], tr_new], axis=1, verify_integrity=True)
        prof["test"] = pd.concat([prof["test"], te_new], axis=1, verify_integrity=True)
        prof["features"] = list(prof["features"]) + new_cols
        materialized.add(key)
        added += new_cols
    prof["model_ops"] = model_ops
    prof["target_specs"] = target_specs
    prof["cv"] = ctx["cv"]
    register("autoeda", f"схема {ctx['fold_scheme']}; принято {len(accepted)} спек -> {added}",
             ctx.get("search_final"), bool(accepted))
    log(f"AutoEDA: {time.time() - t0:.0f}s, признаков {len(prof['features'])}")
    return {"period": ctx["period"], "label_copy": ctx["label_copy"], "accepted": accepted}


def _scored(cands: list[dict]) -> list[dict]:
    """Только оценённые кандидаты: неоценённый не побеждает по несопоставимой оценке."""
    out = []
    for c in cands:
        if c["score"] is None or not np.isfinite(c["score"]):
            log(f"{c['name']}: оценка недоступна — кандидат не участвует в выборе")
            continue
        out.append(c)
    return out


def align_candidates(cands: list[dict]) -> list[dict]:
    """Кандидаты на общих строках train: OOF сопоставлены по индексам строк, классы — в одном порядке.

    Строки без OOF у любого кандидата исключаются (а не считаются нулём).
    Несовпадение таргета на общей строке — ValueError (OOF перепутаны).
    """
    cands = _scored(cands)
    if not cands:
        return []
    common = np.asarray(cands[0]["rows"])
    for c in cands[1:]:
        common = np.intersect1d(common, np.asarray(c["rows"]))
    if len(common) == 0:
        best = max(cands, key=lambda c: len(c["rows"]))
        log(f"У кандидатов нет общих OOF-строк — остаётся {best['name']}")
        cands, common = [best], np.asarray(best["rows"])
    ref = cands[0].get("classes")
    out = []
    for c in cands:
        index = pd.Index(c["rows"])
        if not index.is_unique:
            raise ValueError(f"{c['name']}: повторяющиеся строки OOF")
        idx = index.get_indexer(common)
        oof, test = np.asarray(c["oof"])[idx], np.asarray(c["test"])
        if ref is not None:
            try:
                oof = align_classes(oof, c["classes"], ref)
                test = align_classes(test, c["classes"], ref)
            except ValueError as exc:
                log(f"{c['name']}: {exc} — кандидат исключён")
                continue
        out.append({**c, "rows": common, "y": np.asarray(c["y"])[idx], "oof": oof, "test": test, "classes": ref})
    y0 = out[0]["y"].astype(str)
    for c in out[1:]:
        if not np.array_equal(c["y"].astype(str), y0):
            raise ValueError(f"таргет {c['name']} не совпадает с {out[0]['name']} на общих строках")
    return out


def _cand_score(metric: str, c: dict) -> float:
    if c.get("classes") is None:
        return score(metric, c["y"], c["oof"])
    pred = np.array(c["classes"], dtype=object)[np.argmax(c["oof"], axis=1)]
    return score(metric, c["y"], pred, proba=c["oof"], labels=c["classes"])


def select_final(prof, target, current, quick, cands, fallback, label_copy=False, extra_cands=()):
    """Выбор источника, бленд и постпроцессинг на общих OOF-строках одной схемы валидации."""
    metric = prof["metric"]
    classification = prof["problem_type"] != "regression"
    try:
        cands = align_candidates(cands)
    except ValueError as exc:
        log(f"OOF кандидатов несопоставимы ({exc}) — остаётся последний сабмит")
        cands = []
    if not cands:
        if classification:
            return _tuned_or(tune_quick_weights(prof, quick, target, current), fallback)
        return fallback
    for c in cands:
        c["score"] = _cand_score(metric, c)
        log(f"{c['name']}: {metric}={c['score']:.5f} на {len(c['rows'])} общих строках")
    if len(cands) == 2:
        a, b = cands
        alpha, s = tune_blend(metric, a["y"], a["oof"], b["oof"], classes=a.get("classes"))
        accepted = all(is_better(metric, s, c["score"]) or s == c["score"] for c in cands)
        register("blend", f"смесь {a['name']} и {b['name']}, alpha={alpha:.2f}", s, accepted)
        cands.append({
            "name": "blend", "score": s, "y": a["y"], "rows": a["rows"], "classes": a.get("classes"),
            "oof": alpha * a["oof"] + (1 - alpha) * b["oof"],
            "test": alpha * a["test"] + (1 - alpha) * b["test"],
        })
    win = cands[0]
    for c in cands[1:]:
        if is_better(metric, c["score"], win["score"]):
            win = c
    # Preserve the incumbent's validation population and its existing two-model blend.
    # An optional family may challenge it only on every one of those same rows.
    for candidate in _scored(list(extra_cands)):
        if not np.isin(win["rows"], candidate["rows"]).all():
            register("family_select", f"{candidate['name']}: неполное покрытие OOF", None, False)
            continue
        try:
            incumbent, challenger = align_candidates([win, candidate])
            old, new = _cand_score(metric, incumbent), _cand_score(metric, challenger)
            accepted = np.isfinite(new) and _clearly_better(metric, new, old)
            register("family_select", f"{candidate['name']} против {win['name']} ({old:.6f}); "
                     f"{len(win['rows'])} общих строк; порог 0.2%", new, accepted)
            if accepted:
                win = {**challenger, "score": new}
        except (ValueError, KeyError) as exc:
            register("family_select", f"{candidate['name']}: {exc}", None, False)
    log(f"Финальный источник: {win['name']} ({metric}={win['score']:.5f})")

    if not classification:
        y_all = labeled(prof, target)[target].to_numpy(float)
        test_final, base, new = clip_to_train_range(win["y"], win["oof"], win["test"], metric,
                                                    y_range=(float(y_all.min()), float(y_all.max())))
        register("clip_range", "обрезка предсказаний к диапазону таргета в train", new, new != base)
        current[target] = test_final
        write_submission(prof, current)
        return current[target]

    classes = win["classes"]
    classes_arr = np.array(classes, dtype=object)
    if prof["output_proba"]:
        current[target] = output_from_proba(prof, target, win["test"], classes)
        write_submission(prof, current)
        return current[target]

    proba_test, proba_oof = win["test"], win["oof"]
    if metric in MACRO_METRICS:
        weights, _, new = tune_class_weights_stable(win["y"], win["oof"], classes, metric)
        register("class_weights_stable", "классовые множители (бутстреп-усреднение по OOF)", new, weights is not None)
        if weights is not None:
            proba_test, proba_oof = proba_test * weights, proba_oof * weights
    pred = classes_arr[np.argmax(proba_test, axis=1)]
    current[target] = pred
    write_submission(prof, current)

    # Ступень 5: метки точных дублей train поверх модели (ворота — LOO на train).
    if label_copy:
        try:
            oof_pred = classes_arr[np.argmax(proba_oof, axis=1)]
            out, base, new, n = copy_duplicate_labels(prof, target, win["y"], oof_pred, pred, metric,
                                                      rows=win["rows"])
            register("label_copy_dups", f"метки дублей train для {n} строк test (LOO base={base})",
                     new, out is not None)
            if out is not None:
                changed = int((np.asarray(out, dtype=object) != np.asarray(pred, dtype=object)).sum())
                log(f"label_copy_dups: изменено {changed} предсказаний test")
                current[target] = out
                write_submission(prof, current)
        except Exception:  # noqa: BLE001
            log("label_copy_dups упал:\n" + traceback.format_exc())
    return current[target]


def run_autogluon(prof: dict, target: str, budget: float) -> dict | None:
    """AutoGluon как кандидат на фолдах схемы prof["cv"].

    Фолды покрывают все строки (kfold/group) — бэггинг AutoGluon идёт по ним,
    OOF сопоставим с k-fold LightGBM; target-признаков AutoGluon здесь не
    получает (бэггинг по фолдам протёк бы через них). Temporal — holdout-fit
    на прошлом с оценкой на последнем периоде, затем fit на всём train для
    test; target-признаки строятся только по прошлому (fit-строки — по
    более ранним периодам, holdout и test — по всему fit), утечки нет.
    """
    lab = labeled(prof, target)
    y = lab[target].to_numpy()
    folds = outer_folds(prof, lab, y)
    ids = fold_ids(len(lab), folds)
    regression = prof["problem_type"] == "regression"
    cand = {"name": "autogluon", "score": None}
    if (ids >= 0).all():
        predictor = fit_autogluon(prof, target, budget, folds=ids)
        test_pred, test_proba, classes = ag_predict(prof, predictor, target)
        cand.update(test_pred=test_pred, test=test_pred if regression else test_proba, classes=classes)
        oof = ag_oof(predictor, prof, target)
        if oof is None:
            return cand
        rows, y_oof, oof_pred, oof_proba, oof_classes = oof
        cand.update(rows=rows, y=y_oof, oof=oof_pred if regression else align_classes(oof_proba, oof_classes, classes))
    else:
        t0 = time.time()
        tr, va = folds[-1]
        F_tr, F_va = models.target_features(prof, target, lab.iloc[tr], lab.iloc[va])
        holdout = fit_autogluon(prof, target, 0.35 * budget, rows=lab.index[tr], extra=F_tr)
        h_pred, h_proba, h_classes = ag_predict(prof, holdout, target, frame=lab.iloc[va], extra=F_va)
        rest = budget - (time.time() - t0)
        if rest >= 300:
            F_all, F_te = models.target_features(prof, target, lab, prof["test"])
            predictor = fit_autogluon(prof, target, rest, extra=F_all)
        else:
            log("AutoGluon: на fit по всему train не осталось времени — test предсказывает holdout-модель")
            predictor = holdout
            _, F_te = models.target_features(prof, target, lab.iloc[tr], prof["test"])
            if F_te is not None and F_tr is not None:
                F_te = F_te.reindex(columns=F_tr.columns)
        test_pred, test_proba, classes = ag_predict(prof, predictor, target, extra=F_te)
        oof = h_pred if regression else align_classes(h_proba, h_classes, classes)
        cand.update(test_pred=test_pred, test=test_pred if regression else test_proba, classes=classes,
                    rows=lab.index.to_numpy()[va], y=y[va], oof=oof)
    try:
        cand["score"] = _cand_score(prof["metric"], cand)
    except Exception as exc:  # noqa: BLE001
        log(f"AutoGluon: {prof['metric']} по OOF не посчитан ({exc})")
    return cand


def solve_target(prof: dict, target: str, current: dict[str, np.ndarray], budget: float) -> np.ndarray:
    """Решает одну целевую колонку, по ходу обновляя submission.csv.

    budget — секунды на таргет; этапы считают остаток от абсолютного
    времени окончания, поэтому потраченное на ранние этапы учитывается.
    """
    metric = prof["metric"]
    best_pred = current[target]
    quick = None
    t_end = min(time.time() + budget, deadline())

    def remaining() -> float:
        return t_end - time.time()

    # Ступень 1: быстрый LightGBM — ранний валидный сабмит.
    kfold_reserve = KFOLD_RESERVE_SEC
    try:
        tq = time.time()
        quick = quick_model(prof, target, deadline_ts=t_end)
        # k-fold ≈ N_FOLDS + 1 таких обучений (+ target-признаки): резерв по факту, не больше заданного.
        kfold_reserve = float(np.clip(10 * (time.time() - tq), 120, KFOLD_RESERVE_SEC))
        current[target] = quick["test_pred"]
        write_submission(prof, current)
        best_pred = current[target]
        register("quick_lgbm", "градиентный бустинг с дефолтами как опорная точка", quick["val_score"], True)
    except Exception:  # noqa: BLE001
        log("quick_lgbm упал:\n" + traceback.format_exc())

    # Ступень 1.5: AutoEDA — детекторы, гипотезы Qwen, валидатор на фолдах под тест.
    prof.setdefault("base_features", list(prof["features"]))
    eda = {"period": None, "label_copy": False, "accepted": []}
    spare = remaining() - PREDICT_RESERVE_SEC - kfold_reserve
    eda_budget = min(AUTOEDA_SEC, 0.3 * spare)
    if eda_budget >= MIN_AUTOEDA_SEC:
        # AutoEDA работает на копии профиля под лимитом времени: зависшее построение признаков
        # прерывается, а прерванный этап не оставляет профиль наполовину изменённым.
        work = {**prof, "features": list(prof["features"])}
        eda_limit = eda_budget * 1.25 + AUTOEDA_SLACK_SEC
        try:
            with time_limit(eda_limit):
                res = run_autoeda(work, target, eda_budget)
            eda = res
            for k in EDA_KEYS:
                if k in work:
                    prof[k] = work[k]
        except StageTimeout:
            log(f"AutoEDA не уложился в {eda_limit:.0f}s — продолжаю на исходных признаках")
        except Exception:  # noqa: BLE001
            log("AutoEDA упал, работаю на исходных признаках:\n" + traceback.format_exc())
    else:
        log(f"Пропускаю AutoEDA: мало времени ({eda_budget:.0f}s)")
    if eda["accepted"] and quick is not None:
        # Сравнение на одном фолде схемы cv: база — исходные признаки без target-спек и модельных операций.
        try:
            base_view = {**prof, "features": prof["base_features"], "target_specs": [], "model_ops": {}}
            qb = quick_model(base_view, target, use_cv=True, deadline_ts=t_end)
            q2 = quick_model(prof, target, use_cv=True, deadline_ts=t_end)
            better = is_better(metric, q2["val_score"], qb["val_score"])
            register("quick_lgbm_fe", f"quick-модель на признаках AutoEDA (база {qb['val_score']:.5f})",
                     q2["val_score"], better)
            if better:
                quick = q2
                current[target] = quick["test_pred"]
                write_submission(prof, current)
                best_pred = current[target]
        except Exception:  # noqa: BLE001
            log("quick_lgbm на признаках AutoEDA упал:\n" + traceback.format_exc())

    # Ступень 2: AutoGluon на фолдах схемы cv.
    ag = None
    ag_budget = remaining() - PREDICT_RESERVE_SEC - kfold_reserve
    linear_reserve = LINEAR_SEC if linear_search.recipes(prof, target)[0] else 0.0
    # Reserving a cheap family must not disable an otherwise eligible AutoGluon run.
    if ag_budget >= 300:
        ag_budget -= min(linear_reserve, max(0.0, ag_budget - 300))
    log(f"Бюджет AutoGluon {ag_budget:.0f}s (резерв k-fold {kfold_reserve:.0f}s, predict {PREDICT_RESERVE_SEC:.0f}s)")
    if ag_budget >= 300:
        try:
            ag = run_autogluon(prof, target, ag_budget)
        except Exception:  # noqa: BLE001
            log("AutoGluon упал:\n" + traceback.format_exc())
    else:
        log(f"Пропускаю AutoGluon: осталось мало времени ({ag_budget:.0f}s)")
    if ag is not None and ag.get("test_pred") is not None:
        # Немедленная перезапись сабмита: OOF AutoGluon надёжнее одного
        # сплита quick, поэтому AG отбрасываем только при явном (>2%)
        # преимуществе quick. Финальный выбор ниже — на общих OOF-строках.
        ag_score = ag["score"]
        if quick is None:
            take_ag = True
        elif ag_score is None:
            log("AutoGluon: OOF-оценка недоступна, сабмит quick остаётся")
            take_ag = False
        else:
            qs = quick["val_score"]
            margin = 0.02 * abs(qs) if qs else 1e-9
            shifted = qs - margin if models.HIGHER_BETTER[metric] else qs + margin
            take_ag = not is_better(metric, shifted, ag_score)
        register("autogluon", "ансамбль AutoGluon против опорного бустинга", ag_score, take_ag)
        if take_ag:
            current[target] = ag["test_pred"]
            write_submission(prof, current)
            best_pred = current[target]

    # Ступень 3: k-fold LightGBM на тех же фолдах — база сравнения и материал для бленда.
    cands = [ag] if ag is not None and ag.get("rows") is not None else []
    if remaining() > PREDICT_RESERVE_SEC * 0.5:
        try:
            kq = quick_oof(prof, target, deadline_ts=t_end)
            if kq is not None:
                register("quick_oof", "k-fold LightGBM: OOF-база для бленда", kq["score"], True)
                cands.append({"name": "quick_kfold", **kq})
        except Exception:  # noqa: BLE001
            log("quick_oof упал:\n" + traceback.format_exc())

    # Independent family: compare raw/log within Ridge, then challenge the tree winner.
    linear = linear_search.search(prof, target, min(linear_reserve, remaining() - PREDICT_RESERVE_SEC),
                                  register=register)

    # Ступень 4: бленд, выбор финального источника и постпроцессинг.
    try:
        return select_final(prof, target, current, quick, cands, best_pred, label_copy=eda["label_copy"],
                            extra_cands=[linear] if linear is not None else [])
    except Exception:  # noqa: BLE001
        log("Финальный выбор упал, остаётся последний записанный сабмит:\n" + traceback.format_exc())
        return best_pred


def main() -> None:
    task_path = Path(sys.argv[1] if len(sys.argv) > 1 else "task.md").resolve()
    log(f"Старт. task={task_path}, бюджет={TOTAL_BUDGET_SEC:.0f}s, CPU={models.N_CPUS}")
    start_watchdog(task_path.parent)

    prof = build_profile(task_path)
    targets = prof["target_cols"]

    # Ступень 0: константный сабмит — файл существует с первой минуты.
    current = {t: constant_predictions(prof, t) for t in targets}
    write_submission(prof, current)
    register("constant", "медиана/мода как страховка формата", None, True)

    # Признаки AutoEDA и модельные операции одного таргета не переносятся на другой.
    base_state = {"train": prof["train"], "test": prof["test"], "features": list(prof["features"])}
    for i, t in enumerate(targets):
        prof.update(base_state, features=list(base_state["features"]), target=t, **prof["per_target"][t])
        for k in ("base_features", "model_ops", "target_specs", "cv"):
            prof.pop(k, None)
        # Бюджет — от фактически оставшегося времени на оставшиеся таргеты.
        budget = time_left() / (len(targets) - i)
        try:
            solve_target(prof, t, current, budget)
        except Exception:  # noqa: BLE001
            log(f"solve_target({t}) упал:\n" + traceback.format_exc())
        finally:
            models.cleanup_autogluon()

    log("Журнал экспериментов:")
    for e in EXPERIMENTS:
        log(f"  - {e}")
    log("Готово: submission.csv сформирован.")


if __name__ == "__main__":
    workdir = Path(sys.argv[1] if len(sys.argv) > 1 else "task.md").resolve().parent
    try:
        main()
    except Exception:  # noqa: BLE001
        # Последний рубеж: любая ошибка не должна оставить нас без файла.
        log("Фатальная ошибка:\n" + traceback.format_exc())
        shutdown(0 if (workdir / "submission.csv").exists() else 1, workdir)
    # Фоновые потоки (LLM, брошенные по таймауту этапы) и дочерние процессы не удерживают процесс.
    shutdown(0, workdir)
