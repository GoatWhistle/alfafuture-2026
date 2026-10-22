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

from agent_utils import TOTAL_BUDGET_SEC, log, time_left
import eda_detectors
import fe_dsl
import fe_search
import models
from models import (
    ag_oof,
    ag_predict,
    align_classes,
    clip_to_train_range,
    constant_predictions,
    copy_duplicate_labels,
    fit_autogluon,
    is_better,
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
AUTOEDA_SEC = float(os.environ.get("AGENT_AUTOEDA_SEC", 900))
MACRO_METRICS = ("f1_macro", "f1_weighted", "f1", "balanced_accuracy")
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


def _llm_round1(prof: dict, ctx: dict, summary: dict, known: set[str]) -> list[dict]:
    proposal = llm_propose_specs(prof["task_text"], summary, prof.get("data_dictionary"), list(ctx["features"]))
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
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutTimeout

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
    pool = ThreadPoolExecutor(1)
    fut = pool.submit(_llm_round1, prof, ctx, summary, known) if use_llm else None
    srch.stage_a(specs, t0 + 0.5 * budget)
    if fut is not None:
        try:
            qwen = fut.result(timeout=max(5.0, t0 + 0.6 * budget - time.time()))
        except FutTimeout:
            log("LLM раунд 1 не успел — продолжаю без него")
            qwen = []
        except Exception:  # noqa: BLE001
            log("LLM раунд 1 упал:\n" + traceback.format_exc())
            qwen = []
        srch.stage_a(qwen, t0 + 0.65 * budget)
        if qwen and time.time() < t0 + 0.65 * budget:
            try:
                ref = llm_refine_specs(prof["task_text"], summary, srch.feedback(), list(ctx["features"]))
                qwen2 = _validate_llm(ref["specs"], ctx, known, "qwen2") if ref else []
            except Exception:  # noqa: BLE001
                log("LLM раунд 2 упал:\n" + traceback.format_exc())
                qwen2 = []
            srch.stage_a(qwen2, t0 + 0.8 * budget)
    pool.shutdown(wait=False)
    accepted = srch.stage_b(deadline)

    added = []
    model_ops = {}
    for spec in accepted:
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
        res = fe_dsl.materialize(prof, spec, target, ctx["classes"])
        if res is None:
            continue
        tr_new, te_new = res
        new_cols = [c for c in tr_new.columns if c not in prof["train"].columns]
        prof["train"] = pd.concat([prof["train"], tr_new[new_cols]], axis=1)
        prof["test"] = pd.concat([prof["test"], te_new[new_cols]], axis=1)
        prof["features"] = list(prof["features"]) + new_cols
        added += new_cols
    prof["model_ops"] = model_ops
    register("autoeda", f"схема {ctx['fold_scheme']}; принято {len(accepted)} спек -> {added}",
             ctx.get("search_final"), bool(accepted))
    log(f"AutoEDA: {time.time() - t0:.0f}s, признаков {len(prof['features'])}")
    return {"period": ctx["period"], "label_copy": ctx["label_copy"], "accepted": accepted}


def temporal_mask(prof: dict, target: str, col: str | None) -> np.ndarray | None:
    """Маска строк последнего периода train (в порядке OOF) для temporal_holdout."""
    if col is None:
        return None
    train = prof["train"]
    vals = pd.to_numeric(train.loc[train[target].notna(), col], errors="coerce")
    if vals.notna().sum() == 0:
        log(f"temporal_holdout: колонка {col} не числовая, пропускаю")
        return None
    mask = (vals == vals.max()).to_numpy()
    if mask.sum() < 500 or mask.all():
        log(f"temporal_holdout: в последнем периоде {col}={vals.max()} {mask.sum()} строк, пропускаю")
        return None
    log(f"temporal_holdout: оценка на {col}={vals.max()} ({mask.sum()} строк)")
    return mask


def _cand_score(metric: str, c: dict, mask: np.ndarray) -> float:
    y, oof = c["y"][mask], c["oof"][mask]
    if c.get("classes") is None:
        return score(metric, y, oof)
    pred = np.array(c["classes"], dtype=object)[np.argmax(oof, axis=1)]
    return score(metric, y, pred, proba=oof, labels=c["classes"])


def _apply_mask(metric: str, cands: list[dict], mask: np.ndarray | None) -> np.ndarray | None:
    """Переводит скоры кандидатов на строки маски (temporal); полный OOF — в лог."""
    if mask is None or not cands or any(len(c["y"]) != len(mask) for c in cands):
        return None
    for c in cands:
        full = c["score"]
        c["score"] = _cand_score(metric, c, mask)
        log(f"{c['name']}: OOF {metric}={full:.5f}, temporal {metric}={c['score']:.5f}")
    return mask


def _select_final_regression(prof, target, current, ag, kq, fallback, eval_mask=None):
    metric = prof["metric"]
    ag_score, ag_test_pred, _, ag_y, ag_oof_pred, _, _ = ag
    cands = []
    if ag_score is not None and ag_test_pred is not None:
        cands.append({"name": "autogluon", "score": ag_score, "oof": ag_oof_pred, "test": ag_test_pred, "y": ag_y})
    if kq is not None:
        cands.append({"name": "quick_kfold", "score": kq["score"], "oof": kq["oof"], "test": kq["test"], "y": kq["y"]})
    m = _apply_mask(metric, cands, eval_mask)
    if len(cands) == 2 and len(cands[0]["y"]) == len(cands[1]["y"]):
        y = cands[0]["y"]
        if m is None:
            alpha, s = tune_blend(metric, y, cands[0]["oof"], cands[1]["oof"])
        else:
            alpha, s = tune_blend(metric, y[m], cands[0]["oof"][m], cands[1]["oof"][m])
        accepted = all(is_better(metric, s, c["score"]) or s == c["score"] for c in cands)
        register("blend", f"смесь AutoGluon и k-fold LGBM, alpha={alpha:.2f}" + (" (temporal)" if m is not None else ""),
                 s, accepted)
        cands.append({
            "name": "blend", "score": s, "y": y,
            "oof": alpha * cands[0]["oof"] + (1 - alpha) * cands[1]["oof"],
            "test": alpha * cands[0]["test"] + (1 - alpha) * cands[1]["test"],
        })
    if not cands:
        return fallback
    win = cands[0]
    for c in cands[1:]:
        if is_better(metric, c["score"], win["score"]):
            win = c
    log(f"Финальный источник: {win['name']} ({metric}={win['score']:.5f})")
    test_final, base, new = clip_to_train_range(win["y"], win["oof"], win["test"], metric, eval_mask=m)
    register("clip_range", "обрезка предсказаний к диапазону таргета в train", new, new != base)
    current[target] = test_final
    write_submission(prof, current)
    return test_final


def _select_final_classification(prof, target, current, quick, ag, kq, fallback, eval_mask=None, label_copy=False):
    metric = prof["metric"]
    ag_score, _, ag_test_proba, ag_y, _, ag_oof_proba, ag_classes = ag
    cands = []
    if ag_score is not None and ag_oof_proba is not None and ag_test_proba is not None:
        cands.append({
            "name": "autogluon", "score": ag_score, "y": ag_y,
            "oof": ag_oof_proba, "test": ag_test_proba, "classes": list(ag_classes),
        })
    if kq is not None:
        koof, kte, kcls = kq["oof"], kq["test"], list(kq["classes"])
        if cands and set(kcls) == set(cands[0]["classes"]):
            koof = align_classes(koof, kcls, cands[0]["classes"])
            kte = align_classes(kte, kcls, cands[0]["classes"])
            kcls = cands[0]["classes"]
        cands.append({"name": "quick_kfold", "score": kq["score"], "y": kq["y"], "oof": koof, "test": kte, "classes": kcls})
    m = _apply_mask(metric, cands, eval_mask)
    if len(cands) == 2 and cands[0]["classes"] == cands[1]["classes"] and len(cands[0]["y"]) == len(cands[1]["y"]):
        y = cands[0]["y"]
        rows = slice(None) if m is None else m
        alpha, s = tune_blend(metric, y[rows], cands[0]["oof"][rows], cands[1]["oof"][rows], classes=cands[0]["classes"])
        accepted = all(is_better(metric, s, c["score"]) or s == c["score"] for c in cands)
        register("blend", f"смесь AutoGluon и k-fold LGBM, alpha={alpha:.2f}" + (" (temporal)" if m is not None else ""),
                 s, accepted)
        cands.append({
            "name": "blend", "score": s, "y": y, "classes": cands[0]["classes"],
            "oof": alpha * cands[0]["oof"] + (1 - alpha) * cands[1]["oof"],
            "test": alpha * cands[0]["test"] + (1 - alpha) * cands[1]["test"],
        })
    if not cands:
        return _tuned_or(tune_quick_weights(prof, quick, target, current), fallback)
    win = cands[0]
    for c in cands[1:]:
        if is_better(metric, c["score"], win["score"]):
            win = c
    log(f"Финальный источник: {win['name']} ({metric}={win['score']:.5f})")

    classes = win["classes"]
    classes_arr = np.array(classes, dtype=object)
    if prof["output_proba"]:
        pos = sorted(classes)[-1]
        vals = win["test"][:, classes.index(pos)]
        current[target] = vals
        write_submission(prof, current)
        return vals

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
            out, base, new, n = copy_duplicate_labels(prof, target, win["y"], oof_pred, pred, metric)
            register("label_copy_dups", f"метки дублей train для {n} строк test (LOO base={base})",
                     new, out is not None)
            if out is not None:
                changed = int((np.asarray(out, dtype=object) != np.asarray(pred, dtype=object)).sum())
                log(f"label_copy_dups: изменено {changed} предсказаний test")
                pred = out
                current[target] = pred
                write_submission(prof, current)
        except Exception:  # noqa: BLE001
            log("label_copy_dups упал:\n" + traceback.format_exc())
    return pred


def solve_target(prof: dict, target: str, current: dict[str, np.ndarray], budget: float) -> np.ndarray:
    """Решает одну целевую колонку, по ходу обновляя submission.csv."""
    metric = prof["metric"]
    best_pred = current[target]
    quick = None

    # Ступень 1: быстрый LightGBM — ранний валидный сабмит.
    try:
        quick = quick_model(prof, target)
        best_pred = quick["test_pred"]
        current[target] = best_pred
        write_submission(prof, current)
        register("quick_lgbm", "градиентный бустинг с дефолтами как опорная точка", quick["val_score"], True)
    except Exception:  # noqa: BLE001
        log("quick_lgbm упал:\n" + traceback.format_exc())

    # Ступень 1.5: AutoEDA — детекторы, гипотезы Qwen, валидатор на фолдах под тест.
    prof.setdefault("base_features", list(prof["features"]))
    eda = {"period": None, "label_copy": False, "accepted": []}
    spare = min(budget, time_left()) - PREDICT_RESERVE_SEC - KFOLD_RESERVE_SEC
    eda_budget = min(AUTOEDA_SEC, 0.3 * spare)
    if eda_budget >= 120:
        try:
            eda = run_autoeda(prof, target, eda_budget)
        except Exception:  # noqa: BLE001
            log("AutoEDA упал, работаю на исходных признаках:\n" + traceback.format_exc())
    else:
        log(f"Пропускаю AutoEDA: мало времени ({eda_budget:.0f}s)")
    if eda["accepted"] and quick is not None:
        try:
            q2 = quick_model(prof, target)
            better = is_better(metric, q2["val_score"], quick["val_score"])
            register("quick_lgbm_fe", "quick-модель на признаках AutoEDA", q2["val_score"], better)
            quick = q2
            if better:
                best_pred = quick["test_pred"]
                current[target] = best_pred
                write_submission(prof, current)
        except Exception:  # noqa: BLE001
            log("quick_lgbm на признаках AutoEDA упал:\n" + traceback.format_exc())

    # Ступень 2: AutoGluon.
    predictor = None
    ag_budget = min(budget, time_left()) - PREDICT_RESERVE_SEC - KFOLD_RESERVE_SEC
    if ag_budget >= 300:
        try:
            predictor = fit_autogluon(prof, target, ag_budget)
        except Exception:  # noqa: BLE001
            log("AutoGluon.fit упал:\n" + traceback.format_exc())
    else:
        log(f"Пропускаю AutoGluon: осталось мало времени ({ag_budget:.0f}s)")

    ag_test_pred = ag_test_proba = None
    ag_y = ag_oof_pred = ag_oof_proba = ag_classes = None
    ag_score = None
    if predictor is not None:
        try:
            ag_test_pred, ag_test_proba = ag_predict(prof, predictor, target)
        except Exception:  # noqa: BLE001
            log("AutoGluon.predict упал:\n" + traceback.format_exc())
        if ag_test_pred is not None:
            oof = ag_oof(predictor, prof, target)
            if oof is not None:
                ag_y, ag_oof_pred, ag_oof_proba, ag_classes = oof
                try:
                    ag_score = models.score(metric, ag_y, ag_oof_pred, proba=ag_oof_proba, labels=ag_classes)
                except Exception:  # noqa: BLE001
                    ag_score = None
            # Немедленная перезапись сабмита: OOF AutoGluon надёжнее одного
            # 15%-сплита quick, поэтому AG отбрасываем только при явном (>2%)
            # преимуществе quick. Финальный выбор ниже — по единой OOF-базе.
            if ag_score is None or quick is None:
                take_ag = True
            else:
                qs = quick["val_score"]
                margin = 0.02 * abs(qs) if qs else 1e-9
                shifted = qs - margin if models.HIGHER_BETTER[metric] else qs + margin
                take_ag = not is_better(metric, shifted, ag_score)
            register("autogluon", "ансамбль AutoGluon против опорного бустинга", ag_score, take_ag)
            if take_ag:
                best_pred = ag_test_pred
                current[target] = best_pred
                write_submission(prof, current)

    # Ступень 3: k-fold OOF quick-модели — честная база сравнения и материал для бленда.
    kq = None
    if time_left() > PREDICT_RESERVE_SEC * 0.5:
        try:
            kq = quick_oof(prof, target)
            if kq is not None:
                register("quick_oof", "k-fold LightGBM: OOF-база для бленда", kq["score"], True)
        except Exception:  # noqa: BLE001
            log("quick_oof упал:\n" + traceback.format_exc())

    # Ступень 4: бленд, выбор финального источника и постпроцессинг.
    ag_pack = (ag_score, ag_test_pred, ag_test_proba, ag_y, ag_oof_pred, ag_oof_proba, ag_classes)
    mask = None
    try:
        mask = temporal_mask(prof, target, eda["period"])
    except Exception:  # noqa: BLE001
        log("temporal_holdout упал:\n" + traceback.format_exc())
    try:
        if prof["problem_type"] == "regression":
            return _select_final_regression(prof, target, current, ag_pack, kq, best_pred, eval_mask=mask)
        return _select_final_classification(
            prof, target, current, quick, ag_pack, kq, best_pred,
            eval_mask=mask, label_copy=eda["label_copy"],
        )
    except Exception:  # noqa: BLE001
        log("Финальный выбор упал, остаётся последний записанный сабмит:\n" + traceback.format_exc())
        return best_pred


def main() -> None:
    task_path = Path(sys.argv[1] if len(sys.argv) > 1 else "task.md").resolve()
    log(f"Старт. task={task_path}, бюджет={TOTAL_BUDGET_SEC:.0f}s")

    prof = build_profile(task_path)
    targets = prof["target_cols"]

    # Ступень 0: константный сабмит — файл существует с первой минуты.
    current = {t: constant_predictions(prof, t) for t in targets}
    write_submission(prof, current)
    register("constant", "медиана/мода как страховка формата", None, True)

    per_target_budget = time_left() / len(targets)
    for t in targets:
        if len(targets) > 1:
            # Профиль одноколоночных функций указывает на prof["target"].
            prof["target"] = t
            prof["log_target"] = prof["metric"] == "rmsle"
        try:
            solve_target(prof, t, current, per_target_budget)
        except Exception:  # noqa: BLE001
            log(f"solve_target({t}) упал:\n" + traceback.format_exc())

    log("Журнал экспериментов:")
    for e in EXPERIMENTS:
        log(f"  - {e}")
    log("Готово: submission.csv сформирован.")


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001
        # Последний рубеж: любая ошибка не должна оставить нас без файла.
        log("Фатальная ошибка:\n" + traceback.format_exc())
        sys.exit(1)
