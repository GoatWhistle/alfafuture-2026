"""Профиль задачи: поиск файлов, определение id/target, типа задачи и метрики.

Порядок доверия источникам:
1. sample_submission.csv — истина про id-колонку, целевые колонки и формат вывода;
2. data_dictionary.csv — роли колонок (кросс-чек);
3. dtype/уникальность таргета в train.csv — жёсткие факты о типе задачи;
4. LLM (Qwen через OPENAI_*) — чтение task.md: тип задачи и метрика;
5. дефолты по типу задачи — последний рубеж, если LLM недоступна.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from agent_utils import log

# Канонические имена -> eval_metric AutoGluon.
AG_METRIC = {
    "rmse": "root_mean_squared_error",
    "rmsle": "root_mean_squared_error",  # обучаем на log1p(y)
    "mae": "mean_absolute_error",
    "mape": "mean_absolute_percentage_error",
    "r2": "r2",
    "f1_macro": "f1_macro",
    "f1_weighted": "f1_weighted",
    "f1": "f1",
    "roc_auc": "roc_auc",
    "log_loss": "log_loss",
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
}

REGRESSION_METRICS = {"rmse", "rmsle", "mae", "mape", "r2"}
PROBLEM_TYPES = ("regression", "binary", "multiclass")


def _find_csv(
    workdir: Path, keys: list[str], exclude: list[str] | None = None, recursive: bool = False
) -> Path | None:
    exclude = exclude or []
    it = workdir.rglob("*.csv") if recursive else workdir.glob("*.csv")
    candidates = []
    for p in sorted(it):
        name = p.name.lower()
        if name in exclude:
            continue
        if any(k in name for k in keys):
            candidates.append(p)
    if not candidates:
        return None
    # Самое короткое имя — обычно каноническое (train.csv, а не train_extra.csv).
    return sorted(candidates, key=lambda p: (len(p.name), len(str(p))))[0]


def _paths_from_text(task_text: str, workdir: Path) -> dict[str, Path]:
    """Пути к csv, явно упомянутые в task.md (относительно каталога task.md)."""
    found: dict[str, Path] = {}
    for token in task_text.replace("`", " ").replace("(", " ").replace(")", " ").split():
        token = token.strip(".,;:!?'\"«»—")
        if not token.lower().endswith(".csv"):
            continue
        p = (workdir / token.lstrip("./")).resolve()
        if not p.is_file():
            continue
        name = p.name.lower()
        for key in ("train", "test", "sample", "dictionary", "data_dict"):
            if key in name and key not in found:
                found[key] = p
    return found


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def _hard_facts(y: pd.Series) -> dict:
    """Неоспоримые свойства таргета, против которых LLM идти не может."""
    y = y.dropna()
    nunique = int(y.nunique())
    is_text = y.dtype == object or str(y.dtype) in ("category", "bool")
    is_integer = False
    if not is_text:
        try:
            is_integer = pd.api.types.is_integer_dtype(y) or bool((y % 1 == 0).all())
        except TypeError:
            is_integer = False
    return {"nunique": nunique, "is_text": is_text, "is_integer": is_integer}


def _default_problem_type(facts: dict) -> str:
    if facts["nunique"] <= 2:
        return "binary"
    if facts["is_text"]:
        return "multiclass"
    if facts["is_integer"] and facts["nunique"] <= 20:
        return "multiclass"
    return "regression"


def _validate_problem_type(ptype: str | None, facts: dict) -> str | None:
    """Отклоняет невозможные ответы LLM: текстовый таргет — не регрессия и т.п."""
    if ptype not in PROBLEM_TYPES:
        return None
    if facts["nunique"] <= 2:
        return "binary"
    if ptype == "binary" and facts["nunique"] > 2:
        return "multiclass"
    if ptype == "regression" and facts["is_text"]:
        return None
    if ptype == "multiclass" and not facts["is_text"] and facts["nunique"] > 200:
        return None
    return ptype


def _default_metric(problem_type: str, y: pd.Series) -> str:
    if problem_type == "regression":
        return "rmse"
    if problem_type == "multiclass":
        return "f1_macro"
    vc = y.dropna().value_counts(normalize=True)
    return "f1" if len(vc) and vc.min() < 0.3 else "accuracy"


def _extract_json(text: str) -> dict | None:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _llm_json(prompt: str, tag: str, timeout: float = 60) -> dict | None:
    """Один JSON-ответ LLM: до 2 попыток, любая ошибка -> None."""
    if not os.environ.get("OPENAI_API_KEY"):
        log(f"LLM[{tag}]: OPENAI_API_KEY не задан, работаю на дефолтах")
        return None
    try:
        from openai import OpenAI

        client = OpenAI(timeout=timeout, max_retries=1)
        # Прокси платформы подставляет модель задачи по любому имени.
        model = os.environ.get("OPENAI_MODEL", "qwen/qwen3.8-27b")
        for attempt in (1, 2):
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            data = _extract_json(resp.choices[0].message.content or "")
            if data:
                log(f"LLM[{tag}] (попытка {attempt}): {data}")
                return data
            prompt = "Верни ТОЛЬКО JSON, без рассуждений.\n" + prompt
        log(f"LLM[{tag}]: не удалось получить JSON за 2 попытки")
        return None
    except Exception as exc:  # noqa: BLE001 — решение обязано работать и без LLM
        log(f"LLM[{tag}] недоступна, работаю на дефолтах: {exc}")
        return None


def _llm_parse(task_text: str, columns: list[str], target: str, facts: dict) -> dict | None:
    """Чтение task.md силами LLM: тип задачи и метрика. Любая ошибка -> None."""
    prompt = (
        "Прочитай описание ML-задачи и верни СТРОГО один JSON без пояснений вида "
        '{"task_type": "regression|binary|multiclass", "metric": "<имя>", '
        '"target_column": "<колонка>"}. '
        f"Допустимые значения metric: {sorted(AG_METRIC)}. "
        "Выбирай метрику, названную в тексте ОСНОВНОЙ (основная метрика, метрика оценки); "
        "справочные и дополнительные метрики игнорируй. "
        f"Таргет {target!r}: уникальных значений {facts['nunique']}, "
        f"текстовый={facts['is_text']}. Колонки train: {columns[:80]}.\n\n"
        "Описание задачи:\n" + task_text[:8000]
    )
    return _llm_json(prompt, "task.md")


# ---------------------------------------------------------------------------
# План экспериментов по фактам EDA-зонда.
# ---------------------------------------------------------------------------

# Каталог блоков: LLM выбирает и упорядочивает, но не пишет код.
EDA_BLOCKS = {
    "fe_group": "регрессия: cross-fitted статистики таргета (mean/median/std/count) по группе "
                "key_columns — когда квази-ID здания/объекта повторяется в test и сжимает std таргета; "
                'params: {"key_columns": [1-4 колонки]}',
    "temporal_holdout": "оценка и выбор моделей на последнем периоде train, когда test — будущий период; "
                        'params: {"column": "<колонка периода>"}',
    "fe_log_ratio": "генерация признаков: log1p тяжёлых хвостов + отношения топ-важных числовых",
    "label_copy_dups": "классификация: копировать метку train в test для точных дублей признаков "
                       "без конфликта меток",
    "class_weights": "классификация с macro-метрикой: подбор классовых множителей вероятностей",
    "blend": "смесь AutoGluon и k-fold LightGBM по OOF",
}
REGRESSION_ONLY = {"fe_group"}
CLASSIFICATION_ONLY = {"label_copy_dups", "class_weights"}


def _validate_plan(raw, columns: set[str], problem_type: str) -> list[dict] | None:
    """Отбрасывает неизвестные/неприменимые блоки и параметры с несуществующими колонками."""
    if not isinstance(raw, list):
        return None
    plan, seen = [], set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        name = item.get("block")
        if name not in EDA_BLOCKS or name in seen:
            continue
        if problem_type == "regression" and name in CLASSIFICATION_ONLY:
            continue
        if problem_type != "regression" and name in REGRESSION_ONLY:
            continue
        params = item.get("params") if isinstance(item.get("params"), dict) else {}
        if name == "fe_group":
            keys = params.get("key_columns")
            if not isinstance(keys, list) or not 1 <= len(keys) <= 4 or not set(keys) <= columns:
                log(f"план: fe_group с невалидным ключом {keys!r} отброшен")
                continue
            params = {"key_columns": list(dict.fromkeys(keys))}
        elif name == "temporal_holdout":
            col = params.get("column")
            if col not in columns:
                log(f"план: temporal_holdout с невалидной колонкой {col!r} отброшен")
                continue
            params = {"column": col}
        else:
            params = {}
        seen.add(name)
        plan.append({"block": name, "params": params, "why": str(item.get("why", ""))[:300]})
    return plan or None


def llm_eda_plan(task_text: str, facts: dict, columns: list[str]) -> list[dict] | None:
    """Qwen интерпретирует факты зонда: упорядоченный список блоков из каталога."""
    ptype = facts.get("problem_type")
    catalog = "\n".join(f"- {k}: {v}" for k, v in EDA_BLOCKS.items())
    prompt = (
        "Ты планируешь эксперименты AutoML-агента для табличной задачи. Ниже факты "
        "автоматического EDA и каталог блоков. Выбери полезные блоки и упорядочь их. "
        "Используй только блоки из каталога; колонки — только из фактов. "
        "Каждый блок затем проверяется на валидации, так что включай блок, если факты его поддерживают. "
        'Верни СТРОГО один JSON: {"plan": [{"block": "<имя>", "params": {...}, "why": "<кратко>"}]}.\n\n'
        f"Тип задачи: {ptype}, метрика: {facts.get('metric')}.\n"
        f"Каталог блоков:\n{catalog}\n\n"
        f"Факты EDA (JSON):\n{json.dumps(facts, ensure_ascii=False)[:9000]}\n\n"
        "Описание задачи:\n" + task_text[:4000]
    )
    data = _llm_json(prompt, "eda_plan", timeout=90)
    if not data:
        return None
    return _validate_plan(data.get("plan"), set(columns), ptype)


def default_eda_plan(facts: dict) -> list[dict]:
    """Пороговый план без LLM — те же блоки по тем же фактам."""
    ptype = facts.get("problem_type")
    plan = []
    qid = (facts.get("quasi_id") or {}).get("best")
    if ptype == "regression" and qid and qid.get("coverage", 0) > 0.4 and (qid.get("compression") or 0) > 3:
        plan.append({
            "block": "fe_group", "params": {"key_columns": qid["key"]},
            "why": f"квази-ID покрывает {qid['coverage']:.0%} test, сжатие std {qid['compression']}×",
        })
    temporal = facts.get("temporal_candidates") or []
    if temporal:
        col = temporal[0]["column"]
        plan.append({"block": "temporal_holdout", "params": {"column": col}, "why": f"test — новый период по {col}"})
    dups = facts.get("duplicates") or {}
    if ptype != "regression" and dups.get("test_in_train_share", 0) > 0.02:
        plan.append({
            "block": "label_copy_dups", "params": {},
            "why": f"{dups['test_in_train_share']:.1%} строк test — точные дубли train",
        })
    plan.append({"block": "fe_log_ratio", "params": {}, "why": "базовая генерация признаков"})
    plan.append({"block": "blend", "params": {}, "why": "всегда"})
    if ptype != "regression":
        plan.append({"block": "class_weights", "params": {}, "why": "всегда для macro-метрик"})
    return plan


def build_profile(task_path: Path) -> dict:
    workdir = task_path.parent
    task_text = task_path.read_text(encoding="utf-8", errors="ignore") if task_path.exists() else ""

    # Приоритет: пути из текста task.md -> каталог task.md -> рекурсивный поиск.
    text_paths = _paths_from_text(task_text, workdir)
    train_p = text_paths.get("train") or _find_csv(workdir, ["train"]) or _find_csv(workdir, ["train"], recursive=True)
    test_p = text_paths.get("test") or _find_csv(workdir, ["test"]) or _find_csv(workdir, ["test"], recursive=True)
    sample_p = (
        text_paths.get("sample")
        or _find_csv(workdir, ["sample"])
        or _find_csv(workdir, ["sample"], recursive=True)
        or _find_csv(workdir, ["submission"], exclude=["submission.csv"], recursive=True)
    )
    dict_p = (
        text_paths.get("dictionary")
        or text_paths.get("data_dict")
        or _find_csv(workdir, ["dictionary", "data_dict"])
        or _find_csv(workdir, ["dictionary", "data_dict"], recursive=True)
    )
    if train_p is None or test_p is None:
        raise FileNotFoundError(f"Не найдены train/test csv в {workdir}")
    log(f"Файлы: train={train_p.name}, test={test_p.name}, "
        f"sample={getattr(sample_p, 'name', None)}, dict={getattr(dict_p, 'name', None)}")

    train = _read_csv(train_p)
    test = _read_csv(test_p)
    sample = _read_csv(sample_p) if sample_p else None

    # id и целевые колонки.
    if sample is not None and len(sample.columns) >= 2:
        id_col = sample.columns[0]
        target_cols = [c for c in sample.columns[1:]]
    else:
        id_col = train.columns[0]
        target_cols = [c for c in train.columns if c not in test.columns and c != id_col][:1]

    # Кросс-чек по data_dictionary (role: id/target).
    if dict_p is not None:
        try:
            dd = _read_csv(dict_p)
            cols_lower = {c.lower(): c for c in dd.columns}
            if "role" in cols_lower and "feature" in cols_lower:
                roles = dd.set_index(cols_lower["feature"])[cols_lower["role"]].astype(str).str.lower()
                dd_targets = [f for f, r in roles.items() if "target" in r and f in train.columns]
                if dd_targets and set(dd_targets) != set(target_cols):
                    log(f"data_dictionary указывает target={dd_targets}, sample={target_cols}")
                    if not target_cols or target_cols[0] not in train.columns:
                        target_cols = dd_targets
        except Exception as exc:  # noqa: BLE001
            log(f"Не удалось разобрать data_dictionary: {exc}")

    target_cols = [c for c in target_cols if c in train.columns]
    if not target_cols:
        raise ValueError("Не удалось определить целевую колонку")
    target = target_cols[0]

    features = [c for c in train.columns if c != id_col and c not in target_cols and c in test.columns]

    # Тип задачи и метрика: LLM читает task.md, данные ограничивают, дефолты страхуют.
    facts = _hard_facts(train[target])
    problem_type = None
    metric = None
    llm = _llm_parse(task_text, list(train.columns), target, facts)
    if llm:
        problem_type = _validate_problem_type(llm.get("task_type"), facts)
        llm_metric = llm.get("metric")
        if llm_metric in AG_METRIC:
            metric = llm_metric
    if problem_type is None:
        problem_type = _default_problem_type(facts)
        log(f"Тип задачи взят из свойств данных: {problem_type}")

    # Согласованность метрики и типа задачи + дефолты.
    if metric is not None:
        if (metric in REGRESSION_METRICS) != (problem_type == "regression"):
            log(f"Метрика {metric!r} несовместима с типом {problem_type!r}, беру дефолт")
            metric = None
    if metric is None:
        metric = _default_metric(problem_type, train[target])

    # Формат вывода: метки или вероятности (только для бинарной задачи).
    output_proba = False
    if sample is not None and problem_type == "binary" and len(target_cols) == 1:
        sv = sample[target].dropna()
        if len(sv) and pd.api.types.is_float_dtype(sv):
            labels = set(train[target].dropna().unique().tolist())
            if not set(sv.unique().tolist()) <= labels:
                output_proba = True
    if metric == "roc_auc" and problem_type == "binary" and sample is not None:
        sv = sample[target].dropna()
        if len(sv) and pd.api.types.is_numeric_dtype(sv):
            output_proba = True

    profile = {
        "workdir": workdir,
        "task_text": task_text,
        "train": train,
        "test": test,
        "sample": sample,
        "id_col": id_col,
        "target_cols": target_cols,
        "target": target,
        "features": features,
        "problem_type": problem_type,
        "metric": metric,
        "ag_metric": AG_METRIC[metric],
        "log_target": metric == "rmsle",
        "output_proba": output_proba,
    }
    log(
        f"Профиль: target={target!r}, id={id_col!r}, type={problem_type}, metric={metric}, "
        f"features={len(features)}, train={train.shape}, test={test.shape}, proba={output_proba}"
    )
    return profile
