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
# Семантические гипотезы FE от LLM (спеки DSL, без кода).
# ---------------------------------------------------------------------------

DSL_HELP = """Операции DSL (cols — колонки train; target — таргет):
- group_stat: статистика по группе; cols=[] -> таргет (cross-fitted), cols=[col] -> признак;
  params {"keys": [1-5 колонок], "aggs": ["mean","median","std","count"], "bin": <число квантильных бинов числовых ключей>}
- group_dev: cols=[col] минус среднее col по группе; params {"keys": [...]}
- lag: история объекта в прошлых периодах; cols=[] -> таргет, cols=[col] -> признак;
  params {"keys": [ключ объекта], "period": "<колонка периода>", "aggs": ["last","mean","count","gap"]}
- ratio | diff | prod: cols=[a, b] — a/b, a-b, a*b
- agg_cols: cols=[однородные колонки], params {"aggs": ["mean","std","min","max","sum","range"]}
- log | rank | count_enc | missing_ind: cols=[col]
- bin: cols=[col], params {"bins": q}; interaction: cols=[a, b], params {"bins": q} — код пары
- knn_target: cols=[числовые], params {"k": 10} — таргет ближайших соседей
- drop: cols=[col] — удалить признак (утечка/дрейф)"""


def _dictionary_text(dd: pd.DataFrame | None, columns: list[str], limit: int = 6000) -> str:
    if dd is None:
        return "нет"
    low = {c.lower(): c for c in dd.columns}
    feat = low.get("feature") or dd.columns[0]
    rows = []
    for _, r in dd.iterrows():
        if str(r[feat]) not in columns:
            continue
        rows.append(" | ".join(str(r[c]) for c in dd.columns if c != low.get("role")))
    return "\n".join(rows)[:limit]


def llm_propose_specs(task_text: str, summary: dict, dd: pd.DataFrame | None, columns: list[str]) -> dict | None:
    """Qwen по сводке детекторов и словарю данных предлагает семантические признаки.

    Возвращает {"specs": [сырые спеки], "entity": [...]|None, "period": str|None}
    или None. Спеки валидирует fe_dsl.validate на стороне вызывающего.
    """
    prompt = (
        "Ты — опытный дата-сайентист. AutoML-агент автоматически нашёл зависимости в табличных данных "
        "(сводка ниже). Предложи до 15 НОВЫХ признаков, опираясь на смысл колонок из словаря данных: "
        "физические величины (например, полное потребление = площадь × удельное), амплитуды, суммы "
        "однородных показателей, отношения, история объекта. Не повторяй спеки детекторов. "
        "Каждая гипотеза проверяется на валидации, бесполезные отбрасываются. "
        "Используй ТОЛЬКО операции DSL и ТОЛЬКО колонки из списка; таргет в признаки не подставляй.\n\n"
        f"{DSL_HELP}\n\n"
        'Верни СТРОГО один JSON: {"specs": [{"op": "...", "cols": [...], "params": {...}, "why": "..."}], '
        '"entity": [колонки ключа объекта или пусто], "period": "<колонка периода или null>"}.\n\n'
        f"Колонки: {columns[:200]}\n\n"
        f"Словарь данных:\n{_dictionary_text(dd, columns)}\n\n"
        f"Сводка детекторов (JSON):\n{json.dumps(summary, ensure_ascii=False, default=str)[:6000]}\n\n"
        "Описание задачи:\n" + task_text[:3000]
    )
    data = _llm_json(prompt, "fe_specs", timeout=120)
    if not data or not isinstance(data.get("specs"), list):
        return None
    entity = data.get("entity") if isinstance(data.get("entity"), list) else None
    period = data.get("period") if isinstance(data.get("period"), str) else None
    return {"specs": data["specs"][:20], "entity": entity, "period": period}


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
    dd = None
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
        "data_dictionary": dd,
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
