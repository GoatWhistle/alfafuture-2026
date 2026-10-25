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
import re
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


def _llm_json(prompt: str, tag: str, timeout: float = 60, retries: int = 1) -> dict | None:
    """Один JSON-ответ LLM: до 2 попыток, любая ошибка -> None.

    Весь вызов ограничен по времени (зависший сокет/прокси не держит агента):
    по истечении лимита поток бросается, ответ — None.
    """
    from agent_utils import run_with_timeout, time_left

    if not os.environ.get("OPENAI_API_KEY"):
        log(f"LLM[{tag}]: OPENAI_API_KEY не задан, работаю на дефолтах")
        return None
    limit = min(2 * (retries + 1) * timeout + 10, max(5.0, time_left()))
    done, data = run_with_timeout(_llm_json_call, limit, prompt, tag, timeout, retries)
    if not done:
        log(f"LLM[{tag}]: нет ответа за {limit:.0f}s, работаю без него")
        return None
    return data


def _llm_json_call(prompt: str, tag: str, timeout: float, retries: int) -> dict | None:
    if not os.environ.get("OPENAI_API_KEY"):
        log(f"LLM[{tag}]: OPENAI_API_KEY не задан, работаю на дефолтах")
        return None
    try:
        from openai import OpenAI

        client = OpenAI(timeout=timeout, max_retries=retries)
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
  params {"keys": [1-5 колонок], "aggs": ["mean","median","std","count"], "bin": <число квантильных бинов числовых ключей>,
  "smooth": <сила сглаживания среднего к общему уровню>, "fallback": [[более общие ключи], ...]}
- group_dev: cols=[col] минус среднее col по группе; params {"keys": [...]}
- lag: история объекта в прошлых периодах; cols=[] -> таргет, cols=[col] -> признак;
  params {"keys": [ключ объекта], "period": "<колонка периода>", "aggs": ["last","mean","count","gap"]}
- ratio | diff | prod: cols=[a, b] — a/b, a-b, a*b
- agg_cols: cols=[однородные колонки], params {"aggs": ["mean","std","min","max","sum","range"]}
- log | rank | count_enc | missing_ind: cols=[col]
- missing_count: cols=[колонки с пропусками] — число пропусков в строке
- date_parts: cols=[строка-дата] — год, месяц, день, день недели, день года, час
- cyclic: cols=[col], params {"period": 360|12|24|7} — sin/cos периодической величины
- as_category: cols=[числовая колонка] — целочисленный код (регион, тип, класс) как категория без порядка
- bin: cols=[col], params {"bins": q}; interaction: cols=[a, b], params {"bins": q} — код пары
- knn_target: cols=[числовые], params {"k": 10} — таргет ближайших соседей
- flag: cols=[col], params {"rule": "zero|nan|level|threshold", "value": <порог для threshold>} — индикатор сегмента
- drop: cols=[col] — удалить признак (утечка/дрейф)
Модельные операции (меняют обучение, а не признаки):
- segment_model: cols=[col], params {"rule": "zero|nan|level|threshold", "value": ...} — отдельные модели
  на каждый сегмент (режим поведения), вместо одной общей
- objective: cols=[], params {"name": "huber|l1|fair"} — функция потерь регрессии (тяжёлый хвост ошибок)"""

# Тема -> (вопрос аналитику, ключи фактов из сводки).
LLM_THEMES = {
    "time_objects": (
        "Разные периоды train/test и повторные наблюдения объектов: какой ключ объекта устойчив между "
        "периодами, какая история объекта (лаги таргета и признаков, статистики по объекту) поможет, "
        "какие признаки дрейфуют и мешают переносу на test.",
        ["period", "entity", "adversarial", "duplicates"],
    ),
    "segments": (
        "Смешанные режимы поведения: по найденным сегментам (смена знака корреляций, смещённые остатки) "
        "реши, что лучше — индикатор сегмента, взаимодействия с ним или отдельные модели на сегменты.",
        ["segments", "top_importances"],
    ),
    "errors": (
        "Ошибки модели: что объединяет худшие строки и путаемые классы, какие признаки полезны несмотря "
        "на слабую корреляцию (перестановочная важность), где метки неоднозначны. Предложи признаки и "
        "изменения обучения против этих ошибок.",
        ["tail", "perm_importance", "label_conflicts", "residual_specs"],
    ),
    "semantics": (
        "Смысл колонок по словарю данных: физические величины (например, полное потребление = площадь × "
        "удельное), амплитуды, суммы и агрегаты однородных показателей, отношения.",
        ["top_importances"],
    ),
}


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


def _specs_prompt(question: str, facts: dict, dd_text: str, columns: list[str], task_text: str, n: int,
                  planned: list[str] | None = None) -> str:
    planned_text = ""
    if planned:
        planned_text = ("Эти гипотезы УЖЕ в очереди проверки — НЕ повторяй их; предлагай другие ключи, "
                        "сочетания, пороги threshold, модельные операции:\n- " + "\n- ".join(planned[:40]) + "\n\n")
    return (
        "Ты — опытный дата-сайентист в команде AutoML-агента. Детекторы агента посчитали факты о данных "
        f"(ниже). Вопрос: {question}\n"
        f"Предложи до {n} НОВЫХ гипотез в виде спек DSL. Каждая проверяется переобучением на валидации, "
        "похожей на test; бесполезные отбрасываются, так что предлагай смело, но по фактам. "
        "Используй ТОЛЬКО операции DSL и ТОЛЬКО колонки из списка; таргет в признаки не подставляй.\n\n"
        f"{DSL_HELP}\n\n"
        'Верни СТРОГО один JSON: {"specs": [{"op": "...", "cols": [...], "params": {...}, "why": "..."}], '
        '"entity": [колонки ключа объекта или пусто], "period": "<колонка периода или null>"}.\n\n'
        f"{planned_text}Колонки: {columns[:200]}\n\n"
        f"Словарь данных:\n{dd_text}\n\n"
        f"Факты (JSON):\n{json.dumps(facts, ensure_ascii=False, default=str)[:6000]}\n\n"
        "Описание задачи:\n" + task_text[:2500]
    )


def _parse_specs(data: dict | None, limit: int) -> dict | None:
    if not data or not isinstance(data.get("specs"), list):
        return None
    entity = data.get("entity") if isinstance(data.get("entity"), list) else None
    period = data.get("period") if isinstance(data.get("period"), str) else None
    return {"specs": data["specs"][:limit], "entity": entity, "period": period}


def llm_propose_specs(task_text: str, summary: dict, dd: pd.DataFrame | None, columns: list[str],
                      planned: list[str] | None = None) -> dict | None:
    """Раунд 1: тематические промпты Qwen параллельно, у каждого только свои факты.

    Возвращает {"specs": [сырые спеки с полем theme], "entity", "period"} или None.
    Спеки валидирует fe_dsl.validate на стороне вызывающего.
    """
    from concurrent.futures import ThreadPoolExecutor

    dd_text = _dictionary_text(dd, columns)
    base = {k: summary.get(k) for k in ("problem_type", "metric", "scheme")}

    def ask(theme: str):
        question, keys = LLM_THEMES[theme]
        facts = {**base, **{k: summary.get(k) for k in keys if summary.get(k) is not None}}
        prompt = _specs_prompt(question, facts, dd_text if theme in ("semantics", "time_objects") else "см. колонки",
                               columns, task_text, 8, planned)
        # Reasoning-модель на 4 параллельных запросах отвечает минутами: один долгий запрос без повтора.
        return theme, _parse_specs(_llm_json(prompt, f"fe_{theme}", timeout=300, retries=0), 12)

    with ThreadPoolExecutor(len(LLM_THEMES)) as pool:
        answers = list(pool.map(ask, LLM_THEMES))
    specs, entity, period = [], None, None
    for theme, ans in answers:
        if not ans:
            continue
        specs += [{**s, "theme": theme} for s in ans["specs"] if isinstance(s, dict)]
        entity = entity or ans["entity"]
        period = period or ans["period"]
    return {"specs": specs, "entity": entity, "period": period} if specs else None


def llm_refine_specs(task_text: str, summary: dict, feedback: dict, columns: list[str],
                     planned: list[str] | None = None) -> dict | None:
    """Раунд 2: Qwen видит, какие гипотезы прошли валидатор, и развивает стратегию."""
    question = (
        "Ниже итоги проверки гипотез (delta_rel — относительный прирост метрики, passed — прошла ли порог "
        f"threshold_rel={feedback.get('threshold_rel')}). Сформулируй общую стратегию: какие направления "
        "работают, какие нет. Предложи уточнения прошедших (другие ключи, агрегаты, сочетания, сегменты) "
        "и новые гипотезы в работающих направлениях; не повторяй отклонённые."
    )
    facts = {"results": feedback, **{k: summary.get(k) for k in ("period", "entity", "segments", "tail",
                                                                  "perm_importance")}}
    data = _llm_json(_specs_prompt(question, facts, "см. колонки", columns, task_text, 10, planned), "fe_refine",
                     timeout=240, retries=0)
    return _parse_specs(data, 12)


# Упоминание метрики в тексте задачи: запасной источник, когда LLM недоступна.
METRIC_PATTERNS = {
    "rmsle": r"\brmsle\b",
    "rmse": r"\brmse\b",
    "mape": r"\bmape\b",
    "mae": r"\bmae\b",
    "r2": r"\br2\b|\br\^2|r²",
    "log_loss": r"log[\s_-]?loss|logloss|кросс-энтропи",
    "roc_auc": r"roc[\s_-]?auc|\bauc\b",
    "f1_macro": r"macro[\s_-]?f1|f1[\s_-]?macro|макро[\s-]?f1",
    "f1_weighted": r"weighted[\s_-]?f1|f1[\s_-]?weighted",
    "balanced_accuracy": r"balanced[\s_-]?accuracy",
    "f1": r"\bf1\b",
    "accuracy": r"\baccuracy\b",
}


def _metric_from_text(task_text: str) -> str | None:
    """Первая по тексту метрика; при равной позиции — более длинное совпадение."""
    best = None
    for name, pat in METRIC_PATTERNS.items():
        m = re.search(pat, task_text, re.IGNORECASE)
        if m and (best is None or (m.start(), -len(m.group())) < best[0]):
            best = ((m.start(), -len(m.group())), name)
    return best[1] if best else None


def _declares_regression(task_text: str) -> bool:
    return bool(re.search(r"регресс|regression", task_text, re.IGNORECASE)) and not re.search(
        r"классификац|classif", task_text, re.IGNORECASE)


def _target_profile(task_text: str, train: pd.DataFrame, sample: pd.DataFrame | None, target: str,
                    n_targets: int, multi_proba: bool) -> dict:
    """Тип задачи, метрика и формат вывода одного таргета.

    LLM читает task.md, данные ограничивают, текст и дефолты страхуют.
    """
    from models import positive_label

    facts = _hard_facts(train[target])
    problem_type = metric = None
    llm = _llm_parse(task_text, list(train.columns), target, facts)
    if llm:
        problem_type = _validate_problem_type(llm.get("task_type"), facts)
        if llm.get("metric") in AG_METRIC:
            metric = llm["metric"]
    if metric is None:
        metric = _metric_from_text(task_text)
        if metric:
            log(f"[{target}] метрика из текста задачи: {metric}")
    if problem_type is None:
        numeric_many = not facts["is_text"] and facts["nunique"] > 2
        if numeric_many and (_declares_regression(task_text) or metric in REGRESSION_METRICS):
            problem_type = "regression"
            log(f"[{target}] регрессия задана текстом задачи/метрикой")
        else:
            problem_type = _default_problem_type(facts)
            log(f"[{target}] тип задачи взят из свойств данных: {problem_type}")
    if multi_proba and problem_type == "regression":
        problem_type = "binary" if facts["nunique"] <= 2 else "multiclass"

    if metric is not None and (metric in REGRESSION_METRICS) != (problem_type == "regression"):
        log(f"[{target}] метрика {metric!r} несовместима с типом {problem_type!r}, беру дефолт")
        metric = None
    if metric is None:
        metric = _default_metric(problem_type, train[target])

    # Формат вывода: вероятности требует метрика или шаблон, а не значения заглушек.
    output_proba = multi_proba
    if problem_type == "binary" and metric in ("log_loss", "roc_auc"):
        output_proba = True
    if not output_proba and sample is not None and problem_type == "binary" and target in sample.columns:
        sv = sample[target].dropna()
        if len(sv) and pd.api.types.is_float_dtype(sv):
            labels = set(train[target].dropna().unique().tolist())
            if not set(sv.unique().tolist()) <= labels:
                output_proba = True
    pos_label = positive_label(train[target].dropna().unique()) if problem_type == "binary" else None
    return {
        "problem_type": problem_type,
        "metric": metric,
        "ag_metric": AG_METRIC[metric],
        "log_target": metric == "rmsle",
        "output_proba": output_proba,
        "pos_label": pos_label,
    }


def _match_class_columns(cols: list[str], classes: list, target: str) -> list[tuple[str, object]] | None:
    """Сопоставление колонок sample классам: имя = класс, либо класс — отделённый суффикс/префикс."""
    by_str = {str(v).lower(): v for v in classes}
    out = []
    for col in cols:
        name = col.lower()
        if name.startswith(target.lower()) and len(name) > len(target):
            stripped = name[len(target):].lstrip("_-.: =")
        else:
            stripped = name
        hit = by_str.get(name, by_str.get(stripped))
        if hit is None:
            cands = [
                k for k in by_str
                if re.search(rf"(^|[_\-\s.:=]){re.escape(k)}$", name) or re.search(rf"^{re.escape(k)}([_\-\s.:=]|$)", name)
            ]
            if cands:
                hit = by_str[max(cands, key=len)]
        if hit is None:
            return None
        out.append((col, hit))
    if len({str(c) for _, c in out}) != len(out):
        return None
    return out


def _match_proba_columns(cols: list[str], train: pd.DataFrame, test: pd.DataFrame, id_col: str,
                         dd_targets: list[str]) -> tuple[str | None, list]:
    """Таргет train, классам которого соответствуют колонки вероятностей sample."""
    cands = [t for t in dd_targets if t in train.columns] or [
        c for c in train.columns if c not in test.columns and c != id_col
    ]
    for t in cands:
        classes = train[t].dropna().unique().tolist()
        if not 2 <= len(classes) <= 200:
            continue
        mapping = _match_class_columns(cols, classes, t)
        if mapping is not None:
            return t, mapping
    # Имена не совпали: при равном числе колонок и классов — порядок sample = сортировка классов.
    for t in cands:
        classes = train[t].dropna().unique().tolist()
        if len(classes) == len(cols):
            ordered = sorted(classes, key=lambda v: (isinstance(v, str), v))
            log(f"Колонки {cols} не совпали по именам с классами {t!r}; сопоставляю по порядку {ordered}")
            return t, list(zip(cols, ordered))
    return None, []


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
    proba_cols: dict[str, list[tuple[str, object]]] = {}
    if sample is not None and len(sample.columns) >= 2:
        id_col = sample.columns[0]
        target_cols = [c for c in sample.columns[1:]]
    else:
        id_col = train.columns[0]
        target_cols = [c for c in train.columns if c not in test.columns and c != id_col][:1]

    # Кросс-чек по data_dictionary (role: id/target).
    dd = None
    dd_targets: list[str] = []
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

    # Колонки sample, которых нет в train, — вероятности классов одного таргета.
    if sample is not None and target_cols and not any(c in train.columns for c in target_cols):
        target, mapping = _match_proba_columns(list(sample.columns[1:]), train, test, id_col, dd_targets)
        if target is not None:
            proba_cols[target] = mapping
            target_cols = [target]
            log(f"Вывод вероятностей классов {target!r}: {mapping}")

    target_cols = [c for c in target_cols if c in train.columns]
    if not target_cols:
        raise ValueError("Не удалось определить целевую колонку")
    target = target_cols[0]

    features = [c for c in train.columns if c != id_col and c not in target_cols and c in test.columns]
    per_target = {
        t: _target_profile(task_text, train, sample, t, len(target_cols), t in proba_cols) for t in target_cols
    }

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
        "proba_cols": proba_cols,
        "per_target": per_target,
        **per_target[target],
    }
    for t, tp in per_target.items():
        log(
            f"Профиль: target={t!r}, id={id_col!r}, type={tp['problem_type']}, metric={tp['metric']}, "
            f"features={len(features)}, train={train.shape}, test={test.shape}, proba={tp['output_proba']}, "
            f"pos_label={tp['pos_label']!r}"
        )
    return profile
