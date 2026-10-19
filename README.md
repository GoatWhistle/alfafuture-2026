# Alfa ML Harness

Один ReAct-агент на Qwen исследует ошибки ML-модели, пишет изменения и проверяет гипотезы. Python-harness управляет инструментами, независимой оценкой, памятью экспериментов и ресурсами.

**Статус:** этапы 1–2 завершены. Один вызов CLI проверяет задачу, фиксирует split, обучает CatBoost или LightGBM, независимо считает метрику, делает refit и атомарно сохраняет submission. На обеих публичных задачах путь проверен. Этапы 4–5 добавили память, парную диагностику ошибок, выбор веток, Optuna и ограниченный AutoGluon. Ограниченные ReAct-сессии Qwen, JSONL-память и восстановление реализованы и проверены с fake/HTTP mock; реальный endpoint пока не настроен и не проверен.

## Окружение

Нужны [uv](https://docs.astral.sh/uv/getting-started/installation/) версии 0.9.17 (закреплена в `tool.uv.required-version`) и Python из `.python-version`. Окружение создаётся локально в `.venv`:

```bash
uv python install
make setup
make check
uv run --locked python solution.py --help
```

`make setup` устанавливает runtime, инструменты разработки и extra `automl`. Для окружения без AutoGluon: `uv sync --locked`. Все прямые версии зафиксированы в `pyproject.toml`, разрешённые транзитивные — в `uv.lock`. Constraints удерживают версии библиотек из списка организаторов, даже когда они приходят транзитивно. Dev-инструменты не входят в конкурсное решение.

На macOS для LightGBM/XGBoost может понадобиться системный OpenMP: `brew install libomp`. Linux-проверка отдельно обязательна перед конкурсной сдачей; локальная установка не воспроизводит полностью Docker-образ платформы.

Для будущих live-запусков скопируйте `.env.example` в `.env` и заполните выданные значения. CLI загружает `.env` из текущего каталога без переопределения окружения платформы. После базового ML-запуска CLI проводит одну ReAct-сессию при наличии всех трёх OPENAI_*; без них оставляет baseline. Для запуска без LLM: `--baseline-only`. Файл `.env`, данные и результаты исключены из Git.


## Запуск

```bash
uv run --locked python solution.py data/regression/task.md
uv run --locked python solution.py data/classification/task.md --backend lightgbm
# AutoGluon требует установленный extra automl:
uv run --locked python solution.py data/regression/task.md --backend autogluon --baseline-only
```

Без аргумента читается `task.md` в текущем каталоге. Пути данных и submission разрешаются относительно описания задачи. `--threads` задаёт 1–8 потоков, `--worker-timeout` — deadline одного worker (пока не общий бюджет исследовательского цикла). Каталог `runs/<id>/` содержит frozen split, запросы worker, stdout/stderr и итоговую сводку. Для явного каталога: `--run-dir runs/example`; используйте новый каталог на запуск. Для восстановления готового baseline и памяти: `--resume --run-dir runs/example` (fingerprint и CSV проверяются, baseline не переобучается). `--research-sessions` задаёт число коротких гипотез, `--research-steps` — число ходов на гипотезу, `--llm-token-limit` — общий расход с учётом сохранённой памяти; эти лимиты не являются оценкой стоимости провайдера. Live-проверка запускается отдельно: `RUN_QWEN_LIVE=1 uv run --env-file .env --locked pytest -m live`. Конкурсный сабмит не выполняется.

## Структура

```text
solution.py                   # точка входа платформы
ml_harness/
  domain/                     # сущности, инварианты, решения без I/O
  application/
    ports/                    # интерфейсы внешних эффектов
    use_cases/                # сценарии исследования
  adapters/
    llm/                      # Qwen, форматы сообщений и tools
    data/                     # CSV, описание, профиль данных
    execution/                # дочерние процессы, дедлайны, ресурсы
    ml/                       # модели, метрики, диагностика
    storage/                  # журнал и артефакты
  entrypoints/                # CLI и связывание реализаций
tests/{unit,integration,fixtures}/
docs/                         # архитектура и план
task/                         # ТЗ, исследование и исходные материалы
data/                         # локальные датасеты (игнорируются Git)
runs/                         # будущие результаты запусков (игнорируются Git)
```

Пакет расположен рядом с `solution.py`, чтобы будущий ZIP запускался без установки самого проекта и изменения `sys.path`.

## Документация и проверки

- [Архитектура](docs/ARCHITECTURE.md): ответственность модулей и направление импортов.
- [План](docs/PLAN.md): этапы реализации с критериями готовности.
- [Материалы организаторов](task/reference/README.md): происхождение исходников и зависимостей.
- [Исследование](task/harness-research.md): обоснование решения; текущие решения закреплены в архитектуре и плане.

`make check` проверяет lock-файл, стиль, типы и архитектурные импорты. `make test` запускает тесты доменных инвариантов и подготовки задачи на небольших синтетических CSV без live API. Публичные данные и результаты проверок хранятся локально в игнорируемых каталогах.

Для независимой offline-проверки цепочки ошибок и разных кодовых гипотез:

```bash
uv run --locked python -m ml_harness.entrypoints.verify_research data/regression/task.md runs/stage2/public-regression runs/stage5/regression
```

Эта команда использует сохранённый проверенный baseline и фиксированный protocol. Она явно помечает гипотезы как scripted, сохраняет улучшения и ухудшения в журнале, не вызывает Qwen и не отправляет конкурсный сабмит. Реальное сравнение пользы агента при равных ресурсах ещё предстоит на этапе 7.
