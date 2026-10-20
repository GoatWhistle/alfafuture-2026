# Alfa ML Harness

Qwen предлагает гипотезы по ошибкам ML-модели. Python-harness проверяет
структурированный рецепт, материализует код, выполняет smoke и допуск по ресурсам,
измеряет результат и сохраняет submission. Optuna занимается численным подбором.

**Статус:** реализован pipeline v2 этапа 9: максимум три гипотезы, две коррекции
каждой, свежий бюджет, проверяемые proposal/spec/code hashes и независимый gate.
Существующие baseline, финальный отбор и frozen refit сохранены. LLM-review больше
не блокирует результат; итоговый LLM-текст отключён. Фактические проверки и
открытые критерии — в [плане](docs/PLAN.md) и [отчёте](docs/PIPELINE_RESULT.md).
История прежнего runtime — [NIGHTLY_RESULT.md](docs/NIGHTLY_RESULT.md).

Проверено 20.10.2026: 45/45 baseline-only сравнений (5 задач × 3 backend × 3 seeds) завершились успешным финальным refit и валидным CSV. Распакованный ZIP прошёл 5 ML-задач в Linux без сети (8 CPU/6 GiB). Числа, параметры и оставшиеся проверки — в [плане](docs/PLAN.md); выигрыш полного агента через Qwen ещё не измерен.

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

Для будущих live-запусков скопируйте `.env.example` в `.env` и заполните выданные значения. CLI загружает `.env` из текущего каталога без переопределения окружения платформы. После базового ML-запуска CLI проводит до трёх структурированных гипотез при наличии всех трёх OPENAI_*; без них оставляет baseline. Для запуска без LLM: `--baseline-only`. Файл `.env`, данные и результаты исключены из Git.


## Запуск

```bash
uv run --locked python solution.py data/regression/task.md
uv run --locked python solution.py data/classification/task.md --backend lightgbm
# AutoGluon требует установленный extra automl:
uv run --locked python solution.py data/regression/task.md --backend autogluon --baseline-only
```

Без аргумента читается `task.md` в текущем каталоге. Пути данных и submission разрешаются относительно описания задачи. `--seed` фиксирует разбиение и seed модели (по умолчанию 42), `--threads` задаёт 1–8 потоков, `--worker-timeout` — deadline одного worker. `--time-budget` ограничивает весь запуск (по умолчанию и максимум 10 800 s), включая импорт, подготовку и финализацию. При resume действует исходный deadline. `--memory-limit-mb` ограничивает RSS дерева worker (по умолчанию 28 000 MiB). Каталог `runs/<id>/` содержит frozen split, запросы worker, stdout/stderr и итоговую сводку. Для явного каталога: `--run-dir runs/example`; используйте новый каталог на запуск. Для восстановления готового baseline и памяти: `--resume --run-dir runs/example` (fingerprint и CSV проверяются, baseline не переобучается). `--research-sessions` задаёт число гипотез (не более 3), `--research-steps` — число попыток предложения (не более 3), `--llm-token-limit` — общий расход с учётом сохранённой памяти; эти лимиты не являются оценкой стоимости провайдера. Live-проверка запускается отдельно: `RUN_QWEN_LIVE=1 uv run --env-file .env --locked pytest -m live`. Конкурсный сабмит не выполняется.

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

На достаточно больших данных отделяется контрольная выборка. Она не входит в поиск и ранний refit: финалисты фиксируются до её однократной проверки. После выбора выполняется refit на всех строках. До тяжёлого обучения сохраняется проверенный резервный CSV; запись результата атомарная. Поиск оставляет измеряемый резерв на контроль, до двух refit для ансамбля, predict и запись.

Пакет расположен рядом с `solution.py`. Создать архив без установки самого проекта:

```bash
uv run --locked python scripts/build_submission.py
```

`dist/submission.zip` содержит только entrypoint и Python-пакет. Сборка детерминированная; venv, данные, ключи, результаты и symlinks в архив не включаются. Linux-проверки описаны в [tests/README.md](tests/README.md); точное окружение платформы остаётся отдельным gate.

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


Публичный functional gate v2 (использует выданную Qwen):

```bash
uv run --locked python -m verification.run_agent_comparison \
  data/regression/task.md data/classification/task.md \
  --gate --modes qwen --seeds 42 --sessions 3 \
  --output runs/pipeline-v2/gate-1
```

Gate закрепляет 1800 s / 4 threads / 6000 MiB / 32000 tokens, проверяет обе
публичные задачи и артефакты. Для матрицы используется `--gate-report` принятого
v2-отчёта с теми же исходниками. Старые исследовательские runs нельзя продолжать
новым runtime; создайте новый каталог. Числа live-проверок смотрите в отчёте.
