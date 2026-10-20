# Alfa ML Harness

Qwen предлагает гипотезы по ошибкам ML-модели. Python-harness проверяет
структурированный рецепт, материализует код, выполняет smoke и допуск по ресурсам,
измеряет результат и сохраняет submission. Optuna занимается численным подбором.

**Статус:** реализован pipeline v3 по [REFACTOR-HARNESS.md](docs/REFACTOR-HARNESS.md).
Профиль данных строится по train; Qwen и Optuna продолжают полный pipeline.
Планировщик чередует гипотезы и настройку до резерва завершения (до 12 гипотез,
две коррекции каждой); при отказе API остаётся локальный поиск. Классификация
сохраняет вероятности, ансамбли выбираются на development до control.
Offline macOS/Linux: 699 passed; ZIP проверен без сети на пяти задачах.
Фактическая приёмка и открытые live/quality критерии — [REFACTOR_RESULT.md](docs/REFACTOR_RESULT.md).
История прежнего runtime — [NIGHTLY_RESULT.md](docs/NIGHTLY_RESULT.md).
Текущая выданная модель отличается от ожидаемой на защите Qwen около 27B.
Её конфигурация ожидается; переносимость пока не проверена. Порядок приёмки —
[MODEL_MIGRATION.md](docs/MODEL_MIGRATION.md).

Историческая проверка до v3 (20.10.2026): 45/45 baseline-only сравнений (5 задач × 3 backend × 3 seeds) завершились успешным финальным refit и валидным CSV. Распакованный ZIP прошёл 5 ML-задач в Linux без сети (8 CPU/6 GiB). Числа, параметры и оставшиеся проверки — в [плане](docs/PLAN.md); выигрыш полного агента через Qwen ещё не измерен.

## Веб-панель: быстрый старт для команды

Получите актуальный `main`:

```bash
git clone https://github.com/GoatWhistle/alfa-future.git
cd alfa-future
```

Если репозиторий уже есть, перейдите в него и обновите `main` командой
`git pull --ff-only origin main`, предварительно переключившись на эту ветку.

**Посмотреть интерфейс без обучения:** нужны только Node.js и npm.
Python, датасеты и `.env` не требуются.

```bash
npm ci --prefix web
make web-demo
```

Откройте <http://127.0.0.1:5173>. Это помеченное демо с шестью сценариями:
можно открывать эксперименты, запускать, приостанавливать и повторять демонстрацию.
Запросов к Qwen и настоящего обучения нет.

**Запускать настоящие исследования:** установите Python/uv по разделу ниже,
а также Node.js/npm. Все команды выполняются из корня репозитория:

```bash
uv python install
make setup
make web-setup
(cd web && VITE_DATA_SOURCE=api npm run build)
make web-api
```

Откройте <http://127.0.0.1:8765>. Один сервер отдаёт собранный интерфейс и API;
отдельный Vite не требуется. В «Новом исследовании» загрузите описание задачи
и четыре CSV: `train.csv`, `test.csv`, `data_dictionary.csv`,
`sample_submission.csv`. Датасеты не входят в Git — получите их отдельно у команды.
Обучение начнётся после нажатия «Запустить», одновременно выполняется один запуск.

**Нужен ли `.env`:**

| Режим | Нужны креды Qwen |
|---|---|
| Демо интерфейса | Нет |
| Baseline / Optuna на задаче с однозначным описанием | Нет |
| Исследование с Qwen | Да: `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL` |

Для режима Qwen создайте **свой локальный `.env` в корне репозитория** по
`.env.example` и заполните все три значения из выданной команде конфигурации.
Копировать чей-то полный `.env` не требуется. Если используется общий командный
ключ, все участники расходуют его общую квоту. Реальные ключи не коммитьте;
в браузер они не передаются. Окружение сервера имеет приоритет над `.env`.
Сам запуск панели не проверяет Qwen сетевым запросом и не расходует её токены.

Панель слушает только `127.0.0.1`: каждый участник открывает свой локальный
экземпляр. Для просмотра общего экземпляра через сеть нужна отдельная настройка.
Остановка сервера — `Ctrl+C`; активное обучение прерывайте кнопкой в интерфейсе,
поскольку дочерний процесс может продолжить работу после остановки API.
Подробности разработки и проверок — в [web/README.md](web/README.md).

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

Для будущих live-запусков скопируйте `.env.example` в `.env` и заполните выданные значения. CLI загружает `.env` из текущего каталога без переопределения окружения платформы. После базового ML-запуска CLI чередует гипотезы Qwen (до 12) и локальную настройку по оставшемуся бюджету; без всех трёх OPENAI_* продолжает численный поиск. Для запуска без исследования: `--baseline-only`; для поиска без LLM: `--search-mode optuna`. Файл `.env`, данные и результаты исключены из Git.


## Запуск

```bash
uv run --locked python solution.py data/regression/task.md
uv run --locked python solution.py data/classification/task.md --backend lightgbm
# AutoGluon требует установленный extra automl:
uv run --locked python solution.py data/regression/task.md --backend autogluon --baseline-only
```

Без аргумента читается `task.md` в текущем каталоге. Пути данных и submission разрешаются относительно описания задачи. `--seed` фиксирует разбиение и seed модели (по умолчанию 42), `--threads` задаёт 1–8 потоков, `--worker-timeout` — deadline одного worker. `--time-budget` ограничивает весь запуск (по умолчанию и максимум 10 800 s), включая импорт, подготовку и финализацию. При resume действует исходный deadline. `--memory-limit-mb` ограничивает RSS дерева worker (по умолчанию 28 000 MiB). Каталог `runs/<id>/` содержит frozen split, запросы worker, stdout/stderr и итоговую сводку. Для явного каталога: `--run-dir runs/example`; используйте новый каталог на запуск. Для восстановления готового baseline и памяти: `--resume --run-dir runs/example` (fingerprint и CSV проверяются, baseline не переобучается). `--research-sessions` задаёт число гипотез (не более 12), `--research-steps` — число попыток предложения (не более 3), `--llm-token-limit` — общий расход с учётом сохранённой памяти; эти лимиты не являются оценкой стоимости провайдера. Live-проверка запускается отдельно: `RUN_QWEN_LIVE=1 uv run --env-file .env --locked pytest -m live`. Конкурсный сабмит не выполняется.

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


Публичный functional gate v3 (использует выданную Qwen):

```bash
uv run --locked python -m verification.run_agent_comparison \
  data/regression/task.md data/classification/task.md \
  --gate --modes qwen --seeds 42 --sessions 12 \
  --output runs/pipeline-v3/gate-1
```

Gate закрепляет 1800 s / 4 threads / 6000 MiB / 32000 tokens, проверяет обе
публичные задачи и артефакты. Для матрицы используется `--gate-report` принятого
v3-отчёта с теми же исходниками. Старые исследовательские runs нельзя продолжать
новым runtime; создайте новый каталог. Числа live-проверок смотрите в отчёте.


Итог v3 — `run-result.json`: состояние CSV и поиска, выбранный рецепт,
источник, метрика/протокол, причины остановки/отказов, ресурсы и usage.
CLI, verifier и панель используют эти факты; v2 доступен для чтения,
resume v2 новым исполнителем отклоняется. Используйте новый `--run-dir`.

Сравнительная серия (после gate целевой Qwen):

```bash
uv run --locked python -m verification.benchmark_tasks --output runs/benchmarks/tasks
uv run --locked python -m verification.run_hybrid_comparison \
  data/regression/task.md data/classification/task.md \
  runs/benchmarks/tasks/diabetes/task.md \
  runs/benchmarks/tasks/breast_cancer/task.md runs/benchmarks/tasks/wine/task.md \
  --gate-report runs/pipeline-v3/gate-1/report.json --output runs/pipeline-v3/cohort
```

Скрипт создаёт отдельные benchmark-копии с внешними targets, замораживает
исходники/конфигурацию и последовательно сравнивает старый portfolio+Optuna,
новый Optuna, гибрид, AutoGluon CAT/GBM/XGB и FLAML CPU. Внешняя оценка
выполняется только после полного cohort. Без целевой Qwen допустимы режимы
`--modes frozen optuna autogluon flaml`; это не доказательство пользы гибрида.
`--backend linear --baseline-only` проверяет дешёвый линейный кандидат отдельно;
в начальный portfolio он не входит.
