# Pipeline v2: реализация и проверка

Работа выполняется в `fix/qwen-live-cycle`, исходная точка `4d126be`.
Статус: код реализован, offline macOS/Linux и ZIP приняты; live проверяется.
Публичный functional gate и польза ещё не подтверждены. Исторические gate-v1..v7
не переиспользуются.

## Реализация

- Qwen вызывает только `propose_experiment`. Typed Proposal/ResolvedExperiment,
  строгая внешняя схема, явные task kind/backend, allowlist параметров/диапазонов.
  Seed, потоки, objective, evaluator, split и пути принадлежат harness.
- Harness материализует фабрику, сохраняет requested/resolved/code hashes;
  no-op и ранее исполненный рецепт отклоняются. Два исправления на гипотезу,
  максимум три гипотезы. Следующий запрос получает измеренные факты и свежий бюджет.
- Преобразования: drop/signed_log1p/train-frequency и Python build_transformer
  с fit(X)/transform(X), без y. Проверяются index/order/схема. Реальные worker
  и финальный refit проверены тестом train-fitted Python-преобразования.
- Smoke предшествует численному допуску по baseline/сложности/объёму данных.
  Safety factor 1.5; прогноз, допущения, разрешённое время и ошибка прогноза
  сохраняются. Hard worker deadline остаётся окончательным ограничением.
- Full score/parent delta/error delta сохраняются до следующего Qwen-вызова.
  Отказ последующего API не теряет измеренный результат и refit CSV.
- Mandatory interpretation/review исключены. Итоговое LLM-объяснение отключено
  в первой версии; текст вообще не требуется для functional success.
- `tune=true`: 2 Optuna trials / 30 s, train-only inner split. Feature transforms
  и tuning в одном рецепте пока запрещены: так fit преобразований не видит
  inner-validation. Frozen параметры используются при refit без поиска.
- Gate v2 перепроверяет обе публичные задачи seed 42, runtime/config/provider/input
  hashes, requested→resolved→effective code→фактический worker/full score,
  parent lineage и финальный CSV/refit. Live proposal связан с response trace Qwen.
  Старый boolean gate_passed или непустой summary не являются доказательством.
- Resume v2 сохраняет учёт usage/deadline; старые LLM-журналы отклоняются до
  их изменения. Baseline-only память остаётся совместимой.

## Проверки

- `make check` проходит: lock, Ruff, mypy, два import contracts.
- macOS: `uv run --locked pytest -m 'not live' -q` — 648 passed / 1 deselected,
  89.27 s. `runs/pipeline-v2/macos-offline.log`.
- Дополнительная проверка последнего verifier: 6 passed,
  `/tmp/pipeline-gate-tests.log`.
- Первый полный прогон обнаружил ошибку синтетического Python-преобразования:
  тестовый test.csv содержит строку в числовом столбце. Fixture-преобразование
  исправлено явным to_numeric(errors=coerce); положительный refit проверен повторно.
- Linux: 648 passed / 1 deselected, 92.82 s, `--network none --cpus 2 --memory 6g`;
  исходники смонтированы read-only в существующий `alfa-harness-check:py312`.
  `runs/pipeline-v2/linux-offline.log`.
- ZIP: 60 файлов, побайтово совпадают с исходниками, SHA-256
  `9f8ded485652240e7c9b57f7681c5c19b46c2a385ff8a5ae0568ad4394fc0aef`.
  `runs/pipeline-v2/archive-integrity.json`.
- Распакованный ZIP: regression + diabetes, CatBoost baseline-only, Linux без сети,
  4 CPU / 6 GiB; оба exit=0, финальный CSV проверен. Это не Qwen functional gate.
  `runs/pipeline-v2/linux-archive/report.json`.
- Live: пока в работе, результаты будут добавлены ниже.

## Открытые ограничения

Cost estimate — эвристика, не гарантия времени; hard timeout обязателен.
Python worker не является песочницей для враждебного кода. Только два backend
поддерживаются structured proposals; AutoGluon остаётся стартовым backend.
Точная платформа 8 CPU/32 GiB не проверена. Новый публичный gate и матрица 18
ещё не завершены, улучшение метрик не заявляется. Новые зависимости не добавлены.
