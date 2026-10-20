# Pipeline v2: реализация и проверка

Работа выполняется в `fix/qwen-live-cycle`, исходная точка `4d126be`.
Статус: реализация и независимые проверки выполнены; первая новая live-серия
завершена и не принята. Этап 9 целиком открыт, польза не доказана.
По уточнению пользователя дальнейшие live-запуски ждут конфигурации целевой
Qwen около 27B: [MODEL_MIGRATION.md](MODEL_MIGRATION.md). Gate-2 и матрица на
прежнем endpoint не запускаются. Исторические gate-v1..v7 не переиспользуются.

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

## Исправления после первой live-серии

Исходники gate-1 соответствуют коммиту `1d0a312`; во время серии не менялись.
После её завершения исправлены две границы передачи фактов:

- Коррекция получала только отказ tool и теряла собственный отклонённый рецепт.
  Теперь сохраняются имя tool и точные JSON arguments, включая невалидный JSON.
  В контексте остаётся последняя попытка, в журнале — все попытки.
  `test_correction_receives_exact_rejected_recipe` воспроизвёл KeyError до правки
  и проходит после неё. Логи: `correction-context-confirmed.log`,
  `correction-context-after.log` в `runs/pipeline-v2/`.
- После Optuna следующая гипотеза видела исходную стратегию поиска вместо
  фактически обученных параметров. `executed_recipe` и его hash фиксируют
  параметры worker; parent context использует их с отключённым tuning.
  `test_next_hypothesis_sees_actual_optuna_parameters` проверяет реальный worker,
  verifier также отклоняет подмену `executed_recipe_hash`.

Полный macOS после исправлений: 651 passed / 1 deselected, 96.19 s,
`runs/pipeline-v2/macos-offline-correction.log`. Зависимости не менялись.
Linux после исправлений: 651 passed / 1 deselected, 105.40 s,
`runs/pipeline-v2/linux-offline-correction.log`. Команда:

```bash
docker run --rm --network none --cpus 2 --memory 6g \
  -v "$PWD/runs/pipeline-v2/source-correction:/workspace:ro" -w /workspace \
  alfa-harness-check:py312 python -m pytest -m 'not live' -p no:cacheprovider -q
```

Снимок `source-correction/` содержит проверенные исходники и тесты;
`source-hashes.json` побайтово сопоставлен с текущим runtime. Это существующий
Linux Python 3.12 image, без установки зависимостей и сети.
`make check` после всех правок проходит (`check-correction-final.log`).
Новый `dist/submission.zip`: 60 файлов, все совпадают с исходниками,
SHA-256 `0729030d1c61e80cdaa20cac55546df44f8dd691dd9ed79c7bb6d6109954c4fb`.
Хеши и побайтовая проверка — `archive-integrity-correction.json`.
Распакованный новый ZIP также прошёл regression и diabetes: exit=0 и валидный
CSV для обеих задач, Linux без сети, 4 CPU / 6 GiB, worker memory limit 5000 MiB,
time budget 120 s. Отчёт:
`runs/pipeline-v2/archive-correction-artifacts/verified-final/report.json`;
журнал — `linux-archive-correction-final.log`. Первоначальная попытка
`linux-archive-correction.log` остановилась из-за неверного пути `/data/diabetes`;
повтор использует уже изолированные четыре входных файла smoke без holdout.
Это проверка baseline-only ZIP; исправленный Qwen runtime подтверждён offline,
нового live на нём не было. Исправления и этот отчёт входят в логический коммит
`Preserve rejected proposals and fitted recipes across Qwen steps`.

## Ограниченная live-приёмка

Использована только выданная конфигурация с alias `qwen3.8-flash`; фактический
checkpoint и размер этой модели не установлены. Пользователь уточнил, что она
сильнее целевой модели; результаты ниже на целевую Qwen не переносятся.
Использован один smoke и одна серия публичного gate; новых API-проб не было.
Control не передавался Qwen, diabetes holdout не открывался.

### Smoke diabetes

`runs/pipeline-v2/smoke/`: exit=0, 80.15 s, три предложенных Qwen и успешно
измеренных изменённых full, 15 956 reported/charged tokens, usage известен.
Baseline validation RMSE 62.477927; full гипотез: 62.345246, 63.583051,
62.701732. Это validation на одной задаче, не сравнение пользы с Optuna.
Финальный ensemble/refit CSV: 88 строк, независимо проверен `verify_run`,
`runs/pipeline-v2/smoke/verified.json`; первоначальная сводка — `smoke.log`.
Worker видел только task.md и четыре разрешённых входных файла.

### Публичный gate-1

Команда:

```bash
uv run --locked python -m verification.run_agent_comparison \
  data/regression/task.md data/classification/task.md \
  --gate --modes qwen --seeds 42 --sessions 3 \
  --output runs/pipeline-v2/gate-1
```

Одинаковые ceilings: 1800 s, 4 threads, 6000 MiB, worker timeout 300 s,
32 000 tokens, максимум три гипотезы. API overhead включён в wall time.

| Задача | Изменённые full Qwen | Wall, s | Peak RSS, bytes | CSV строк / refit | Reported / charged tokens |
|---|---:|---:|---:|---|---:|
| regression | 0 | 165.34 | 908886016 | 22449 / verified | 3601 / 3601 |
| classification | 1 | 1050.42 | 1257504768 | 41166 / verified | 7843 / 22218 |

Gate **не принят**. Regression: num_leaves=127 отклонён до worker, затем две
HTTP 503 `capacity_unavailable`, каждая около 60.7 s; status=api_error.
Classification: один успешный full CatBoost с class balancing и frequency
признаками, validation Macro F1 0.529763 против baseline 0.565243.
Следующий запрос получил HTTP 503 (61.34 s), retry вернул рецепт с запрещёнными
random_strength/bagging_temperature, он отклонён; research status=budget_exhausted.
В обоих запусках usage_known=false: reported usage не выдаётся за полный расход;
при неучтённом расходе следующие вызовы прекращаются.

Победителем остался baseline. Его независимые control-метрики:
regression RMSE 40.355049, classification Macro F1 0.539817.
Они не использовались для новых гипотез и не являются score платформы.
Сводки и API traces сохранены локально в `runs/pipeline-v2/gate-1/`;
`report.json`, `configuration.json`, `progress.json`, manifests и `gate-1.log`
сохраняют все попытки. Секреты и endpoint URL в Git не добавлены.

После исправления verifier прежние измерения перепроверены его **замороженной**
версией из `1d0a312`: все source hashes совпали с gate configuration, CSV и
trace hashes совпали с отчётом. Результат — `gate-1-independent.json`,
исходники — `source-gate-1/` в `runs/pipeline-v2/`.
Новый verifier требует executed_recipe metadata; старые артефакты не дополнялись
задним числом и не используются как разрешение на новый cohort.

Матрица Qwen/Optuna — **0/18**, поскольку gate не принят и конфигурация целевой
модели ожидается. Парный прирост, критерий пользы и переносимость не измерены.

## Сверка критериев этапа 9

| Критерии | Проверяемое основание / оставшийся пробел |
|---|---|
| 9.1, контракты, allowlist, harness-owned поля | `tests/unit/test_proposals.py`, строгая `ProposalArguments`, вызов resolve до worker; неверный live num_leaves также отклонён без обучения. |
| 9.1, Python fit/transform и материализация | `test_python_fit_receives_train_only_and_schema_is_enforced`, `test_real_feature_transform_refit_survives_next_provider_failure`; requested/resolved/effective hashes проверяет verifier. |
| 9.2, свежий бюджет, стоимость, deadline/CSV | `test_session_repairs_have_fresh_budget_and_are_bounded`, `test_expensive_recipe_smoke_never_starts_full_and_repeat_deduplicates`, `test_runtime_budget.py`, `test_cli_budget.py`, `test_qwen_retry_budget.py`; реальные 503 не потеряли оба CSV. |
| 9.3, следующий эксперимент и сохранение фактов | `test_new_hypotheses_get_short_sessions_with_prior_evidence` исполняет два изменённых full; исправления коррекции и fitted parent дополнительно проверены новыми тестами. |
| 9.3, train-only Optuna/frozen refit/resume | `test_tuning_automl.py`, `test_worker_tuning.py`, `test_memory_recovery.py`, `test_live_regressions.py` входят в полный offline suite; новая интеграция проверяет передачу выбранных параметров потомку. |
| 9.4, независимая проверка и отрицательные случаи | `tests/integration/test_agent_comparison.py`: отсутствующая задача, пустой/legacy gate, config/runtime/provider mismatch, forged hashes, incomplete measurement, CSV mismatch. Оба реальных CSV gate-1 проверены замороженным verifier. |
| 9.4, обе публичные задачи с changed full | Не выполнено: regression 0, classification 1; gate_passed=false. |
| 9.4, macOS/Linux/ZIP | Полные журналы и archive integrity выше; точные 8 CPU/32 GiB не проверены. |
| 9.5, ограниченные попытки и матрица | Старый endpoint: smoke 1/1, gate 1/2. Следующие попытки отложены по MODEL_MIGRATION; целевая модель не проверена, сравнение 0/18, польза не доказана. |

## Открытые ограничения

Cost estimate — эвристика, не гарантия времени; hard timeout обязателен.
Python worker не является песочницей для враждебного кода. Только два backend
поддерживаются structured proposals; AutoGluon остаётся стартовым backend.
Точная платформа 8 CPU/32 GiB не проверена. Не выполнены: успешный gate на обеих
публичных задачах, live на целевой модели, полная матрица 18 и критерий пользы.
Повторяющиеся 503 — наблюдаемый внешний отказ, их инфраструктурная причина
не установлена. Эти ограничения не закрываются offline-тестами.
