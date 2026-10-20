# Ночная стабилизация Qwen: фактический результат

Промежуточный отчёт: работа продолжается одним агентом в fix/qwen-live-cycle.
Короткий измеренный цикл принят, включая новый review интерпретации.
Пять публичных gate не приняты; следующий повтор — gate-v6 на том же runtime. Сравнение — 0/18,
польза Qwen относительно Optuna ещё не измерена.

## 503 и контракт tools

503 воспроизводились примерно за 60–66 s с capacity_unavailable. Снятие forced
tool_choice не устранило сбой. Capacity, внутренний timeout и routing без
телеметрии endpoint не разделены; перегрузка не доказана. Alias — qwen3.8-flash;
реальное семейство/размер за alias не установлены. Использован только выданный endpoint.

Раунд 1: одинаковый payload smoke-v6 в ABBA forced/auto/auto/forced дал
503, 503, HTTP 200 с inspect_data, HTTP 200 с inspect_data, хотя разрешён только
edit_file. Аргументы inspect_data тоже невалидны. Короткие messages с прежними
tools/max_tokens дали валидный edit_file в обоих режимах. Замена только system
prompt и удаление только повторных страниц дали по два 503. Исходные messages/tools
с max_tokens=1536 вместо 4096 дали два валидных edit_file, но последующие smoke
подтвердили, что уменьшение output cap не гарантирует устранения 503.

Раунд 2: четыре default forced/auto/auto/forced на payload smoke-v3 дали 503.
С enable_thinking=False forced/auto дали валидный edit_file. В обратной паре
default тоже восстановился, а false снова успешен. False — 3/3 валидных ответа;
это не доказывает, что все 503 вызваны thinking.
[Документация Qwen](https://docs.qwencloud.com/developer-guides/tool-calling/function-calling)
предписывает отключать thinking перед forced выбором функции. Исправление
применено только для единственной принудительно выбранной функции, оба параметра
сохраняются в трассе. Application дополнительно отклоняет tools вне текущей схемы.

Раунд 3: replay интерпретаций без ML. И исходный запрос, и код без annotations
иногда отвечают верно: единственная причина ошибок не выделена, AST-переписывание
в runtime не добавлено. Явный охват диагностики и проверка ошибочного черновика
той же Qwen исправляли основные ложные утверждения. Выявлены также stale omission
counts и отсутствующий backend родителя. Replay не заменяет приёмку harness.

| Диагностика | HTTP-попытки | Reported tokens | Консервативно учтено |
|---|---:|---:|---:|
| Раунд 1 | 12 | 13 740 | 91 070 |
| Раунд 2 | 8 | 10 667 | 52 747 |
| Раунд 3 | 12 | 33 079 | 33 079 |

Итого 32 HTTP-попытки, 57 486 reported tokens, консервативно 176 896.
Все три диагностических раунда завершены; дополнительных API probes за ночь нет.
Учёт отделён от исследовательских запусков: runs/nightly/provider-round{1,2,3}/accounting.json.
Для ответов без usage резерв: UTF-8 prompt bytes + запас + max_tokens.
Точный биллинг не подтверждён; некоторые ответы сообщают 1–2 output tokens и
полностью cached prompt при нескольких аргументах tools. Это reported counters,
не доказательство полного расхода. Ключи, данные, полные payload/ответы и логи
остаются локально в игнорируемом runs/, в Git не включены.

## Проверенные изменения

Коммиты отправлены в origin/fix/qwen-live-cycle от настроенного Git-пользователя,
без соавторства и переписывания истории. Merge и конкурсная отправка не выполнялись.

| Commit | Изменение |
|---|---|
| d46c7c6 | Tools вне схемы запроса отклоняются до executor; ошибка возвращается модели, фаза/расход сохраняются. |
| a40d167 | Bounded generation до 1536 tokens, отдельный system prompt action-фаз, сверка гипотезы с исполняемым кодом. |
| 8319ce9 | Non-thinking для forced tool choice; параметры режима/выбора записываются в трассу. |
| 272fab3 | Ограниченная строка worker exception и failure_phase доступны Qwen; failed-кандидат возвращает её в edit_file. |
| 980cb43 | В контексте доступны measured baseline validation/refit seconds, worker/smoke timeout и потоки. |
| 895708f | Verifier отличает эксперименты по hypothesis_source, а не префиксу baseline в ID. |
| 1ede55d | Полный охват/счётчики изменений по классам, backend родителя, корректные class/feature counts; review черновика той же Qwen до model_finished в прежнем бюджете. |
| bed869d | Точная семантика фабрики (task kind vs backend, default CatBoost) сохраняется в run/interpretation/review compaction. Поведение ML-фабрики не меняется. |

Регрессии воспроизвели дефекты на границах SDK/session/worker/verifier, затем прошли.
Незавершённый review не считается принятой интерпретацией; draft/code/measurement
сохраняются при compaction. Scripted-тесты не считаются live-приёмкой.

## Короткие реальные запуски

Diabetes в каждом запуске скопирована через verification.verify_archive.copy_task:
условие и четыре входных файла. Holdout targets не переданы и не открывались.

| Каталог в runs/nightly | Результат |
|---|---|
| smoke-v1 | Changed full: RMSE 62.4347247663 против 62.4779271142, refit CSV проверен. Семантический отказ: объяснение приписало результат отсутствующим параметрам. 7 API-попыток без 503, 24 334 reported/charged tokens; затем budget_exhausted. |
| smoke-v2 | Два 503 на edit_file; changed=0, interpreted=0. Baseline CSV/refit проверены. |
| smoke-v3 | Два 503 на edit_file, несмотря на отдельный system prompt; changed=0, interpreted=0. Baseline CSV/refit проверены. |
| smoke-v4 | Принят на 8319ce9: CatBoost depth=4/lr=0.03/l2=9/600 итераций; full RMSE 60.1898931077 против 62.4779271142, корректная интерпретация и 88 строк refit CSV. 5 запросов, 14 606 reported/charged tokens, без ошибок. |
| smoke-v5 | Принят на 1ede55d: источник задаёт CatBoost 400/d4/lr0.05/l2=5 и Optuna 8 trials/40 s. Full RMSE 60.1943223042 против 62.4779271142. Отдельный review завершён, scope диагностики соблюдён; 88 строк CSV совпадают с выбранным frozen refit. 6 API-попыток, 17 208 reported/charged tokens, без tool errors/503. |

В smoke-v5 Optuna выбрала depth=3/lr=0.0286449874; стартовые значения в коде
не выдаются за фиксированные параметры обученного победителя. Числовой поиск
выполняла Optuna. verification.json, semantic-review.json и source-hashes.json
сохранены в каталоге запуска. Это инженерная приёмка цикла, не измерение пользы.

## Проверки и ZIP последнего runtime

- make check успешен; полный macOS offline — 625 passed, 1 live deselected,
  84.94 s; runs/nightly/offline-tests-v8.log.
- Полный Linux offline на bed869d — 625 passed, 1 live deselected, 91.17 s,
  network=none, 2 CPU / 6 GiB; runs/nightly/linux-offline-v6/pytest.log.
- ZIP: 56 файлов, побайтовое совпадение с исходниками; SHA-256
  78ce1a505111793feb48b59d81e547a98d5c2b420b355b0f646e3c2e0fb8cc6f.
- Распакованный entrypoint bed869d проверен в Linux вне checkout, без сети,
  Python 3.12.12, 4 CPU / 6 GiB: regression 22 449 строк, diabetes 88,
  оба exit=0, валидный CSV и успешный refit. runs/nightly/linux-archive-v6/report.json.
  Это search-mode=none, не Qwen gate. Посторонние контейнеры не останавливались.

Предыдущие проверки/артефакты сохранены. Первый технический запуск вспомогательного
archive-скрипта остановился до harness из-за PYTHONPATH; повтор исправлен.
Точная среда 8 CPU / 32 GB остаётся непроверенной.

## Публичные задачи

| Серия / задача | Приёмка |
|---|---|
| gate-v1 / regression | Не принята: changed=0, worker exception недоступен модели; 101.6 s. CSV/refit проверены, 22 449 строк. |
| gate-v1 / classification | Не принята: changed=0, worker exception недоступен модели; 552.7 s. CSV/refit проверены, 41 166 строк. |
| gate-v2 / regression | Не принята: changed full есть (RMSE 42.0071741965 против 42.1966696526), но интерпретация ошибочно утверждает LOO вместо smoothed group means. 139.0 s, CSV/refit проверены. |
| gate-v2 / classification | Не принята: changed full есть (macro-F1 0.5944175769 против 0.5652431446), но интерпретация заявляет падение precision во всех классах; в 3/10 он вырос. 756.4 s, CSV/refit проверены. |
| gate-v3 / regression | Не принята по содержанию: changed=1, interpreted=1, CSV/refit проверены, 113.5 s. Summary утверждает два LightGBM, но wrapper вызывает base_model без backend и фактически создаёт два CatBoost. |
| gate-v3 / classification | Принята на старом runtime: changed=1, interpreted=1, корректная интерпретация SqrtBalanced/scope, CSV/refit 41 166 строк; 1090.8 s. Второй кандидат timed_out на 300 s, сохранён. На новый код этот результат не переносится. |
| gate-v4 / regression | Не принята: api_error после повторных 503 на edit_file; changed=0, 238.2 s. |
| gate-v4 / classification | Не принята: catboost-balanced-w700d8 timed_out на full, changed=0, budget_exhausted; 935.2 s. |
| gate-v5 / regression | Не принята: changed full RMSE 41.1507446438 против 42.1966696526, но interpreted=0; после одного 503 и успешного retry не хватило token budget на интерпретацию. 199.0 s, CSV/refit 22 449 строк проверены. |
| gate-v5 / classification | Не принята: api_error после двух edit-попыток (~61 s каждая), changed=0, interpreted=0; 624.4 s. CSV/refit 41 166 строк проверены. |

У gate-v2 автоматический changed=0 для classification оказался ложным отказом:
ID baseline-catboost-sqrtbalanced был исключён по префиксу. Исправленный verifier
подтвердил по одному changed full на обеих задачах; counter-audit.json сохранён
отдельно. Оригинальный report не переписан, post-hoc replay не подставлен вместо
исходных summary. Из-за семантических отказов и изменения исходников старые gate
не допускают сравнение. Control scores для исправлений не просматривались.

Gate-v3 завершён после проверок и smoke-v5; общий автоматический флаг true,
но ручная приёмка серии false из-за regression. Gate-v4 также завершён и не принят: обе заданные публичные задачи,
seed 42, до трёх сессий, прежние ceilings. Runtime и verification во время серии
не меняются. Требуется проверка содержимого отчёта, кода, интерпретации и refit CSV,
а не только gate_passed. Локатор процесса — runs/nightly/ACTIVE.json;
живость подтверждать tool handle или PID/create_time, а не одним файлом статуса.

После gate-v3 исправлена потеря контракта фабрики:
kind задаёт тип задачи, backend выбирается через options; default — CatBoost.
Короткий candidate_runtime_contract в bed869d сохраняется в run/interpretation
и review compaction; модель больше не получает ложную подсказку, что полный
contract там уже доступен. Регрессии сначала воспроизвели потерю поля, затем
проверили весь вызывающий путь и соответствие default реальной фабрике. В gate-v3/backend-probe.json сохранена проверка реального frozen
wrapper без fit: task kind=regression, parameters={}, backend=catboost,
estimator=CatBoostRegressor. Это не API probe и не повторное ML-обучение.
Числовая метрика 44.5598818679 против 42.1966696526 рассчитана корректно,
но нельзя объявить протестированной LightGBM-гипотезу или приписать ухудшение
только log-blend при одновременно другом backend. Результат classification
сохранён и принят на 1ede55d; нужен новый gate на bed869d.
Gate-v4 завершился до следующих изменений runtime. Его payload подтверждает доставку
candidate_runtime_contract, forced edit_file и enable_thinking=False; это не устранило 503.

После gate-v4 исправлен учёт резерва: неудачный/прерванный worker расходует
wall-clock, но не повышает прогноз стоимости успешных финалистов; fallback
учитывается как refit, а не validation. Три быстрые регрессии воспроизвели
дефекты до исправления. make check и macOS offline: 628 passed / 1 live
deselected (84.97 s); Linux без сети: 628 passed / 1 live deselected (90.97 s).
ZIP: 56 файлов, SHA-256 6f512560ea4f5208b04e4450c5e2c03c002668ac2c9f9c27b0812580c4895429.
Распакованный ZIP побайтово совпадает с исходниками; Linux без сети, 4 CPU/6 GiB:
regression/diabetes дали валидные CSV и успешный refit. Следующий — gate-v5
на frozen runtime 13b8c80.

## Сравнение и следующий шаг

Матрица — 0/18, не начата. Измеренного эффекта Qwen против Optuna нет.
Диабетический holdout verifier открывает только после полного cohort.

1. Повторить обе публичные задачи в свежей серии gate-v6 на том же runtime
   13b8c80: gate-v5 завершён, обе задачи не приняты. CSV/refit обеих задач
   проверены, но regression без интерпретации, classification без changed full.
   После завершения gate-v6 проверить код, full-метрики, scope и review.
2. Только после этого выполнить:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
uv run --locked --all-extras python -m verification.run_agent_comparison \
  data/regression/task.md data/classification/task.md runs/stage7/tasks/diabetes/task.md \
  --gate-report runs/nightly/gate-v6/report.json --output runs/nightly/comparison-v1
```

3. Сохранить все 18 попыток и парные эффекты по трём seed, обновить этот отчёт
   фактами; отрицательный эффект допустим. После просмотра final control/holdout
   новые изменения не подбирать. Gate и сравнение остаются незавершёнными до
   фактической проверки; готовность агента по этой записи не объявляется.
