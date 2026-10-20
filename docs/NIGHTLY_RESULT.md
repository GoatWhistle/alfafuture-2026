# Ночная стабилизация Qwen: фактический результат

> Отчёт о предыдущем runtime. Текущая работа и новые критерии — этап 9
> `PLAN.md`, задание — `PIPELINE_HANDOFF.md`. Порядок продолжения ниже
> сохранён как история и заменён новым планом.

Работа выполнена одним агентом в fix/qwen-live-cycle; цель полностью не достигнута.
Полный короткий цикл Qwen принят на diabetes на 1ede55d, но публичная приёмка
последнего runtime fec115c не пройдена. Все семь серий gate завершены и сохранены.
В gate-v7 regression снова получила 503; classification не получила changed full.
Сравнение Qwen против Optuna — 0/18, измеренного эффекта нет.

Продолжение упирается в повторяющийся отказ выданного endpoint. Перегрузка как
первопричина не доказана; три разрешённых диагностических раунда исчерпаны.
Готовность агента не объявляется. Ниже сохранены изменения, проверки и точный
порядок возобновления после восстановления доступности endpoint.

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
| 13b8c80 | Failed/timed_out worker расходует deadline, но не завышает прогноз стоимости успешных финалистов; fallback учитывается как refit. |
| fec115c | Review при тесном бюджете сохраняет код/метрики, удаляя дубли инструкций, поисковые подсказки и пути/хеши артефактов. |

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

Проверен runtime fec115c; после него менялась только документация.

- make check успешен: lock, Ruff, mypy, оба import contracts.
- Полный macOS offline: 629 passed / 1 live deselected, 85.27 s;
  runs/nightly/offline-tests-v10.log.
- Полный Linux offline: 629 passed / 1 live deselected, 90.73 s;
  network=none, 2 CPU / 6 GiB; runs/nightly/linux-offline-v8/pytest.log.
- ZIP: 56 файлов, SHA-256
  03983dd361bd9f63fe7958554ec5f35b3f3746f8e8c46787b0338e20612409eb.
  После запуска gate-v7 архив повторно побайтово сопоставлен с текущими исходниками.
- Распакованный entrypoint проверен вне checkout в Linux без сети,
  Python 3.12.12, 4 CPU / 6 GiB: regression 22 449 строк, diabetes 88,
  оба exit=0, валидный CSV и успешный refit;
  runs/nightly/linux-archive-v8/report.json. Это search-mode=none, не Qwen gate.

Зависимости pyproject.toml/uv.lock не менялись. Посторонние контейнеры не
останавливались. Предыдущие артефакты сохранены; первый вспомогательный
archive-скрипт ранее остановился из-за PYTHONPATH и был исправлен до проверки.
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
| gate-v6 / regression | Не принята: forced edit HTTP 503 после навигации, changed=0; 104.4 s. Два read_file с file_path отклонены структурированно. CSV/refit проверены. |
| gate-v6 / classification | Не принята: первый full timed_out; исправленный кандидат прошёл full (macro-F1 0.5241378916 против 0.5652431446), но review не уместился в token budget. 1043.8 s; CSV/refit 41 166 строк проверены. |
| gate-v7 / regression | Не принята: первый запрос 503, retry успешен, но budget_exhausted до edit; changed=0, interpreted=0; 99.9 s. CSV/refit 22 449 строк проверены. |
| gate-v7 / classification | Не принята: TotalPrecision вызвал smoke failure; исправленный 600/d7 кандидат timed_out на full, затем budget_exhausted; changed=0, interpreted=0; 919.7 s. CSV/refit 41 166 строк проверены. |

У gate-v2 автоматический changed=0 для classification оказался ложным отказом:
ID baseline-catboost-sqrtbalanced был исключён по префиксу. Исправленный verifier
подтвердил по одному changed full на обеих задачах; counter-audit.json сохранён
отдельно. Оригинальный report не переписан, post-hoc replay не подставлен вместо
исходных summary. Из-за семантических отказов и изменения исходников старые gate
не допускают сравнение. Control scores для исправлений не просматривались.

Все серии использовали обе публичные задачи, seed 42, до трёх сессий и
прежние ceilings. Runtime/verification не менялись до terminal status каждой
серии. Булев gate_passed не заменяет проверку фактического кода, интерпретации
и CSV/refit. Последний locator runs/nightly/ACTIVE.json имеет terminal status; живого
процесса gate нет. При возобновлении проверять handle или PID/create_time,
а не полагаться на старый файл статуса.

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
сохранён и принят на 1ede55d; он не переносится на следующие исходники.
Gate-v4 завершился до следующих изменений runtime. Его payload подтверждает доставку
candidate_runtime_contract, forced edit_file и enable_thinking=False; это не устранило 503.

После gate-v4 в 13b8c80 исправлен учёт резерва: неудачный worker расходует
wall-clock, но не повышает прогноз стоимости успешных финалистов; fallback
учитывается как refit. Три быстрые регрессии воспроизвели дефекты до исправления.
В gate-v6 после full timeout Qwen действительно получила возможность ремонта
и успешно измерила меньший кандидат; затем остановилась перед review.

После terminal status gate-v6 воспроизведён отдельный дефект review-context:
при оставшихся 7 728 tokens исходный код/метрики с повторными инструкциями и
служебными полями не проходили консервативный request limit. Offline replay
исходного контекста, без API/ML, завершался после draft с budget_exhausted;
синтетическая регрессия подтвердила тот же отказ при неподвижных часах.
Исправление убирает из тесного review-контекста повторную инструкцию,
поисковый budget/direction, пути и хеши артефактов. Код, factory semantics,
гипотеза, full/parent/smoke-метрики, вся показанная диагностика и draft сохранены.
Offline replay теперь доходит до mocked review (5 918 bytes); это не новая
Qwen-интерпретация и не замена исходного gate report. 19 связанных тестов прошли.
Сводка оригинальных research API traces/session accounting:
runs/nightly/live-accounting.json. В smoke-v1…v5 и gate-v1…v7 — 106 попыток,
301 434 reported tokens, 336 504 списано по session accounting. Пять сессий
завершились без покрытия неопределённого расхода последней ошибки; продолжение
их сессий запрещено. Поэтому эта сумма не является верхней оценкой полного
расхода или фактическим биллингом. Диагностические 32 попытки учтены отдельно.

## Сравнение и следующий шаг

Матрица — 0/18, не начата. Измеренного эффекта Qwen против Optuna нет.
Диабетический holdout verifier открывает только после полного cohort.

После восстановления выданного endpoint повторить обе публичные задачи
в новом каталоге gate-v8 на том же fec115c, без объединения успешных частей
старых серий. Диагностические API probes за эту ночь больше не запускать.
Старые summary/report не исправлять post-hoc. Команда gate:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
uv run --locked --all-extras python -m verification.run_agent_comparison \
  data/regression/task.md data/classification/task.md --gate --modes qwen \
  --seeds 42 --sessions 3 --output runs/nightly/gate-v8
```

Проверить фактический код, full-метрики, scope, завершённый review и CSV/refit
обеих задач. Если исходники менялись, нужны проверки и новый gate того кода.

Только после приёмки обеих задач выполнить:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
uv run --locked --all-extras python -m verification.run_agent_comparison \
  data/regression/task.md data/classification/task.md runs/stage7/tasks/diabetes/task.md \
  --gate-report runs/nightly/gate-v8/report.json --output runs/nightly/comparison-v1
```

Сохранить все 18 попыток и парные эффекты по трём seed, обновить этот отчёт
   фактами; отрицательный эффект допустим. После просмотра final control/holdout
   новые изменения не подбирать. Gate и сравнение остаются незавершёнными до
   фактической проверки; готовность агента по этой записи не объявляется.

Остаются публичный gate обеих задач на последнем runtime, все 18 сравнений,
измерение эффекта и точная платформа 8 CPU / 32 GB. В ходе ночной работы control scores
и diabetes holdout не открывались. Все исходные отчёты сохранены;
неудачные серии не исключаются из истории. Merge/main/конкурсная отправка не выполнялись.
