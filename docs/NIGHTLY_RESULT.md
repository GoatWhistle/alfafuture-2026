# Ночная стабилизация Qwen: фактический результат

Промежуточный отчёт, работа продолжается в `fix/qwen-live-cycle` одним агентом.
Короткий измеренный цикл принят. Первый публичный gate не принят, второй выполняется; сравнение с Optuna
ещё не разрешено. Польза агента пока не измерена.

## 503 и некорректные tool calls

503 воспроизводятся примерно за 60–66 s, с `capacity_unavailable`. Снятие
forced tool_choice не устраняет сбой. Серверные capacity, timeout и routing
без телеметрии endpoint не разделены; перегрузка не доказана.

Первый раунд (`runs/nightly/provider-round1`): 12 последовательных HTTP-попыток,
SDK retries выключены. Исходный payload smoke-v6 в ABBA forced/auto/auto/forced
дал 503, 503, HTTP 200 с inspect_data, HTTP 200 с inspect_data; разрешён только
edit_file. Аргументы этих inspect_data также невалидны. Короткие messages
с прежними tools/max_tokens дали два валидных edit_file. Замена только system
prompt и удаление только повторных страниц дали по два 503. Исходные messages/tools
с max_tokens=1536 вместо 4096 дали два валидных edit_file, но позднейшие smoke
подтвердили, что уменьшение output cap не гарантирует устранения 503.

Второй раунд (`runs/nightly/provider-round2`): 8 HTTP-попыток на payload smoke-v3.
Четыре default forced/auto/auto/forced дали 503. С enable_thinking=False
forced/auto дали валидный edit_file. В обратной паре default тоже восстановился,
а false снова успешен. Итого false — 3/3 валидных ответа; это не доказывает,
что все 503 вызваны thinking. Реальное семейство за alias qwen3.8-flash не установлено.

[Документация Qwen](https://docs.qwencloud.com/developer-guides/tool-calling/function-calling)
предписывает отключать thinking перед forced выбором функции. Исправление
применено только для единственной принудительно выбранной функции; в трассе
сохраняются tool_choice и enable_thinking. Корректность tools дополнительно
проверяется в application до исполнения, независимо от поведения endpoint.

| Диагностика | HTTP-попытки | Reported tokens | Консервативно учтено |
|---|---:|---:|---:|
| Раунд 1 | 12 | 13 740 | 91 070 |
| Раунд 2 | 8 | 10 667 | 52 747 |

Учёт отделён от исследовательских запусков: accounting.json в каждом каталоге.
Для ответов без usage использована верхняя граница UTF-8 prompt bytes + запас +
max_tokens. Точный биллинг не подтверждён. Некоторые ответы сообщают 1–2 output
tokens и полностью cached prompt при нескольких аргументах tools; эти значения
сохранены как reported, а не как доказанный полный расход. Ключи, полные payload,
ответы и логи остаются в игнорируемом runs/; в Git не включены.

## Изменения

Все коммиты отправлены в origin/fix/qwen-live-cycle от настроенного Git-пользователя,
без соавторства и переписывания истории.

| Commit | Изменение |
|---|---|
| d46c7c6 | Инструменты вне схемы текущего запроса отклоняются до executor; ошибка возвращается модели, фаза и расход сохраняются. |
| a40d167 | Bounded generation ограничена 1536 tokens; action-фазы получают system prompt без стартовой навигации. Запуск и интерпретация обязаны сверять гипотезу с исполняемым candidate_code. |
| 8319ce9 | Thinking отключён для forced tool choice; параметры выбора и режима сохраняются в API-трассе. |
| 272fab3 | Ограниченная строка исключения stderr и failure_phase доступны модели; failed-кандидат возвращает её в edit_file для исправления. |
| 980cb43 | В контекст Qwen передаются измеренные baseline validation/refit seconds, worker/smoke timeout и число потоков. |

Регрессии сначала воспроизвели нарушения на реальных границах SDK/session,
затем прошли после исправлений. Scripted-тесты не считаются live-приёмкой.

## Реальные короткие запуски

Каждая diabetes скопирована через verification.verify_archive.copy_task:
только условие и четыре входных файла. Holdout targets не переданы и не открывались.

| Каталог в runs/nightly | Результат |
|---|---|
| smoke-v1 | Изменённый full-кандидат: RMSE 62.4347247663 против 62.4779271142. CSV/refit проверены, но интерпретация приписала результат отсутствующим параметрам. Семантический отказ, несмотря на interpreted_successes=1 у verifier. 7 API-попыток без 503, 24 334 reported/charged tokens; затем budget_exhausted. |
| smoke-v2 | API error на edit_file после двух 503. Изменённых full-результатов нет; baseline CSV/refit проверены. |
| smoke-v3 | API error на edit_file после двух 503, несмотря на отдельный system prompt фазы. Изменённых full-результатов нет; baseline CSV/refit проверены. |
| smoke-v4 | Принят: Qwen написала CatBoost depth=4, lr=0.03, l2=9, iterations=600; full RMSE 60.1898931077 против 62.4779271142. Гипотеза и интерпретация соответствуют коду/метрике, есть оговорка об одном split. verify_run подтвердил 88 строк CSV и совпадение с выбранным refit. 5 API-попыток, 14 606 reported/charged tokens, usage_known=true, без tool errors/503. |

В каталогах сохранены verification.json; в smoke-v1 и smoke-v4 дополнительно
semantic-review.json. Последний smoke подтверждает инженерный цикл, а не
переносимость выигрыша на скрытый тест или пользу относительно Optuna.

## Проверки и ZIP

- Последний runtime 980cb43: make check успешен; SDK-тесты проверяют тело/трассу,
  integration-регрессия — настоящий worker failure и исправление в той же сессии.
- Полный macOS offline-прогон: 619 passed, 1 live deselected, 82.42 s;
  runs/nightly/offline-tests-v6.log.
- Полный Linux offline-прогон 980cb43: 619 passed, 1 live deselected, 84.88 s,
  network=none, 2 CPU / 6 GiB; runs/nightly/linux-offline-v4/pytest.log.
- ZIP последнего runtime: 56 файлов, побайтовое совпадение с исходниками;
  SHA-256 286f31b3ebb10e21ebac900710619b90b0ea52a4dcb279e71e20ed59bf7024fb.
- Распакованный entrypoint 980cb43 проверен в Linux вне checkout, без сети,
  Python 3.12.12, 4 CPU / 6 GiB: regression 22 449 строк, diabetes 88,
  оба exit=0, валидный CSV, успешный refit. runs/nightly/linux-archive-v4/report.json.
  Это режим search-mode=none, не Qwen gate. Посторонние контейнеры не останавливались.

Предыдущие проверки и артефакты сохранены; первый технический запуск вспомогательного
archive-скрипта остановился до harness из-за PYTHONPATH, повтор исправлен.
Точная среда 8 CPU / 32 GB остаётся непроверенной.

## Публичный gate и сравнение

Серия runs/nightly/gate-v1 завершена на зафиксированном runtime 8319ce9:
regression и classification, seed 42, Qwen, до трёх сессий. Runtime и verification
во время серии не меняются. Требуется ручная проверка содержания отчёта и
интерпретации, одного gate_passed недостаточно.

| Публичная задача | Приёмка |
|---|---|
| regression | Не принята: changed=0, interpreted=0, budget_exhausted; CSV/refit проверены (22 449 строк), wall=101.6 s. |
| classification | Не принята: changed=0, interpreted=0, budget_exhausted; CSV/refit проверены (41 166 строк), wall=552.7 s. |

Оба процесса exit=0; report.json gate_passed=false и acceptance-review.json
сохранены. Control scores при диагностике не просматривались. Причины:
LightGBM early_stopping_rounds без eval set и CatBoost class_weights="Balanced".
Модель получала лишь generic worker exit, а stderr был недоступен через tools.
Исправлена общая граница (272fab3): ограниченная последняя строка stderr в error,
сохранение failure_phase и переход к edit_file после неуспешного эксперимента.
Регрессия с настоящим LightGBM/worker сначала упала, затем исправила кандидата
в той же ReAct-сессии и завершила full за 2.49 s. Полный macOS offline-прогон 272fab3 — 619 passed (81.85 s); Linux — 619 passed
(86.41 s), оба с 1 live deselected. В 980cb43 дополнительно переданы measured
baseline timings и worker/smoke limits: реальный baseline classification
занял 93.6/120.1 s, а Qwen предложила 3000 итераций. Проверка выявила отсутствие
этих полей на границе entrypoint → model, затем прошла. make check и полный
macOS offline 980cb43 — 619 passed / 1 live deselected (82.42 s).
Последние Linux/ZIP проверки успешны (см. выше).
На frozen runtime 980cb43 выполняется новый gate-v2 на обеих задачах.
Regression выполняется; classification ожидает последовательного запуска.
Исходники runtime/verification не меняются; после окончания необходимо
проверить содержание report, код кандидатов и интерпретации.

Сравнение — 0/18, ещё не запущено. Измеренного эффекта Qwen против Optuna нет.
Полная матрица разрешена только после приёмки обеих публичных задач на этих исходниках;
diabetes holdout открывается verifier только после полного cohort. Control/holdout
не используются для выбора новых runtime-изменений.

## Открытые пункты и следующий шаг

1. Дождаться текущего gate-v2 через runner.log/manifest/session и проверить
   обе задачи: изменение кода, full, корректная интерпретация, refit CSV.
   Живой процесс не перезапускать из-за таймаута наблюдения.
2. После приёмки выполнить 18 запусков с параметрами PLAN через:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
uv run --locked --all-extras python -m verification.run_agent_comparison \
  data/regression/task.md data/classification/task.md runs/stage7/tasks/diabetes/task.md \
  --gate-report runs/nightly/gate-v2/report.json --output runs/nightly/comparison-v1
```

3. Записать фактические парные эффекты и оставшиеся отказы; отрицательный результат
   допустим. После просмотра final control/holdout новые изменения не подбирать.
   Merge в main и конкурсная отправка не выполнялись и не входят в эту работу.
