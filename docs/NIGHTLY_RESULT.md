# Ночная стабилизация Qwen: фактический результат

Промежуточный отчёт, работа продолжается в `fix/qwen-live-cycle` одним агентом.
Короткий измеренный цикл принят. Публичный gate выполняется; сравнение с Optuna
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

- Последний runtime 8319ce9: make check успешен; 75 связанных unit-тестов,
  отдельно 29 SDK-тестов с проверкой фактического тела и трассы.
- Полный macOS offline-прогон: 618 passed, 1 live deselected, 80.24 s;
  runs/nightly/offline-tests-v4.log.
- Полный Linux offline-прогон a40d167: 618 passed, 1 live deselected, 83.74 s,
  network=none, 2 CPU / 6 GiB; runs/nightly/linux-offline-v2/pytest.log.
- ZIP последнего runtime: 56 файлов, побайтовое совпадение с исходниками;
  SHA-256 b76d8ad7b9bc0a1f93a5b455e99b80bfd0af392ed9aa75feaf5aa5ce7d71155d.
- Распакованный entrypoint 8319ce9 проверен в Linux вне checkout, без сети,
  Python 3.12.12, 4 CPU / 6 GiB: regression 22 449 строк, diabetes 88,
  оба exit=0, валидный CSV, успешный refit. runs/nightly/linux-archive-v3/report.json.
  Это режим search-mode=none, не Qwen gate. Посторонние контейнеры не останавливались.

Предыдущие проверки и артефакты сохранены; первый технический запуск вспомогательного
archive-скрипта остановился до harness из-за PYTHONPATH, повтор исправлен.
Точная среда 8 CPU / 32 GB остаётся непроверенной.

## Публичный gate и сравнение

Новая серия runs/nightly/gate-v1 запущена на зафиксированном runtime 8319ce9:
regression и classification, seed 42, Qwen, до трёх сессий. Runtime и verification
во время серии не меняются. Требуется ручная проверка содержания отчёта и
интерпретации, одного gate_passed недостаточно.

| Публичная задача | Приёмка |
|---|---|
| regression | Выполняется |
| classification | Ожидает последовательного запуска |

Сравнение — 0/18, ещё не запущено. Измеренного эффекта Qwen против Optuna нет.
Полная матрица разрешена только после приёмки обеих публичных задач на этих исходниках;
diabetes holdout открывается verifier только после полного cohort. Control/holdout
не используются для выбора новых runtime-изменений.

## Открытые пункты и следующий шаг

1. Дождаться текущего gate через runner.log/manifest/session и проверить оба
   публичных запуска: самостоятельное изменение кода, успешный full, корректная
   интерпретация и CSV выбранного refit. Текущий процесс не перезапускать по таймауту наблюдения.
2. После приёмки выполнить 18 запусков с параметрами PLAN через:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
uv run --locked --all-extras python -m verification.run_agent_comparison \
  data/regression/task.md data/classification/task.md runs/stage7/tasks/diabetes/task.md \
  --gate-report runs/nightly/gate-v1/report.json --output runs/nightly/comparison-v1
```

3. Записать фактические парные эффекты и оставшиеся отказы; отрицательный результат
   допустим. После просмотра final control/holdout новые изменения не подбирать.
   Merge в main и конкурсная отправка не выполнялись и не входят в эту работу.
