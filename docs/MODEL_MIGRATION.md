# Приёмка целевой модели

20.10.2026 пользователь уточнил: текущая выданная Qwen сильнее, на защите будет
другая Qwen примерно 27B. Конфигурация целевой модели ожидается. Число параметров
не определяет checkpoint, model ID, API и поддержку thinking. Никакие результаты
на текущем alias `qwen3.8-flash` не доказывают переносимость на целевую модель.
Новый gate и сравнительную серию с Qwen запускать после получения целевой
конфигурации. Прежняя gate-1 и старые v2-результаты сохраняют исторический статус;
они не принимаются как доказательство v3 или целевой модели. Исправления и
проверки offline/Linux/ZIP выполняются независимо.

## Порядок после получения конфигурации

1. Зафиксировать точный запрошенный model ID, возвращаемый `response.model`,
   fingerprint endpoint, код и зависимости, дату/лимиты. Сам URL и ключ в отчёты
   не включать. Если provider раскрывает checkpoint — записать без догадок;
   alias и фактический checkpoint различать.
2. Ограниченная контрактная проверка без обучения: проверить tool calling
   с `propose_experiment` и совместимость набора `propose_experiment` и
   ограниченного read-only `column_details`; проверить JSON arguments,
   finish_reason и usage
   input/output/total, ограничения timeout/output и известность учёта retries.
   Для единственного `propose_experiment` текущий адаптер отправляет forced
   tool_choice и `enable_thinking=false`; для двух tools эти поля не отправляются.
   Фактические параметры обоих запросов сверить с trace.
   Поддержку этих полей у новой модели проверить отдельно; при 400/несовместимости
   сохранить ответ, адаптировать транспорт и проверить его offline до smoke.
   Не предполагать, что отключение thinking поддерживается или необходимо.
3. Новый изолированный diabetes smoke, без копирования/чтения holdout targets:
   через `verification.verify_archive.copy_task` копируются только четыре
   входных файла и task.md. Ceiling 600 s, 4 threads, 6000 MiB, worker 120 s,
   32000 tokens, 3 гипотезы. Сохранить source hashes и API trace.
   `verify_run` должен подтвердить предложенный целевой моделью changed full,
   совпадение фактического рецепта/параметров и финального CSV с frozen refit.
   Проверить, что следующая гипотеза видит выбранные Optuna параметры и факты.
4. На неизменных исходниках — обе публичные задачи, seed 42:

   ```bash
   uv run --locked python -m verification.run_agent_comparison \
     data/regression/task.md data/classification/task.md \
     --gate --modes qwen --seeds 42 --sessions 12 \
     --output runs/pipeline-v3/target-gate-1
   ```

   Ceiling 1800 s / 4 threads / 6000 MiB / worker 300 s / 32000 tokens.
   На каждой задаче нужен хотя бы один новый успешно измеренный full и verified
   CSV/refit. Не переносить boolean, traces или score из старого gate. При
   provider failure сохранить частичный результат; не повторять до зелёного.
5. Только после принятого gate v3 на целевой модели — новая последовательная
   серия из 75 запусков: пять режимов × пять задач × seeds 7/42/2026.
   Режимы: замороженный прежний portfolio+Optuna (`frozen`), новый Optuna,
   гибрид Qwen, AutoGluon CAT/GBM/XGB и FLAML CPU. Подготовить три локальные
   задачи и запустить отдельный исполнитель сравнений:

   ```bash
   uv run --locked python -m verification.benchmark_tasks \
     --output runs/benchmarks/tasks
   uv run --locked python -m verification.run_hybrid_comparison \
     data/regression/task.md data/classification/task.md \
     runs/benchmarks/tasks/diabetes/task.md \
     runs/benchmarks/tasks/breast_cancer/task.md \
     runs/benchmarks/tasks/wine/task.md \
     --modes frozen optuna qwen autogluon flaml --seeds 7 42 2026 \
     --time-budget 1800 --threads 4 \
     --gate-report runs/pipeline-v3/target-gate-1/report.json \
     --output runs/pipeline-v3/target-cohort
   ```

   Исполнитель закрепляет исходники, baseline ref и конфигурацию, создаёт
   benchmark-копии всех пяти задач с отдельными внешними targets и проверяет
   свежий `--gate-report` против текущего кода, модели и fingerprint endpoint.
   Публичные исходники для functional gate остаются исходными. Для каждого
   запуска ceiling — 1800 s / 4 threads / 6000 MiB; worker timeout в этом
   исполнителе равен общему time-budget. Qwen использует лимиты CLI v3:
   максимум 12 гипотез и 32000 tokens; фактическое число ограничено бюджетом.
   Control не участвует в поиске; внешняя оценка всех пяти задач открывается
   только после завершения и проверки всего cohort. Повтор после просмотра
   внешних результатов требует нового заранее зафиксированного разбиения.

   Без целевой конфигурации допустима отдельная offline-серия
   `--modes frozen optuna autogluon flaml`; она не подтверждает пользу Qwen.
   Результаты и денежный объём live-серии согласуются с известной квотой
   провайдера до запуска. Использовать новые каталоги: старый v2-run доступен
   для чтения, его resume исполнителем v3 отклоняется.

Функциональную надёжность, качество гипотез и парную пользу фиксировать отдельно.
При смене модели/transport settings/runtime новый gate обязателен. Точное
окружение платформы 8 CPU/32 GiB остаётся самостоятельным непроверенным условием.
