# Приёмка целевой модели

20.10.2026 пользователь уточнил: текущая выданная Qwen сильнее, на защите будет
другая Qwen примерно 27B. Конфигурация целевой модели ожидается. Число параметров
не определяет checkpoint, model ID, API и поддержку thinking. Никакие результаты
на текущем alias `qwen3.8-flash` не доказывают переносимость на целевую модель.
Новую gate-2 и матрицу на прежнем endpoint не запускать. Уже запущенная gate-1
завершается; исправления и offline/Linux/ZIP выполняются независимо.

## Порядок после получения конфигурации

1. Зафиксировать точный запрошенный model ID, возвращаемый `response.model`,
   fingerprint endpoint, код и зависимости, дату/лимиты. Сам URL и ключ в отчёты
   не включать. Если provider раскрывает checkpoint — записать без догадок;
   alias и фактический checkpoint различать.
2. Один ограниченный контрактный запрос без обучения: проверить tool calling
   с единственным `propose_experiment`, JSON arguments, finish_reason и usage
   input/output/total, ограничения timeout/output и известность учёта retries.
   Текущий адаптер отправляет forced tool_choice и `enable_thinking=false`.
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
     --gate --modes qwen --seeds 42 --sessions 3 \
     --output runs/pipeline-v2/target-gate-1
   ```

   Ceiling 1800 s / 4 threads / 6000 MiB / worker 300 s / 32000 tokens.
   На каждой задаче нужен хотя бы один новый успешно измеренный full и verified
   CSV/refit. Не переносить boolean, traces или score из старого gate. При
   provider failure сохранить частичный результат; не повторять до зелёного.
5. Только после принятого gate на целевой модели — 18 последовательных запусков
   Qwen/Optuna на двух публичных задачах и diabetes, seeds 7/42/2026, те же
   ресурсные ceilings и 3 гипотезы. Передать `--gate-report` свежего gate;
   verifier перепроверит артефакты и fingerprint. Control не участвует в поиске,
   diabetes holdout открывается только после всего cohort.

Функциональную надёжность, качество гипотез и парную пользу фиксировать отдельно.
При смене модели/transport settings/runtime новый gate обязателен. Точное
окружение платформы 8 CPU/32 GiB остаётся самостоятельным непроверенным условием.
