# Реализация harness v3

Работа в отдельной ветке `feat/refactor-harness-v3`, исходная точка `3327e07`.
Требования — [REFACTOR-HARNESS.md](REFACTOR-HARNESS.md), этап 10
[PLAN.md](PLAN.md). Исторические live-критерии этапа 9 остаются открытыми.

## Изменения

| Этап | Реализация | Проверяемые сценарии |
|---|---|---|
| A | `adapters/data/profile.py`, `domain/dataset_profile.py`, context/proposal tools, диагностика всех признаков | train-only статистики, поздний проблемный признак, сохранение бюджета/метрики/рецепта/свидетельства, ограниченные read-only запросы, неизвестный target_error |
| B | FeatureRecipe + TunedCandidate, общий каталог параметров/балансировки, PipelineRecipe | новый pipeline каждого trial, fit только inner-train, group/time до drop, early stopping, fingerprint study, наследование fitted параметров/признаков, frozen worker refit |
| C | application `hybrid_search.py`, общий numeric recipe/admission, checkpoint publisher | больше трёх гипотез, чередование, local-only после API/unknown usage/стагнации, сохранение стагнации при resume, реальные worker после отказа API, CSV без control |
| D | вероятности worker, согласование ID/классов, три фиксированных веса на development | переставленные классы/ID, замороженные веса, control не меняет ranking, сохранение label-rule одиночной модели, линейный кандидат отдельно |
| E | `run-result.json` v3, CLI/verifier/панель, обновлённые инструкции | отдельные состояния CSV/поиска, происхождение/ветка/отказы/local-only, восстановление checkpoint после interruption, reject resume v2, просмотр legacy |

`ExperimentResult` использует существующий dataclass/state machine `Experiment`;
`PipelineRecipe` — совместимое имя полного `ResolvedExperiment`. Работающие
runner/evaluator/finalization сохранены. Новые интерфейсы не дублируют их.

Пустые поля признаков предложения наследуют преобразования и Python родителя;
непустые заменяют соответствующее поле. Параметры того же backend наследуются
с явными overrides; смена backend сохраняет признаки и задаёт defaults новой модели.
Численный шаг развивает лучший полный pipeline; ранний stopping count родителя
не ограничивает новый поиск одним деревом — следующий рецепт явно возвращает
потолок минимум 200 деревьев с независимым smoke/admission.

Итог содержит выбранный recipe (включая members/weights ансамбля), score/protocol,
происхождение и отдельные result/search status. В interruption canonical summary
связывается с текущим CSV по checksum checkpoint; неизвестный usage не объявляется
нулевым. Peak RSS дерева worker измеряется выборочно с интервалом 50 ms;
это наблюдаемый пик, а не доказательство абсолютного максимума между выборками.

FLAML установлен как optional extra `automl`, версия 2.3.6 соответствует
`task/reference/requirements.txt`, lock обновлён. Его выбранная конфигурация
переобучается на всём разрешённом train и сохраняется для refit без AutoML-поиска.
Тесты выявили и закрыли внутреннее удержание 10% CatBoost при refit.
AutoGluon comparison включает CAT/GBM/XGB и weighted ensemble без stacking;
исторический CAT-only режим сохранён. Линейная модель не добавлена в portfolio.

## Проверки

Проверки выполнены на исходниках этой ветки; live-тест исключён явно.

- `make check`: lock, Ruff, format, mypy, оба import contracts проходят.
- `make web-check`: build, 6 frontend tests и 30 API tests прошли.
- Отдельные тесты реального FLAML regression/multiclass/group/time/refit прошли.
- `uv run --locked pytest -m 'not live' -q`: **699 passed / 1 deselected**, macOS, 125.21 s.
- Linux: **699 passed / 1 deselected**, 129.28 s, Python 3.12, контейнер
  `alfa-harness-v3:py312`, `--network none --cpus 4 --memory 6g`.
- После последней правки полей сравнительного отчёта: ещё 2 comparator-теста прошли.
- ZIP `dist/submission-v3.zip`: 69 файлов, побайтовое совпадение с исходниками;
  SHA-256 `b2a9ebf9ca9e151f6dd4edb619b369ff12e441830c27707e4bfa2438fa03c790`.
  Независимый результат сверки: `runs/pipeline-v3/checks/archive-bytes.json`.
- Реальный Linux AutoGluon smoke без сети: обучены `LightGBM`, `CatBoost`,
  `XGBoost`, `WeightedEnsemble_L2`; выбран `WeightedEnsemble_L2_FULL`.
  Вероятности проверены по форме/конечности/сумме. Артефакт:
  `runs/pipeline-v3/checks/autogluon-full-smoke.json`.
- Распакованный ZIP, первый проход: 5/5 CSV независимо валидны (22 449,
  41 166, 88, 113, 36 строк). Regression и три локальные задачи завершили refit.
  Classification на коротком лимите 120 s завершилась `fallback_preserved`:
  deadline сохранил ранний CSV; это не засчитано как успешный полный refit.
  Отчёт: `runs/pipeline-v3/archive-check/report.json`.
- Свежая classification с лимитом 600 s: **refit succeeded**, 41 166 строк,
  wall 506.63 s. Отчёт: `runs/pipeline-v3/archive-classification-600/report.json`.
- Все пять итоговых CSV независимо сравнены с предсказаниями выбранных frozen
  refit-моделей; SHA-256 worker-артефактов сверены с finalization state.
  **5/5 refit matches CSV**, без сети. Доказательство:
  `runs/pipeline-v3/checks/archive-refits.json`.
- Логи всех финальных проверок: `runs/pipeline-v3/checks/`.
  Эти результаты закрывают offline-корректность, а не независимый прирост качества.

Предварительные прогоны во время параллельной разработки не считаются итоговой
приёмкой: freeze-тесты правильно отклоняли изменение source hashes во время теста.
Также адаптированы старые тесты, ожидавшие завершение поиска после трёх гипотез:
policy проверяется с управляемыми часами, а семантические тесты ограничивают
независимый numeric fallback. Реальный worker continuation имеет отдельный тест.

## Независимое качество и открытые условия

`verification/run_hybrid_comparison.py` создаёт отдельные benchmark-копии,
оставляя исходные публичные материалы неизменными. Он фиксирует commit старого
portfolio, source hashes нового runtime, параметры и входы; запускает режимы
последовательно и открывает внешние targets только после всего cohort.
В Qwen-режиме требует принятого functional gate с теми же исходниками/endpoint.

Полная серия: пять задач, seeds 7/42/2026, режимы frozen/optuna/qwen/autogluon/flaml,
1800 s / 4 CPU / 6000 MiB. Зафиксировать старую исходную точку явно:
`--baseline-ref 3327e07`. По умолчанию закреплён `3327e07`; другой baseline задаётся явным историческим SHA.

Серия не запускалась: конфигурация целевой Qwen ещё не выдана. Transport/tool-call,
smoke, functional gate и сравнение на целевой модели остаются открытыми.
Старый endpoint не вызывается. Код сравнительных движков и тесты не доказывают
прирост качества; критерий пользы из задания ещё не измерен.

Полный конкурсный режим 3 часа / 8 CPU / 32 GB также не принят. Доступная Docker
VM имеет 12 CPU и около 7.65 GiB RAM; проверка контейнера на 4 CPU / 6 GiB не
заменяет проверку на 32 GB. Результаты запусков, ZIP и логи остаются локально
в игнорируемых путях; ключи и данные в Git не добавляются.


Команды Linux и сборки:

```bash
docker build -f verification/Dockerfile -t alfa-harness-v3:py312 .
docker run --rm --network none --cpus 4 --memory 6g \
  -v "$PWD/runs:/workspace/runs" alfa-harness-v3:py312 \
  python -m pytest -m 'not live' -p no:cacheprovider -q
uv run --locked python scripts/build_submission.py --output dist/submission-v3.zip
```


ZIP проверялся через `verification.verify_archive` в контейнере с теми же
`--network none --cpus 4 --memory 6g`, `--threads 4 --memory-limit-mb 5000`.
Первый запуск: обе публичные задачи и `runs/stage7/tasks/{diabetes,breast_cancer,wine}/task.md`,
`--time-budget 120 --output runs/pipeline-v3/archive-check`.
Второй: только `data/classification/task.md`,
`--time-budget 600 --output runs/pipeline-v3/archive-classification-600`.
Исходники/ZIP не менялись между запусками. Скрипт независимой сверки сохранён
локально: `runs/pipeline-v3/checks/verify_archive_refits.py`.
