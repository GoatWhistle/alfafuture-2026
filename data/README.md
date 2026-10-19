# Локальные данные

Источник — [папка организаторов](https://drive.google.com/drive/folders/1eUjfuovujgEoUbtotU9uRGWLo-RacY3T).

При подготовке прогонов разместить материалы подпапок «Регрессия» и «Многоклассовая классификация» соответственно в `data/regression/` и `data/classification/`. По ТЗ ожидаются train.csv, test.csv, data_dictionary.csv и sample_submission.csv. Состав и task.md проверить при скачивании.

Содержимое каталогов с данными исключено из Git. Тестовые синтетические примеры хранятся отдельно в `tests/fixtures/`. Артефакты обучения пишутся в `runs/`.

19.10.2026 скачаны оригинальные task.md и четыре CSV для каждой задачи. Официальные описания содержат вложенные относительные пути, поэтому локальная структура:

- `data/regression/task.md` и `data/regression/regression/participants/*.csv`;
- `data/classification/task.md` и `data/classification/multiclass/participants/*.csv`.

Оригинальные task.md не правились. В обоих словарях пять колонок, включая unit; роли target/id совпадают с train/sample. Данные не коммитятся.
