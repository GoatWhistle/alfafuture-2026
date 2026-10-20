"""Builds task.md from the user's description; data files are always siblings of task.md."""

import json
import re

from .config import FILE_NAMES

_PATH_PREFIX = re.compile(
    r"(?:[\w.\-]+/)+(" + "|".join(re.escape(n) for n in FILE_NAMES.values()) + r")"
)
_ABSOLUTE_CSV = re.compile(r"(?<![\w.])(?:/|[A-Za-z]:[\\/]|~/)[^\s`'\")]*\.csv", re.IGNORECASE)


_PARENT_SEGMENT = re.compile(r"(^|[\s/\\`'\"(])\.\.([/\\]|$)", re.MULTILINE)
_METADATA = re.compile(r"```(?:task-json|json)\s*\n(.*?)```", re.S)
_INPUT_PATHS = {
    "train_path": FILE_NAMES["train"],
    "test_path": FILE_NAMES["test"],
    "dictionary_path": FILE_NAMES["dictionary"],
    "sample_path": FILE_NAMES["sample"],
}


class TaskTextError(ValueError):
    pass


def build_task_md(title: str, description: str) -> str:
    """Reject paths leaving the run directory and point every data reference to its own files."""
    text = title + "\n" + description
    if "\n" in title or "\r" in title:
        raise TaskTextError("Название должно быть одной строкой")
    if _PARENT_SEGMENT.search(text):
        raise TaskTextError("Описание содержит путь с «..»")
    if _ABSOLUTE_CSV.search(text):
        raise TaskTextError("Описание ссылается на абсолютный путь; используйте загруженные файлы")
    body = _PATH_PREFIX.sub(lambda m: m.group(1), description.strip())
    match = _METADATA.search(body)
    metadata = {}
    if match:
        try:
            metadata = json.loads(match.group(1))
        except ValueError:
            raise TaskTextError("Некорректный JSON в описании задачи") from None
        if not isinstance(metadata, dict):
            raise TaskTextError("task-json должен содержать объект")
    for key, filename in _INPUT_PATHS.items():
        if key in metadata and metadata[key] != filename:
            raise TaskTextError(f"{key} должен ссылаться на загруженный {filename}")
    output = metadata.get("submission_path", "submission.csv")
    if (
        not isinstance(output, str)
        or not re.fullmatch(r"[\w-]+\.csv", output)
        or output in FILE_NAMES.values()
    ):
        raise TaskTextError("submission_path должен быть отдельным именем CSV без каталогов")
    # Explicit paths take precedence over free-text hints in the harness parser.
    # Decode JSON before validation: escaped slashes must not bypass confinement.
    metadata.update(_INPUT_PATHS, submission_path=output)
    block = "```task-json\n" + json.dumps(metadata, ensure_ascii=False, indent=2) + "\n```"
    if match:
        body = body[: match.start()] + block + body[match.end() :]
    else:
        body = block + "\n\n" + body
    data = "\n".join(f"- `{name}`" for name in FILE_NAMES.values())
    return (
        f"# {title.strip()}\n\n{body}\n\n"
        "## Данные\n\nПути указаны относительно каталога этого Markdown-файла:\n\n"
        f"{data}\n"
    )
