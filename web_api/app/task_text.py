"""Builds task.md from the user's description; data files are always siblings of task.md."""

import re

from .config import FILE_NAMES

_PATH_PREFIX = re.compile(
    r"(?:[\w.\-]+/)+(" + "|".join(re.escape(n) for n in FILE_NAMES.values()) + r")"
)
_ABSOLUTE_CSV = re.compile(r"(?<![\w.])(?:/|[A-Za-z]:[\\/]|~/)[^\s`'\")]*\.csv", re.IGNORECASE)


_PARENT_SEGMENT = re.compile(r"(^|[\s/\\`'\"(])\.\.([/\\]|$)", re.MULTILINE)


class TaskTextError(ValueError):
    pass


def build_task_md(title: str, description: str) -> str:
    """Reject paths leaving the run directory and point every data reference to its own files."""
    if _PARENT_SEGMENT.search(description):
        raise TaskTextError("Описание содержит путь с «..»")
    if _ABSOLUTE_CSV.search(description):
        raise TaskTextError("Описание ссылается на абсолютный путь; используйте загруженные файлы")
    body = _PATH_PREFIX.sub(lambda m: m.group(1), description.strip())
    data = "\n".join(f"- `{name}`" for name in FILE_NAMES.values())
    return (
        f"# {title.strip()}\n\n{body}\n\n"
        "## Данные\n\nПути указаны относительно каталога этого Markdown-файла:\n\n"
        f"{data}\n"
    )
