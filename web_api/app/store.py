"""Per-run directories: inputs, harness artifacts, logs and a small atomic meta file."""

import json
import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

_ID = re.compile(r"^[0-9a-f]{32}$")


class NotFound(LookupError):
    pass


def is_run_id(value: str) -> bool:
    return bool(_ID.match(value))


def atomic_write(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


class RunStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def new_id(self) -> str:
        return uuid.uuid4().hex

    def path(self, run_id: str) -> Path:
        """Only generated identifiers resolve; anything else cannot name a path."""
        if not is_run_id(run_id) or not (self.root / run_id).is_dir():
            raise NotFound(run_id)
        return self.root / run_id

    def create(self, run_id: str, meta: dict[str, Any]) -> Path:
        directory = self.root / run_id
        (directory / "inputs").mkdir(parents=True)
        (directory / "harness").mkdir()
        self.write_meta(run_id, meta)
        return directory

    def write_meta(self, run_id: str, meta: dict[str, Any]) -> None:
        atomic_write(
            self.root / run_id / "meta.json", json.dumps(meta, ensure_ascii=False, indent=2)
        )

    def meta(self, run_id: str) -> dict[str, Any]:
        directory = self.path(run_id)
        try:
            value = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise NotFound(run_id) from error
        if not isinstance(value, dict):
            raise NotFound(run_id)
        return value

    def ids(self) -> list[str]:
        return sorted(p.name for p in self.root.iterdir() if p.is_dir() and is_run_id(p.name))


def iso_now() -> str:
    return datetime.now().isoformat(timespec="seconds")
