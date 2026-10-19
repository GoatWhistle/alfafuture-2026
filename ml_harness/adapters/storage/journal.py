"""Fsynced JSONL facts, explicit tail recovery and content-addressed candidates."""

import fcntl
import hashlib
import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

from ml_harness.domain.models import ContractError


def _event(value: Any) -> dict[str, Any]:
    if not (
        isinstance(value, dict)
        and isinstance(value.get("event_id"), str)
        and isinstance(value.get("kind"), str)
        and isinstance(value.get("payload"), dict)
    ):
        raise ContractError("Journal event requires string event_id/kind and dictionary payload")
    return value


def _invalid_constant(value: str) -> None:
    raise ValueError("Nonfinite JSON constant")


class JsonlExperimentStore:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "events.jsonl"

    @contextmanager
    def _lock(self) -> Iterator[None]:
        with (self.root / ".journal.lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def _read(self) -> list[dict[str, Any]]:
        data = self.path.read_bytes() if self.path.exists() else b""
        events: dict[str, dict[str, Any]] = {}
        offset = 0
        lines = data.splitlines(keepends=True)
        for index, line in enumerate(lines):
            try:
                value = json.loads(line.decode("utf-8"), parse_constant=_invalid_constant)
            except (UnicodeDecodeError, ValueError):
                if index != len(lines) - 1 or line.endswith(b"\n"):
                    raise ContractError(f"Broken journal record at byte {offset}") from None
                recovery = self.root / "recovery"
                recovery.mkdir(exist_ok=True)
                with (recovery / f"{uuid4().hex}.bin").open("xb") as stream:
                    stream.write(data[offset:])
                    stream.flush()
                    os.fsync(stream.fileno())
                with self.path.open("r+b") as stream:
                    stream.truncate(offset)
                    stream.flush()
                    os.fsync(stream.fileno())
                break
            event = _event(value)
            event_id = event["event_id"]
            if event_id in events and events[event_id] != event:
                raise ContractError(f"Conflicting journal event ID: {event_id}")
            events[event_id] = event
            offset += len(line)
        return list(events.values())

    def read_events(self) -> list[dict[str, Any]]:
        with self._lock():
            return self._read()

    def append(self, event: dict[str, Any]) -> None:
        _event(event)
        try:
            encoded = json.dumps(event, ensure_ascii=False, allow_nan=False).encode("utf-8")
            normalized = _event(json.loads(encoded))
        except (ValueError, TypeError, UnicodeError):
            raise ContractError("Journal event must contain finite JSON data") from None
        with self._lock():
            existing = {item["event_id"]: item for item in self._read()}
            if normalized["event_id"] in existing:
                if existing[normalized["event_id"]] != normalized:
                    raise ContractError("Conflicting journal event ID")
                return
            with self.path.open("a+b") as stream:
                stream.seek(0, os.SEEK_END)
                if stream.tell():
                    stream.seek(-1, os.SEEK_END)
                    if stream.read(1) != b"\n":
                        stream.write(b"\n")
                stream.write(encoded + b"\n")
                stream.flush()
                os.fsync(stream.fileno())

    def save_code(self, code: bytes) -> tuple[str, str]:
        digest = hashlib.sha256(code).hexdigest()
        directory = self.root / "versions"
        directory.mkdir(exist_ok=True)
        destination = directory / f"{digest}.py"
        with self._lock():
            if destination.exists():
                if (
                    destination.is_symlink()
                    or hashlib.sha256(destination.read_bytes()).hexdigest() != digest
                ):
                    raise ContractError("Candidate version hash mismatch")
            else:
                temporary = directory / f".{uuid4().hex}.tmp"
                try:
                    with temporary.open("xb") as stream:
                        stream.write(code)
                        stream.flush()
                        os.fsync(stream.fileno())
                    temporary.chmod(0o444)
                    os.link(temporary, destination)
                finally:
                    temporary.unlink(missing_ok=True)
            destination.chmod(0o444)
        return digest, str(destination)
