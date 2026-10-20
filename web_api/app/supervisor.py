"""Starts solution.py as a separate process; one active ML run at a time."""

import os
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import psutil

from .config import Settings
from .store import RunStore, atomic_write, iso_now


class Conflict(RuntimeError):
    pass


class AbortFailed(RuntimeError):
    pass


def _alive(info: dict[str, Any] | None) -> bool:
    if not info:
        return False
    try:
        process = psutil.Process(int(info["pid"]))
        same = abs(process.create_time() - float(info["create_time"])) < 1.0
        return same and process.status() != psutil.STATUS_ZOMBIE
    except (psutil.Error, KeyError, ValueError, TypeError):
        return False


def command_for(settings: Settings, meta: dict[str, Any], run_dir: Path) -> list[str]:
    options = meta["options"]
    command = [
        settings.python,
        str(settings.solution),
        str(run_dir / "inputs" / "task.md"),
        "--run-dir",
        str(run_dir / "harness"),
        "--seed",
        str(options["seed"]),
        "--threads",
        str(min(8, max(1, options["threads"]))),
        "--time-budget",
        str(min(10800, options["time_sec"])),
        "--worker-timeout",
        str(options["worker_timeout_sec"]),
        "--memory-limit-mb",
        str(min(28000, options["memory_gb"] * 1024)),
        "--llm-token-limit",
        str(options["tokens"]),
    ]
    if options["mode"] == "baseline":
        command.append("--baseline-only")
    else:
        command += ["--search-mode", options["mode"]]
    return command


class Supervisor:
    def __init__(self, settings: Settings, store: RunStore) -> None:
        self.settings = settings
        self.store = store
        self._lock = threading.Lock()

    # -- process state -------------------------------------------------

    def _process_info(self, run_id: str) -> dict[str, Any] | None:
        import json

        try:
            value = json.loads((self.store.path(run_id) / "process.json").read_text("utf-8"))
        except (OSError, ValueError):
            return None
        return value if isinstance(value, dict) else None

    def is_running(self, run_id: str) -> bool:
        return _alive(self._process_info(run_id))

    def active_run(self) -> str | None:
        return next((i for i in self.store.ids() if self.is_running(i)), None)

    def exit_code(self, run_id: str) -> int | None:
        try:
            return int((self.store.path(run_id) / "exit.code").read_text("utf-8").strip())
        except (OSError, ValueError):
            return None

    def aborted(self, run_id: str) -> bool:
        return (self.store.path(run_id) / "abort.flag").exists()

    # -- actions ---------------------------------------------------------

    def start(self, run_id: str) -> None:
        with self._lock:
            run_dir = self.store.path(run_id)
            meta = self.store.meta(run_id)
            if meta.get("started_at") or (run_dir / "process.json").exists():
                raise Conflict("Исследование уже запущено")
            active = self.active_run()
            if active is not None:
                raise Conflict("Уже выполняется другое исследование")
            command = command_for(self.settings, meta, run_dir)
            wrapper = [
                "/bin/sh",
                "-c",
                '"$@"; echo $? > "$0"',
                str(run_dir / "exit.code"),
                *command,
            ]
            with (
                (run_dir / "stdout.log").open("ab") as out,
                (run_dir / "stderr.log").open("ab") as err,
            ):
                process = subprocess.Popen(
                    wrapper,
                    cwd=self.settings.repo_root,
                    stdout=out,
                    stderr=err,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True,
                )
            created = psutil.Process(process.pid).create_time()
            atomic_write(
                run_dir / "process.json",
                f'{{"pid": {process.pid}, "create_time": {created}}}',
            )
            meta["started_at"] = iso_now()
            meta["started_unix"] = time.time()
            self.store.write_meta(run_id, meta)
            threading.Thread(target=process.wait, daemon=True).start()  # reap zombie

    def abort(self, run_id: str, grace: float = 5.0, limit: float = 15.0) -> None:
        """Ends the whole tree and returns only after the process is observed gone."""
        run_dir = self.store.path(run_id)
        info = self._process_info(run_id)
        if not _alive(info):
            raise Conflict("Запуск не выполняется")
        atomic_write(run_dir / "abort.flag", iso_now())
        try:
            root = psutil.Process(int(info["pid"]))  # type: ignore[index]
            tree = [root, *root.children(recursive=True)]
        except psutil.Error:
            tree = []
        group = int(info["pid"])  # type: ignore[index]
        for sig in (signal.SIGTERM, signal.SIGKILL):
            for process in tree:
                try:
                    process.send_signal(sig)
                except psutil.Error:
                    pass
            try:
                os.killpg(group, sig)  # catches workers spawned after the snapshot
            except OSError:
                pass
            _, alive = psutil.wait_procs(tree, timeout=grace if sig == signal.SIGTERM else 2.0)
            if not alive:
                break
        deadline = time.monotonic() + limit
        while self.is_running(run_id) and time.monotonic() < deadline:
            time.sleep(0.1)
        if self.is_running(run_id):
            raise AbortFailed("Процесс не завершился; запуск по-прежнему выполняется")
