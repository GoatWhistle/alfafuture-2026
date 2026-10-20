"""Assembles API documents from meta, process state and harness artifacts."""

import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from . import reader
from .config import REPO_ROOT, Settings
from .store import RunStore
from .supervisor import Supervisor


def qwen_status() -> dict[str, Any]:
    model = os.environ.get("OPENAI_MODEL", "")
    configured = all(os.environ.get(k) for k in ("OPENAI_API_KEY", "OPENAI_BASE_URL")) and bool(
        model
    )
    if not configured:
        env_file = REPO_ROOT / ".env"
        text = env_file.read_text(encoding="utf-8") if env_file.is_file() else ""
        keys = {line.split("=", 1)[0].strip() for line in text.splitlines() if "=" in line}
        configured = {"OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL"} <= keys
        for line in text.splitlines():
            if line.startswith("OPENAI_MODEL="):
                model = line.split("=", 1)[1].strip().strip("\"'")
    return {
        "available": configured,
        "model": model or "Qwen",
        "detail": "Конфигурация найдена; доступность сети не проверялась запросом"
        if configured
        else "Не найдены OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL в окружении или .env",
    }


class Service:
    def __init__(self, settings: Settings, store: RunStore, supervisor: Supervisor) -> None:
        self.settings = settings
        self.store = store
        self.supervisor = supervisor

    def status_of(self, run_id: str, facts: dict[str, Any]) -> str:
        meta = self.store.meta(run_id)
        if not meta.get("started_at"):
            return "stopped"  # created but never launched
        if self.supervisor.is_running(run_id):
            return "running"
        if self.supervisor.aborted(run_id):
            return "stopped"
        code = self.supervisor.exit_code(run_id)
        if code == 0 and facts["hasSummary"]:
            return "completed"
        return "failed"

    def run(self, run_id: str) -> dict[str, Any]:
        run_dir = self.store.path(run_id)
        meta = self.store.meta(run_id)
        facts = reader.run_facts(run_dir)
        status = self.status_of(run_id, facts)
        options = meta["options"]
        started = meta.get("started_unix")
        if started is None:
            elapsed = 0
        elif status == "running":
            elapsed = time.time() - started
        else:
            ends = [
                p.stat().st_mtime
                for p in (run_dir / "exit.code", run_dir / "abort.flag")
                if p.exists()
            ]
            elapsed = (max(ends) if ends else time.time()) - started
        csv_path = self._csv(facts, run_dir)
        stage = facts["stage"]
        note = None
        if status == "stopped" and meta.get("started_at"):
            note = "Запуск прерван пользователем. Существующий CSV сохранён."
        elif status == "failed":
            note = "Процесс завершился с ошибкой. Журнал сохранён, существующий CSV доступен."
        elif status == "completed":
            parts = []
            if facts["researchStatus"] not in (None, "completed"):
                parts.append(f"Поиск завершён со статусом «{facts['researchStatus']}».")
            if facts["selected"]:
                chosen = facts["selected"]
                parts.append(
                    f"В итоговый CSV вошёл {chosen['id']} "
                    f"({facts['metric']} {reader.format_score(chosen['score'], facts['metric'])} "
                    "на полной validation)."
                )
            note = " ".join(parts) or None
        return {
            "id": run_id,
            "title": meta["title"],
            "taskKind": facts["taskKind"],
            "metric": facts["metric"] if facts["baseline"] is not None else "—",
            "direction": facts["direction"],
            "status": status,
            "stage": stage,
            "baseline": facts["baseline"],
            "best": facts["best"],
            "elapsedSec": round(elapsed),
            "startedAt": meta.get("started_at") or meta["created_at"],
            "demo": False,
            "model": meta.get("model") or "модель не записана",
            "mode": options["mode"],
            "budget": {
                "timeSec": options["time_sec"],
                "tokens": options["tokens"],
                "threads": options["threads"],
            },
            "used": {
                "timeSec": round(elapsed),
                "tokens": facts["tokens"],
                "tokensUncertain": facts["tokensUncertain"],
            },
            "experiments": facts["experiments"],
            **{
                key: facts[key]
                for key in (
                    "resultStatus",
                    "researchStatus",
                    "stopReason",
                    "selectedRecipe",
                    "selectedOrigin",
                    "rejections",
                    "searchDecisions",
                    "canonicalVersion",
                )
            },
            "csvAvailable": csv_path is not None,
            "finalCsv": csv_path is not None
            and (
                facts["resultStatus"] == "refitted"
                if facts["canonicalVersion"] == 3
                else status == "completed"
            ),
            "note": note,
        }

    def summaries(self) -> list[dict[str, Any]]:
        runs = [self.run(i) for i in self.store.ids()]
        for r in runs:
            r.pop("experiments")
            r.pop("budget"), r.pop("used"), r.pop("model"), r.pop("mode")
            r.pop("csvAvailable"), r.pop("finalCsv"), r.pop("note")
        return sorted(runs, key=lambda r: r["startedAt"], reverse=True)

    def _csv(self, facts: dict[str, Any], run_dir: Path) -> Path | None:
        """The run's own submission file; paths outside the run directory are never served."""
        path = facts["submission"]
        if path is None:
            return None
        try:
            resolved = path.resolve()
            resolved.relative_to(run_dir.resolve())
        except (OSError, ValueError):
            return None
        return resolved if resolved.is_file() else None

    def csv_path(self, run_id: str) -> Path | None:
        return self._csv(reader.run_facts(self.store.path(run_id)), self.store.path(run_id))

    def report_text(self, run_id: str) -> str:
        run = self.run(run_id)
        started = run["startedAt"]
        try:
            started = datetime.fromisoformat(started).strftime("%d.%m.%Y %H:%M")
        except ValueError:
            pass
        lines = [
            f"# Отчёт: {run['title']}",
            "",
            f"- Состояние: {run['status']}",
            f"- Начало: {started}",
            f"- Метрика: {run['metric']}",
            f"- Baseline: {run['baseline']}",
            f"- Лучший результат: {run['best']}",
            "",
            "## Эксперименты",
            *[f"- {e['id']} ({e['status']}): {e['measured']}" for e in run["experiments"]],
        ]
        return "\n".join(lines) + "\n"
