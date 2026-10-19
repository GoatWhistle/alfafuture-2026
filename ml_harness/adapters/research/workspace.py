"""Versioned candidate attempts and honest smoke/full measurement for research tools."""

import hashlib
import json
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import psutil  # type: ignore[import-untyped]

from ml_harness.adapters.data.task import read_csv
from ml_harness.adapters.storage.journal import JsonlExperimentStore
from ml_harness.application.ports.execution import Evaluator, ExperimentRunner
from ml_harness.application.use_cases.experiment import execute_experiment
from ml_harness.domain.models import (
    ContractError,
    Experiment,
    Hypothesis,
    PreparedTask,
    TaskKind,
    ValidationProtocol,
)

STARTER = '''"""Starter candidate: change model/feature logic here; never change evaluation."""
from ml_harness.adapters.ml.candidates import TabularCandidate


def build_model(kind, parameters, threads, seed):
    options = {"backend": "catboost", "iterations": 200, "depth": 6}
    options.update(parameters)
    return TabularCandidate(kind, options, threads, seed)
'''


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    temporary.replace(path)


def smoke_task(task: PreparedTask) -> PreparedTask:
    """Keep disjoint partitions and every class; smoke is a different protocol/level."""
    _, rows = read_csv(Path(task.spec.train_path))
    partitions = []
    for source in (task.protocol.train_indices, task.protocol.validation_indices):
        if task.spec.kind == TaskKind.REGRESSION:
            selected = source[: min(256, len(source))]
        else:
            strata: dict[str, list[int]] = defaultdict(list)
            for index in source:
                strata[rows[index][task.spec.target]].append(index)
            selected = tuple(sorted(index for values in strata.values() for index in values[:20]))
        partitions.append(tuple(selected))
    fingerprint = hashlib.sha256(
        json.dumps((task.protocol.fingerprint, partitions, "smoke")).encode()
    ).hexdigest()
    protocol = ValidationProtocol(
        fingerprint,
        task.protocol.data_fingerprint,
        task.protocol.kind,
        task.protocol.seed,
        partitions[0],
        partitions[1],
        level="smoke",
    )
    return replace(task, protocol=protocol)


class ResearchWorkspace:
    def __init__(
        self,
        task: PreparedTask,
        path: Path,
        runner: ExperimentRunner,
        evaluator: Evaluator,
        *,
        threads: int = 8,
        timeout_seconds: float = 600,
    ) -> None:
        self.task, self.path = task, path.resolve()
        self.runner, self.evaluator = runner, evaluator
        self.threads, self.timeout_seconds = threads, timeout_seconds
        self.path.mkdir(parents=True, exist_ok=True)
        candidate = self.path / "candidate.py"
        if not candidate.exists():
            candidate.write_text(STARTER, encoding="utf-8")
        manifest = self.path / "prepared.json"
        if manifest.exists():
            existing = json.loads(manifest.read_text(encoding="utf-8"))
            if existing["protocol"]["fingerprint"] != task.protocol.fingerprint:
                raise ContractError("Research workspace belongs to another task/protocol")
        else:
            write_json(manifest, asdict(task))
        self.store = JsonlExperimentStore(self.path / "memory")
        self.reports: list[dict[str, Any]] = []
        self._restore()
        self._profile: dict[str, Any] | None = None

    def inspect_data(self) -> dict[str, Any]:
        if self._profile is None:
            _, train = read_csv(Path(self.task.spec.train_path))
            _, test = read_csv(Path(self.task.spec.test_path))
            spec = self.task.spec
            features = []
            for column in spec.features:
                features.append(
                    {
                        "name": column,
                        "train_missing": sum(row[column] == "" for row in train),
                        "test_missing": sum(row[column] == "" for row in test),
                        "sample_values": list(dict.fromkeys(row[column] for row in train[:500]))[
                            :5
                        ],
                    }
                )
            targets: dict[str, Any]
            if spec.kind == TaskKind.REGRESSION:
                values = [float(row[spec.target]) for row in train]
                targets = {
                    "min": min(values),
                    "max": max(values),
                    "mean": sum(values) / len(values),
                }
            else:
                targets = dict(Counter(row[spec.target] for row in train))
            self._profile = {
                "task": asdict(spec),
                "train_rows": len(train),
                "test_rows": len(test),
                "features": features,
                "target_profile": targets,
                "protocol": self.task.protocol.fingerprint,
            }
        return self._profile

    def read_experiments(self) -> list[dict[str, Any]]:
        return deepcopy(self.reports)

    def _restore(self) -> None:
        experiments: dict[str, dict[str, Any]] = {}
        sessions = []
        for event in self.store.read_events():
            if event["kind"] in ("experiment_started", "experiment_finished"):
                record = event["payload"]
                experiments[record["id"]] = record
            elif event["kind"] == "session_finished":
                sessions.append(event["payload"])
        for report in experiments.values():
            if "code_path" in report:
                version = Path(report["code_path"])
                if (
                    not version.resolve().is_relative_to(self.store.root / "versions")
                    or hashlib.sha256(version.read_bytes()).hexdigest() != report["code_hash"]
                ):
                    raise ContractError("Stored candidate version failed integrity check")
            if report["status"] == "running":
                self._recover_attempt(report)
            self.reports.append(report)
        for session in sessions:
            self._apply_session(session)

    def _recover_attempt(self, report: dict[str, Any]) -> None:
        directory = self.path / "experiments" / report["id"]
        for manifest in directory.glob("*/process.json"):
            handle = json.loads(manifest.read_text(encoding="utf-8"))
            try:
                process = psutil.Process(handle["pid"])
                if (
                    abs(process.create_time() - handle["created_at"]) < 0.01
                    and process.is_running()
                    and process.status() != psutil.STATUS_ZOMBIE
                ):
                    raise ContractError(
                        "Prior worker is still running; wait for its recorded handle"
                    )
            except psutil.NoSuchProcess:
                pass
        report.update(status="failed", error="Interrupted prior process", recovered=True)
        full = directory / "full/result.json"
        if full.exists():
            try:
                result = json.loads(full.read_text(encoding="utf-8"))
                report["full_score"] = asdict(self.evaluator.evaluate(self.task, result))
                report.update(
                    status="succeeded",
                    error=None,
                    elapsed_seconds=result.get("worker_seconds"),
                    timing_source="worker",
                )
                smoke = directory / "smoke/result.json"
                if smoke.exists():
                    report["smoke_score"] = asdict(
                        self.evaluator.evaluate(
                            smoke_task(self.task), json.loads(smoke.read_text(encoding="utf-8"))
                        )
                    )
            except (ValueError, OSError):
                report.update(status="failed", error="Interrupted result failed validation")
        self._finish_record(report)

    def _finish_record(self, report: dict[str, Any]) -> None:
        self.store.append(
            {
                "event_id": "experiment:" + report["id"] + ":finished",
                "kind": "experiment_finished",
                "payload": report,
            }
        )
        write_json(self.path / "experiments" / report["id"] / "experiment.json", report)
        # Artifacts are referenced by immutable experiment IDs; tools cannot edit them.
        directory = self.path / "experiments" / report["id"]
        for artifact in directory.rglob("*"):
            if artifact.is_file() and not artifact.is_symlink():
                artifact.chmod(0o444)

    def _apply_session(self, session: dict[str, Any]) -> None:
        for report in self.reports:
            if report["id"] in session["experiment_ids"]:
                report["llm_commentary"] = session["summary"]
                report["usage"] = {
                    "scope": "shared_session",
                    "session_id": session["id"],
                    **session["usage"],
                }

    def register_session(self, identifier: str, session: dict[str, Any]) -> None:
        payload = {
            "id": identifier,
            "summary": session["summary"],
            "experiment_ids": [report["id"] for report in session["reports"]],
            "reason": session["reason"],
            "usage": {
                key: session[key]
                for key in ("input_tokens", "output_tokens", "usage_known", "api_attempts")
            },
        }
        self.store.append(
            {"event_id": "session:" + identifier, "kind": "session_finished", "payload": payload}
        )
        self._apply_session(payload)

    def run_experiment(self, arguments: dict[str, Any]) -> dict[str, Any]:
        identifier = arguments["experiment_id"]
        candidate = self.path / "candidate.py"
        code = candidate.read_bytes()
        digest, version_path = self.store.save_code(code)
        previous = next((report for report in self.reports if report["id"] == identifier), None)
        if previous is not None:
            if previous["code_hash"] != digest or any(
                previous.get(key) != arguments.get(key)
                for key in ("hypothesis", "expected_effect", "parent_id")
            ):
                raise ContractError("Experiment ID already belongs to another candidate version")
            return previous
        directory = self.path / "experiments" / identifier
        if directory.exists():
            raise ContractError("Experiment directory already exists; inspect it and use a new ID")
        directory.mkdir(parents=True)
        frozen = directory / "candidate.py"
        frozen.write_bytes(code)
        frozen.chmod(0o444)
        report: dict[str, Any] = {
            "id": identifier,
            "parent_id": arguments.get("parent_id"),
            "hypothesis": arguments["hypothesis"],
            "expected_effect": arguments["expected_effect"],
            "code_hash": digest,
            "parameters": {},
            "parameters_hash": hashlib.sha256(b"{}").hexdigest(),
            "code_path": version_path,
            "artifacts": [str(directory / "smoke"), str(directory / "full")],
            "error_report": {},
            "usage": None,
            "data_fingerprint": self.task.protocol.data_fingerprint,
            "protocol_id": self.task.protocol.fingerprint,
            "level": "full",
            "status": "running",
            "measured_by": "harness",
            "llm_commentary": None,
        }
        self.store.append(
            {
                "event_id": "experiment:" + identifier + ":started",
                "kind": "experiment_started",
                "payload": report,
            }
        )
        write_json(directory / "experiment.json", report)
        experiment = Experiment(
            identifier,
            Hypothesis(
                arguments["hypothesis"], arguments["expected_effect"], arguments.get("parent_id")
            ),
            digest,
            report["parameters_hash"],
            self.task.protocol.data_fingerprint,
            self.task.protocol.fingerprint,
        )
        try:
            report.update(
                execute_experiment(
                    experiment,
                    self.task,
                    smoke_task(self.task),
                    self.runner,
                    self.evaluator,
                    candidate_path=str(frozen),
                    output_dir=str(directory),
                    threads=self.threads,
                    timeout_seconds=self.timeout_seconds,
                )
            )
        except (ContractError, OSError, ValueError) as exc:
            report.update(status="failed", error=str(exc)[:2000], elapsed_seconds=0.0)
        self._finish_record(report)
        self.reports.append(report)
        return report
