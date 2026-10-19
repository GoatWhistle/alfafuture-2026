"""Versioned candidate attempts and honest smoke/full measurement for research tools."""

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from ml_harness.adapters.data.task import read_csv
from ml_harness.application.ports.execution import Evaluator, ExperimentRunner, RunRequest
from ml_harness.domain.models import (
    ContractError,
    ExperimentStatus,
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
        write_json(self.path / "prepared.json", asdict(task))
        self.reports: list[dict[str, Any]] = []
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
        return list(self.reports)

    def run_experiment(self, arguments: dict[str, Any]) -> dict[str, Any]:
        identifier = arguments["experiment_id"]
        candidate = self.path / "candidate.py"
        code = candidate.read_bytes()
        digest = hashlib.sha256(code).hexdigest()
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
            "data_fingerprint": self.task.protocol.data_fingerprint,
            "protocol_id": self.task.protocol.fingerprint,
            "level": "full",
            "status": "running",
            "measured_by": "harness",
            "llm_commentary": None,
        }
        write_json(directory / "experiment.json", report)
        elapsed = 0.0
        try:
            smoke = smoke_task(self.task)
            for level, prepared in (("smoke", smoke), ("full", self.task)):
                outcome = self.runner.run(
                    RunRequest(
                        prepared,
                        str(directory / level),
                        candidate_path=str(frozen),
                        threads=self.threads,
                    ),
                    min(60, self.timeout_seconds) if level == "smoke" else self.timeout_seconds,
                )
                elapsed += outcome.elapsed_seconds
                if outcome.status != ExperimentStatus.SUCCEEDED or outcome.output is None:
                    report.update(
                        status=outcome.status.value, failure_phase=level, error=outcome.detail
                    )
                    break
                score = self.evaluator.evaluate(prepared, outcome.output)
                report[level + "_score"] = asdict(score)
            else:
                report["status"] = "succeeded"
        except (ContractError, OSError, ValueError) as exc:
            report.update(status="failed", error=str(exc)[:2000])
        report["elapsed_seconds"] = elapsed
        write_json(directory / "experiment.json", report)
        self.reports.append(report)
        return report
