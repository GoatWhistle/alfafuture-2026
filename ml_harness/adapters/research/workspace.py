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
from ml_harness.adapters.ml.diagnostics import IndependentErrorAnalyzer
from ml_harness.adapters.storage.journal import JsonlExperimentStore
from ml_harness.application.ports.execution import ErrorAnalyzer, Evaluator, ExperimentRunner
from ml_harness.application.use_cases.context import diagnostic_summary
from ml_harness.application.use_cases.directions import ResearchDirection, validate_parent
from ml_harness.application.use_cases.experiment import execute_experiment
from ml_harness.domain.models import (
    ContractError,
    Experiment,
    Hypothesis,
    PreparedTask,
    Score,
    TaskKind,
    ValidationProtocol,
)

CANDIDATE_API = """# Candidate interface
Only candidate.py is editable. Read task.md and data_dictionary.csv for task semantics.
Do not read package source files; the supported interface is documented here.

Export build_model(kind, parameters, threads, seed), returning an unfitted object.
fit(X, y) receives a pandas DataFrame of features and a Series of training targets.
predict(X) returns a 1D array. Classification must expose classes_ and, when required,
predict_proba(X) with columns in classes_ order. All learned transformations fit on X/y
from fit only; reuse them at prediction. Never read targets or split files yourself.

from ml_harness.adapters.ml.candidates import build_model as base_model

def build_model(kind, parameters, threads, seed):
    options = {"backend": "catboost", "iterations": 200, "depth": 6}
    options.update(parameters)  # Required: honor frozen refit overrides.
    return base_model(kind, options, threads, seed)

CatBoost accepts iterations, depth, learning_rate, l2_leaf_reg, auto_class_weights
(classification only). LightGBM uses backend="lightgbm", n_estimators, num_leaves,
learning_rate, min_child_samples. Missing values/categories are handled by the base model.
Optional numeric search: options["tuning"]={"trials":5,"time_limit":60}.
Target/feature wrappers must forward classes_, predict_proba, configure_validation,
and refit_parameters_ / tuning_report_ when wrapping a tuned base model.
A hypothesis requires an actual code change before run_experiment.
"""

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

    def sample(source: tuple[int, ...] | list[int], count: int) -> tuple[int, ...]:
        if len(source) <= count:
            return tuple(source)
        return tuple(source[round(i * (len(source) - 1) / (count - 1))] for i in range(count))

    partitions = []
    for source in (task.protocol.train_indices, task.protocol.validation_indices):
        if task.spec.kind == TaskKind.REGRESSION:
            selected = sample(source, 256)
        else:
            strata: dict[str, list[int]] = defaultdict(list)
            for index in source:
                strata[rows[index][task.spec.target]].append(index)
            selected = tuple(
                sorted(index for values in strata.values() for index in sample(values, 20))
            )
        for column in (task.spec.group_column, task.spec.time_column):
            if column and len({rows[index][column] for index in selected}) < 2:
                other = next(
                    (index for index in source if rows[index][column] != rows[selected[0]][column]),
                    None,
                )
                if other is not None:
                    selected = (*selected, other)
        partitions.append(tuple(sorted(set(selected))))
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
        analyzer: ErrorAnalyzer | None = None,
    ) -> None:
        self.task, self.path = task, path.resolve()
        self.runner, self.evaluator = runner, evaluator
        self.analyzer = analyzer or IndependentErrorAnalyzer()
        self.baseline_report: dict[str, Any] | None = None
        self.active_parent_id: str | None = None
        self.active_direction: ResearchDirection | None = None
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
        for name, source in (
            ("task.md", Path(task.spec.task_path)),
            ("data_dictionary.csv", Path(task.spec.dictionary_path)),
        ):
            destination = self.path / name
            if not destination.exists() and source.is_file():
                destination.write_bytes(source.read_bytes())
                destination.chmod(0o444)
        api = self.path / "candidate_api.md"
        if not api.exists():
            api.write_text(CANDIDATE_API, encoding="utf-8")
            api.chmod(0o444)
        self.store = JsonlExperimentStore(self.path / "memory")
        self.reports: list[dict[str, Any]] = []
        self._restore()
        self._profile: dict[str, Any] | None = None

    def inspect_data(self) -> dict[str, Any]:
        if self._profile is None:
            _, all_train = read_csv(Path(self.task.spec.train_path))
            visible = sorted(
                self.task.protocol.train_indices + self.task.protocol.validation_indices
            )
            train = [all_train[index] for index in visible]
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
                "train_rows": self.task.train_rows,
                "development_rows": len(train),
                "control_rows_withheld": len(self.task.protocol.control_indices),
                "test_rows": len(test),
                "features": features,
                "target_profile": targets,
                "protocol": self.task.protocol.fingerprint,
                "baseline_errors": diagnostic_summary(
                    (self.baseline_report or {}).get("error_report", {})
                ),
            }
        write_json(self.path / "profile.json", self._profile)
        return self._profile

    def read_experiments(self) -> list[dict[str, Any]]:
        return deepcopy(([self.baseline_report] if self.baseline_report else []) + self.reports)

    def numeric_completed(self) -> int:
        return sum(r["id"].startswith("numeric-") for r in self.reports)

    def numeric_batch(self, backend: str, index: int) -> bool:
        storage = "sqlite:///" + str(self.path / "numeric-studies.sqlite3")
        options = {
            "backend": backend,
            "tuning": {
                "trials": 5,
                "time_limit": 60,
                "storage": storage,
                "study_name": backend,
                "batch": index,
            },
        }
        code = (
            "from ml_harness.adapters.ml.candidates import build_model as base_model\n"
            "def build_model(kind, parameters, threads, seed):\n"
            f"    options = {options!r}\n"
            "    options.update(parameters)\n"
            "    return base_model(kind, options, threads, seed)\n"
        )
        (self.path / "candidate.py").write_text(code)
        result = self.run_experiment(
            {
                "experiment_id": f"numeric-{index:05d}",
                "parent_id": "baseline",
                "hypothesis": f"Continue train-only {backend} study, batch {index}",
                "hypothesis_source": "deterministic_optuna",
                "expected_effect": "Improve development score",
            }
        )
        return result.get("status") == "succeeded"

    def register_starting_point(self, item: dict[str, Any]) -> None:
        identifier = "baseline-" + item["backend"]
        if any(r["id"] == identifier for r in self.reports):
            return
        raw = item["score"]
        output_path = Path(item["validation_dir"]) / "result.json"
        output = json.loads(output_path.read_text())
        score = self.evaluator.evaluate(self.task, output)
        if asdict(score) != raw:
            raise ContractError("Portfolio score disagrees with independent evaluation")
        code = (
            "from ml_harness.adapters.ml.candidates import build_model as base_model\n"
            "def build_model(kind, parameters, threads, seed):\n"
            f"    options = {{'backend': {item['backend']!r}}}\n"
            "    options.update(parameters)\n"
            "    return base_model(kind, options, threads, seed)\n"
        ).encode()
        digest, path = self.store.save_code(code)
        report = {
            "id": identifier,
            "parent_id": "baseline",
            "hypothesis": "Alternative starting model",
            "status": "succeeded",
            "backend": item["backend"],
            "parameters": {},
            "code_hash": digest,
            "code_path": path,
            "full_score": raw,
            "level": "full",
            "protocol_id": self.task.protocol.fingerprint,
            "measured_by": "harness",
            "prediction_path": str(output_path),
            "prediction_hash": hashlib.sha256(output_path.read_bytes()).hexdigest(),
            "error_report": self.analyzer.analyze(self.task, output),
        }
        self.store.append(
            {
                "event_id": "experiment:" + identifier + ":finished",
                "kind": "experiment_finished",
                "payload": report,
            }
        )
        self.reports.append(report)

    def register_baseline(
        self,
        score: Score,
        output_path: Path | None,
        *,
        backend: str = "catboost",
        parameters: dict[str, Any] | None = None,
    ) -> None:
        options = {"backend": backend, **(parameters or {})}
        code = (
            "from ml_harness.adapters.ml.candidates import build_model as base_model\n\n"
            "def build_model(kind, parameters, threads, seed):\n"
            f"    options = {options!r}\n"
            "    options.update(parameters)\n"
            "    return base_model(kind, options, threads, seed)\n"
        ).encode()
        digest, code_path = self.store.save_code(code)
        if self.baseline_report is not None:
            if self.baseline_report["code_hash"] != digest or self.baseline_report[
                "full_score"
            ] != asdict(score):
                raise ContractError("Saved baseline code/score changed")
            if output_path is not None and self.baseline_report.get("prediction_hash"):
                if (
                    hashlib.sha256(output_path.read_bytes()).hexdigest()
                    != self.baseline_report["prediction_hash"]
                ):
                    raise ContractError("Saved baseline predictions failed integrity check")
            return
        report: dict[str, Any] = {
            "id": "baseline",
            "backend": backend,
            "parent_id": None,
            "status": "succeeded",
            "level": "full",
            "measured_by": "harness",
            "full_score": asdict(score),
            "code_hash": digest,
            "code_path": code_path,
            "parameters": options,
            "protocol_id": self.task.protocol.fingerprint,
            "data_fingerprint": self.task.protocol.data_fingerprint,
            "hypothesis": "Fixed initial pipeline",
            "llm_commentary": None,
            "error_report": {},
            "diagnostic_status": "unavailable_baseline_predictions",
        }
        if output_path is not None and output_path.exists():
            output = json.loads(output_path.read_text(encoding="utf-8"))
            measured = self.evaluator.evaluate(self.task, output)
            if measured != score:
                raise ContractError("Baseline summary disagrees with independent reevaluation")
            report.update(
                prediction_path=str(output_path.resolve()),
                prediction_hash=hashlib.sha256(output_path.read_bytes()).hexdigest(),
                error_report=self.analyzer.analyze(self.task, output),
                diagnostic_status="measured",
            )
        self.store.append(
            {"event_id": "baseline", "kind": "baseline_registered", "payload": report}
        )
        self.baseline_report = report
        self._profile = None

    def prepare_hypothesis(self, decision: ResearchDirection, identifier: str) -> None:
        parent = (
            self.baseline_report
            if decision.parent_id == "baseline"
            else next(
                (report for report in self.reports if report["id"] == decision.parent_id), None
            )
        )
        if parent is None:
            raise ContractError("Selected parent is unavailable")
        validate_parent(self.task, parent)
        code = Path(parent["code_path"]).read_bytes()
        if hashlib.sha256(code).hexdigest() != parent["code_hash"]:
            raise ContractError("Selected parent code failed integrity check")
        temporary = self.path / "candidate.py.tmp"
        temporary.write_bytes(code)
        temporary.replace(self.path / "candidate.py")
        self.active_parent_id = decision.parent_id
        self.active_direction = decision
        payload = {"session_id": identifier, **asdict(decision)}
        self.store.append(
            {
                "event_id": "hypothesis:" + identifier,
                "kind": "hypothesis_started",
                "payload": payload,
            }
        )
        write_json(self.path / "decisions" / (identifier + ".json"), payload)

    def _parent_output(self, parent_id: str | None) -> dict[str, Any] | None:
        if parent_id is None:
            return None
        parent = (
            self.baseline_report
            if parent_id == "baseline"
            else next((report for report in self.reports if report["id"] == parent_id), None)
        )
        if parent is None:
            if parent_id == "baseline":
                return None  # Direct offline use may have no initial-pipeline artifact.
            raise ContractError("Unknown experiment parent")
        validate_parent(self.task, parent)
        path = parent.get("prediction_path")
        if path is None and parent_id != "baseline":
            path = str(self.path / "experiments" / parent_id / "full/result.json")
        if path is None:
            return None
        file = Path(path)
        if (
            parent.get("prediction_hash")
            and hashlib.sha256(file.read_bytes()).hexdigest() != parent["prediction_hash"]
        ):
            raise ContractError("Parent predictions failed integrity check")
        value = json.loads(file.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ContractError("Parent predictions must be an object")
        return value

    def _attach_diagnostics(self, report: dict[str, Any]) -> None:
        if report["status"] != "succeeded":
            return
        file = self.path / "experiments" / report["id"] / "full/result.json"
        output = json.loads(file.read_text(encoding="utf-8"))
        parent_output = self._parent_output(report.get("parent_id"))
        report.update(
            prediction_path=str(file),
            prediction_hash=hashlib.sha256(file.read_bytes()).hexdigest(),
            error_report=self.analyzer.analyze(self.task, output, parent_output),
            diagnostic_status="measured",
            backend=output.get("backend", report.get("backend", "custom")),
        )
        if "tuning_report" in output:
            report["tuning_report"] = output["tuning_report"]
            if not isinstance(output.get("refit_parameters"), dict):
                raise ContractError(
                    "Tuned candidate must export refit_parameters_ before finalization"
                )
            report["refit_parameters"] = output["refit_parameters"]
            if output.get("refit_factory") == "tabular":
                code = (
                    b"from ml_harness.adapters.ml.candidates import build_model as base_model\n"
                    b"def build_model(kind, parameters, threads, seed):\n"
                    b"    return base_model(kind, parameters, threads, seed)\n"
                )
                digest, path = self.store.save_code(code)
                report.update(refit_code_hash=digest, refit_code_path=path)

    def _restore(self) -> None:
        experiments: dict[str, dict[str, Any]] = {}
        sessions = []
        for event in self.store.read_events():
            if event["kind"] in ("experiment_started", "experiment_finished"):
                record = event["payload"]
                experiments[record["id"]] = record
            elif event["kind"] == "session_finished":
                sessions.append(event["payload"])
            elif event["kind"] == "baseline_registered":
                self.baseline_report = event["payload"]
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
        if report["status"] == "succeeded":
            self._attach_diagnostics(report)
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
            "tool_failures": session.get("tool_failures", [])[-3:],
            "error_detail": session.get("error_detail"),
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
        arguments = dict(arguments)
        identifier = arguments["experiment_id"]
        if identifier == "baseline":
            raise ContractError("The baseline experiment ID is reserved")
        parent_id = arguments.get("parent_id") or self.active_parent_id
        if self.active_parent_id is not None and parent_id != self.active_parent_id:
            raise ContractError("Parent differs from this hypothesis; start a new session")
        arguments["parent_id"] = parent_id
        self._parent_output(parent_id)  # Validate lineage before spending training time.
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
        parent = (
            self.baseline_report
            if parent_id == "baseline"
            else next((r for r in self.reports if r["id"] == parent_id), None)
        )
        if parent is not None and digest == parent["code_hash"]:
            raise ContractError(
                "Candidate unchanged: edit candidate.py to implement the hypothesis before training"
            )
        duplicate = next(
            (
                r
                for r in self.reports
                if r["code_hash"] == digest
                and r.get("parameters", {}) == arguments.get("parameters", {})
                and r.get("status") == "succeeded"
            ),
            None,
        )
        if duplicate is not None:
            output = self._parent_output(duplicate["id"])
            if output is None:
                raise ContractError("Completed duplicate has no verified prediction artifact")
            reused = {
                **deepcopy(duplicate),
                "id": identifier,
                "parent_id": parent_id,
                "hypothesis": arguments["hypothesis"],
                "expected_effect": arguments["expected_effect"],
                "parent_score": parent.get("full_score") if parent else None,
                "parent_code_hash": parent.get("code_hash") if parent else None,
                "reused_from": duplicate["id"],
                "hypothesis_source": arguments.get("hypothesis_source", "tool_call"),
                "elapsed_seconds": 0.0,
                "llm_commentary": None,
                "usage": None,
                "error_report": self.analyzer.analyze(
                    self.task, output, self._parent_output(parent_id)
                ),
            }
            self._finish_record(reused)
            self.reports.append(reused)
            return reused
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
            "hypothesis_source": arguments.get("hypothesis_source", "tool_call"),
            "target_errors": (
                deepcopy(self.active_direction.target_errors) if self.active_direction else ()
            ),
            "expected_effect": arguments["expected_effect"],
            "code_hash": digest,
            "parameters": arguments.get("parameters", {}),
            "parameters_hash": hashlib.sha256(
                json.dumps(arguments.get("parameters", {}), sort_keys=True).encode()
            ).hexdigest(),
            "parent_code_hash": parent.get("code_hash") if parent else None,
            "parent_score": parent.get("full_score") if parent else None,
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
        try:
            self._attach_diagnostics(report)
        except (ContractError, OSError, ValueError) as exc:
            report.update(diagnostic_status="failed", diagnostic_error=str(exc)[:2000])
        self._finish_record(report)
        self.reports.append(report)
        return report
