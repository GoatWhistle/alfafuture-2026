"""Freeze before control, recover once-only executions, and publish checked refits."""

import hashlib
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import psutil  # type: ignore[import-untyped]

from ml_harness.adapters.data.task import read_csv
from ml_harness.adapters.ml.ensemble import combine_predictions
from ml_harness.adapters.ml.evaluation import validate_predictions
from ml_harness.adapters.research.workspace import ResearchWorkspace, write_json
from ml_harness.application.ports.execution import (
    Evaluator,
    ExperimentRunner,
    RunRequest,
    SubmissionWriter,
)
from ml_harness.application.use_cases.finalization import select_finalists
from ml_harness.domain.finalization import CandidateRef, Finalist, rank_finalists
from ml_harness.domain.models import ContractError, ExperimentStatus, Metric, PreparedTask, Score
from ml_harness.domain.protocols import control_task


class _PendingWorker(ContractError):
    """Keep the saved answer while an existing worker may still be active."""


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _score(value: dict[str, Any]) -> Score:
    return Score(value["value"], Metric(value["metric"]), value["protocol_id"], value["level"])


class FinalizationWorkspace:
    def __init__(
        self,
        task: PreparedTask,
        workspace: ResearchWorkspace,
        runner: ExperimentRunner,
        evaluator: Evaluator,
        writer: SubmissionWriter,
        threads: int = 8,
        timeout_seconds: float = 600,
    ) -> None:
        self.task, self.workspace, self.runner = task, workspace, runner
        self.evaluator, self.writer = evaluator, writer
        self.threads, self.timeout_seconds = threads, timeout_seconds
        self.manifest = workspace.path / "finalists.json"
        self.state_path = workspace.path / "finalization-state.json"
        self.state: dict[str, Any] = {}
        self.diagnostics: list[str] = []

    def _ids(self, task: PreparedTask, refit: bool = False) -> list[str]:
        _, rows = read_csv(Path(task.spec.test_path if refit else task.spec.train_path))
        selected = rows if refit else [rows[i] for i in task.protocol.validation_indices]
        return [row[task.spec.id_column] for row in selected]

    def _check_code(self, member: CandidateRef) -> None:
        if member.candidate_path is None or member.code_hash is None:
            raise ContractError("Finalist requires an immutable code version")
        if hashlib.sha256(Path(member.candidate_path).read_bytes()).hexdigest() != member.code_hash:
            raise ContractError("Finalist code checksum mismatch")

    def _freeze(self) -> tuple[Finalist, ...]:
        if self.manifest.exists():
            frozen = json.loads(self.manifest.read_text())
            plan = frozen["plan"]
            if (
                _digest(plan) != frozen["checksum"]
                or plan["protocol"] != self.task.protocol.fingerprint
            ):
                raise ContractError("Frozen finalist plan checksum/protocol mismatch")
            finalists = tuple(
                Finalist(
                    item["id"],
                    tuple(CandidateRef(**member) for member in item["members"]),
                    _score(item["score"]),
                    tuple(item["weights"]),
                )
                for item in plan["finalists"]
            )
            for finalist in finalists:
                for member in finalist.members:
                    self._check_code(member)
            return finalists
        candidates: list[Finalist] = []
        outputs: dict[str, dict[str, Any]] = {}
        reports = [self.workspace.baseline_report] if self.workspace.baseline_report else []
        reports += self.workspace.reports
        for report in reports:
            if (
                report.get("status") != "succeeded"
                or report.get("measured_by") != "harness"
                or report.get("level") != "full"
                or report.get("protocol_id") != self.task.protocol.fingerprint
            ):
                continue
            try:
                path = report.get("prediction_path")
                if not path:
                    if report["id"] == "baseline":
                        continue
                    path = str(
                        self.workspace.path / "experiments" / report["id"] / "full/result.json"
                    )
                raw = Path(path).read_bytes()
                if (
                    report.get("prediction_hash")
                    and hashlib.sha256(raw).hexdigest() != report["prediction_hash"]
                ):
                    raise ContractError("Search prediction checksum mismatch")
                output = json.loads(raw)
                if not isinstance(output, dict):
                    raise ContractError("Search predictions must be an object")
                measured = self.evaluator.evaluate(self.task, output)
                if measured != _score(report["full_score"]):
                    raise ContractError("Search score disagrees with independent evaluation")
                member = CandidateRef(
                    report["id"],
                    report["code_path"],
                    report["code_hash"],
                    report.get(
                        "backend", (report.get("parameters") or {}).get("backend", "catboost")
                    ),
                    report.get("parameters", {}),
                )
                self._check_code(member)
                candidates.append(Finalist(member.id, (member,), measured))
                outputs[member.id] = output
            except Exception as error:
                self.diagnostics.append(f"Excluded finalist {report.get('id')}: {error}")
        ranked = rank_finalists(candidates, len(candidates))
        if len(ranked) >= 2:
            members = ranked[:2]
            combined = combine_predictions(
                self.task.spec, self._ids(self.task), [outputs[item.id] for item in members]
            )
            measured = self.evaluator.evaluate(self.task, combined)
            if measured.better_than(ranked[0].score):
                candidates.append(
                    Finalist(
                        "ensemble:" + "+".join(item.id for item in members),
                        tuple(item.members[0] for item in members),
                        measured,
                        (0.5, 0.5),
                    )
                )
        finalists_list = list(rank_finalists(candidates, 3))
        baseline = next((item for item in candidates if item.id == "baseline"), None)
        if baseline and all(item.id != "baseline" for item in finalists_list):
            finalists_list[-1] = baseline
        plan = {
            "protocol": self.task.protocol.fingerprint,
            "finalists": [asdict(item) for item in finalists_list],
        }
        write_json(self.manifest, {"plan": plan, "checksum": _digest(plan)})
        self.manifest.chmod(0o444)
        return tuple(finalists_list)

    def _save(self) -> None:
        write_json(self.state_path, self.state)

    @staticmethod
    def _alive(directory: Path) -> bool:
        path = directory / "process.json"
        if not path.exists():
            return False
        try:
            info = json.loads(path.read_text())
            process = psutil.Process(info["pid"])
            return (
                process.is_running()
                and process.status() != psutil.STATUS_ZOMBIE
                and math.isclose(
                    process.create_time(), info["created_at"], rel_tol=0, abs_tol=0.001
                )
            )
        except psutil.NoSuchProcess:
            return False
        except (OSError, ValueError, KeyError, TypeError, psutil.AccessDenied):
            raise _PendingWorker(
                "Frozen finalist worker liveness unavailable; preserve artifacts"
            ) from None

    def _member_output(
        self, member: CandidateRef, task: PreparedTask, phase: str
    ) -> dict[str, Any] | None:
        table = self.state.setdefault(phase, {})
        key = member.id
        directory = (
            self.workspace.path
            / "finalization"
            / phase
            / hashlib.sha256(key.encode()).hexdigest()[:20]
        )
        artifact = directory / "result.json"
        previous = table.get(key)
        if previous is None and directory.exists() and any(directory.iterdir()):
            # The runtime state can lag artifacts after a crash: existing worker evidence
            # is still an execution attempt and can never authorize another control fit.
            table[key] = {"status": "started", "output_dir": str(directory), "recovered": True}
            self._save()
            previous = table[key]
        if previous:
            if self._alive(directory):
                raise _PendingWorker(
                    "Frozen finalist worker is still running; preserve its artifacts"
                )
            if previous["status"] == "failed":
                return None
            if not artifact.exists():
                previous.update(
                    status="failed", error="Interrupted execution has no complete result"
                )
                self._save()
                return None
            raw = artifact.read_bytes()
            if (
                previous.get("result_hash")
                and hashlib.sha256(raw).hexdigest() != previous["result_hash"]
            ):
                raise ContractError("Finalist result checksum mismatch")
            output = json.loads(raw)
        else:
            self._check_code(member)
            table[key] = {
                "status": "started",
                "started_at": time.time(),
                "output_dir": str(directory),
            }
            self._save()
            outcome = self.runner.run(
                RunRequest(
                    task,
                    str(directory),
                    mode=phase,
                    candidate_path=member.candidate_path,
                    backend=member.backend,
                    parameters=member.parameters,
                    threads=self.threads,
                ),
                self.timeout_seconds,
            )
            if outcome.status != ExperimentStatus.SUCCEEDED or outcome.output is None:
                table[key].update(status="failed", error=outcome.detail)
                self._save()
                return None
            output = outcome.output
            if not artifact.exists():
                write_json(artifact, output)
        if not isinstance(output, dict):
            raise ContractError("Finalist predictions must be an object")
        validate_predictions(task.spec, self._ids(task, phase == "refit"), output)
        entry = table[key]
        if phase == "control" and "score" not in entry:
            entry["score"] = asdict(self.evaluator.evaluate(task, output))
        entry.update(
            status="completed", result_hash=hashlib.sha256(artifact.read_bytes()).hexdigest()
        )
        self._save()
        return cast(dict[str, Any], output)

    def _selection(self, finalists: tuple[Finalist, ...]) -> tuple[str, ...]:
        if "selection" in self.state:
            return tuple(self.state["selection"])
        scores: dict[str, Score] = {}
        if not self.task.protocol.control_indices:
            scores = {item.id: item.score for item in finalists}
            self.state["selection_protocol"] = "search_no_control_partition"
        else:
            task = control_task(self.task)
            outputs: dict[str, dict[str, Any]] = {}
            for finalist in finalists:
                for member in finalist.members:
                    if member.id in outputs:
                        continue
                    try:
                        output = self._member_output(member, task, "control")
                        if output is not None:
                            outputs[member.id] = output
                    except Exception as error:
                        if isinstance(error, _PendingWorker):
                            raise
                        self.diagnostics.append(f"Control member {member.id}: {error}")
            checked = self.state.setdefault("control_scores", {})
            for finalist in finalists:
                if any(member.id not in outputs for member in finalist.members):
                    continue
                if finalist.id not in checked:
                    if len(finalist.members) == 1:
                        measured = _score(self.state["control"][finalist.members[0].id]["score"])
                    else:
                        combined = combine_predictions(
                            task.spec,
                            self._ids(task),
                            [outputs[member.id] for member in finalist.members],
                            finalist.weights,
                        )
                        measured = self.evaluator.evaluate(task, combined)
                    checked[finalist.id] = asdict(measured)
                    self._save()
                scores[finalist.id] = _score(checked[finalist.id])
            self.state["selection_protocol"] = task.protocol.fingerprint
        selected = select_finalists(finalists, scores)
        self.state["selection"] = list(selected)
        self.state["selected_once"] = selected[0] if selected else None
        self._save()
        return selected

    def finish(self) -> dict[str, Any]:
        finalists = self._freeze()
        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text())
        checksum = json.loads(self.manifest.read_text())["checksum"]
        if self.state.get("plan_checksum", checksum) != checksum:
            raise ContractError("Finalization runtime state belongs to another frozen plan")
        self.state["plan_checksum"] = checksum
        selected = self._selection(finalists)
        status = "failed_baseline_preserved" if finalists and not selected else "not_needed"
        winner = "baseline"
        if self.state.get("published"):
            return dict(self.state["published"])
        for identifier in selected:
            finalist = next(item for item in finalists if item.id == identifier)
            try:
                outputs = [
                    self._member_output(member, self.task, "refit") for member in finalist.members
                ]
                if any(output is None for output in outputs):
                    raise ContractError("Selected refit failed")
                combined = combine_predictions(
                    self.task.spec,
                    self._ids(self.task, True),
                    [output for output in outputs if output is not None],
                    finalist.weights,
                )
                self.writer.write(self.task, combined)
                status, winner = "succeeded", identifier
                break
            except Exception as error:
                if isinstance(error, _PendingWorker):
                    raise
                self.diagnostics.append(f"Refit finalist {identifier}: {error}")
                status = "failed_baseline_preserved"
        report = {
            "selected": winner,
            "refit_status": status,
            "finalists": [asdict(item) for item in finalists],
            "control_scores": self.state.get("control_scores", {}),
            "selection_protocol": self.state["selection_protocol"],
            "selection_reason": (
                "no_valid_finalists"
                if not finalists
                else "control_unavailable_preserved_csv"
                if self.task.protocol.control_indices and not selected
                else "checked_control"
                if self.task.protocol.control_indices
                else "no_control_partition"
            ),
            "diagnostics": self.diagnostics,
        }
        normalized: dict[str, Any] = json.loads(json.dumps(report, allow_nan=False))
        self.state["published"] = normalized
        self._save()
        return normalized
