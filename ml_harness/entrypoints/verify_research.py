"""Offline evidence runs for a few distinct hypotheses, without pretending to be Qwen."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ml_harness.adapters.data.task import CsvTaskData
from ml_harness.adapters.execution.runner import SubprocessRunner
from ml_harness.adapters.research.workspace import ResearchWorkspace, write_json
from ml_harness.adapters.storage.lease import RunLease
from ml_harness.adapters.storage.submission import IndependentEvaluator
from ml_harness.application.use_cases.context import diagnostic_summary
from ml_harness.application.use_cases.directions import ResearchDirection, diagnostic_focus
from ml_harness.domain.models import ContractError, Metric, Score, TaskKind


def candidate_code(options: dict[str, Any], *, transform: bool = False) -> str:
    factory = (
        "from ml_harness.adapters.ml.candidates import build_model as base_model\n"
        "import numpy as np\n\n"
        "def build_model(kind, parameters, threads, seed):\n"
        f"    options = {options!r}\n"
        "    options.update(parameters)\n"
        "    model = base_model(kind, options, threads, seed)\n"
    )
    if not transform:
        return factory + "    return model\n"
    return (
        factory
        + """    return SignedLogTarget(model)

class SignedLogTarget:
    classes_ = []
    def __init__(self, model):
        self.model = model
    def fit(self, X, y):
        self.model.fit(X, np.sign(y) * np.log1p(np.abs(y)))
        return self
    def predict(self, X):
        z = self.model.predict(X)
        return np.sign(z) * np.expm1(np.abs(z))
"""
    )


def verify(
    task_path: Path,
    baseline_dir: Path,
    output_dir: Path,
    *,
    threads: int = 8,
    timeout_seconds: float = 600,
    hypotheses: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    request = json.loads((baseline_dir / "validation/request.json").read_text())
    frozen = request["protocol"]
    task = CsvTaskData(reserve_control=bool(frozen.get("control_indices"))).prepare(
        str(task_path), frozen["seed"]
    )
    if task.protocol.fingerprint != frozen["fingerprint"]:
        raise ContractError("Verification task differs from the saved baseline protocol")
    baseline_file = baseline_dir / "baseline.json"
    if not baseline_file.exists():
        baseline_file = baseline_dir / "summary.json"
    baseline = json.loads(baseline_file.read_text())
    raw = baseline["score"]
    score = Score(raw["value"], Metric(raw["metric"]), raw["protocol_id"], raw["level"])
    with RunLease(output_dir / ".run.lock"):
        workspace = ResearchWorkspace(
            task,
            output_dir,
            SubprocessRunner(),
            IndependentEvaluator(),
            threads=threads,
            timeout_seconds=timeout_seconds,
        )
        workspace.register_baseline(
            score,
            baseline_dir / "validation/result.json",
            backend=request["backend"],
            parameters=request.get("parameters"),
        )
        regression = task.spec.kind == TaskKind.REGRESSION
        variants = [
            (
                "capacity",
                "More tree capacity can reduce underfit in the measured worst segments",
                {"backend": "catboost", "iterations": 600, "depth": 7},
                False,
                "baseline",
            ),
            (
                "backend",
                "Leaf-wise trees and train-fitted category handling may change segment errors",
                {"backend": "lightgbm", "n_estimators": 600, "num_leaves": 31},
                False,
                "baseline",
            ),
        ]
        if regression:
            variants.append(
                (
                    "target_transform",
                    "Signed-log target may reduce errors in skewed segments",
                    {"backend": "catboost", "iterations": 600, "depth": 7},
                    True,
                    "capacity",
                )
            )
        else:
            variants.append(
                (
                    "class_balance",
                    "Class weights may improve F1/recall of supported weak classes",
                    {
                        "backend": "catboost",
                        "iterations": 200,
                        "depth": 6,
                        "auto_class_weights": "Balanced",
                    },
                    False,
                    "baseline",
                )
            )
        variants.append(
            (
                "inner_tuning",
                "Train-only numeric tuning may improve full validation",
                {
                    "backend": "catboost",
                    "iterations": 200,
                    "tuning": {"trials": 3, "time_limit": 60},
                },
                False,
                "baseline",
            )
        )
        if hypotheses is not None:
            available = {variant[0] for variant in variants}
            if set(hypotheses) - available:
                raise ContractError("Unsupported verification hypothesis for this task kind")
            variants = [variant for variant in variants if variant[0] in hypotheses]
        outcomes = []
        for identifier, hypothesis, options, transform, parent_id in variants:
            parent = next((item for item in workspace.reports if item["id"] == parent_id), None)
            if parent_id != "baseline" and (parent is None or parent["status"] != "succeeded"):
                parent_id = "baseline"
                if transform:
                    options = dict((workspace.baseline_report or {})["parameters"])
            parent_report = workspace.baseline_report if parent_id == "baseline" else parent
            targets = tuple(diagnostic_focus((parent_report or {}).get("error_report", {})))
            direction = ResearchDirection(
                parent_id, parent_id, (), targets, "Offline scripted verification", ()
            )
            workspace.prepare_hypothesis(direction, "verify-" + identifier)
            (workspace.path / "candidate.py").write_text(
                candidate_code(options, transform=transform)
            )
            report = workspace.run_experiment(
                {
                    "experiment_id": identifier,
                    "parent_id": parent_id,
                    "hypothesis": hypothesis,
                    "expected_effect": "Measured change; improvement is not assumed",
                    "hypothesis_source": "offline_scripted_verification",
                }
            )
            summary = {
                "id": identifier,
                "parent_id": parent_id,
                "status": report["status"],
                "full_score": report.get("full_score"),
                "elapsed_seconds": report.get("elapsed_seconds"),
                "errors": diagnostic_summary(report.get("error_report", {})),
            }
            outcomes.append(summary)
            print(json.dumps(summary, ensure_ascii=False), flush=True)
        result = {
            "source": "offline_scripted_verification",
            "qwen_verified": False,
            "protocol_id": task.protocol.fingerprint,
            "baseline_score": asdict(score),
            "experiments": outcomes,
        }
        write_json(workspace.path / "verification.json", result)
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", type=Path)
    parser.add_argument("baseline_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--threads", type=int, choices=range(1, 9), default=8)
    parser.add_argument("--worker-timeout", type=float, default=600)
    parser.add_argument(
        "--hypotheses",
        nargs="+",
        choices=("capacity", "backend", "target_transform", "class_balance", "inner_tuning"),
    )
    args = parser.parse_args()
    verify(
        args.task,
        args.baseline_dir,
        args.output_dir,
        threads=args.threads,
        timeout_seconds=args.worker_timeout,
        hypotheses=tuple(args.hypotheses) if args.hypotheses else None,
    )


if __name__ == "__main__":
    main()
