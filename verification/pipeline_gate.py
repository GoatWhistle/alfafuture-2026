"""Version 2 acceptance rechecks immutable recipes, measurements and the public cohort."""

from dataclasses import asdict
from pathlib import Path
from typing import Any

from ml_harness.adapters.llm.proposals import digest, materialize
from ml_harness.adapters.storage.submission import IndependentEvaluator
from ml_harness.domain.proposals import Proposal, frozen_recipe, require_resolved, resolve

VERSION = 2


def verify_recipe(report: dict[str, Any], task: Any, reports: list[dict[str, Any]]) -> None:
    import hashlib
    import json

    if (
        report.get("pipeline_version") != VERSION
        or report.get("hypothesis_source") != "qwen_proposal"
    ):
        raise ValueError("Not a version 2 Qwen proposal")
    raw = report["requested_proposal"]
    if digest(raw) != report["proposal_hash"]:
        raise ValueError("Proposal hash mismatch")
    recipe = require_resolved(
        resolve(
            Proposal(**{**raw, "transforms": tuple(raw["transforms"])}),
            task.spec.kind.value,
            task.spec.features,
        )
    )
    spec = asdict(recipe)
    if digest(spec) != report["resolved_hash"] or digest(report["resolved_spec"]) != digest(spec):
        raise ValueError("Resolved recipe hash mismatch")
    code = Path(report["code_path"]).read_bytes()
    if (
        code != materialize(recipe).encode()
        or hashlib.sha256(code).hexdigest() != report["code_hash"]
        or report["effective_code_hash"] != report["code_hash"]
    ):
        raise ValueError("Effective code hash mismatch")
    parent = next((r for r in reports if r["id"] == report["parent_id"]), None)
    if parent is None or parent["code_hash"] != report["parent_code_hash"]:
        raise ValueError("Parent provenance mismatch")
    if report["code_hash"] == parent["code_hash"] or report.get("reused_from"):
        raise ValueError("Unchanged or reused candidate is not a new experiment")
    if parent.get("executed_recipe") or parent.get("resolved_spec"):
        parent_spec = parent.get("executed_recipe") or parent["resolved_spec"]
    elif parent["backend"] not in ("catboost", "lightgbm"):
        parent_spec = {"backend": parent["backend"], "parameters": parent.get("parameters", {})}
    else:
        parent_spec = asdict(
            require_resolved(
                resolve(
                    Proposal(
                        parent["id"],
                        task.spec.kind.value,
                        parent["backend"],
                        "baseline",
                        "baseline",
                        "baseline",
                        {k: v for k, v in parent.get("parameters", {}).items() if k != "backend"},
                    ),
                    task.spec.kind.value,
                    task.spec.features,
                )
            )
        )
    if digest(parent_spec) == digest(spec):
        raise ValueError("Semantically unchanged recipe")
    if (
        report.get("status") != "succeeded"
        or report.get("level") != "full"
        or not report.get("admission", {}).get("allowed")
        or report.get("protocol_id") != task.protocol.fingerprint
        or report.get("data_fingerprint") != task.protocol.data_fingerprint
    ):
        raise ValueError("Incomplete or incompatible full measurement")
    if raw["parent_id"] != report["parent_id"] or report["backend"] != recipe.backend:
        raise ValueError("Proposal lineage/backend mismatch")
    request = json.loads((Path(report["prediction_path"]).parent / "request.json").read_text())
    if (
        Path(request["candidate_path"]).read_bytes() != code
        or request["protocol"]["fingerprint"] != task.protocol.fingerprint
        or request["spec"]["kind"] != recipe.task_kind
    ):
        raise ValueError("Executed worker recipe/protocol differs from provenance")
    predictions = Path(report["prediction_path"])
    if hashlib.sha256(predictions.read_bytes()).hexdigest() != report["prediction_hash"]:
        raise ValueError("Prediction hash mismatch")
    output = json.loads(predictions.read_text())
    fitted = frozen_recipe(spec, output.get("refit_parameters"))
    if (
        digest(fitted) != report["executed_recipe_hash"]
        or digest(report["executed_recipe"]) != digest(fitted)
        or report.get("refit_parameters") != output.get("refit_parameters")
    ):
        raise ValueError("Fitted recipe differs from measured worker parameters")
    score = IndependentEvaluator().evaluate(task, output)
    if asdict(score) != report["full_score"]:
        raise ValueError("Measured full score mismatch")
    if hashlib.sha256(Path(parent["code_path"]).read_bytes()).hexdigest() != parent["code_hash"]:
        raise ValueError("Parent code hash mismatch")
    if parent.get("prediction_path"):
        parent_score = IndependentEvaluator().evaluate(
            task, json.loads(Path(parent["prediction_path"]).read_text())
        )
        if asdict(parent_score) != parent["full_score"]:
            raise ValueError("Parent measured score mismatch")
    if report["parent_score"] != parent["full_score"]:
        raise ValueError("Parent score mismatch")
    if report["parent_delta"] != score.value - parent["full_score"]["value"]:
        raise ValueError("Parent delta mismatch")


def validate_gate_shape(
    config: dict[str, Any], runs: list[dict[str, Any]], expected_tasks: dict[str, Any]
) -> None:
    if config.get("gate_version") != VERSION or not config.get("gate"):
        raise ValueError("Legacy or non-gate configuration")
    if config.get("seeds") != [42] or config.get("modes") != ["qwen"]:
        raise ValueError("Public gate requires Qwen seed 42")
    actual = {item["path"]: item["inputs"] for item in config["tasks"]}
    if len(config["tasks"]) != 2 or actual != expected_tasks:
        raise ValueError("Gate must contain both original public tasks and exact inputs")
    if len(runs) != 2 or {r["task"] for r in runs} != set(expected_tasks):
        raise ValueError("Missing or empty public cohort")
    if any(r["seed"] != 42 or r["mode"] != "qwen" for r in runs):
        raise ValueError("Invalid gate run configuration")
    if (
        config.get("sessions") != 3
        or config.get("threads") != 4
        or config.get("time_budget") != 1800
        or config.get("worker_timeout") != 300
        or config.get("token_limit") != 32000
        or config.get("memory_limit_mb") != 6000
    ):
        raise ValueError("Gate resource configuration mismatch")
