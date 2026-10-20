"""External proposal schema, deterministic materialization and resource admission."""

import hashlib
import json
from dataclasses import asdict
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ml_harness.adapters.data.profile import column_details, dataset_profile
from ml_harness.adapters.research.workspace import ResearchWorkspace, smoke_task, write_json
from ml_harness.application.ports.clock import Clock
from ml_harness.application.ports.language_model import ToolCall
from ml_harness.application.use_cases.directions import diagnostic_focus
from ml_harness.application.use_cases.observations import experiment_observation
from ml_harness.domain.budget import WallClockBudget
from ml_harness.domain.models import ContractError
from ml_harness.domain.proposals import (
    CLASSIFICATION_CHOICES,
    RANGES,
    Proposal,
    ResolvedExperiment,
    cost_estimate,
    require_resolved,
    resolve,
)


class ProposalArguments(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    parent_id: str
    task_kind: Literal["regression", "binary", "multiclass"]
    backend: Literal["catboost", "lightgbm"]
    target_error: str
    rationale: str
    expected_effect: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    transforms: list[dict[str, Any]] = Field(default_factory=list)
    python_transform: Annotated[str, Field(max_length=12000)] = ""
    tune: bool = False


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def materialize(recipe: ResolvedExperiment) -> str:
    options = {"backend": recipe.backend, **recipe.parameters}
    if recipe.tuning:
        options["tuning"] = recipe.tuning
    prefix = (
        "from ml_harness.adapters.ml.candidates import build_model as base_model\n"
        "from ml_harness.adapters.ml.feature_recipe import FeatureRecipe\n"
        "def build_model(kind, parameters, threads, seed):\n"
        f"    if kind != {recipe.task_kind!r}:\n        raise ValueError('Task kind changed')\n"
        f"    options = {options!r}\n"
        "    options.update(parameters)\n"
    )
    if recipe.transforms or recipe.python_transform:
        return prefix + (
            f"    return FeatureRecipe(kind, options, threads, seed, "
            f"{list(recipe.transforms)!r}, {recipe.python_transform!r})\n"
        )
    return prefix + "    return base_model(kind, options, threads, seed)\n"


class ProposalTools:
    def __init__(
        self,
        workspace: ResearchWorkspace,
        wall: WallClockBudget,
        clock: Clock,
        identifier: str,
        baseline_seconds: float,
    ):
        self.workspace, self.wall, self.clock = workspace, wall, clock
        self.identifier, self.baseline_seconds = identifier, baseline_seconds
        self.attempt = 0
        self.results: dict[str, dict[str, Any]] = {}
        self.detail_requests = 0
        self.profile: dict[str, Any] | None = None

    @property
    def schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "column_details",
                    "description": (
                        "Read-only profile for up to eight named features; "
                        "at most three requests per hypothesis."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "columns": {
                                "type": "array",
                                "items": {"type": "string"},
                                "minItems": 1,
                                "maxItems": 8,
                            }
                        },
                        "required": ["columns"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "propose_experiment",
                    "description": (
                        "Submit one executable hypothesis. Numeric backend capabilities: "
                        + json.dumps(
                            {
                                backend: {
                                    name: [low, high, kind.__name__]
                                    for name, (low, high, kind) in ranges.items()
                                }
                                for backend, ranges in RANGES.items()
                            }
                        )
                        + ". Classification-only balancing choices: "
                        + json.dumps(CLASSIFICATION_CHOICES)
                        + ". "
                        "Transforms: {operation: drop/signed_log1p/frequency, columns: [feature]}. "
                        "python_transform exports build_transformer() with fit(X), transform(X); "
                        "Return DataFrame, stable index/order/schema. No files/network/targets. "
                        "tune=true: 2 train-only Optuna trials, 30s on the complete pipeline. "
                        "Each transform fits only inner-train. Numeric capability ranges define "
                        "search; tree count is capped by the requested recipe count. "
                        "Harness owns seed, threads, objectives, metric, paths and splits."
                    ),
                    "parameters": ProposalArguments.model_json_schema(),
                },
            },
        ]

    def execute(self, call: ToolCall) -> dict[str, Any]:
        if call.name == "column_details":
            try:
                if self.detail_requests >= 3:
                    raise ContractError("Column detail request limit reached")
                self.detail_requests += 1
                raw_details = json.loads(call.arguments)
                if (
                    set(raw_details) != {"columns"}
                    or not isinstance(raw_details["columns"], list)
                    or any(not isinstance(name, str) for name in raw_details["columns"])
                ):
                    raise ContractError("Expected columns: list of feature names")
                if self.profile is None:
                    self.profile = dataset_profile(self.workspace.task)
                details = column_details(self.profile, raw_details["columns"])
                encoded = json.dumps(details, ensure_ascii=False)
                if len(encoded) > 10000:
                    raise ContractError(
                        "Column details exceed response limit; request fewer columns"
                    )
                return {"ok": True, "column_details": details}
            except (ContractError, ValueError, TypeError) as exc:
                return {
                    "ok": False,
                    "error": {"type": "invalid_column_request", "message": str(exc)},
                }
        key = hashlib.sha256((call.name + call.arguments).encode()).hexdigest()
        if key in self.results:
            return self.results[key]
        self.attempt += 1
        identifier = f"{self.identifier}-{self.attempt}"
        try:
            if call.name != "propose_experiment":
                raise ContractError("Only propose_experiment is available")
            raw = ProposalArguments.model_validate_json(call.arguments).model_dump()
            proposal = Proposal(**{**raw, "transforms": tuple(raw["transforms"])})
            if proposal.parent_id != self.workspace.active_parent_id:
                raise ContractError("Use the parent_id selected in the current context")
            parent = next(
                r for r in self.workspace.read_experiments() if r["id"] == proposal.parent_id
            )
            diagnostic = parent.get("error_report", {})
            measured_keys = (
                {item["key"] for item in diagnostic_focus(diagnostic, limit=1000000)}
                if diagnostic.get("measured_by") == "harness"
                and diagnostic.get("protocol_id") == self.workspace.task.protocol.fingerprint
                and diagnostic.get("level") == "full"
                else set()
            )
            if proposal.target_error not in measured_keys and not (
                proposal.target_error.startswith("exploratory:")
                and proposal.target_error.partition(":")[2].strip()
            ):
                raise ContractError(
                    "target_error must reference a measured parent error "
                    "or use exploratory: with a hypothesis"
                )
            parent_recipe = parent.get("executed_recipe") or parent.get("resolved_spec")
            if parent_recipe is None and parent["backend"] not in ("catboost", "lightgbm"):
                parent_recipe = {
                    "backend": parent["backend"],
                    "parameters": parent.get("parameters", {}),
                }
            if parent_recipe is None:
                parent_recipe = asdict(
                    require_resolved(
                        resolve(
                            Proposal(
                                proposal.parent_id,
                                proposal.task_kind,
                                parent["backend"],
                                "baseline",
                                "baseline",
                                "baseline",
                                {
                                    k: v
                                    for k, v in parent.get("parameters", {}).items()
                                    if k != "backend"
                                },
                            ),
                            proposal.task_kind,
                            self.workspace.task.spec.features,
                        )
                    )
                )
            inherited = (
                ResolvedExperiment(**parent_recipe)
                if parent_recipe.get("backend") in ("catboost", "lightgbm")
                else None
            )
            recipe = require_resolved(
                resolve(
                    proposal,
                    self.workspace.task.spec.kind.value,
                    self.workspace.task.spec.features,
                    parent_recipe=inherited,
                )
            )
            resolved = asdict(recipe)
            if digest(resolved) == digest(parent_recipe):
                raise ContractError("Recipe is unchanged from parent; propose a semantic change")
            prior = next(
                (r for r in self.workspace.reports if r.get("resolved_hash") == digest(resolved)),
                None,
            )
            if prior is not None:
                raise ContractError("Recipe already attempted: " + prior["id"])
            code = materialize(recipe)
            compile(code, "candidate.py", "exec")
            if recipe.python_transform:
                compile(recipe.python_transform, "feature_transform.py", "exec")
            self.workspace.path.joinpath("candidate.py").write_text(code)
            provenance = {
                "pipeline_version": 3,
                "requested_proposal": raw,
                "proposal_hash": digest(raw),
                "resolved_spec": resolved,
                "resolved_hash": digest(resolved),
                "effective_code_hash": hashlib.sha256(code.encode()).hexdigest(),
            }
            baseline = self.workspace.baseline_report or {}
            smoke = smoke_task(self.workspace.task)

            def admit(seconds: float) -> dict[str, Any]:
                estimate = cost_estimate(
                    recipe,
                    seconds,
                    len(self.workspace.task.protocol.train_indices),
                    len(smoke.protocol.train_indices),
                    self.baseline_seconds,
                    baseline.get("backend", ""),
                    baseline.get("parameters", {}),
                )
                available = min(
                    self.workspace.timeout_seconds, self.wall.available(self.clock.now())
                )
                allowed = estimate["estimated_full_seconds"] <= available
                return {
                    **estimate,
                    "allowed": allowed,
                    "available_seconds": available,
                    "reason": "admitted"
                    if allowed
                    else (
                        f"Estimated {estimate['estimated_full_seconds']:.1f}s "
                        f"exceeds available {available:.1f}s"
                    ),
                }

            report = self.workspace.run_experiment(
                {
                    "experiment_id": identifier,
                    "parent_id": proposal.parent_id,
                    "hypothesis": proposal.rationale,
                    "expected_effect": proposal.expected_effect,
                    "hypothesis_source": "qwen_proposal",
                    "recipe_provenance": provenance,
                },
                admit_full=admit,
            )
            result = {
                "ok": report["status"] == "succeeded",
                "result": experiment_observation(report),
            }
        except (ContractError, ValidationError, ValueError, SyntaxError, KeyError) as exc:
            result = {
                "ok": False,
                "error": {"type": "invalid_proposal", "message": str(exc)[:1200]},
            }
        write_json(
            self.workspace.path / "proposals" / (identifier + ".json"),
            {
                "tool": call.name,
                "arguments": call.arguments,
                "response": result,
            },
        )
        self.results[key] = result
        return result
