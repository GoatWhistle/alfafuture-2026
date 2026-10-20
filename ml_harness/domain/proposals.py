"""Executable experiment recipes; resource and evaluation controls are never model input."""

from dataclasses import asdict, dataclass, field
from math import isfinite
from typing import Any

from ml_harness.domain.models import ContractError


@dataclass(frozen=True)
class Proposal:
    parent_id: str
    task_kind: str
    backend: str
    target_error: str
    rationale: str
    expected_effect: str
    parameters: dict[str, Any] = field(default_factory=dict)
    transforms: tuple[dict[str, Any], ...] = ()
    python_transform: str = ""
    tune: bool = False


@dataclass(frozen=True)
class ResolvedExperiment:
    task_kind: str
    backend: str
    parameters: dict[str, Any]
    transforms: tuple[dict[str, Any], ...]
    python_transform: str
    tuning: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# The complete executable pipeline retains the established serialization contract.
PipelineRecipe = ResolvedExperiment


@dataclass(frozen=True)
class ValidationResult:
    resolved: ResolvedExperiment | None
    errors: tuple[str, ...] = ()


RANGES = {
    "catboost": {
        "iterations": (10, 600, int),
        "depth": (2, 8, int),
        "learning_rate": (0.005, 0.3, float),
        "l2_leaf_reg": (0, 100, float),
    },
    "lightgbm": {
        "n_estimators": (10, 600, int),
        "num_leaves": (4, 63, int),
        "learning_rate": (0.005, 0.3, float),
        "min_child_samples": (5, 200, int),
        "reg_lambda": (0, 100, float),
    },
}


CLASSIFICATION_CHOICES: dict[str, dict[str, tuple[Any, ...]]] = {
    "catboost": {"auto_class_weights": (None, "Balanced", "SqrtBalanced")},
    "lightgbm": {"class_weight": (None, "balanced")},
}


def resolve(
    proposal: Proposal,
    task_kind: str,
    features: tuple[str, ...],
    parent_recipe: ResolvedExperiment | dict[str, Any] | None = None,
) -> ValidationResult:
    """Empty feature fields inherit the executed parent; nonempty fields replace them.

    Model parameters override the same-backend parent. Backend switches retain features
    and receive backend defaults. Tuning is always an explicit request, never inherited.
    """
    parent = (
        parent_recipe.to_dict() if isinstance(parent_recipe, ResolvedExperiment) else parent_recipe
    )
    transforms = (
        proposal.transforms or tuple(parent.get("transforms", ()))
        if parent
        else proposal.transforms
    )
    python_transform = (
        proposal.python_transform or parent.get("python_transform", "")
        if parent
        else proposal.python_transform
    )
    errors = []
    if proposal.task_kind != task_kind or task_kind not in ("regression", "binary", "multiclass"):
        errors.append("task_kind must equal the supplied task kind")
    if proposal.backend not in RANGES:
        return ValidationResult(None, ("backend must be catboost or lightgbm",))
    for name in ("parent_id", "target_error", "rationale", "expected_effect"):
        if not isinstance(getattr(proposal, name), str) or not getattr(proposal, name).strip():
            errors.append(name + " must be nonempty")
    options = (
        {"iterations": 200, "depth": 6}
        if proposal.backend == "catboost"
        else {"n_estimators": 200, "num_leaves": 31}
    )
    if parent and parent.get("backend") == proposal.backend:
        options.update(parent.get("parameters", {}))
    for key, value in proposal.parameters.items():
        if key in CLASSIFICATION_CHOICES[proposal.backend]:
            choices = CLASSIFICATION_CHOICES[proposal.backend][key]
            if task_kind == "regression" or value not in choices:
                errors.append(f"{key} requires classification and one of {choices}")
        elif key in RANGES[proposal.backend]:
            low, high, kind = RANGES[proposal.backend][key]
            if (
                type(value) not in (int, float)
                or not isfinite(value)
                or not low <= value <= high
                or (kind is int and type(value) is not int)
            ):
                errors.append(f"{key} must be {kind.__name__} in [{low}, {high}]")
        else:
            errors.append(f"Unsupported or harness-owned parameter: {key}")
        options[key] = value
    remaining_features = set(features)
    for transform in transforms:
        if set(transform) != {"operation", "columns"} or transform.get("operation") not in (
            "drop",
            "signed_log1p",
            "frequency",
        ):
            errors.append("Transforms require operation=drop/signed_log1p/frequency and columns")
            continue
        columns = transform["columns"]
        if (
            not isinstance(columns, list)
            or not columns
            or any(not isinstance(c, str) or c not in features for c in columns)
            or len(set(columns)) != len(columns)
        ):
            errors.append("Transform columns must be distinct known features")
        elif any(c not in remaining_features for c in columns):
            errors.append("Transform refers to previously removed feature")
        elif transform["operation"] == "drop":
            remaining_features.difference_update(columns)
            if not remaining_features:
                errors.append("Cannot drop all input features")
    if type(proposal.tune) is not bool:
        errors.append("tune must be boolean")
    if len(python_transform) > 12000:
        errors.append("Python transform exceeds 12000 characters")
    if errors:
        return ValidationResult(None, tuple(errors))
    return ValidationResult(
        ResolvedExperiment(
            task_kind,
            proposal.backend,
            options,
            transforms,
            python_transform,
            {"trials": 2, "time_limit": 30} if proposal.tune else {},
        )
    )


def complexity(backend: str, parameters: dict[str, Any]) -> float:
    if backend == "catboost":
        return float(parameters.get("iterations", 200)) * 2 ** float(parameters.get("depth", 6))
    return float(parameters.get("n_estimators", 200)) * float(parameters.get("num_leaves", 31))


def cost_estimate(
    recipe: ResolvedExperiment,
    smoke_seconds: float,
    train_rows: int,
    smoke_rows: int,
    baseline_seconds: float,
    baseline_backend: str,
    baseline_parameters: dict[str, Any],
) -> dict[str, Any]:
    """Conservative admission heuristic, never a guarantee; worker deadlines remain mandatory."""
    ratio = max(1.0, train_rows / max(1, smoke_rows))
    # Startup dominates tiny smoke samples: sublinear estimate plus calibrated same-backend floor.
    sampled = smoke_seconds * ratio**0.65
    calibrated = 0.0
    if baseline_backend == recipe.backend:
        calibrated = (
            baseline_seconds
            * complexity(recipe.backend, recipe.parameters)
            / complexity(baseline_backend, baseline_parameters)
        )
    tuning_factor = int(recipe.tuning.get("trials", 2)) + 1 if recipe.tuning else 1
    estimate = 1.5 * max(sampled, calibrated * tuning_factor)
    return {
        "estimated_full_seconds": estimate,
        "safety_factor": 1.5,
        "method": "max(smoke*row_ratio^0.65, same_backend_baseline*complexity*tuning)*1.5",
        "assumptions": "Startup and feature costs vary; heuristic is not a runtime guarantee",
        "smoke_seconds": smoke_seconds,
        "train_rows": train_rows,
        "smoke_rows": smoke_rows,
        "baseline_seconds": baseline_seconds,
        "same_backend_calibration": calibrated,
    }


def require_resolved(result: ValidationResult) -> ResolvedExperiment:
    if result.resolved is None:
        raise ContractError("; ".join(result.errors))
    return result.resolved


def frozen_recipe(
    recipe: dict[str, Any], refit_parameters: dict[str, Any] | None
) -> dict[str, Any]:
    """Separate the requested search strategy from the fitted learner used by descendants."""
    from copy import deepcopy

    result = deepcopy(recipe)
    if refit_parameters is not None:
        if refit_parameters.get("backend", result["backend"]) != result["backend"]:
            raise ContractError("Fitted backend differs from the resolved recipe")
        result["parameters"] = {
            k: v for k, v in refit_parameters.items() if k not in ("backend", "tuning")
        }
        result["tuning"] = {}
    return result
