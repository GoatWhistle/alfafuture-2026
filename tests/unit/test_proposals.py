"""Validate the execution boundary, compute admission, and train-only feature contract."""

import json
from dataclasses import replace

import pandas as pd
import pytest

from ml_harness.adapters.ml.feature_recipe import FeatureRecipe
from ml_harness.application.ports.language_model import ModelReply, ToolCall
from ml_harness.application.use_cases.proposal_session import run_proposal_session
from ml_harness.domain.models import ContractError, RunBudget
from ml_harness.domain.proposals import Proposal, cost_estimate, require_resolved, resolve


def proposal(**kwargs):
    return replace(
        Proposal(
            "baseline",
            "regression",
            "catboost",
            "exploratory: variance",
            "shallower trees",
            "lower RMSE",
            {"depth": 3},
        ),
        **kwargs,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"backend": "regression"},
        {"task_kind": "multiclass"},
        {"parameters": {"eval_metric": "TotalPrecision"}},
        {"parameters": {"thread_count": 20}},
        {"parameters": {"depth": True}},
        {"parameters": {"depth": 7.5}},
        {"parameters": {"learning_rate": float("nan")}},
        {"parameters": {"random_seed": 1}},
        {"parameters": {"auto_class_weights": "Balanced"}},
        {"parameters": {"loss_function": "RMSE"}},
        {"transforms": ({"operation": "drop", "columns": ["target"]},)},
        {"tune": True, "python_transform": "pass"},
    ],
)
def test_invalid_recipes_rejected_before_execution(kwargs):
    result = resolve(proposal(**kwargs), "regression", ("a",))
    assert result.resolved is None and result.errors


def test_estimate_accounts_for_complexity_and_is_not_guarantee():
    small = require_resolved(resolve(proposal(), "regression", ("a",)))
    large = replace(small, parameters={"iterations": 600, "depth": 8})
    base = dict(
        smoke_seconds=1,
        train_rows=40000,
        smoke_rows=256,
        baseline_seconds=90,
        baseline_backend="catboost",
        baseline_parameters={},
    )
    assert cost_estimate(large, **base)["estimated_full_seconds"] > 300
    assert cost_estimate(small, **base)["estimated_full_seconds"] < 300
    assert "not a runtime guarantee" in cost_estimate(small, **base)["assumptions"]


def test_python_fit_receives_train_only_and_schema_is_enforced():
    code = """
class Center:
    def fit(self, X):
        self.center = X.a.mean()
    def transform(self, X):
        result = X.copy()
        result["centered"] = result.a - self.center
        return result
def build_transformer():
    return Center()
"""
    model = FeatureRecipe("regression", {"backend": "catboost", "iterations": 10}, 1, 42, [], code)
    train = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]}, index=[7, 3, 8, 2])
    model.fit(train, pd.Series([1.0, 2.0, 3.0, 4.0], index=train.index))
    test = pd.DataFrame({"a": [100.0]}, index=[19])
    assert model._transform(test).centered.iloc[0] == 97.5
    assert list(model._transform(test).index) == [19]
    broken = code.replace("return result", "return result.reset_index(drop=True)")
    bad = FeatureRecipe("regression", {"backend": "catboost", "iterations": 10}, 1, 42, [], broken)
    with pytest.raises(ContractError, match="row index"):
        bad.fit(train, pd.Series([1.0, 2.0, 3.0, 4.0]))


def test_session_repairs_have_fresh_budget_and_are_bounded():
    class Clock:
        value = 0

        def now(self):
            return self.value

    clock = Clock()
    seen = []

    class Model:
        def complete(self, messages, tools):
            seen.append(json.loads(messages[1]["content"])["budget"])
            clock.value += 10
            return ModelReply("", (ToolCall("x", "propose_experiment", "{}"),), 5, 5)

    class Tools:
        schemas = []

        def execute(self, call):
            clock.value += 20
            return {"ok": False, "error": "rejected"}

    result = run_proposal_session(
        Model(),
        Tools(),
        "{}",
        RunBudget(1000, 100, 9),
        clock,
        max_steps=12,
        reserve_seconds=lambda: 30,
    )
    assert result.steps == 3 and result.reason == "correction_limit"
    assert [x["seconds_remaining"] for x in seen] == [970, 940, 910]
    assert [x["tokens_remaining"] for x in seen] == [100, 90, 80]


@pytest.mark.parametrize(
    "arguments, rejected",
    [
        (
            '{"backend": "lightgbm", "parameters": {"num_leaves": 127}}',
            {"backend": "lightgbm", "parameters": {"num_leaves": 127}},
        ),
        ('{"backend":', '{"backend":'),
    ],
)
def test_correction_receives_exact_rejected_recipe(arguments, rejected):
    class Clock:
        def now(self):
            return 0

    seen = []

    class Model:
        def complete(self, messages, tools):
            seen.append(json.loads(messages[1]["content"]))
            return ModelReply("", (ToolCall("bad", "propose_experiment", arguments),))

    class Tools:
        schemas = []

        def execute(self, call):
            return {"ok": False, "error": {"message": "num_leaves must be int in [4, 63]"}}

    run_proposal_session(
        Model(),
        Tools(),
        "{}",
        RunBudget(1000, 1000, 9),
        Clock(),
        max_steps=2,
        reserve_seconds=lambda: 30,
    )
    assert seen[1]["previous_attempts"][-1]["requested_proposal"] == rejected
