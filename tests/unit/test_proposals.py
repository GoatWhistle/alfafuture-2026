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


def test_unknown_target_error_rejected_before_training(tmp_path):
    from types import SimpleNamespace

    from ml_harness.adapters.llm.proposals import ProposalTools

    task = SimpleNamespace(
        spec=SimpleNamespace(kind=SimpleNamespace(value="regression"), features=("a",)),
        protocol=SimpleNamespace(fingerprint="split"),
    )
    parent = {
        "id": "baseline",
        "error_report": {
            "measured_by": "harness",
            "protocol_id": "split",
            "level": "full",
            "segments": [],
        },
    }
    workspace = SimpleNamespace(
        task=task, path=tmp_path, active_parent_id="baseline", read_experiments=lambda: [parent]
    )
    tools = ProposalTools(workspace, None, None, "check", 1)
    raw = {
        "parent_id": "baseline",
        "task_kind": "regression",
        "backend": "catboost",
        "target_error": "segment:invented",
        "rationale": "test",
        "expected_effect": "test",
    }
    result = tools.execute(ToolCall("x", "propose_experiment", json.dumps(raw)))
    assert not result["ok"]
    assert "measured parent error" in result["error"]["message"]


def test_tuning_accepts_complete_feature_pipeline():
    result = resolve(
        proposal(tune=True, transforms=({"operation": "signed_log1p", "columns": ["a"]},)),
        "regression",
        ("a",),
    )
    assert result.resolved is not None
    assert result.resolved.transforms and result.resolved.tuning


def test_descendant_inherits_executed_pipeline_and_small_early_stop_count():
    parent = {
        "task_kind": "regression",
        "backend": "catboost",
        "parameters": {"iterations": 1, "depth": 4},
        "transforms": [{"operation": "signed_log1p", "columns": ["a"]}],
        "python_transform": "",
        "tuning": {},
    }
    recipe = resolve(
        proposal(parameters={"depth": 5}, tune=True), "regression", ("a",), parent
    ).resolved
    assert recipe.parameters == {"iterations": 1, "depth": 5}
    assert recipe.transforms == tuple(parent["transforms"])
    assert recipe.tuning


def test_read_only_column_details_precede_proposal_without_consuming_corrections():
    class Clock:
        def now(self):
            return 0

    seen = []

    class Model:
        def complete(self, messages, tools):
            seen.append(json.loads(messages[1]["content"]))
            name = "column_details" if len(seen) == 1 else "propose_experiment"
            return ModelReply("", (ToolCall("x", name, "{}"),))

    class Tools:
        schemas = []

        def execute(self, call):
            if call.name == "column_details":
                return {"ok": True, "column_details": {"late": {"description": "useful"}}}
            return {"ok": True, "result": {"id": "feature", "status": "succeeded"}}

    result = run_proposal_session(
        Model(),
        Tools(),
        "{}",
        RunBudget(1000, 100, 9),
        Clock(),
        max_steps=1,
        reserve_seconds=lambda: 30,
    )
    assert result.reason == "experiment_completed"
    assert result.tool_failures == ()
    assert seen[1]["column_details"]["column_details"]["late"]["description"] == "useful"


def test_unknown_usage_stops_llm_even_with_conservative_charge():
    class Clock:
        def now(self):
            return 0

    class Model:
        calls = 0

        def complete(self, messages, tools):
            self.calls += 1
            return ModelReply(
                "",
                (ToolCall("details", "column_details", "{}"),),
                usage_known=False,
                unreported_tokens_upper_bound=10,
            )

    class Tools:
        schemas = []

        def execute(self, call):
            pytest.fail("Unknown usage must stop the semantic step")

    model = Model()
    result = run_proposal_session(
        model,
        Tools(),
        "{}",
        RunBudget(1000, 100, 9),
        Clock(),
        reserve_seconds=lambda: 30,
    )
    assert model.calls == 1
    assert result.reason == "unknown_usage"
    assert result.tokens_charged == 10 and result.budget_accounted
    assert not result.usage_known
