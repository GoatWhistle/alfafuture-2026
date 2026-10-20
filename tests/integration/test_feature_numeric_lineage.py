"""Structured semantic feature proposal survives numeric tuning and real frozen refit."""

import json
from pathlib import Path

import numpy as np
from test_prepare_task import task_fixture

from ml_harness.adapters.data.task import CsvTaskData
from ml_harness.adapters.execution.clock import SystemClock
from ml_harness.adapters.execution.runner import SubprocessRunner
from ml_harness.adapters.llm.proposals import ProposalTools, materialize
from ml_harness.adapters.research.workspace import ResearchWorkspace
from ml_harness.adapters.storage.submission import IndependentEvaluator
from ml_harness.application.ports.execution import RunRequest
from ml_harness.application.ports.language_model import ToolCall
from ml_harness.domain.budget import WallClockBudget
from ml_harness.domain.models import ExperimentStatus, Metric, Score
from ml_harness.domain.proposals import ResolvedExperiment

SOURCE = """
import pandas as pd
class Transformer:
    def fit(self, X):
        self.center = X.period.mean()
    def transform(self, X):
        return pd.DataFrame({"useful": (X.period - self.center) ** 2}, index=X.index)
def build_transformer():
    return Transformer()
"""


def test_structured_feature_numeric_and_frozen_refit_chain(tmp_path):
    task = CsvTaskData().prepare(str(task_fixture(tmp_path)), 42)
    runner, clock = SubprocessRunner(), SystemClock()
    workspace = ResearchWorkspace(
        task,
        tmp_path / "research",
        runner,
        IndependentEvaluator(),
        threads=1,
        timeout_seconds=30,
    )
    workspace.register_baseline(Score(1000, Metric.RMSE, task.protocol.fingerprint), None)
    workspace.active_parent_id = "baseline"
    tools = ProposalTools(workspace, WallClockBudget(clock.now() + 120), clock, "feature", 0.01)
    response = tools.execute(
        ToolCall(
            "proposal",
            "propose_experiment",
            json.dumps(
                {
                    "parent_id": "baseline",
                    "task_kind": "regression",
                    "backend": "catboost",
                    "target_error": "exploratory: centered nonlinear period captures signal",
                    "rationale": "Introduce a fitted nonlinear period feature",
                    "expected_effect": "lower RMSE",
                    "parameters": {"iterations": 200, "depth": 2},
                    "python_transform": SOURCE,
                }
            ),
        )
    )
    assert response.get("ok"), response
    semantic = workspace.reports[-1]
    assert semantic["status"] == "succeeded"
    assert semantic["requested_proposal"]["python_transform"] == SOURCE
    assert semantic["executed_recipe"]["python_transform"] == SOURCE

    assert workspace.numeric_batch("best", 0)
    numeric = workspace.reports[-1]
    assert numeric["parent_id"] == semantic["id"]
    assert numeric["hypothesis_source"] == "deterministic_optuna"
    assert numeric["resolved_spec"]["python_transform"] == SOURCE
    assert numeric["executed_recipe"]["python_transform"] == SOURCE
    assert numeric["executed_recipe"]["tuning"] == {}
    assert numeric["executed_recipe"]["parameters"] == {
        key: value
        for key, value in numeric["refit_parameters"].items()
        if key not in ("backend", "tuning")
    }
    code = Path(numeric.get("refit_code_path", numeric["code_path"]))
    assert SOURCE in numeric["executed_recipe"]["python_transform"]
    frozen = runner.run(
        RunRequest(
            task,
            str(tmp_path / "frozen"),
            mode="refit",
            candidate_path=str(code),
            parameters=numeric["refit_parameters"],
            threads=1,
        ),
        30,
    )
    assert frozen.status == ExperimentStatus.SUCCEEDED, frozen.detail
    assert "tuning_report" not in frozen.output
    request = json.loads((tmp_path / "frozen/request.json").read_text())
    assert request["parameters"] == numeric["refit_parameters"]
    assert request["candidate_path"] == str(code)

    recipe = ResolvedExperiment(**numeric["executed_recipe"])
    independent = tmp_path / "independent.py"
    independent.write_text(materialize(recipe))
    repeated = runner.run(
        RunRequest(
            task,
            str(tmp_path / "repeated"),
            mode="refit",
            candidate_path=str(independent),
            threads=1,
        ),
        30,
    )
    assert repeated.status == ExperimentStatus.SUCCEEDED, repeated.detail
    assert "tuning_report" not in repeated.output
    np.testing.assert_allclose(frozen.output["predictions"], repeated.output["predictions"])
    # Changing the only generated feature changes predictions under the frozen parameters.
    ablated = tmp_path / "ablated.py"
    ablated.write_text(
        materialize(
            ResolvedExperiment(
                **{
                    **numeric["executed_recipe"],
                    "python_transform": SOURCE.replace("(X.period - self.center) ** 2", "X.period"),
                }
            )
        )
    )
    without_feature = runner.run(
        RunRequest(
            task,
            str(tmp_path / "ablated"),
            mode="refit",
            candidate_path=str(ablated),
            threads=1,
        ),
        30,
    )
    assert without_feature.status == ExperimentStatus.SUCCEEDED, without_feature.detail
    assert not np.allclose(frozen.output["predictions"], without_feature.output["predictions"])
