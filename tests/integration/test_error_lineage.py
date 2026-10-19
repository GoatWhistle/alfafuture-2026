"""A measured parent, immutable checkout and paired diagnostics stay linked."""

import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest
from test_prepare_task import task_fixture

from ml_harness.adapters.data.task import CsvTaskData
from ml_harness.adapters.execution.runner import SubprocessRunner
from ml_harness.adapters.ml.diagnostics import IndependentErrorAnalyzer
from ml_harness.adapters.research.workspace import STARTER, ResearchWorkspace
from ml_harness.adapters.storage.submission import IndependentEvaluator
from ml_harness.application.ports.execution import RunRequest
from ml_harness.application.use_cases.directions import ResearchDirection, choose_direction
from ml_harness.domain.models import ContractError, ExperimentStatus


def test_baseline_parent_deltas_and_checkout_survive_restart(tmp_path):
    task = CsvTaskData().prepare(str(task_fixture(tmp_path)), 42)
    runner, evaluator = SubprocessRunner(), IndependentEvaluator()
    baseline_dir = tmp_path / "baseline"
    baseline = runner.run(
        RunRequest(task, str(baseline_dir), parameters={"iterations": 3}, threads=1), 10
    )
    assert baseline.status == ExperimentStatus.SUCCEEDED
    score = evaluator.evaluate(task, baseline.output)
    work = ResearchWorkspace(
        task,
        tmp_path / "research",
        runner,
        evaluator,
        analyzer=IndependentErrorAnalyzer(min_support=2),
        threads=1,
        timeout_seconds=10,
    )
    work.register_baseline(score, baseline_dir / "result.json", parameters={"iterations": 3})
    decision = choose_direction(task, score, [], baseline_report=work.baseline_report)
    work.prepare_hypothesis(decision, "one")
    baseline_code = (work.path / "candidate.py").read_bytes()
    code = STARTER.replace('"iterations": 200', '"iterations": 5')
    (work.path / "candidate.py").write_text(code)
    report = work.run_experiment(
        {
            "experiment_id": "child",
            "hypothesis": "capacity",
            "expected_effect": "measured",
            "parent_id": None,
        }
    )
    assert report["status"] == "succeeded" and report["parent_id"] == "baseline"
    diagnostic = report["error_report"]
    assert diagnostic["parent_score"] == asdict(score)
    assert diagnostic["delta"]["value"] == report["full_score"]["value"] - score.value
    assert diagnostic["interpretation"] is None
    assert diagnostic["segments"] and all("delta" in segment for segment in diagnostic["segments"])
    (work.path / "candidate.py").write_text('raise RuntimeError("corrupted draft")')
    child = ResearchDirection("child", "child", (), (), "explicit branch", ())
    work.prepare_hypothesis(child, "two")
    assert (work.path / "candidate.py").read_text() == code
    with pytest.raises(ContractError, match="Parent differs"):
        work.run_experiment(
            {
                "experiment_id": "wrong",
                "hypothesis": "bad lineage",
                "expected_effect": "none",
                "parent_id": "baseline",
            }
        )
    assert not (work.path / "experiments/wrong").exists()
    restored = ResearchWorkspace(task, work.path, runner, evaluator, threads=1)
    assert restored.reports[0]["error_report"] == diagnostic
    restored.prepare_hypothesis(decision, "three")
    assert (work.path / "candidate.py").read_bytes() == baseline_code
    assert Path(restored.reports[0]["code_path"]).read_text() == code
    with pytest.raises(ContractError, match="code/score changed"):
        restored.register_baseline(
            replace(score, value=score.value + 1),
            baseline_dir / "result.json",
            parameters={"iterations": 3},
        )


def test_unknown_parent_and_reserved_baseline_never_train(tmp_path):
    task = CsvTaskData().prepare(str(task_fixture(tmp_path)), 42)

    class NeverRun:
        def run(self, *args):
            raise AssertionError("No work for invalid lineage")

    work = ResearchWorkspace(task, tmp_path / "research", NeverRun(), IndependentEvaluator())
    for identifier, parent in [("new", "missing"), ("baseline", None)]:
        with pytest.raises(ContractError):
            work.run_experiment(
                {
                    "experiment_id": identifier,
                    "hypothesis": "x",
                    "expected_effect": "y",
                    "parent_id": parent,
                }
            )
    assert work.reports == []


def test_parent_prediction_corruption_is_explicit(tmp_path):
    task = CsvTaskData().prepare(str(task_fixture(tmp_path)), 42)
    runner, evaluator = SubprocessRunner(), IndependentEvaluator()
    output = tmp_path / "validation"
    result = runner.run(RunRequest(task, str(output), parameters={"iterations": 3}, threads=1), 10)
    score = evaluator.evaluate(task, result.output)
    work = ResearchWorkspace(task, tmp_path / "research", runner, evaluator)
    work.register_baseline(score, output / "result.json", parameters={"iterations": 3})
    (output / "result.json").write_text(json.dumps({"ids": []}))
    with pytest.raises(ContractError, match="integrity"):
        work.run_experiment(
            {
                "experiment_id": "new",
                "hypothesis": "x",
                "expected_effect": "y",
                "parent_id": "baseline",
            }
        )
