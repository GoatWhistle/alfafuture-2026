"""Application failure boundaries, without launching models."""

from unittest.mock import Mock

import pytest

from ml_harness.application.ports.execution import ExecutionResult
from ml_harness.application.use_cases.baseline import run_baseline
from ml_harness.domain.models import ContractError, ExperimentStatus, Metric, Score


def test_baseline_evaluates_before_refit_and_write():
    task, runner, evaluator, writer = Mock(), Mock(), Mock(), Mock()
    output, refit_output = {"kind": "validation"}, {"kind": "refit"}
    runner.run.side_effect = [
        ExecutionResult(ExperimentStatus.SUCCEEDED, 2, output),
        ExecutionResult(ExperimentStatus.SUCCEEDED, 3, refit_output),
    ]
    evaluator.evaluate.return_value = Score(2, Metric.RMSE, "frozen")
    writer.write.return_value = "submission.csv"
    result = run_baseline(task, runner, evaluator, writer, work_dir="work")
    evaluator.evaluate.assert_called_once_with(task, output)
    writer.write.assert_called_once_with(task, refit_output)
    assert [call.args[0].mode for call in runner.run.call_args_list] == ["validation", "refit"]
    assert result.validation_seconds == 2 and result.refit_seconds == 3


@pytest.mark.parametrize("failed_phase", ["validation", "refit", "evaluation"])
def test_baseline_never_writes_failed_results(failed_phase):
    task, runner, evaluator, writer = Mock(), Mock(), Mock(), Mock()
    good = ExecutionResult(ExperimentStatus.SUCCEEDED, 1, {"predictions": []})
    bad = ExecutionResult(ExperimentStatus.TIMED_OUT, 2, detail="deadline")
    runner.run.side_effect = [bad] if failed_phase == "validation" else [good, bad]
    if failed_phase == "evaluation":
        evaluator.evaluate.side_effect = ContractError("invalid predictions")
    with pytest.raises(ContractError):
        run_baseline(task, runner, evaluator, writer, work_dir="work")
    writer.write.assert_not_called()
    assert runner.run.call_count == (2 if failed_phase == "refit" else 1)
