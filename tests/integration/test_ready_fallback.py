"""A valid early submission survives failures and never consumes control targets."""

import csv
import json
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import Mock

import pytest

from ml_harness.adapters.execution.worker import run
from ml_harness.adapters.storage.submission import (
    AtomicSubmissionWriter,
    ensure_ready_submission,
    load_submission,
)
from ml_harness.application.ports.execution import ExecutionResult
from ml_harness.application.use_cases.baseline import run_baseline
from ml_harness.domain.models import (
    ContractError,
    ExperimentStatus,
    Metric,
    PredictionKind,
    PreparedTask,
    Score,
    SplitKind,
    TaskKind,
    TaskSpec,
    ValidationProtocol,
)

pytestmark = pytest.mark.integration


def write_csv(path: Path, columns: list[str], rows: list[list]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(rows)


@pytest.fixture
def task(tmp_path: Path) -> PreparedTask:
    write_csv(
        tmp_path / "train.csv",
        ["id", "x", "y"],
        [[str(i), i, y] for i, y in enumerate([1, 3, 9, 100, 1e100, -1e100])],
    )
    write_csv(tmp_path / "test.csv", ["id", "x"], [["002", 2], ["001", 1]])
    spec = TaskSpec(
        "task",
        str(tmp_path / "train.csv"),
        str(tmp_path / "test.csv"),
        "dict",
        "sample",
        str(tmp_path / "submission.csv"),
        "y",
        "id",
        ("x",),
        TaskKind.REGRESSION,
        Metric.RMSE,
        PredictionKind.VALUES,
        ("id", "y"),
    )
    protocol = ValidationProtocol("split", "data", SplitKind.RANDOM, 1, (0, 1, 2), (3,), (4, 5))
    return PreparedTask(spec, protocol, 6, 2)


def test_statistical_regression_uses_only_training_median(task: PreparedTask) -> None:
    result = ensure_ready_submission(task, AtomicSubmissionWriter())
    assert result["status"] == "statistical_fallback_written"
    assert load_submission(task)["predictions"] == [3, 3]
    assert load_submission(task)["ids"] == ["002", "001"]


def test_valid_existing_bytes_are_preserved_even_when_training_unavailable(
    task: PreparedTask,
) -> None:
    destination = Path(task.spec.submission_path)
    original = b"id,y\n002,7.000\n001,8.000\n"
    destination.write_bytes(original)
    Path(task.spec.train_path).unlink()
    result = ensure_ready_submission(task, AtomicSubmissionWriter())
    assert result["status"] == "preserved_existing"
    assert destination.read_bytes() == original


@pytest.mark.parametrize(
    "invalid",
    [
        "id,y\n002,nan\n001,1\n",
        "id,y\n001,1\n002,2\n",
        "wrong,y\n002,1\n001,2\n",
        "id,y\n002,1\n002,2\n",
        "id,y\n002,inf\n001,1\n",
        "id,y\n002,1\n",
    ],
)
def test_invalid_existing_submission_is_atomically_replaced(
    task: PreparedTask, invalid: str
) -> None:
    destination = Path(task.spec.submission_path)
    destination.write_text(invalid)
    with pytest.raises(ContractError):
        load_submission(task)
    assert ensure_ready_submission(task, AtomicSubmissionWriter())["status"] == (
        "statistical_fallback_written"
    )
    assert load_submission(task)["predictions"] == [3, 3]
    assert list(destination.parent.glob(".submission-*")) == []


@pytest.mark.parametrize("classes", [("NA", "001"), (2, -1.5)])
@pytest.mark.parametrize("probability", [False, True])
def test_class_fallback_order_majority_and_laplace_probability(
    task: PreparedTask,
    classes: tuple,
    probability: bool,
) -> None:
    spec = replace(
        task.spec,
        kind=TaskKind.BINARY,
        classes=classes,
        metric=Metric.LOG_LOSS if probability else Metric.ACCURACY,
        prediction_kind=PredictionKind.PROBABILITIES if probability else PredictionKind.LABELS,
    )
    task = replace(task, spec=spec)
    write_csv(
        Path(spec.train_path),
        ["id", "x", "y"],
        [
            [str(i), i, y]
            for i, y in enumerate(
                [
                    classes[0],
                    classes[0],
                    classes[1],
                    "ignored",
                    "unknown_control",
                    "unknown_control",
                ]
            )
        ],
    )
    ensure_ready_submission(task, AtomicSubmissionWriter())
    output = load_submission(task)
    assert output["classes"] == list(classes)
    if probability:
        assert output["predictions"] == [[0.6, 0.4], [0.6, 0.4]]
        with Path(spec.submission_path).open() as stream:
            rows = list(csv.reader(stream))
        assert [float(row[1]) for row in rows[1:]] == [0.4, 0.4]
    else:
        assert output["predictions"] == [classes[0], classes[0]]


def test_multiclass_probability_submission(task: PreparedTask) -> None:
    spec = replace(
        task.spec,
        kind=TaskKind.MULTICLASS,
        classes=("z", "a", "m"),
        metric=Metric.LOG_LOSS,
        prediction_kind=PredictionKind.PROBABILITIES,
        submission_columns=("id", "z", "a", "m"),
    )
    task = replace(task, spec=spec)
    write_csv(
        Path(spec.train_path),
        ["id", "x", "y"],
        [[str(i), i, y] for i, y in enumerate(["z", "a", "m", "x", "x", "x"])],
    )
    ensure_ready_submission(task, AtomicSubmissionWriter())
    assert load_submission(task)["predictions"] == [[1 / 3] * 3] * 2


@pytest.mark.parametrize("bad", ["nan", "missing", ""])
def test_bad_training_targets_fail_explicitly_and_keep_existing_bytes(
    task: PreparedTask, bad: str
) -> None:
    destination = Path(task.spec.submission_path)
    destination.write_text("invalid existing bytes")
    write_csv(
        Path(task.spec.train_path), ["id", "x", "y"], [["0", 0, bad], ["1", 1, 3], ["2", 2, 9]]
    )
    with pytest.raises(ContractError):
        ensure_ready_submission(task, AtomicSubmissionWriter())
    assert destination.read_text() == "invalid existing bytes"


@pytest.mark.parametrize("control", [False, True])
def test_baseline_early_fit_uses_fallback_only_with_control(
    task: PreparedTask, control: bool
) -> None:
    if not control:
        task = replace(task, protocol=replace(task.protocol, control_indices=()))
    success = ExecutionResult(ExperimentStatus.SUCCEEDED, 1, {"predictions": []})
    runner = Mock()
    runner.run.return_value = success
    evaluator = Mock()
    evaluator.evaluate.return_value = Score(1, Metric.RMSE, "split")
    writer = Mock()
    writer.write.return_value = task.spec.submission_path
    run_baseline(task, runner, evaluator, writer, work_dir="work")
    requests = [call.args[0] for call in runner.run.call_args_list]
    assert requests[0].mode == "validation"
    assert requests[1].mode == ("fallback" if control else "refit")


def test_worker_fallback_excludes_control_but_final_refit_uses_all_rows(
    task: PreparedTask,
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "candidate.py"
    candidate.write_text("""import json
from pathlib import Path
class Model:
    def fit(self, X, y):
        Path(__file__).with_name('seen.json').write_text(json.dumps(X['x'].tolist()))
    def predict(self, X):
        return [0] * len(X)
def build_model(kind, parameters, threads, seed):
    return Model()
""")
    for mode, expected in [("fallback", [0, 1, 2, 3]), ("refit", [0, 1, 2, 3, 4, 5])]:
        payload = json.loads(
            json.dumps(
                {
                    "spec": asdict(task.spec),
                    "protocol": asdict(task.protocol),
                    "mode": mode,
                    "threads": 1,
                    "candidate_path": str(candidate),
                    "parameters": {},
                    "output_dir": str(tmp_path / mode),
                }
            )
        )
        run(payload)
        assert json.loads((tmp_path / "seen.json").read_text()) == expected
        assert json.loads((tmp_path / mode / "result.json").read_text())["ids"] == ["002", "001"]
