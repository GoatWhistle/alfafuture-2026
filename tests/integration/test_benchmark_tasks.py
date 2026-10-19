"""Bundled offline benchmarks have reproducible, independently parsed task contracts."""

import importlib.util
from pathlib import Path

import pytest

from ml_harness.adapters.data.task import CsvTaskData, read_csv
from ml_harness.domain.models import Metric, PredictionKind, TaskKind

pytestmark = pytest.mark.integration


def generator():
    script = Path(__file__).resolve().parents[2] / "verification/benchmark_tasks.py"
    spec = importlib.util.spec_from_file_location("benchmark_tasks", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.generate_tasks


def snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_three_bundled_benchmark_contracts_and_stable_files(tmp_path: Path) -> None:
    generate = generator()
    first = tmp_path / "first"
    second = tmp_path / "second"
    paths = generate(first)
    saved = snapshot(first)
    generate(first)
    generate(second)
    assert snapshot(first) == snapshot(second) == saved
    expected = {
        "diabetes": (TaskKind.REGRESSION, Metric.RMSE, PredictionKind.VALUES, ()),
        "breast_cancer": (TaskKind.BINARY, Metric.ROC_AUC, PredictionKind.PROBABILITIES, (0, 1)),
        "wine": (TaskKind.MULTICLASS, Metric.MACRO_F1, PredictionKind.LABELS, (1, 2, 3)),
    }
    assert paths.keys() == expected.keys()
    for name, path in paths.items():
        task = CsvTaskData().prepare(str(path), 2026)
        assert (task.spec.kind, task.spec.metric, task.spec.prediction_kind, task.spec.classes) == (
            expected[name]
        )
        train_columns, train = read_csv(Path(task.spec.train_path))
        test_columns, test = read_csv(Path(task.spec.test_path))
        holdout_columns, holdout = read_csv(path.parent / "holdout_targets.csv")
        assert set(train_columns) - set(test_columns) == {task.spec.target}
        assert holdout_columns == (task.spec.id_column, task.spec.target)
        assert [r[task.spec.id_column] for r in test] == [r[task.spec.id_column] for r in holdout]
        assert not (
            {r[task.spec.id_column] for r in train} & {r[task.spec.id_column] for r in test}
        )
        assert task.train_rows == len(train)
        assert task.test_rows == len(test)
        assert not task.protocol.control_indices
        if name == "breast_cancer":
            assert task.spec.submission_columns == ("case_key", "diagnosis_probability")
