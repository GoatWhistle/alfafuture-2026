"""Preserve an already valid CSV whenever a new result fails validation or writing."""

from pathlib import Path

import pytest
from test_prepare_task import task_fixture

from ml_harness.adapters.data.task import CsvTaskData, read_csv
from ml_harness.adapters.storage.submission import AtomicSubmissionWriter
from ml_harness.domain.models import ContractError


def test_atomic_output_order_and_previous_file_on_error(tmp_path):
    task = CsvTaskData().prepare(str(task_fixture(tmp_path)), 42)
    writer = AtomicSubmissionWriter()
    good = {"ids": ["102", "101"], "predictions": [1.25, 9.5], "classes": []}
    path = Path(writer.write(task, good))
    old = path.read_bytes()
    columns, rows = read_csv(path)
    assert columns == ("key", "response")
    assert [row["key"] for row in rows] == ["102", "101"]
    assert [row["response"] for row in rows] == ["1.25", "9.5"]
    for bad in (
        {**good, "ids": ["101", "102"]},
        {**good, "ids": ["102", "102"]},
        {**good, "predictions": [float("nan"), 1]},
        {**good, "predictions": [1]},
    ):
        with pytest.raises(ContractError):
            writer.write(task, bad)
        assert path.read_bytes() == old
    assert not list(tmp_path.glob(".submission-*"))


def test_replace_failure_preserves_old_file(tmp_path, monkeypatch):
    import ml_harness.adapters.storage.submission as storage

    task = CsvTaskData().prepare(str(task_fixture(tmp_path)), 42)
    good = {"ids": ["102", "101"], "predictions": [1, 2], "classes": []}
    path = Path(storage.AtomicSubmissionWriter().write(task, good))
    old = path.read_bytes()

    def fail(*args):
        raise OSError("synthetic disk failure")

    monkeypatch.setattr(storage.os, "replace", fail)
    with pytest.raises(OSError, match="disk failure"):
        storage.AtomicSubmissionWriter().write(task, {**good, "predictions": [3, 4]})
    assert path.read_bytes() == old
    assert not list(tmp_path.glob(".submission-*"))


def test_binary_probability_column_is_positive_class(tmp_path):
    task = CsvTaskData().prepare(
        str(task_fixture(tmp_path, labels=[0, 1] * 15, metric="roc_auc")), 42
    )
    # Worker columns are reversed; output must still contain P(spec.classes[1]).
    path = AtomicSubmissionWriter().write(
        task,
        {
            "ids": ["102", "101"],
            "predictions": [[0.7, 0.3], [0.1, 0.9]],
            "classes": [1, 0],
        },
    )
    _, rows = read_csv(Path(path))
    assert [float(row["response"]) for row in rows] == [0.7, 0.1]
