"""Small generated CSV fixtures exercise the same contract as public data."""

import csv
import json
from dataclasses import replace
from pathlib import Path

import pytest

from ml_harness.adapters.data.task import CsvTaskData
from ml_harness.application.use_cases.prepare_task import prepare_task
from ml_harness.domain.models import ContractError, Metric, SplitKind, TaskKind


def write_csv(path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(rows)


def task_fixture(root, *, labels=None, metric="rmse", role="feature", metadata=None):
    labels = labels if labels is not None else [float(i) / 3 for i in range(30)]
    write_csv(
        root / "input/train.csv",
        ["key", "x", "period", "response"],
        [[i, "" if i % 3 == 0 else i, i // 3, label] for i, label in enumerate(labels)],
    )
    write_csv(root / "input/test.csv", ["key", "x", "period"], [[102, "new", 20], [101, "", 21]])
    write_csv(
        root / "input/sample_submission.csv",
        ["key", "response"],
        [[101, labels[0]], [102, labels[0]]],
    )
    dtype = "float64" if isinstance(labels[0], int | float) else "object"
    write_csv(
        root / "input/data_dictionary.csv",
        ["feature", "role", "dtype"],
        [
            ["key", "id", "int64"],
            ["x", "feature", "object"],
            ["period", role, "int64"],
            ["response", "target", dtype],
        ],
    )
    meta = {"metric": metric, **(metadata or {})}
    path = root / "task.md"
    path.write_text(
        "\n".join(
            [
                "# Unfamiliar task",
                "`input/train.csv` `input/test.csv`",
                "`input/data_dictionary.csv` `input/sample_submission.csv`",
                "```task-json",
                json.dumps(meta),
                "```",
                "Output: `answer.csv`",
            ]
        )
    )
    return path


def prepared(path):
    return prepare_task(str(path), CsvTaskData())


def test_regression_paths_schema_reproducibility_and_fingerprint(tmp_path, monkeypatch):
    path = task_fixture(tmp_path)
    monkeypatch.chdir(tmp_path.parent)
    first = prepared(path)
    assert first.spec.target == "response"
    assert first.spec.id_column == "key"
    assert first.spec.kind == TaskKind.REGRESSION
    assert first.spec.metric == Metric.RMSE
    assert first.spec.features == ("x", "period")
    assert first.spec.submission_path == str(tmp_path / "answer.csv")
    assert first.protocol.kind == SplitKind.RANDOM
    assert first == prepared(path)
    assert set(first.protocol.train_indices + first.protocol.validation_indices) == set(range(30))
    assert first.protocol != prepare_task(str(path), CsvTaskData(), seed=19).protocol
    train_path = Path(first.spec.train_path)
    train_path.write_text(train_path.read_text().replace("0.3333333333333333", "0.34"))
    assert prepared(path).protocol.fingerprint != first.protocol.fingerprint


@pytest.mark.parametrize(
    "labels,kind,classes",
    [
        ([0, 1] * 15, TaskKind.BINARY, (0, 1)),
        ([2, 7, 9] * 10, TaskKind.MULTICLASS, (2, 7, 9)),
        (["alpha", "beta", "gamma"] * 10, TaskKind.MULTICLASS, ("alpha", "beta", "gamma")),
    ],
)
def test_classification_including_numeric_labels(tmp_path, labels, kind, classes):
    task = prepared(task_fixture(tmp_path, labels=labels, metric="macro_f1"))
    assert task.spec.kind == kind
    assert task.spec.classes == classes
    assert task.protocol.kind == SplitKind.STRATIFIED
    assert len(task.protocol.validation_indices) == 6


@pytest.mark.parametrize("role,split", [("group", SplitKind.GROUP), ("time", SplitKind.TIME)])
def test_group_and_time_do_not_leak(tmp_path, role, split):
    task = prepared(task_fixture(tmp_path, role=role))
    assert task.protocol.kind == split
    train_groups = {i // 3 for i in task.protocol.train_indices}
    validation_groups = {i // 3 for i in task.protocol.validation_indices}
    assert train_groups.isdisjoint(validation_groups)
    if role == "time":
        assert max(train_groups) < min(validation_groups)


@pytest.mark.parametrize(
    "metadata,match",
    [
        ({"target": "wrong"}, "contradicts"),
        ({"kind": "binary"}, "contradicts"),
        ({"prediction_kind": "probabilities"}, "contradicts"),
        ({"metric": "unknown"}, "Unsupported"),
        ({"time_column": "missing"}, "not a feature"),
    ],
)
def test_conflicting_description_is_rejected(tmp_path, metadata, match):
    with pytest.raises(ContractError, match=match):
        prepared(task_fixture(tmp_path, metadata=metadata))


def test_conflicting_sample_schema(tmp_path):
    path = task_fixture(tmp_path)
    write_csv(tmp_path / "input/sample_submission.csv", ["key", "wrong"], [[101, 0], [102, 0]])
    with pytest.raises(ContractError, match="Submission target"):
        prepared(path)


@pytest.mark.parametrize(
    "case,match",
    [
        ("dictionary", "Dictionary features"),
        ("ids", "IDs differ"),
        ("duplicate", "duplicate ID"),
        ("target", "missing values"),
        ("nonfinite", "nonfinite"),
        ("schema", "schema differs"),
    ],
)
def test_bad_materials(tmp_path, case, match):
    path = task_fixture(tmp_path)
    if case == "dictionary":
        file = tmp_path / "input/data_dictionary.csv"
        file.write_text(file.read_text().replace("response,target", "other,target"))
    elif case == "ids":
        write_csv(tmp_path / "input/sample_submission.csv", ["key", "response"], [[999, 0]])
    elif case == "duplicate":
        write_csv(tmp_path / "input/test.csv", ["key", "x", "period"], [[101, 0, 1], [101, 1, 2]])
    elif case in ("target", "nonfinite"):
        file = tmp_path / "input/train.csv"
        file.write_text(
            file.read_text().replace("0,,0,0.0", "0,,0," + ("nan" if case == "nonfinite" else ""))
        )
    else:
        file = tmp_path / "input/test.csv"
        file.write_text(file.read_text().replace("key,x,period", "key,x,another"))
    with pytest.raises(ContractError, match=match):
        prepared(path)


def test_rare_class_refuses_invalid_stratification(tmp_path):
    with pytest.raises(ContractError, match="at least two"):
        prepared(task_fixture(tmp_path, labels=[0] * 29 + [1], metric="accuracy"))


def test_primary_metric_and_qualified_paths(tmp_path):
    path = task_fixture(tmp_path, labels=["A", "B"] * 15, metric="accuracy")
    text = path.read_text()
    text = text[: text.index("```task-json")]
    path.write_text(
        text + "\nВ train.csv есть target.\nОсновная метрика — Macro F1.\n"
        "Дополнительно Accuracy и Weighted F1.\n"
    )
    assert prepared(path).spec.metric == Metric.MACRO_F1


def test_ambiguous_metric_requires_clarification(tmp_path):
    path = task_fixture(tmp_path)
    path.write_text(path.read_text().split("```task-json")[0] + "RMSE and MAE")
    with pytest.raises(ContractError, match="primary metric"):
        prepared(path)


def test_multiclass_probability_order(tmp_path):
    path = task_fixture(
        tmp_path,
        labels=["A", "B", "C"] * 10,
        metric="log_loss",
        metadata={"classes": ["C", "B", "A"]},
    )
    sample = tmp_path / "input/sample_submission.csv"
    write_csv(sample, ["key", "C", "B", "A"], [[102, 0.3, 0.3, 0.4], [101, 0.2, 0.3, 0.5]])
    assert prepared(path).spec.classes == ("C", "B", "A")
    write_csv(sample, ["key", "A", "B", "C"], [[102, 0.3, 0.3, 0.4], [101, 0.2, 0.3, 0.5]])
    with pytest.raises(ContractError, match="class order"):
        prepared(path)


def test_application_uses_port_only(tmp_path):
    expected = prepared(task_fixture(tmp_path))

    class FakeData:
        def prepare(self, task_path, seed):
            assert (task_path, seed) == ("virtual.md", 13)
            return replace(expected, warnings=("fake",))

    assert prepare_task("virtual.md", FakeData(), seed=13).warnings == ("fake",)


def test_prose_target_conflict(tmp_path):
    path = task_fixture(tmp_path)
    path.write_text(path.read_text() + "\nTarget: `different`\n")
    with pytest.raises(ContractError, match="contradicts"):
        prepared(path)
