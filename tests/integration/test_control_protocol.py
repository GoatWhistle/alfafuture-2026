"""Frozen control holdouts preserve split families and stay outside worker training."""

import csv
import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from ml_harness.adapters.data.task import CsvTaskData
from ml_harness.adapters.execution.worker import run
from ml_harness.domain.models import ContractError, PreparedTask, SplitKind
from ml_harness.domain.protocols import control_task

pytestmark = pytest.mark.integration


def make_task(
    root: Path, *, count: int = 1200, kind: str = "random", rare: bool = False, groups: int = 12
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    classification = kind in ("stratified", "group_class", "time_class") or rare
    role = "group" if kind.startswith("group") else "time" if kind.startswith("time") else "feature"
    metric = "accuracy" if classification else "rmse"
    labels = [int(i % 2 == 0) if classification else i / 10 for i in range(count)]
    if kind == "group_class":
        labels = [(i // groups) % 2 for i in range(count)]
    if rare:
        labels = [int(i < 10) for i in range(count)]
    data = {
        "train.csv": (
            ["id", "x", "period", "y"],
            [
                [str(i), i, i % groups if role == "group" else i // 5, labels[i]]
                for i in range(count)
            ],
        ),
        "test.csv": (["id", "x", "period"], [["test", 3, 3000]]),
        "sample_submission.csv": (["id", "y"], [["test", labels[0]]]),
        "data_dictionary.csv": (
            ["feature", "role", "dtype"],
            [
                ["id", "id", "object"],
                ["x", "feature", "float64"],
                ["period", role, "int64"],
                ["y", "target", "float64"],
            ],
        ),
    }
    for name, (header, rows) in data.items():
        with (root / name).open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            writer.writerows(rows)
    path = root / "task.md"
    path.write_text("```task-json\n" + json.dumps({"metric": metric}) + "\n```")
    return path


@pytest.mark.parametrize(
    "kind,expected",
    [
        ("random", SplitKind.RANDOM),
        ("stratified", SplitKind.STRATIFIED),
        ("group", SplitKind.GROUP),
        ("group_class", SplitKind.GROUP),
        ("time", SplitKind.TIME),
        ("time_class", SplitKind.TIME),
    ],
)
def test_control_coverage_disjointness_and_reproducibility(
    tmp_path: Path,
    kind: str,
    expected: SplitKind,
) -> None:
    path = make_task(tmp_path, kind=kind)
    task = CsvTaskData().prepare(str(path), 42)
    protocol = task.protocol
    assert task == CsvTaskData().prepare(str(path), 42)
    assert protocol.kind == expected
    partitions = [
        set(protocol.train_indices),
        set(protocol.validation_indices),
        set(protocol.control_indices),
    ]
    assert all(partitions)
    assert not (
        partitions[0] & partitions[1]
        or partitions[0] & partitions[2]
        or partitions[1] & partitions[2]
    )
    assert set.union(*partitions) == set(range(1200))
    assert 0.05 <= len(protocol.control_indices) / 1200 <= 0.2
    assert not task.warnings
    if "class" in kind or kind == "stratified":
        for partition in partitions:
            assert {(i // 12 if kind == "group_class" else i) % 2 for i in partition} == {0, 1}
    if kind.startswith("group"):
        group_sets = [{i % 12 for i in partition} for partition in partitions]
        assert not (
            group_sets[0] & group_sets[1]
            or group_sets[0] & group_sets[2]
            or group_sets[1] & group_sets[2]
        )
    if kind.startswith("time"):
        assert max(i // 5 for i in partitions[0]) < min(i // 5 for i in partitions[1])
        assert max(i // 5 for i in partitions[1]) < min(i // 5 for i in partitions[2])
    historical = CsvTaskData(reserve_control=False).prepare(str(path), 42)
    assert not historical.protocol.control_indices
    assert historical.protocol.fingerprint != protocol.fingerprint


@pytest.mark.parametrize(
    "count,kind,rare,groups,reason",
    [
        (100, "random", False, 12, "1000"),
        (1200, "stratified", True, 12, "20"),
        (1200, "group", False, 4, "five groups"),
    ],
)
def test_control_not_justified_preserves_historical_split(
    tmp_path: Path,
    count: int,
    kind: str,
    rare: bool,
    groups: int,
    reason: str,
) -> None:
    path = make_task(tmp_path, count=count, kind=kind, rare=rare, groups=groups)
    current = CsvTaskData().prepare(str(path), 42)
    historical = CsvTaskData(False).prepare(str(path), 42)
    assert current.protocol == historical.protocol
    assert not current.protocol.control_indices
    assert reason in current.warnings[0]


def test_time_control_missing_classes_is_not_replaced_with_random(tmp_path: Path) -> None:
    path = make_task(tmp_path, kind="time_class")
    train = tmp_path / "train.csv"
    rows = list(csv.reader(train.open()))
    for i, row in enumerate(rows[1:]):
        row[-1] = "0" if i >= 1080 else str(i % 2)
    with train.open("w", newline="") as stream:
        csv.writer(stream).writerows(rows)
    task = CsvTaskData().prepare(str(path), 42)
    assert task.protocol.kind == SplitKind.TIME
    assert not task.protocol.control_indices
    assert "miss classes" in task.warnings[0]


def test_control_task_freezes_development_and_holdout(tmp_path: Path) -> None:
    task = CsvTaskData().prepare(str(make_task(tmp_path)), 42)
    controlled = control_task(task)
    assert controlled == control_task(task)
    assert set(controlled.protocol.train_indices) == set(
        task.protocol.train_indices + task.protocol.validation_indices
    )
    assert controlled.protocol.validation_indices == task.protocol.control_indices
    assert controlled.protocol.control_indices == ()
    assert controlled.protocol.fingerprint != task.protocol.fingerprint
    assert controlled.protocol.level == "full"
    with pytest.raises(ContractError):
        control_task(controlled)
    with pytest.raises(ContractError):
        control_task(replace(task, protocol=replace(task.protocol, level="smoke")))


def request(task: PreparedTask, tmp_path: Path, mode: str) -> dict:
    candidate = tmp_path / "candidate.py"
    candidate.write_text("""import json
from pathlib import Path
class Model:
    def configure_validation(self, settings):
        self.settings = settings
    def fit(self, X, y):
        Path(__file__).with_name('seen.json').write_text(json.dumps({
            'rows': X['x'].tolist(), 'targets': y.tolist(), 'columns': X.columns.tolist(),
            'settings': self.settings}))
    def predict(self, X):
        return [0] * len(X)
def build_model(kind, parameters, threads, seed):
    return Model()
""")
    return json.loads(
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


def test_worker_control_fit_excludes_holdout_targets_and_reports_timing(tmp_path: Path) -> None:
    task = CsvTaskData().prepare(str(make_task(tmp_path)), 42)
    converted = control_task(task)
    run(request(converted, tmp_path, "control"))
    seen = json.loads((tmp_path / "seen.json").read_text())
    assert set(seen["rows"]) == set(converted.protocol.train_indices)
    assert not set(seen["rows"]) & set(task.protocol.control_indices)
    assert "y" not in seen["columns"]
    result = json.loads((tmp_path / "control/result.json").read_text())
    assert result["ids"] == [str(i) for i in task.protocol.control_indices]
    assert result["fit_seconds"] >= 0
    assert result["predict_seconds"] >= 0
    assert seen["settings"]["kind"] == "random"
    run(request(task, tmp_path, "refit"))
    assert len(json.loads((tmp_path / "seen.json").read_text())["rows"]) == 1200


@pytest.mark.parametrize("mutation", ["empty", "overlap", "unconverted", "negative"])
def test_worker_rejects_invalid_control_partitions(tmp_path: Path, mutation: str) -> None:
    task = CsvTaskData().prepare(str(make_task(tmp_path)), 42)
    payload = request(control_task(task), tmp_path, "control")
    protocol = payload["protocol"]
    if mutation == "empty":
        protocol["validation_indices"] = []
    elif mutation == "overlap":
        protocol["validation_indices"] = protocol["train_indices"][:1]
    elif mutation == "negative":
        protocol["validation_indices"] = [-1]
    else:
        protocol["control_indices"] = [1]
    with pytest.raises(ValueError):
        run(payload)
    assert not (tmp_path / "seen.json").exists()
