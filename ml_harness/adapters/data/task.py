"""CSV/task description adapter; refuses ambiguous or contradictory contracts."""

import csv
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from ml_harness.domain.models import (
    ContractError,
    Metric,
    PredictionKind,
    PreparedTask,
    Scalar,
    SplitKind,
    TaskKind,
    TaskSpec,
    ValidationProtocol,
)


def read_csv(path: Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = tuple(reader.fieldnames or ())
        if not columns or len(set(columns)) != len(columns):
            raise ContractError(f"{path.name}: missing or duplicate columns")
        rows = []
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ContractError(f"{path.name}: malformed CSV row")
            rows.append(row)
    if not rows:
        raise ContractError(f"{path.name}: no rows")
    return columns, rows


def _metadata(text: str) -> dict[str, Any]:
    """Optional structured clarification, also usable for synthetic/private cases."""
    match = re.search(r"```(?:task-json|json)\s*\n(.*?)```", text, re.S)
    if match is None:
        return {}
    try:
        result = json.loads(match.group(1))
    except json.JSONDecodeError as exc:
        raise ContractError(f"Invalid task JSON: {exc}") from exc
    if not isinstance(result, dict):
        raise ContractError("Task JSON must be an object")
    return result


def _path(root: Path, text: str, meta: dict[str, Any], key: str, default: str) -> Path:
    value = meta.get(key)
    if value is None and key == "submission_path":
        outputs = re.findall(
            r"(?:Файл|Output|submission_path)\s*[:=]?\s*[`\"]([\w./-]+\.csv)[`\"]",
            text,
            re.I,
        )
        if len(set(outputs)) > 1:
            raise ContractError("Ambiguous submission output path")
        if outputs:
            value = outputs[0]
    if value is None:
        matches = [
            match
            for match in re.findall(r"[\w./-]*" + re.escape(default), text)
            if Path(match).name == default
        ]
        qualified = [match for match in matches if "/" in match]
        candidates = qualified or matches
        value = next(iter(dict.fromkeys(candidates)), default)
        if len(set(candidates)) > 1:
            raise ContractError(f"Ambiguous {key}: {sorted(set(matches))}")
    if not isinstance(value, str) or not value:
        raise ContractError(f"Invalid path for {key}")
    return (root / value).resolve()


def _metric(text: str, meta: dict[str, Any]) -> Metric:
    if "metric" in meta:
        try:
            return Metric(meta["metric"])
        except ValueError as exc:
            raise ContractError("Unsupported metric") from exc
    patterns = {
        Metric.RMSE: r"\bRMSE\b|root mean squared error",
        Metric.MAE: r"\bMAE\b|mean absolute error",
        Metric.MACRO_F1: r"macro[ _-]*f1|f1[ _-]*macro",
        Metric.ACCURACY: r"\baccuracy\b",
        Metric.ROC_AUC: r"ROC[ _-]*AUC",
        Metric.LOG_LOSS: r"log[ _-]*loss|cross[ _-]*entropy",
    }
    # Prefer an explicit primary-metric line, not auxiliary diagnostics.
    primary = [
        line
        for line in text.splitlines()
        if re.search(r"основн\w*\s+метрик|primary\s+metric", line, re.I)
    ]
    scope = "\n".join(primary) if primary else text
    found = [metric for metric, pattern in patterns.items() if re.search(pattern, scope, re.I)]
    if len(found) != 1:
        raise ContractError(f"Specify one supported primary metric; found {found}")
    return found[0]


def _number(value: str) -> int | float:
    try:
        number = float(value)
    except ValueError as exc:
        raise ContractError(f"Non-numeric target: {value!r}") from exc
    if not math.isfinite(number):
        raise ContractError("Target contains nonfinite values")
    return int(number) if number.is_integer() else number


def _roles(rows: list[dict[str, str]], role: str) -> list[str]:
    aliases = {
        "target": {"target", "таргет", "целевая", "целевая переменная"},
        "id": {"id", "identifier", "идентификатор"},
        "group": {"group", "group_id", "группа"},
        "time": {"time", "timestamp", "время"},
    }
    return [row["feature"] for row in rows if row["role"].lower().strip() in aliases[role]]


def _validation(
    spec: TaskSpec, rows: list[dict[str, str]], fingerprint: str, seed: int
) -> ValidationProtocol:
    rng = random.Random(seed)
    n = len(rows)
    if n < 5:
        raise ContractError("At least five training rows required")
    indices = list(range(n))
    if spec.time_column:
        column = spec.time_column
        try:
            values: list[Any] = [float(row[column]) for row in rows]
            if not all(math.isfinite(value) for value in values):
                raise ValueError("nonfinite time")
        except ValueError:
            try:
                values = [
                    datetime.fromisoformat(row[column].replace("Z", "+00:00")) for row in rows
                ]
            except ValueError as exc:
                raise ContractError("Time column must contain numbers or ISO timestamps") from exc
        try:
            indices.sort(key=lambda index: values[index])
        except TypeError as exc:
            raise ContractError("Inconsistent timestamp timezones") from exc
        boundary = max(1, int(n * 0.8))
        # Equal timestamps must never cross the training/validation boundary.
        while boundary > 0 and values[indices[boundary - 1]] == values[indices[boundary]]:
            boundary -= 1
        train, validation = indices[:boundary], indices[boundary:]
        kind = SplitKind.TIME
    elif spec.group_column:
        groups: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(rows):
            group = row[spec.group_column]
            if not group:
                raise ContractError("Group column contains missing values")
            groups[group].append(index)
        keys = sorted(groups)
        if len(keys) < 2:
            raise ContractError("Group validation requires at least two groups")
        rng.shuffle(keys)
        count = min(len(keys) - 1, max(1, round(len(keys) * 0.2)))
        validation = [index for key in keys[:count] for index in groups[key]]
        train = [index for key in keys[count:] for index in groups[key]]
        kind = SplitKind.GROUP
    elif spec.kind != TaskKind.REGRESSION:
        strata: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(rows):
            strata[row[spec.target]].append(index)
        train, validation = [], []
        for label, members in sorted(strata.items()):
            if len(members) < 2:
                raise ContractError(f"Class {label!r} needs at least two rows for stratification")
            rng.shuffle(members)
            count = min(len(members) - 1, max(1, round(len(members) * 0.2)))
            train.extend(members[count:])
            validation.extend(members[:count])
        kind = SplitKind.STRATIFIED
    else:
        rng.shuffle(indices)
        count = max(1, round(n * 0.2))
        validation, train = indices[:count], indices[count:]
        kind = SplitKind.RANDOM
    if not train or not validation:
        raise ContractError("Validation cannot produce nonempty disjoint partitions")
    if spec.kind != TaskKind.REGRESSION:
        labels = {row[spec.target] for row in rows}
        if {rows[index][spec.target] for index in train} != labels:
            raise ContractError("Training partition is missing classes")
        if {rows[index][spec.target] for index in validation} != labels:
            raise ContractError("Validation partition is missing classes")
    payload = (fingerprint, kind.value, seed, sorted(train), sorted(validation), "full")
    protocol_id = hashlib.sha256(json.dumps(payload).encode()).hexdigest()
    return ValidationProtocol(
        protocol_id, fingerprint, kind, seed, tuple(sorted(train)), tuple(sorted(validation))
    )


class CsvTaskData:
    def prepare(self, task_path: str, seed: int) -> PreparedTask:
        try:
            return self._prepare(Path(task_path).resolve(), seed)
        except (OSError, KeyError, TypeError, csv.Error) as exc:
            raise ContractError(f"Cannot prepare task: {exc}") from exc

    def _prepare(self, path: Path, seed: int) -> PreparedTask:
        text = path.read_text(encoding="utf-8")
        meta = _metadata(text)
        root = path.parent
        paths = {
            key: _path(root, text, meta, key, default)
            for key, default in (
                ("train_path", "train.csv"),
                ("test_path", "test.csv"),
                ("dictionary_path", "data_dictionary.csv"),
                ("sample_path", "sample_submission.csv"),
            )
        }
        train_cols, train = read_csv(paths["train_path"])
        test_cols, test = read_csv(paths["test_path"])
        dict_cols, dictionary = read_csv(paths["dictionary_path"])
        sample_cols, sample = read_csv(paths["sample_path"])
        if not {"feature", "role", "dtype"}.issubset(dict_cols):
            raise ContractError("Dictionary requires feature, role, dtype")
        if Counter(row["feature"] for row in dictionary) != Counter(train_cols):
            raise ContractError("Dictionary features must match train columns exactly once")
        target_candidates = _roles(dictionary, "target")
        id_candidates = _roles(dictionary, "id")
        if len(target_candidates) != 1 or len(id_candidates) != 1:
            raise ContractError("Dictionary must identify exactly one target and one ID")
        target, id_column = target_candidates[0], id_candidates[0]
        if set(train_cols) - set(test_cols) != {target} or set(test_cols) - set(train_cols):
            raise ContractError("Train/test schema differs beyond the target")
        for key, expected in (("target", target), ("id_column", id_column)):
            if key in meta and meta[key] != expected:
                raise ContractError(f"Task {key} contradicts dictionary/data")
        for name, rows in (("train", train), ("test", test), ("sample", sample)):
            ids = [row.get(id_column, "") for row in rows]
            if any(not value for value in ids) or len(ids) != len(set(ids)):
                raise ContractError(f"{name}: missing/duplicate ID")
        if set(row[id_column] for row in test) != set(row[id_column] for row in sample):
            raise ContractError("Sample and test IDs differ")
        if any(not row[target] for row in train):
            raise ContractError("Target contains missing values")
        # Explicit declarations in prose receive the same checks as task-json.
        for key, pattern in (
            ("target", r"(?:target|таргет|целевая переменная)\s*[:=]\s*[`\"]?([\w]+)"),
            ("id_column", r"id_column\s*[:=]\s*[`\"]?([\w]+)"),
        ):
            declarations = re.findall(pattern, text, re.I)
            expected = target if key == "target" else id_column
            if declarations and set(declarations) != {expected}:
                raise ContractError(f"Task {key} contradicts dictionary/data")
        metric = _metric(text, meta)
        regression = metric in (Metric.RMSE, Metric.MAE)
        if regression and re.search(r"классификац|\bclassification\b", text, re.I):
            raise ContractError("Description declares classification with a regression metric")
        if not regression and re.search(r"регресси[яо]|\bregression\b", text, re.I):
            raise ContractError("Description declares regression with a classification metric")
        classes: tuple[Scalar, ...] = ()
        if regression:
            for row in train:
                _number(row[target])
            kind = TaskKind.REGRESSION
            prediction_kind = PredictionKind.VALUES
        else:
            dtype = next(row["dtype"].lower() for row in dictionary if row["feature"] == target)
            numeric = any(token in dtype for token in ("int", "float", "number", "numeric"))
            labels = {_number(row[target]) if numeric else row[target] for row in train}
            classes = tuple(sorted(labels, key=lambda value: (isinstance(value, str), value)))
            if len(classes) < 2:
                raise ContractError("Classification needs at least two classes")
            kind = TaskKind.BINARY if len(classes) == 2 else TaskKind.MULTICLASS
            prediction_kind = (
                PredictionKind.PROBABILITIES
                if metric in (Metric.ROC_AUC, Metric.LOG_LOSS)
                else PredictionKind.LABELS
            )
            if "classes" in meta:
                if not isinstance(meta["classes"], list) or set(meta["classes"]) != set(classes):
                    raise ContractError("Explicit class order contradicts training labels")
                classes = tuple(meta["classes"])
        if "kind" in meta and meta["kind"] != kind.value:
            raise ContractError("Explicit task kind contradicts metric/classes")
        if "prediction_kind" in meta and meta["prediction_kind"] != prediction_kind.value:
            raise ContractError("Explicit prediction format contradicts metric")
        if prediction_kind == PredictionKind.LABELS:
            for row in sample:
                if target not in row:
                    raise ContractError("Sample target differs from dictionary")
                label = _number(row[target]) if numeric else row[target]
                if label not in classes:
                    raise ContractError("Sample contains unknown labels")
        if prediction_kind == PredictionKind.PROBABILITIES and len(classes) > 2:
            if sample_cols[1:] != tuple(str(label) for label in classes):
                raise ContractError("Probability columns must follow explicit class order")
        options: dict[str, str | None] = {}
        for role in ("time", "group"):
            candidates = _roles(dictionary, role)
            explicit = meta.get(role + "_column")
            if len(candidates) > 1 or (explicit and candidates and candidates != [explicit]):
                raise ContractError(f"Ambiguous {role} column")
            options[role] = explicit or (candidates[0] if candidates else None)
        spec = TaskSpec(
            task_path=str(path),
            **{key: str(value) for key, value in paths.items()},
            submission_path=str(_path(root, text, meta, "submission_path", "submission.csv")),
            target=target,
            id_column=id_column,
            features=tuple(col for col in train_cols if col not in (target, id_column)),
            kind=kind,
            metric=metric,
            prediction_kind=prediction_kind,
            submission_columns=sample_cols,
            classes=classes,
            time_column=options["time"],
            group_column=options["group"],
        )
        digest = hashlib.sha256()
        digest.update(text.encode())
        for key, file in paths.items():
            digest.update(key.encode())
            with file.open("rb") as stream:
                while block := stream.read(1024 * 1024):
                    digest.update(block)
        fingerprint = digest.hexdigest()
        protocol = _validation(spec, train, fingerprint, seed)
        return PreparedTask(spec, protocol, len(train), len(test))
