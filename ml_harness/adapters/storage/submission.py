"""Validate the whole CSV before atomically replacing the previous answer."""

import csv
import os
import tempfile
from math import isfinite
from pathlib import Path
from statistics import median
from typing import Any

from ml_harness.adapters.data.task import read_csv
from ml_harness.adapters.ml.evaluation import _class_index, evaluate, validate_predictions
from ml_harness.application.ports.execution import SubmissionWriter
from ml_harness.domain.models import ContractError, PredictionKind, PreparedTask, Score, TaskKind


class IndependentEvaluator:
    def evaluate(self, task: PreparedTask, output: dict[str, Any]) -> Score:
        _, train = read_csv(Path(task.spec.train_path))
        return evaluate(task.spec, task.protocol, train, output)


class AtomicSubmissionWriter:
    def write(self, task: PreparedTask, output: dict[str, Any]) -> str:
        spec = task.spec
        _, test = read_csv(Path(spec.test_path))
        ids = [row[spec.id_column] for row in test]
        predictions = validate_predictions(spec, ids, output)
        destination = Path(spec.submission_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                dir=destination.parent,
                prefix=".submission-",
                suffix=".csv",
                encoding="utf-8",
                newline="",
                delete=False,
            ) as stream:
                temporary = stream.name
                writer = csv.writer(stream)
                writer.writerow(spec.submission_columns)
                for identifier, prediction in zip(ids, predictions, strict=True):
                    values: list[Any]
                    if spec.prediction_kind == PredictionKind.PROBABILITIES:
                        if not isinstance(prediction, list):
                            raise ValueError("Expected a probability row")
                        values = [prediction[1]] if spec.kind == TaskKind.BINARY else prediction
                    else:
                        values = [prediction]
                    writer.writerow([identifier, *values])
                stream.flush()
                os.fsync(stream.fileno())
            columns, written = read_csv(Path(temporary))
            if columns != spec.submission_columns or len(written) != len(test):
                raise ValueError("Written submission schema/row count mismatch")
            if [row[spec.id_column] for row in written] != ids:
                raise ValueError("Written submission ID order mismatch")
            os.replace(temporary, destination)
        finally:
            if temporary is not None:
                Path(temporary).unlink(missing_ok=True)
        return str(destination)


def load_submission(task: PreparedTask) -> dict[str, Any]:
    """Load and validate the existing CSV using literal test ID and class order."""
    spec = task.spec
    try:
        columns, rows = read_csv(Path(spec.submission_path))
        _, test = read_csv(Path(spec.test_path))
        if columns != spec.submission_columns:
            raise ContractError("Submission columns disagree with task")
        ids = [row[spec.id_column] for row in test]
        predictions: list[Any] = []
        for row in rows:
            if spec.prediction_kind == PredictionKind.VALUES:
                prediction: Any = float(row[columns[1]])
            elif spec.prediction_kind == PredictionKind.LABELS:
                prediction = spec.classes[_class_index(row[columns[1]], spec.classes)]
            elif spec.kind == TaskKind.BINARY:
                positive = float(row[columns[1]])
                prediction = [1 - positive, positive]
            else:
                prediction = [float(row[column]) for column in columns[1:]]
            predictions.append(prediction)
        result = {
            "ids": [row[spec.id_column] for row in rows],
            "predictions": predictions,
            "classes": list(spec.classes),
        }
        result["predictions"] = validate_predictions(spec, ids, result)
        return result
    except (OSError, KeyError, TypeError, ValueError, OverflowError, csv.Error) as error:
        raise ContractError("Existing submission violates the task contract") from error


def ensure_ready_submission(task: PreparedTask, writer: SubmissionWriter) -> dict[str, Any]:
    """Preserve a valid CSV or atomically prepare a training-only statistical fallback."""
    try:
        load_submission(task)
    except ContractError:
        pass
    else:
        return {"path": task.spec.submission_path, "status": "preserved_existing"}
    spec = task.spec
    try:
        columns, rows = read_csv(Path(spec.train_path))
        _, test = read_csv(Path(spec.test_path))
        if not {spec.id_column, spec.target, *spec.features}.issubset(columns):
            raise ContractError("Fallback training schema disagrees with task")
        train_ids = [row[spec.id_column] for row in rows]
        if (
            len(rows) != task.train_rows
            or any(not identifier for identifier in train_ids)
            or len(set(train_ids)) != len(train_ids)
        ):
            raise ContractError("Fallback training rows or IDs disagree with task")
        indices = task.protocol.train_indices
        if not indices or any(
            not isinstance(i, int) or isinstance(i, bool) or i < 0 or i >= len(rows)
            for i in indices
        ):
            raise ContractError("Fallback training partition is invalid")
        values = [rows[i][spec.target] for i in indices]
        if spec.kind == TaskKind.REGRESSION:
            numeric = [float(value) for value in values]
            if not all(isfinite(value) for value in numeric):
                raise ContractError("Fallback training targets must be finite")
            prediction: Any = median(numeric)
        else:
            counts = [0] * len(spec.classes)
            for value in values:
                counts[_class_index(value, spec.classes)] += 1
            if any(count == 0 for count in counts):
                raise ContractError("Fallback training partition is missing task classes")
            if spec.prediction_kind == PredictionKind.LABELS:
                prediction = spec.classes[max(range(len(counts)), key=lambda i: counts[i])]
            else:
                denominator = len(values) + len(counts)
                prediction = [(count + 1) / denominator for count in counts]
        output = {
            "ids": [row[spec.id_column] for row in test],
            "predictions": [prediction for _ in test],
            "classes": list(spec.classes),
        }
        validate_predictions(spec, output["ids"], output)
        path = writer.write(task, output)
        load_submission(task)
        return {"path": path, "status": "statistical_fallback_written"}
    except (
        OSError,
        KeyError,
        IndexError,
        TypeError,
        ValueError,
        OverflowError,
        csv.Error,
    ) as error:
        if isinstance(error, ContractError):
            raise
        raise ContractError("Cannot construct fallback from the training contract") from error
