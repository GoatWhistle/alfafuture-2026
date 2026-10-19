"""Validate the whole CSV before atomically replacing the previous answer."""

import csv
import os
import tempfile
from pathlib import Path
from typing import Any

from ml_harness.adapters.data.task import read_csv
from ml_harness.adapters.ml.evaluation import evaluate, validate_predictions
from ml_harness.domain.models import PredictionKind, PreparedTask, Score, TaskKind


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
