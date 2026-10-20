"""Actual worker processes exercise fitted preprocessing and candidate boundaries."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml_harness.adapters.ml.candidates import build_model


@pytest.mark.parametrize("backend", ["catboost", "lightgbm"])
@pytest.mark.parametrize("labels", [None, [1.5, 3.5] * 10, ["blue", "red"] * 10])
def test_candidates_train_only_preprocessing(backend, labels):
    frame = pd.DataFrame({"number": [1.0, np.nan] * 10, "category": ["a", None] * 10})
    parameters = {"backend": backend}
    parameters["iterations" if backend == "catboost" else "n_estimators"] = 3
    model = build_model("regression" if labels is None else "binary", parameters, 1, 42)
    target = pd.Series(range(20) if labels is None else labels)
    model.fit(frame, target)
    unseen = pd.DataFrame({"number": [99999.0, np.nan], "category": ["unseen", None]})
    prediction = model.predict(unseen)
    assert prediction.shape == (2,)
    assert model.medians["number"] == 1.0
    assert "unseen" not in model.categories["category"]
    if labels is None:
        assert np.isfinite(prediction).all()
    else:
        assert set(prediction) <= set(labels)
        assert model.predict_proba(unseen).shape == (2, 2)
        assert model.classes_.tolist() == sorted(set(labels))


@pytest.mark.parametrize("mode", ["validation", "refit"])
@pytest.mark.parametrize("probabilities", [False, True])
def test_worker_candidate_split_order_and_shape(tmp_path, mode, probabilities):
    train = tmp_path / "train.csv"
    test = tmp_path / "test.csv"
    pd.DataFrame({"key": [8, 5, 9, 2], "x": [1, 2, 3, 4], "y": [7, 9, 7, 9]}).to_csv(
        train, index=False
    )
    pd.DataFrame({"key": [101, 100], "x": [50, 60]}).to_csv(test, index=False)
    candidate = tmp_path / "candidate.py"
    candidate.write_text(
        """import numpy as np
class Model:
    classes_ = np.array([9, 7])
    def fit(self, X, y):
        assert list(X.columns) == ["x"]
        assert X.x.tolist() == EXPECTED_TRAIN
        assert y.tolist() == EXPECTED_LABELS
    def predict(self, X):
        return np.array([[int(x)] for x in X.x])
    def predict_proba(self, X):
        return np.array([[0.25, 0.75] for _ in X.x])
def build_model(kind, parameters, threads, seed):
    assert parameters == {"custom": 1}
    assert threads == 1 and seed == 42
    return Model()
""".replace("EXPECTED_TRAIN", "[3, 1]" if mode == "validation" else "[1, 2, 3, 4]").replace(
            "EXPECTED_LABELS", "[7, 7]" if mode == "validation" else "[7, 9, 7, 9]"
        )
    )
    request = {
        "spec": {
            "train_path": str(train),
            "test_path": str(test),
            "features": ["x"],
            "target": "y",
            "id_column": "key",
            "kind": "binary",
            "prediction_kind": "probabilities" if probabilities else "labels",
        },
        "protocol": {"seed": 42, "train_indices": [2, 0], "validation_indices": [3, 1]},
        "candidate_path": str(candidate),
        "parameters": {"custom": 1},
        "threads": 1,
        "mode": mode,
        "backend": "catboost",
        "output_dir": str(tmp_path),
    }
    path = tmp_path / "request.json"
    path.write_text(json.dumps(request))
    result = subprocess.run(
        [sys.executable, "-m", "ml_harness.adapters.execution.worker", str(path)],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    output = json.loads((tmp_path / "result.json").read_text())
    assert output["ids"] == (["2", "5"] if mode == "validation" else ["101", "100"])
    assert output["classes"] == [9, 7]
    assert output["probabilities"] == [[0.25, 0.75]] * 2
    expected = [4, 2] if mode == "validation" else [50, 60]
    assert output["predictions"] == ([[0.25, 0.75]] * 2 if probabilities else expected)


def test_worker_rejects_nonfinite_without_overwriting_result(tmp_path):
    from ml_harness.adapters.execution.worker import run

    train = tmp_path / "train.csv"
    pd.DataFrame({"id": [1, 2], "x": [3, 4], "y": [5, 6]}).to_csv(train, index=False)
    candidate = tmp_path / "bad.py"
    candidate.write_text("""class Model:
    def fit(self, X, y): pass
    def predict(self, X): return [float("nan")] * len(X)
def build_model(*args): return Model()
""")
    previous = tmp_path / "result.json"
    previous.write_text('{"previous": true}')
    with pytest.raises(ValueError, match="JSON compliant"):
        run(
            {
                "spec": {
                    "train_path": str(train),
                    "features": ["x"],
                    "target": "y",
                    "id_column": "id",
                    "kind": "regression",
                    "prediction_kind": "values",
                },
                "protocol": {"seed": 42, "train_indices": [0], "validation_indices": [1]},
                "candidate_path": str(candidate),
                "parameters": {},
                "threads": 1,
                "mode": "validation",
                "backend": "catboost",
                "output_dir": str(tmp_path),
            }
        )
    assert previous.read_text() == '{"previous": true}'


@pytest.mark.parametrize("labels", [["NA", "001"], ["0", "1"]])
@pytest.mark.parametrize("mode", ["validation", "refit"])
def test_worker_preserves_literal_ids_and_string_labels(tmp_path, labels, mode):
    from ml_harness.adapters.execution.worker import run

    train = tmp_path / "train.csv"
    test = tmp_path / "test.csv"
    pd.DataFrame({"id": ["001", "NA"] * 10, "x": [1, 2] * 10, "y": labels * 10}).to_csv(
        train, index=False
    )
    test.write_text("id,x\nNA,\n001,3\n")
    run(
        {
            "spec": {
                "train_path": str(train),
                "test_path": str(test),
                "features": ["x"],
                "target": "y",
                "id_column": "id",
                "kind": "binary",
                "classes": labels,
                "prediction_kind": "labels",
            },
            "protocol": {
                "seed": 42,
                "train_indices": list(range(18)),
                "validation_indices": [19, 18],
            },
            "candidate_path": None,
            "parameters": {"iterations": 3},
            "threads": 1,
            "mode": mode,
            "backend": "catboost",
            "output_dir": str(tmp_path),
        }
    )
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["ids"] == ["NA", "001"]
    assert set(result["classes"]) == set(labels)
    assert set(result["predictions"]) <= set(labels)
