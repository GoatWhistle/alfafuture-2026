"""Fit/predict subprocess. Scoring belongs exclusively to the parent evaluator."""

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from ml_harness.adapters.ml.candidates import build_model


def run(request: dict[str, Any]) -> None:
    """Materialize the specified split and write finite JSON predictions."""
    spec = request["spec"]
    protocol = request["protocol"]
    if request["mode"] not in ("validation", "refit"):
        raise ValueError("Unknown worker mode")
    if request["threads"] < 1:
        raise ValueError("threads must be positive")
    factory = build_model
    if request.get("candidate_path") is not None:
        path = Path(request["candidate_path"])
        module_spec = importlib.util.spec_from_file_location("harness_candidate", path)
        if module_spec is None or module_spec.loader is None:
            raise ValueError("Cannot load candidate module")
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[module_spec.name] = module
        module_spec.loader.exec_module(module)
        factory = module.build_model
    parameters = dict(request.get("parameters", {}))
    if request.get("candidate_path") is None:
        parameters["backend"] = request["backend"]
    model = factory(spec["kind"], parameters, request["threads"], protocol["seed"])
    features = list(spec["features"])
    dtypes = {spec["id_column"]: str}
    if spec.get("classes") and all(isinstance(label, str) for label in spec["classes"]):
        dtypes[spec["target"]] = str
    # Missing markers apply to features only: IDs and class labels are literal CSV text.
    missing = {column: [""] for column in features}
    train = pd.read_csv(spec["train_path"], dtype=dtypes, keep_default_na=False, na_values=missing)
    if request["mode"] == "validation":
        fit_frame = train.iloc[protocol["train_indices"]]
        predict_frame = train.iloc[protocol["validation_indices"]]
    else:
        fit_frame = train
        predict_frame = pd.read_csv(
            spec["test_path"],
            dtype={spec["id_column"]: str},
            keep_default_na=False,
            na_values=missing,
        )
    model.fit(fit_frame[features], fit_frame[spec["target"]])
    if spec["prediction_kind"] == "probabilities":
        predictions = np.asarray(model.predict_proba(predict_frame[features]))
    else:
        predictions = np.asarray(model.predict(predict_frame[features])).reshape(-1)
    result = {
        "ids": predict_frame[spec["id_column"]].tolist(),
        "predictions": predictions.tolist(),
        "classes": np.asarray(getattr(model, "classes_", [])).tolist(),
    }
    serialized = json.dumps(result, ensure_ascii=False, allow_nan=False)
    output = Path(request["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / "result.json.tmp"
    temporary.write_text(serialized, encoding="utf-8")
    temporary.replace(output / "result.json")


def main() -> None:
    if len(sys.argv) != 2 or not Path(sys.argv[1]).is_absolute():
        raise SystemExit(
            "Usage: python -m ml_harness.adapters.execution.worker /absolute/request.json"
        )
    run(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
