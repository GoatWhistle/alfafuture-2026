"""Fit/predict subprocess. Scoring belongs exclusively to the parent evaluator."""

import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from ml_harness.adapters.ml.candidates import build_model


def run(request: dict[str, Any]) -> None:
    """Materialize the specified split and write finite JSON predictions."""
    started = time.monotonic()
    spec = request["spec"]
    protocol = request["protocol"]
    if request["mode"] not in ("validation", "control", "fallback", "refit"):
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
    if request["mode"] in ("validation", "control"):
        fit_indices = protocol["train_indices"]
        predict_indices = protocol["validation_indices"]
        all_indices = fit_indices + predict_indices
        if (
            not fit_indices
            or not predict_indices
            or len(set(all_indices)) != len(all_indices)
            or any(
                not isinstance(i, int) or isinstance(i, bool) or i < 0 or i >= len(train)
                for i in all_indices
            )
            or (
                request["mode"] == "control"
                and (protocol.get("control_indices") or protocol.get("level") != "full")
            )
        ):
            raise ValueError("Invalid worker validation partitions")
        fit_frame = train.iloc[protocol["train_indices"]]
        predict_frame = train.iloc[protocol["validation_indices"]]
    else:
        if request["mode"] == "fallback":
            development = protocol["train_indices"] + protocol["validation_indices"]
            controls = set(protocol.get("control_indices", []))
            if (
                not development
                or len(set(development)) != len(development)
                or controls.intersection(development)
                or any(
                    not isinstance(i, int) or isinstance(i, bool) or i < 0 or i >= len(train)
                    for i in development
                )
            ):
                raise ValueError("Invalid fallback development partition")
            fit_frame = train.iloc[development]
        else:
            fit_frame = train
        predict_frame = pd.read_csv(
            spec["test_path"],
            dtype={spec["id_column"]: str},
            keep_default_na=False,
            na_values=missing,
        )
    configure = getattr(model, "configure_validation", None)
    if configure is not None:
        configure(
            {
                "kind": protocol["kind"],
                "time_column": spec.get("time_column"),
                "group_column": spec.get("group_column"),
                "metric": spec["metric"],
            }
        )
    fit_started = time.monotonic()
    try:
        model.fit(fit_frame[features], fit_frame[spec["target"]])
    finally:
        tuning = getattr(model, "tuning_report_", None)
        if tuning:
            output = Path(request["output_dir"])
            output.mkdir(parents=True, exist_ok=True)
            (output / "tuning.json").write_text(
                json.dumps(tuning, allow_nan=False), encoding="utf-8"
            )
    fit_seconds = time.monotonic() - fit_started
    predict_started = time.monotonic()
    if spec["prediction_kind"] == "probabilities":
        predictions = np.asarray(model.predict_proba(predict_frame[features]))
    else:
        predictions = np.asarray(model.predict(predict_frame[features])).reshape(-1)
    result = {
        "ids": predict_frame[spec["id_column"]].tolist(),
        "predictions": predictions.tolist(),
        "classes": np.asarray(getattr(model, "classes_", [])).tolist(),
        "worker_seconds": time.monotonic() - started,
        "fit_seconds": fit_seconds,
        "predict_seconds": time.monotonic() - predict_started,
    }
    if hasattr(model, "tuning_report_"):
        result["tuning_report"] = model.tuning_report_
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
