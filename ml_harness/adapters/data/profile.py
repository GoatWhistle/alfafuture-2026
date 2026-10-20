"""Read-only dataset facts fitted exclusively on the development training partition."""

from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ml_harness.adapters.data.task import read_csv
from ml_harness.domain.dataset_profile import DatasetProfile
from ml_harness.domain.models import ContractError, PreparedTask


def dataset_profile(task: PreparedTask) -> dict[str, Any]:
    import pandas as pd  # type: ignore[import-untyped]

    train = pd.read_csv(task.spec.train_path).iloc[list(task.protocol.train_indices)]
    test = pd.read_csv(task.spec.test_path)
    _, dictionary = read_csv(Path(task.spec.dictionary_path))
    descriptions = {row["feature"]: row for row in dictionary}
    columns: dict[str, Any] = {}
    hashes: dict[int, list[str]] = {}
    duplicates = []
    for name in task.spec.features:
        series = train[name]
        numeric = pd.to_numeric(series, errors="coerce")
        is_numeric = numeric.notna().sum() == series.notna().sum() and series.notna().any()
        detail: dict[str, Any] = {
            "name": name,
            "dictionary": descriptions.get(name, {}),
            "type": "numeric" if is_numeric else "categorical",
            "missing": int(series.isna().sum()),
            "rows": len(series),
            "unique": int(series.nunique()),
            "constant": bool(series.nunique() <= 1),
            "test_missing": int(test[name].isna().sum()) if name in test else None,
            "test_rows": len(test),
        }
        if is_numeric:
            detail["quantiles"] = {
                str(q): float(v)
                for q, v in numeric.quantile([0, 0.25, 0.5, 0.75, 1]).items()
                if pd.notna(v) and abs(v) != float("inf")
            }
        else:
            detail["frequent"] = [
                {"value": str(k)[:160], "count": int(v)}
                for k, v in series.value_counts().head(5).items()
            ]
        columns[name] = detail
        fingerprint = int(pd.util.hash_pandas_object(series, index=False).sum())
        for previous in hashes.get(fingerprint, []):
            if series.equals(train[previous]):
                duplicates.append([previous, name])
                break
        hashes.setdefault(fingerprint, []).append(name)
    target = train[task.spec.target]
    target_summary: dict[str, Any] = {"rows": len(target), "missing": int(target.isna().sum())}
    if task.spec.kind.value == "regression":
        target_summary["quantiles"] = {
            str(q): float(v)
            for q, v in target.quantile([0, 0.25, 0.5, 0.75, 1]).items()
            if pd.notna(v)
        }
    else:
        target_summary["counts"] = dict(Counter(str(value) for value in target))
    return asdict(
        DatasetProfile(
            fit_partition="train",
            columns=columns,
            target=target_summary,
            duplicate_features=duplicates,
            schema={
                "train_only": sorted(set(train) - set(test)),
                "test_only": sorted(set(test) - set(train)),
            },
        )
    )


def column_details(profile: dict[str, Any], names: list[str]) -> dict[str, Any]:
    if not 1 <= len(names) <= 8 or len(set(names)) != len(names):
        raise ContractError("Request between one and eight distinct feature names")
    if any(name not in profile["columns"] for name in names):
        raise ContractError("Unknown feature name")
    return {name: profile["columns"][name] for name in names}
