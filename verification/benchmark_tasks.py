"""Generate reproducible offline fixtures from sklearn's bundled tabular datasets."""

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

from sklearn.datasets import (  # type: ignore[import-untyped]
    load_breast_cancer,
    load_diabetes,
    load_wine,
)

SEED = 2026


def _write_csv(path: Path, columns: list[str], rows: list[list[Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(columns)
        writer.writerows(rows)


def generate_tasks(output: Path, seed: int = SEED) -> dict[str, Path]:
    """Use only bundled data; holdout targets remain local and separate from test features."""
    definitions = (
        ("diabetes", load_diabetes, "glycemic_progression", "regression", "rmse", "values"),
        (
            "breast_cancer",
            load_breast_cancer,
            "diagnosis_code",
            "binary",
            "roc_auc",
            "probabilities",
        ),
        ("wine", load_wine, "cultivar_code", "multiclass", "macro_f1", "labels"),
    )
    tasks = {}
    for name, loader, target, kind, metric, prediction in definitions:
        dataset = loader()
        features = [str(value) for value in dataset.feature_names]
        values = dataset.data.tolist()
        targets = dataset.target.tolist()
        # Nonzero numeric labels expose pipelines that accidentally treat them as regression.
        if name == "wine":
            targets = [int(value) + 1 for value in targets]
        root = output / name
        root.mkdir(parents=True, exist_ok=True)
        rng = random.Random(seed)
        if kind == "regression":
            members = list(range(len(targets)))
            rng.shuffle(members)
            heldout = members[: round(len(members) * 0.2)]
        else:
            heldout = []
            for label in sorted(set(targets)):
                members = [i for i, value in enumerate(targets) if value == label]
                rng.shuffle(members)
                heldout.extend(members[: max(1, round(len(members) * 0.2))])
            rng.shuffle(heldout)
        heldout_set = set(heldout)
        training = [i for i in range(len(values)) if i not in heldout_set]
        id_column = "case_key"

        def identifier(index: int, name: str = name) -> str:
            return f"{name}_{index:05d}"

        _write_csv(
            root / "train.csv",
            [id_column, *features, target],
            [[identifier(i), *values[i], targets[i]] for i in training],
        )
        _write_csv(
            root / "test.csv",
            [id_column, *features],
            [[identifier(i), *values[i]] for i in heldout],
        )
        _write_csv(
            root / "holdout_targets.csv",
            [id_column, target],
            [[identifier(i), targets[i]] for i in heldout],
        )
        _write_csv(
            root / "data_dictionary.csv",
            ["feature", "role", "dtype"],
            [
                [id_column, "id", "object"],
                *[[feature, "feature", "float64"] for feature in features],
                [target, "target", "float64" if kind == "regression" else "int64"],
            ],
        )
        prediction_column = "diagnosis_probability" if prediction == "probabilities" else target
        sample_value = (
            0.5 if prediction == "probabilities" else 0.0 if kind == "regression" else targets[0]
        )
        _write_csv(
            root / "sample_submission.csv",
            [id_column, prediction_column],
            [[identifier(i), sample_value] for i in heldout],
        )
        metadata: dict[str, Any] = {
            "kind": kind,
            "metric": metric,
            "prediction_kind": prediction,
            "target": target,
            "id_column": id_column,
            "submission_path": "submission.csv",
        }
        if kind != "regression":
            metadata["classes"] = sorted(set(targets))
        text = (
            f"# Local offline benchmark: {name}\n\n"
            f"Generated from sklearn bundled data with holdout seed {seed}. "
            "This fixture is not an organizer task. No downloads are used.\n\n"
            "Train on train.csv and predict test.csv using data_dictionary.csv and "
            "sample_submission.csv. holdout_targets.csv is reserved for offline evaluation; "
            "never use it for training or model selection.\n\n"
            "```task-json\n" + json.dumps(metadata, ensure_ascii=False, indent=2) + "\n```\n"
        )
        path = root / "task.md"
        path.write_text(text, encoding="utf-8")
        tasks[name] = path
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/stage7/tasks"))
    args = parser.parse_args()
    print(
        json.dumps(
            {name: str(path.resolve()) for name, path in generate_tasks(args.output).items()},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
