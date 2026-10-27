"""Public-case benchmarking preserves the task contract and hides holdout labels."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "verification"))
import research_curve


@pytest.mark.parametrize("kind,sub,target,metric,labels", [
    ("classification", "multiclass", "attack_cat", "f1_macro", ["Normal", "Worms"]),
    ("regression", "regression", "site_eui", "rmse", [12., 27.]),
])
def test_public_case_preserves_original_contract(tmp_path, monkeypatch, kind, sub, target, metric, labels):
    public = tmp_path / "data"
    original = public / kind
    participants = original / sub / "participants"
    participants.mkdir(parents=True)
    dictionary = f"feature,role,description\n{target},target,original units\nx,feature,original feature\n"
    (participants / "data_dictionary.csv").write_text(dictionary)
    (original / "task.md").write_text(f"Predict {target}; read {sub}/participants/train.csv; metric {metric}.")
    train = pd.DataFrame({"id": [1, 2], "x": [3., 4.], "target": labels})
    test = pd.DataFrame({"id": [3, 4], "x": [5., 6.]})
    truth = pd.DataFrame({"id": [3, 4], "target": labels[::-1]})
    monkeypatch.setattr(research_curve, "dataset", lambda *a: (train, test, truth, metric, "fixture"))
    args = SimpleNamespace(public_root=public, output=tmp_path / "out", max_train_rows=100000, max_test_rows=100000)
    root = args.output / "task"
    got, got_metric, _, got_target, n_train, n_test = research_curve.prepare_case(args, "public_" + kind, 17, root)
    assert got_target == target and got_metric == metric and (n_train, n_test) == (2, 2)
    assert target in pd.read_csv(root / "train.csv")
    assert target not in pd.read_csv(root / "test.csv")
    assert list(pd.read_csv(root / "sample_submission.csv")) == ["id", target]
    assert (root / "data_dictionary.csv").read_text() == dictionary
    assert f"{sub}/participants/" not in (root / "task.md").read_text()
    assert target in (root / "task.md").read_text()
    assert list(got[target]) == labels[::-1]
    assert (args.output / "holdout_labels" / f"public_{kind}-17.csv").exists()
    assert sorted(p.name for p in root.iterdir()) == ["data_dictionary.csv", "sample_submission.csv", "task.md", "test.csv", "train.csv"]
