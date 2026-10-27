"""Issue #5: вложенные сегменты доходят до сравнения, эквивалентные — один раз."""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import eda_detectors
import fe_search


def test_agreement_is_symmetric_and_invariant():
    fine = np.repeat(np.arange(4), 100)
    coarse = fine // 2
    renamed = np.array([8, 2, 9, 5])[fine]
    order = np.random.RandomState(7).permutation(len(fine))
    assert eda_detectors._agreement(fine, coarse) == 0.5
    assert eda_detectors._agreement(coarse, fine) == 0.5
    assert eda_detectors._agreement(fine[order], coarse[order]) == 0.5
    assert eda_detectors._agreement(fine, renamed) == 1.0


def context(reverse=False):
    rng = np.random.RandomState(42)
    fine = np.tile(np.arange(4), 400)
    x, z = rng.randn(2, len(fine))
    y = np.array([-3, -1, 1, 3])[fine] * x + 0.8 * rng.randn(len(fine))
    df = pd.DataFrame({"coarse": fine // 2, "fine": fine,
                       "renamed": np.array([8, 2, 9, 5])[fine], "x": x, "z": z})
    return {"df": df, "features": list(reversed(df.columns)) if reverse else list(df.columns),
            "period": None, "resid": y, "y": y, "facts": {}, "importances": {"x": 1, "z": 1},
            "classes": None, "ptype": "regression", "metric": "rmse", "cv": {"kind": "kfold"},
            "deadline": time.time() + 120}


@pytest.mark.parametrize("reverse", [False, True])
def test_nested_candidates_survive_detector_and_equivalent_columns_merge(reverse):
    ctx = context(reverse)
    specs = eda_detectors.detect_segments(ctx)
    segments = [s for s in fe_search.interleave(specs) if s["op"] == "segment_model"]
    assert len(segments) == 2
    columns = {s["cols"][0] for s in segments}
    assert "coarse" in columns
    assert len(columns & {"fine", "renamed"}) == 1
    facts = ctx["facts"]["segments"]
    fine_fact = next(f for f in facts if f["col"] in {"fine", "renamed"})
    assert set([fine_fact["col"], *fine_fact["same_as"]]) == {"fine", "renamed"}


def test_small_distinct_partition_is_not_dropped(monkeypatch):
    ctx = context()
    a = np.repeat([0, 1], 800)
    b = a.copy()
    b[0] = 1
    monkeypatch.setattr(eda_detectors, "_segment_candidates", lambda ctx: [
        ("coarse", {"rule": "level"}, a), ("fine", {"rule": "level"}, b)])
    assert 0.98 < eda_detectors._agreement(a, b) < 1.0
    specs = eda_detectors.detect_segments(ctx)
    assert len([s for s in specs if s["op"] == "segment_model"]) == 2


def test_search_evaluates_both_partitions_and_selects_by_quality(monkeypatch):
    ctx = context()
    specs = [s for s in eda_detectors.detect_segments(ctx) if s["op"] == "segment_model"]
    # The partition columns select separate models; they are not direct predictors.
    ctx["features"] = ["x", "z"]
    rows = np.arange(len(ctx["df"]))
    ctx["folds"] = [(rows[rows % 8 < 4], rows[rows % 8 >= 4])]
    monkeypatch.setattr(fe_search, "SEG_MIN_ROWS", 100)
    monkeypatch.setattr(fe_search.models, "N_CPUS", 2)
    search = fe_search.Search(ctx)
    search.base = search.ev.run([])
    search.rel_min = 0.002
    tested = search.stage_a(specs, ctx["deadline"])
    assert len(tested) == 2
    assert search.ev.n_evals == 3  # base + two distinct partitions; renamed is deduplicated
    winner = search.stage_b(ctx["deadline"])
    assert len(winner) == 1
    assert winner[0]["cols"][0] in {"fine", "renamed"}
    assert ctx["search_final"] < ctx["search_base"]
