"""Benchmark creation preserves source data and keeps external targets outside worker inputs."""

import csv
import importlib
import os
import sys
from pathlib import Path

import pytest
from test_prepare_task import task_fixture

from ml_harness.adapters.data.task import CsvTaskData


def test_external_benchmark_copy_and_independent_score(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]))
    module = importlib.import_module("verification.run_hybrid_comparison")
    benchmark_copy, external_score, write_csv = (
        module.benchmark_copy,
        module.external_score,
        module.write_csv,
    )
    copy_task = importlib.import_module("verification.verify_archive").copy_task
    source_dir = tmp_path / "original"
    source_dir.mkdir()
    source = task_fixture(source_dir)
    original = (source_dir / "input/train.csv").read_bytes()
    benchmark = benchmark_copy(source, tmp_path / "benchmark")
    prepared = CsvTaskData().prepare(str(benchmark), 42)
    with (benchmark.parent / "external_targets.csv").open() as stream:
        truth = list(csv.DictReader(stream))
    assert len(truth) > 0
    assert (source_dir / "input/train.csv").read_bytes() == original
    submission = tmp_path / "predictions.csv"
    write_csv(submission, [prepared.spec.id_column, prepared.spec.target], truth)
    assert external_score(benchmark, submission)["value"] == 0
    isolated = tmp_path / "isolated"
    isolated.mkdir()
    copy_task(Path(__file__).resolve().parents[2], benchmark, isolated / "inputs", isolated, 42)
    assert not (isolated / "inputs" / "external_targets.csv").exists()


def test_bundled_classification_benchmark_copies_preserve_label_contract(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]))
    module = importlib.import_module("verification.run_hybrid_comparison")
    generate = importlib.import_module("verification.benchmark_tasks").generate_tasks
    for name, path in generate(tmp_path / "originals").items():
        copied = module.benchmark_copy(path, tmp_path / "copies" / name)
        prepared = CsvTaskData().prepare(str(copied), 42)
        assert prepared.spec.kind in ("binary", "multiclass", "regression")


@pytest.mark.parametrize("configured_model", [None, "platform-model"])
def test_comparison_cli_loads_root_env_without_overriding_platform(
    tmp_path, monkeypatch, configured_model
):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]))
    module = importlib.import_module("verification.run_hybrid_comparison")
    (tmp_path / ".env").write_text("OPENAI_MODEL=fixture-model\nOPENAI_API_KEY=fixture-key\n")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    if configured_model:
        monkeypatch.setenv("OPENAI_MODEL", configured_model)
    captured = []

    def capture(args):
        captured.append((os.environ["OPENAI_MODEL"], os.environ["OPENAI_API_KEY"], args.modes))

    monkeypatch.setattr(module, "run", capture)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "comparison",
            str(tmp_path / "task.md"),
            "--output",
            str(tmp_path / "out"),
            "--modes",
            "qwen",
        ],
    )
    module.main()
    assert captured == [(configured_model or "fixture-model", "fixture-key", ["qwen"])]
