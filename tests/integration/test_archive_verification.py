"""Archive checks isolate imports, task inputs, execution cwd and saved predictions."""

import csv
import importlib.util
import json
import stat
from pathlib import Path
from zipfile import ZipFile, ZipInfo

import pytest

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]


def verifier():
    spec = importlib.util.spec_from_file_location(
        "archive_verifier", ROOT / "verification/verify_archive.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def archive(root: Path, *, synthetic_entrypoint: bool = False) -> Path:
    path = root / "submission.zip"
    with ZipFile(path, "w") as output:
        for source in sorted((ROOT / "ml_harness").rglob("*.py")):
            output.write(source, source.relative_to(ROOT).as_posix())
        if synthetic_entrypoint:
            solution = """import json, os, sys
from pathlib import Path
import ml_harness
from ml_harness.adapters.data.task import CsvTaskData
from ml_harness.adapters.storage.submission import AtomicSubmissionWriter, ensure_ready_submission
prepared = CsvTaskData().prepare(sys.argv[1], int(sys.argv[sys.argv.index('--seed') + 1]))
result = ensure_ready_submission(prepared, AtomicSubmissionWriter())
run_dir = Path(sys.argv[sys.argv.index('--run-dir') + 1])
run_dir.mkdir(parents=True)
summary = {'status': 'synthetic_fallback_only', 'ready': result,
           'cwd': os.getcwd(), 'import_origin': ml_harness.__file__,
           'injected_env_present': any(k.startswith(('OPENAI_', 'QWEN_')) or k == 'PYTHONPATH'
                                       for k in os.environ)}
(run_dir / 'summary.json').write_text(json.dumps(summary))
print(json.dumps(summary))
"""
        else:
            solution = (ROOT / "solution.py").read_text()
        output.writestr("solution.py", solution)
    return path


def source_task(root: Path, *, metadata: dict | None = None) -> Path:
    (root / "input").mkdir(parents=True)
    files = {
        "train.csv": (["id", "x", "y"], [[str(i), i, i % 5] for i in range(30)]),
        "test.csv": (["id", "x"], [["002", 2], ["001", 1]]),
        "sample_submission.csv": (["id", "y"], [["001", 0], ["002", 0]]),
        "data_dictionary.csv": (
            ["feature", "role", "dtype"],
            [["id", "id", "object"], ["x", "feature", "float64"], ["y", "target", "float64"]],
        ),
    }
    for name, (columns, rows) in files.items():
        with (root / "input" / name).open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(columns)
            writer.writerows(rows)
    meta = {
        "metric": "rmse",
        "train_path": "input/train.csv",
        "test_path": "input/test.csv",
        "dictionary_path": "input/data_dictionary.csv",
        "sample_path": "input/sample_submission.csv",
        "submission_path": "output/submission.csv",
        **(metadata or {}),
    }
    path = root / "task.md"
    path.write_bytes(
        ("Original task bytes\n```task-json\n" + json.dumps(meta) + "\n```\n").encode()
    )
    (root / "holdout_targets.csv").write_text("SECRET_HOLDOUT")
    (root / "output").mkdir()
    (root / "output/submission.csv").write_text("PRIOR_SUBMISSION")
    (root / ".env").write_text("OPENAI_API_KEY=SECRET")
    return path


@pytest.mark.parametrize(
    "name",
    [
        "../escape.py",
        "/absolute.py",
        "ml_harness/../escape.py",
        "ml_harness\\escape.py",
        "ml_harness//escape.py",
        "ml_harness/.hidden.py",
        ".env",
        "data.csv",
        "ml_harness/data.csv",
        "ml_harness/__pycache__/leak.py",
    ],
)
def test_archive_rejects_non_allowlisted_or_traversing_entries(tmp_path: Path, name: str) -> None:
    path = tmp_path / "bad.zip"
    with ZipFile(path, "w") as output:
        output.writestr("solution.py", "")
        output.writestr("ml_harness/__init__.py", "")
        output.writestr(name, "secret")
    with pytest.raises(ValueError, match="allowlist"):
        verifier().extract_archive(path, tmp_path / "unpacked")
    assert not (tmp_path / "unpacked").exists()


def test_archive_rejects_symlink(tmp_path: Path) -> None:
    path = tmp_path / "bad.zip"
    with ZipFile(path, "w") as output:
        output.writestr("solution.py", "")
        output.writestr("ml_harness/__init__.py", "")
        entry = ZipInfo("ml_harness/escape.py")
        entry.create_system = 3
        entry.external_attr = (stat.S_IFLNK | 0o777) << 16
        output.writestr(entry, "/outside/secret")
    with pytest.raises(ValueError, match="allowlist"):
        verifier().extract_archive(path, tmp_path / "unpacked")


def test_bootstrap_and_isolated_copy_only_required_inputs(tmp_path: Path) -> None:
    module = verifier()
    source = source_task(tmp_path / "source")
    saved = source.read_bytes()
    unpacked = tmp_path / "unpacked"
    module.extract_archive(archive(tmp_path), unpacked)
    cwd = tmp_path / "external"
    cwd.mkdir()
    result = module.copy_task(unpacked, source, tmp_path / "copy", cwd, 42)
    assert source.read_bytes() == (tmp_path / "copy/task.md").read_bytes() == saved
    assert sorted(
        p.relative_to(tmp_path / "copy").as_posix()
        for p in (tmp_path / "copy").rglob("*")
        if p.is_file()
    ) == [
        "input/data_dictionary.csv",
        "input/sample_submission.csv",
        "input/test.csv",
        "input/train.csv",
        "task.md",
    ]
    assert Path(result["import_origin"]).is_relative_to(unpacked)
    assert result["source_task_sha256"] == result["copied_task_sha256"]
    assert not (tmp_path / "copy/output/submission.csv").exists()
    assert (tmp_path / "source/output/submission.csv").read_text() == "PRIOR_SUBMISSION"


@pytest.mark.parametrize("kind", ["input_absolute", "output_external"])
def test_external_paths_are_rejected_before_training(tmp_path: Path, kind: str) -> None:
    root = tmp_path / "source"
    metadata = (
        {"train_path": str(root / "input/train.csv")}
        if kind == "input_absolute"
        else {"submission_path": str(tmp_path / "outside.csv")}
    )
    source = source_task(root, metadata=metadata)
    module = verifier()
    unpacked = tmp_path / "unpacked"
    module.extract_archive(archive(tmp_path), unpacked)
    cwd = tmp_path / "external"
    cwd.mkdir()
    with pytest.raises(ValueError, match="[Ee]xternal|absolute"):
        module.copy_task(unpacked, source, tmp_path / "copy", cwd, 42)
    assert not (tmp_path / "outside.csv").exists()


def test_synthetic_entrypoint_external_cwd_clean_environment_and_validation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "MUST_NOT_LEAK")
    monkeypatch.setenv("QWEN_TOKEN", "MUST_NOT_LEAK")
    monkeypatch.setenv("PYTHONPATH", str(ROOT))
    source = source_task(tmp_path / "source")
    output = tmp_path / "verification"
    result = verifier().verify_archive(
        archive(tmp_path, synthetic_entrypoint=True),
        [source],
        output,
        time_budget=10,
        threads=1,
        memory_limit_mb=1024,
    )
    assert result["all_successful"]
    record = result["results"][0]
    summary = record["summary"]
    assert not summary["injected_env_present"]
    assert Path(summary["import_origin"]).is_relative_to(output / "unpacked")
    assert not Path(summary["cwd"]).is_relative_to(ROOT)
    assert not Path(summary["cwd"]).is_relative_to(output / "unpacked")
    assert record["submission"]["status"] == "validated"
    assert len(record["submission"]["sha256"]) == 64
    assert (output / "task-00/stdout.log").exists()
    assert (output / "task-00/stderr.log").exists()
    assert set(result["environment"]) >= {
        "python",
        "platform",
        "cgroup_cpu_max",
        "cgroup_memory_max",
    }
    assert json.loads((output / "report.json").read_text())["all_successful"]


def test_actual_cli_tiny_budget_exits_without_heavy_training(tmp_path: Path) -> None:
    source = source_task(tmp_path / "source")
    result = verifier().verify_archive(
        archive(tmp_path),
        [source],
        tmp_path / "verification",
        time_budget=0.0001,
        threads=1,
        memory_limit_mb=1024,
    )
    record = result["results"][0]
    assert not result["all_successful"]
    assert record["returncode"] == 124
    assert record["submission"]["status"] == "invalid_or_unavailable"
    assert not (tmp_path / "verification/task-00/harness/validation/request.json").exists()
    assert (tmp_path / "source/output/submission.csv").read_text() == "PRIOR_SUBMISSION"
