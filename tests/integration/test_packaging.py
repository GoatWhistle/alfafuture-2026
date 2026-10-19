"""Source packaging is deterministic, allowlisted and runnable outside its source cwd."""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/build_submission.py"
pytestmark = pytest.mark.integration


def builder():
    spec = importlib.util.spec_from_file_location("submission_builder", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_submission


def source_fixture(root: Path) -> None:
    paths = {
        "solution.py": "from ml_harness.main import message\nprint(message)\n",
        "ml_harness/__init__.py": "",
        "ml_harness/main.py": 'message = "ready"\n',
        "ml_harness/nested/__init__.py": "",
        "ml_harness/nested/module.py": "VALUE=1\n",
        ".env": "API_KEY=secret",
        "secret.py": "SECRET",
        "README.md": "docs",
        "data/train.csv": "private",
        "runs/result.json": "private",
        ".venv/site.py": "private",
        "ml_harness/.env": "secret",
        "ml_harness/model.pkl": "private",
        "ml_harness/__pycache__/cached.py": "private",
        "ml_harness/runs/__init__.py": "",
        "ml_harness/runs/private.py": "private",
        "ml_harness/data/__init__.py": "",
        "ml_harness/data/private.py": "private",
        "ml_harness/.hidden/__init__.py": "",
        "ml_harness/.hidden/private.py": "private",
        "ml_harness/unpackaged/private.py": "private",
    }
    for name, content in paths.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)


def test_safe_allowlist_and_fixed_metadata(tmp_path: Path) -> None:
    source_fixture(tmp_path)
    output = builder()(tmp_path, tmp_path / "dist/submission.zip")
    with ZipFile(output) as archive:
        assert archive.namelist() == [
            "ml_harness/__init__.py",
            "ml_harness/main.py",
            "ml_harness/nested/__init__.py",
            "ml_harness/nested/module.py",
            "solution.py",
        ]
        for item in archive.infolist():
            assert not item.filename.startswith("/")
            assert ".." not in Path(item.filename).parts
            assert item.date_time == (1980, 1, 1, 0, 0, 0)
            assert item.external_attr >> 16 == 0o100644


def test_identical_bytes_despite_mtimes_and_permissions(tmp_path: Path) -> None:
    source_fixture(tmp_path)
    build = builder()
    first = build(tmp_path, tmp_path / "first.zip").read_bytes()
    for path in tmp_path.rglob("*.py"):
        os.utime(path, (1700000000, 1700000000))
        path.chmod(0o755)
    assert build(tmp_path, tmp_path / "second.zip").read_bytes() == first
    assert build(tmp_path, tmp_path / "first.zip").read_bytes() == first


@pytest.mark.parametrize("target", ["file", "directory", "entrypoint", "package"])
def test_symlinks_cannot_include_external_sources(tmp_path: Path, target: str) -> None:
    root = tmp_path / "source"
    source_fixture(root)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "__init__.py").write_text("secret")
    if target == "file":
        (root / "ml_harness/escape.py").symlink_to(outside / "__init__.py")
    elif target == "directory":
        (root / "ml_harness/escape").symlink_to(outside, target_is_directory=True)
    elif target == "entrypoint":
        (root / "solution.py").unlink()
        (root / "solution.py").symlink_to(outside / "__init__.py")
    else:
        import shutil

        shutil.rmtree(root / "ml_harness")
        (root / "ml_harness").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="[Ss]ymlink"):
        builder()(root, tmp_path / "submission.zip")
    assert not (tmp_path / "submission.zip").exists()


def test_extracted_source_runs_from_an_unrelated_working_directory(tmp_path: Path) -> None:
    source, extracted, unrelated = (
        tmp_path / name for name in ("source", "extracted", "unrelated")
    )
    source_fixture(source)
    output = builder()(source, tmp_path / "submission.zip")
    with ZipFile(output) as archive:
        archive.extractall(extracted)
    unrelated.mkdir()
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    process = subprocess.run(
        [sys.executable, str(extracted / "solution.py")],
        cwd=unrelated,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert process.returncode == 0, process.stderr
    assert process.stdout.strip() == "ready"


def test_actual_cli_archive_runs_help_outside_repository(tmp_path: Path) -> None:
    output = tmp_path / "actual.zip"
    built = subprocess.run(
        [sys.executable, str(SCRIPT), "--output", str(output)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert built.returncode == 0, built.stderr
    with ZipFile(output) as archive:
        names = archive.namelist()
        assert "solution.py" in names
        assert all(
            name == "solution.py" or (name.startswith("ml_harness/") and name.endswith(".py"))
            for name in names
        )
        extracted = tmp_path / "extracted"
        archive.extractall(extracted)
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    checked = subprocess.run(
        [sys.executable, str(extracted / "solution.py"), "--help"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert checked.returncode == 0, checked.stderr
    assert "--time-budget" in checked.stdout
