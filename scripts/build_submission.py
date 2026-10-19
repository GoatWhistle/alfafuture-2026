"""Build a deterministic source-only competition archive without external tools."""

import argparse
import os
import tempfile
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

_EXCLUDED = frozenset(
    {
        "__pycache__",
        "runs",
        "dist",
        "build",
        "venv",
        "node_modules",
    }
)


def _package_files(directory: Path) -> list[Path]:
    files = []
    for path in sorted(directory.iterdir()):
        if path.name.startswith(".") or path.name in _EXCLUDED:
            continue
        # adapters/data is production source; root-level data is a local artifact folder.
        if directory.name == "ml_harness" and path.name == "data":
            continue
        if path.is_symlink():
            raise ValueError("Symlinks are forbidden inside the packaged source tree")
        if path.is_file() and path.suffix == ".py":
            files.append(path)
        elif path.is_dir() and (path / "__init__.py").is_file():
            files.extend(_package_files(path))
    return files


def build_submission(root: Path, output: Path) -> Path:
    """Include solution.py and Python files in regular ml_harness packages only."""
    root = root.resolve()
    entrypoint, package = root / "solution.py", root / "ml_harness"
    if entrypoint.is_symlink() or package.is_symlink():
        raise ValueError("Submission entrypoint and package must not be symlinks")
    if not entrypoint.is_file() or not (package / "__init__.py").is_file():
        raise ValueError("Source root requires solution.py and ml_harness/__init__.py")
    files = sorted([entrypoint, *_package_files(package)], key=lambda path: path.relative_to(root))
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".zip", delete=False) as stream:
            temporary = stream.name
        with ZipFile(temporary, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
            for path in files:
                info = ZipInfo(path.relative_to(root).as_posix(), date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = ZIP_DEFLATED
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                archive.writestr(info, path.read_bytes(), compresslevel=9)
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("dist/submission.zip"))
    args = parser.parse_args()
    print(build_submission(Path(__file__).resolve().parents[1], args.output))


if __name__ == "__main__":
    main()
