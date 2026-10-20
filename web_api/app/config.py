"""Paths and limits; everything is local and derived from the repository root."""

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FILE_NAMES = {
    "train": "train.csv",
    "test": "test.csv",
    "dictionary": "data_dictionary.csv",
    "sample": "sample_submission.csv",
}
MAX_UPLOAD_BYTES = 512 * 1024 * 1024
MAX_DESCRIPTION_CHARS = 100_000


@dataclass(frozen=True)
class Settings:
    runs_root: Path
    repo_root: Path
    python: str
    solution: Path
    allowed_origins: frozenset[str]
    allowed_hosts: frozenset[str]
    port: int

    @staticmethod
    def from_env() -> "Settings":
        port = int(os.environ.get("WEB_API_PORT", "8765"))
        venv_python = REPO_ROOT / ".venv" / "bin" / "python"
        origins = os.environ.get(
            "WEB_API_ORIGINS",
            ",".join(
                f"http://{host}:{p}" for host in ("127.0.0.1", "localhost") for p in (5173, port)
            ),
        )
        return Settings(
            runs_root=Path(os.environ.get("WEB_API_RUNS", REPO_ROOT / "runs" / "web")).resolve(),
            repo_root=REPO_ROOT,
            python=os.environ.get(
                "WEB_API_HARNESS_PYTHON", str(venv_python if venv_python.exists() else "python3")
            ),
            solution=Path(os.environ.get("WEB_API_SOLUTION", REPO_ROOT / "solution.py")).resolve(),
            allowed_origins=frozenset(o.strip() for o in origins.split(",") if o.strip()),
            allowed_hosts=frozenset(f"{h}:{port}" for h in ("127.0.0.1", "localhost")),
            port=port,
        )
