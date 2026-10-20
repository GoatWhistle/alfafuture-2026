import sys
from pathlib import Path

import pytest
from app.config import Settings
from app.main import create_app
from fastapi.testclient import TestClient

HERE = Path(__file__).parent
ORIGIN = "http://127.0.0.1:5173"


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        runs_root=tmp_path / "runs",
        repo_root=tmp_path,
        python=sys.executable,
        solution=HERE / "fake_solution.py",
        allowed_origins=frozenset({ORIGIN}),
        allowed_hosts=frozenset({"127.0.0.1:8765"}),
        port=8765,
    )


@pytest.fixture
def client(settings: Settings) -> TestClient:
    return TestClient(
        create_app(settings), base_url="http://127.0.0.1:8765", headers={"Origin": ORIGIN}
    )


def files() -> dict[str, tuple[str, bytes, str]]:
    return {
        key: (name, b"id,x\n1,2\n", "text/csv")
        for key, name in {
            "train": "train.csv",
            "test": "test.csv",
            "dictionary": "data_dictionary.csv",
            "sample": "sample_submission.csv",
        }.items()
    }


def create(client: TestClient, **form: str) -> str:
    data = {"title": "Тест", "description": "Задача", **form}
    response = client.post("/api/runs", data=data, files=files())
    assert response.status_code == 201, response.text
    return response.json()["id"]
