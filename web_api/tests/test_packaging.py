import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def test_competition_archive_excludes_web_and_api(tmp_path):
    sys.path.insert(0, str(REPO / "scripts"))
    try:
        import build_submission
    finally:
        sys.path.pop(0)
    archive = build_submission.build_submission(REPO, tmp_path / "submission.zip")
    names = zipfile.ZipFile(archive).namelist()
    assert "solution.py" in names
    assert not [n for n in names if n.split("/")[0] in ("web", "web_api")]
    assert all(n == "solution.py" or n.startswith("ml_harness/") for n in names)
