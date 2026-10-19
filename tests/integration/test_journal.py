"""Durability/recovery tests exercise real bytes and a separate writer process."""

import hashlib
import json
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from ml_harness.adapters.storage.journal import JsonlExperimentStore
from ml_harness.domain.models import ContractError


def event(identifier="first", **payload):
    return {"event_id": identifier, "kind": "experiment_finished", "payload": payload}


def line(value):
    return json.dumps(value).encode() + b"\n"


def test_process_restart_and_idempotent_append(tmp_path):
    code = """import sys
from pathlib import Path
from ml_harness.adapters.storage.journal import JsonlExperimentStore
store = JsonlExperimentStore(Path(sys.argv[1]))
store.append({"event_id": "first", "kind": "experiment_finished", "payload": {"score": 0.4}})
"""
    subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)],
        check=True,
        cwd=Path(__file__).resolve().parents[2],
        timeout=10,
    )
    store = JsonlExperimentStore(tmp_path)
    original = store.path.read_bytes()
    store.append(event(score=0.4))
    assert store.path.read_bytes() == original
    assert store.read_events() == [event(score=0.4)]
    with pytest.raises(ContractError, match="Conflicting"):
        store.append(event(score=0.5))
    assert store.path.read_bytes() == original


@pytest.mark.parametrize("tail", [b'{"event_id":', b'{"event_id":"\xd0', b"not-json"])
def test_incomplete_tail_quarantined_exactly_once(tmp_path, tail):
    store = JsonlExperimentStore(tmp_path)
    whole = line(event())
    store.path.write_bytes(whole + tail)
    assert store.read_events() == [event()]
    assert store.path.read_bytes() == whole
    quarantined = list((tmp_path / "recovery").glob("*.bin"))
    assert len(quarantined) == 1
    assert quarantined[0].read_bytes() == tail
    assert JsonlExperimentStore(tmp_path).read_events() == [event()]
    assert len(list((tmp_path / "recovery").glob("*.bin"))) == 1
    store.append(event("second"))
    assert store.read_events() == [event(), event("second")]


def test_append_recovers_tail_and_completes_missing_newline(tmp_path):
    store = JsonlExperimentStore(tmp_path)
    store.path.write_bytes(line(event()).rstrip(b"\n"))
    assert store.read_events() == [event()]
    store.append(event("second"))
    assert store.path.read_bytes() == line(event()) + line(event("second"))
    with store.path.open("ab") as stream:
        stream.write(b'{"partial":')
    store.append(event("third"))
    assert store.read_events() == [event(), event("second"), event("third")]


@pytest.mark.parametrize(
    "contents",
    [
        line(event()) + b"broken\n" + line(event("third")),
        line(event()) + b"broken\n",
        line(event()) + line({"event_id": "bad", "payload": {}}),
        line(event()) + b'{"event_id":"bad","kind":"x","payload":{"n":NaN}}\n',
    ],
)
def test_broken_complete_records_are_not_silently_removed(tmp_path, contents):
    store = JsonlExperimentStore(tmp_path)
    store.path.write_bytes(contents)
    with pytest.raises(ContractError):
        store.read_events()
    assert store.path.read_bytes() == contents
    assert not (tmp_path / "recovery").exists()


def test_read_deduplicates_identical_ids_and_rejects_conflicts(tmp_path):
    store = JsonlExperimentStore(tmp_path)
    store.path.write_bytes(line(event()) + line(event()) + line(event("other")))
    assert store.read_events() == [event(), event("other")]
    contents = line(event(score=1)) + line(event(score=2))
    store.path.write_bytes(contents)
    with pytest.raises(ContractError, match="Conflicting"):
        store.read_events()
    assert store.path.read_bytes() == contents


@pytest.mark.parametrize(
    "value",
    [
        {},
        {"event_id": 1, "kind": "a", "payload": {}},
        {"event_id": "a", "kind": "a", "payload": []},
        event(score=float("inf")),
        event(score=object()),
    ],
)
def test_append_checks_event_contract_before_writing(tmp_path, value):
    store = JsonlExperimentStore(tmp_path)
    with pytest.raises(ContractError):
        store.append(value)
    assert not store.path.exists()


def test_candidate_is_content_addressed_readonly_and_not_overwritten(tmp_path):
    store = JsonlExperimentStore(tmp_path)
    code = b"def build_model(*args): return None\n"
    digest, saved = store.save_code(code)
    path = Path(saved)
    assert path.is_absolute()
    assert digest == hashlib.sha256(code).hexdigest()
    assert path.name == digest + ".py"
    assert path.read_bytes() == code
    assert stat.S_IMODE(path.stat().st_mode) == 0o444
    inode = path.stat().st_ino
    assert JsonlExperimentStore(tmp_path).save_code(code) == (digest, saved)
    assert path.stat().st_ino == inode
    assert not list(path.parent.glob("*.tmp"))
    path.chmod(0o644)
    path.write_bytes(b"corruption")
    with pytest.raises(ContractError, match="hash mismatch"):
        store.save_code(code)
    assert path.read_bytes() == b"corruption"


def test_candidate_symlink_is_rejected_without_touching_target(tmp_path):
    store = JsonlExperimentStore(tmp_path)
    code = b"original"
    digest = hashlib.sha256(code).hexdigest()
    target = tmp_path / "outside.py"
    target.write_bytes(code)
    versions = tmp_path / "versions"
    versions.mkdir()
    (versions / (digest + ".py")).symlink_to(target)
    with pytest.raises(ContractError, match="hash mismatch"):
        store.save_code(code)
    assert stat.S_IMODE(target.stat().st_mode) != 0o444
