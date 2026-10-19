"""Research tools validate external calls before touching files or callbacks."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

from ml_harness.adapters.llm.tools import ResearchTools
from ml_harness.application.ports.language_model import ToolCall


@pytest.fixture
def tools(tmp_path: Path) -> ResearchTools:
    return ResearchTools(
        tmp_path,
        lambda: {"rows": 4},
        lambda args: args,
        lambda: [{"id": str(i)} for i in range(20)],
    )


def call(tools: ResearchTools, name: str, args: Any, identity: str = "call") -> dict[str, Any]:
    return tools.execute(ToolCall(identity, name, json.dumps(args)))


def test_generated_schemas_for_all_five_tools(tools: ResearchTools) -> None:
    assert {schema["function"]["name"] for schema in tools.schemas} == {
        "inspect_data",
        "read_file",
        "edit_file",
        "run_experiment",
        "read_experiments",
    }
    for schema in tools.schemas:
        assert schema["type"] == "function"
        assert schema["function"]["parameters"]["additionalProperties"] is False


@pytest.mark.parametrize("arguments", ["{", "null", "[]", '"string"', ""])
def test_malformed_arguments(tools: ResearchTools, arguments: str) -> None:
    response = tools.execute(ToolCall("id", "inspect_data", arguments))
    assert response["ok"] is False
    assert response["error"]["type"] == "invalid_arguments"


def test_unknown_tool(tools: ResearchTools) -> None:
    assert call(tools, "bad", {})["error"]["type"] == "unknown_tool"


@pytest.mark.parametrize(
    "name,args",
    [
        ("inspect_data", {"extra": 1}),
        ("read_file", {}),
        ("edit_file", {"path": "candidate.py"}),
        ("read_file", {"path": 1}),
        ("read_file", {"path": "candidate.py", "max_chars": "4"}),
        ("read_file", {"path": "candidate.py", "max_chars": True}),
        ("read_file", {"path": "candidate.py", "max_chars": 0}),
        ("read_file", {"path": "candidate.py", "max_chars": 12001}),
        ("read_experiments", {"limit": 31}),
        ("read_experiments", {"limit": "1"}),
        ("read_experiments", {"limit": 0}),
        ("edit_file", {"path": "candidate.py", "content": 4}),
        ("edit_file", {"path": "candidate.py", "content": "x" * 100001}),
        ("run_experiment", {}),
        ("run_experiment", {"experiment_id": "../bad", "hypothesis": "h", "expected_effect": "e"}),
        ("run_experiment", {"experiment_id": "x" * 65, "hypothesis": "h", "expected_effect": "e"}),
        ("run_experiment", {"experiment_id": "x", "hypothesis": " ", "expected_effect": "e"}),
        ("run_experiment", {"experiment_id": "x", "hypothesis": "h", "expected_effect": ""}),
    ],
)
def test_schema_rejections(tools: ResearchTools, name: str, args: dict[str, Any]) -> None:
    assert call(tools, name, args)["error"]["type"] == "invalid_arguments"


def test_read_file_truncation_and_edit_utf8(tools: ResearchTools, tmp_path: Path) -> None:
    assert call(tools, "edit_file", {"path": "candidate.py", "content": "привет"})["ok"]
    response = call(tools, "read_file", {"path": "candidate.py", "max_chars": 3}, "read")
    assert response == {"ok": True, "result": {"text": "при", "truncated": True, "next_offset": 3}}
    response = call(tools, "read_file", {"path": "candidate.py"}, "read_all")
    assert response["result"] == {"text": "привет", "truncated": False, "next_offset": None}
    assert {p.name for p in tmp_path.iterdir()} == {"candidate.py", "tools.jsonl"}


def test_only_allowlisted_files_can_be_edited(tools: ResearchTools, tmp_path: Path) -> None:
    (tmp_path / "data.csv").write_text("data")
    assert call(tools, "read_file", {"path": "data.csv"})["ok"]
    assert not call(tools, "edit_file", {"path": "data.csv", "content": "bad"}, "edit")["ok"]
    assert (tmp_path / "data.csv").read_text() == "data"


@pytest.mark.parametrize("path", ["../secret", "sub/../../secret"])
def test_rejects_traversal(tools: ResearchTools, path: str) -> None:
    assert not call(tools, "read_file", {"path": path})["ok"]


def test_absolute_and_symlink_escape_rejected(tools: ResearchTools, tmp_path: Path) -> None:
    outside = tmp_path.parent / (tmp_path.name + "-secret")
    outside.write_text("secret")
    (tmp_path / "escape").symlink_to(outside)
    (tmp_path / "candidate.py").symlink_to(outside)
    for i, path in enumerate([str(outside), "escape"]):
        response = call(tools, "read_file", {"path": path}, str(i))
        assert not response["ok"]
        assert str(outside) not in response["error"]["message"]
    assert not call(tools, "edit_file", {"path": "candidate.py", "content": "bad"}, "edit")["ok"]
    assert outside.read_text() == "secret"


def test_directory_and_missing_read_paths(tools: ResearchTools) -> None:
    assert not call(tools, "read_file", {"path": "."})["ok"]
    assert not call(tools, "read_file", {"path": "missing"}, "missing")["ok"]


def test_inspect_and_journal_limits(tools: ResearchTools) -> None:
    assert call(tools, "inspect_data", {}) == {"ok": True, "result": {"rows": 4}}
    assert len(call(tools, "read_experiments", {}, "journal")["result"]) == 10
    assert call(tools, "read_experiments", {"limit": 1}, "last")["result"] == [{"id": "19"}]


def test_experiment_callback_receives_validated_defaults(tools: ResearchTools) -> None:
    args = {"experiment_id": "exp-01", "hypothesis": "h", "expected_effect": "e"}
    assert call(tools, "run_experiment", args)["result"] == args | {"parent_id": None}


def test_repeated_id_deduplicates_and_conflicting_payload_is_rejected(tmp_path: Path) -> None:
    callback = Mock(return_value={"status": "done"})
    tools = ResearchTools(tmp_path, lambda: {}, callback, lambda: [])
    args = {"experiment_id": "exp", "hypothesis": "h", "expected_effect": "e"}
    first = call(tools, "run_experiment", args)
    first["result"]["status"] = "mutated"
    assert call(tools, "run_experiment", args)["result"]["status"] == "done"
    assert callback.call_count == 1
    assert call(tools, "run_experiment", args | {"hypothesis": "changed"})["error"]["type"] == (
        "call_id_conflict"
    )
    assert call(tools, "inspect_data", {})["error"]["type"] == "call_id_conflict"
    assert callback.call_count == 1


def test_repeated_edit_does_not_write_again(tools: ResearchTools, tmp_path: Path) -> None:
    args = {"path": "candidate.py", "content": "first"}
    assert call(tools, "edit_file", args)["ok"]
    (tmp_path / "candidate.py").write_text("external")
    assert call(tools, "edit_file", args)["ok"]
    assert (tmp_path / "candidate.py").read_text() == "external"


def test_nested_editable_file_created_atomically(tmp_path: Path) -> None:
    tools = ResearchTools(tmp_path, lambda: {}, lambda args: args, lambda: [], ("code/model.py",))
    assert call(tools, "edit_file", {"path": "code/model.py", "content": "x"})["ok"]
    assert (tmp_path / "code/model.py").read_text() == "x"
    assert len(list((tmp_path / "code").iterdir())) == 1


def test_callback_failures_are_sanitized_and_cached(tmp_path: Path) -> None:
    callback = Mock(side_effect=RuntimeError("/private/path API_KEY=secret"))
    tools = ResearchTools(tmp_path, callback, lambda args: args, lambda: [])
    response = call(tools, "inspect_data", {})
    assert not response["ok"]
    assert "secret" not in str(response)
    assert call(tools, "inspect_data", {}) == response
    assert callback.call_count == 1


def test_repeated_unchanged_page_is_short_and_new_candidate_can_be_read(tools, tmp_path):
    path = tmp_path / "candidate.py"
    path.write_text("x" * 3000)
    first = call(tools, "read_file", {"path": "candidate.py"}, "first")
    second = call(tools, "read_file", {"path": "candidate.py"}, "second")
    assert len(first["result"]["text"]) == 3000
    assert second["result"]["already_read"] and len(second["result"]["text"]) == 500
    path.write_text("changed")
    third = call(tools, "read_file", {"path": "candidate.py"}, "third")
    assert third["result"]["text"] == "changed" and "already_read" not in third["result"]
