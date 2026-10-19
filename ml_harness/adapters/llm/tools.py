"""Validated research tools with workspace confinement and call deduplication."""

import os
import tempfile
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ml_harness.application.ports.language_model import ToolCall


class Arguments(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")


class ReadFile(Arguments):
    path: str
    max_chars: Annotated[int, Field(ge=1, le=12000)] = 8000


class EditFile(Arguments):
    path: str
    content: Annotated[str, Field(max_length=100000)]


class RunExperiment(Arguments):
    experiment_id: Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")]
    hypothesis: Annotated[str, Field(min_length=1, pattern=r"\S")]
    expected_effect: Annotated[str, Field(min_length=1, pattern=r"\S")]
    parent_id: str | None = None


class ReadExperiments(Arguments):
    limit: Annotated[int, Field(ge=1, le=30)] = 10


_MODELS: dict[str, type[Arguments]] = {
    "inspect_data": Arguments,
    "read_file": ReadFile,
    "edit_file": EditFile,
    "run_experiment": RunExperiment,
    "read_experiments": ReadExperiments,
}


def _error(kind: str, message: str) -> dict[str, Any]:
    return {"ok": False, "error": {"type": kind, "message": message}}


class ResearchTools:
    def __init__(
        self,
        workspace: Path,
        inspect: Callable[[], dict[str, Any]],
        run_experiment: Callable[[dict[str, Any]], dict[str, Any]],
        read_experiments: Callable[[], list[dict[str, Any]]],
        editable_paths: tuple[str, ...] = ("candidate.py",),
    ) -> None:
        self.workspace = workspace.resolve()
        self.inspect = inspect
        self.run_experiment = run_experiment
        self.read_experiments = read_experiments
        self.editable_paths = frozenset(self._path(path) for path in editable_paths)
        self._cache: dict[str, tuple[str, str, dict[str, Any]]] = {}

    @property
    def schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": name.replace("_", " "),
                    "parameters": model.model_json_schema(),
                },
            }
            for name, model in _MODELS.items()
        ]

    def _path(self, value: str) -> Path:
        path = Path(value)
        if ".." in path.parts:
            raise ValueError("Path traversal is forbidden")
        resolved = (self.workspace / path).resolve()
        if not resolved.is_relative_to(self.workspace):
            raise ValueError("Path must stay inside workspace")
        return resolved

    def execute(self, call: ToolCall) -> dict[str, Any]:
        previous = self._cache.get(call.id)
        if previous is not None:
            name, arguments, response = previous
            if (name, arguments) != (call.name, call.arguments):
                return _error(
                    "call_id_conflict", "Tool call ID already used with different payload"
                )
            return deepcopy(response)
        response = self._execute(call)
        self._cache[call.id] = (call.name, call.arguments, deepcopy(response))
        return response

    def _execute(self, call: ToolCall) -> dict[str, Any]:
        model = _MODELS.get(call.name)
        if model is None:
            return _error("unknown_tool", "Unknown research tool")
        try:
            arguments = model.model_validate_json(call.arguments)
        except ValidationError:
            return _error("invalid_arguments", "Tool arguments do not match the schema")
        try:
            if isinstance(arguments, ReadFile):
                path = self._path(arguments.path)
                if not path.is_file():
                    return _error("invalid_path", "Read path must be a regular file")
                with path.open(encoding="utf-8") as stream:
                    text = stream.read(arguments.max_chars + 1)
                result: Any = {
                    "text": text[: arguments.max_chars],
                    "truncated": len(text) > arguments.max_chars,
                }
            elif isinstance(arguments, EditFile):
                path = self._path(arguments.path)
                if path not in self.editable_paths:
                    return _error("invalid_path", "File is not editable")
                # Resolve again: an editable file may have become an escaping symlink.
                if not path.resolve().is_relative_to(self.workspace):
                    return _error("invalid_path", "Path must stay inside workspace")
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary: str | None = None
                try:
                    with tempfile.NamedTemporaryFile(
                        mode="w",
                        encoding="utf-8",
                        dir=path.parent,
                        delete=False,
                    ) as stream:
                        temporary = stream.name
                        stream.write(arguments.content)
                    os.replace(temporary, path)
                finally:
                    if temporary is not None:
                        Path(temporary).unlink(missing_ok=True)
                result = {"written": True}
            elif isinstance(arguments, RunExperiment):
                result = self.run_experiment(arguments.model_dump())
            elif isinstance(arguments, ReadExperiments):
                result = self.read_experiments()[-arguments.limit :]
            else:
                result = self.inspect()
            return {"ok": True, "result": result}
        except (ValueError, OSError):
            return _error("tool_error", "Tool failed: invalid path, content or operation")
        except Exception:
            return _error("tool_error", "Tool execution failed")
