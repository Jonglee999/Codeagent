"""Confined file and empty-directory deletion tool."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from codeagent.tools.base import BaseTool, ToolResult


class DeleteFileTool(BaseTool):
    category = "mutation"
    risk_level = "high"
    idempotent = False
    name = "delete_file"
    description = "Delete one file, symbolic link, or empty directory inside the project workspace"
    parameters = {
        "type": "object",
        "properties": {"file_path": {"type": "string", "minLength": 1}},
        "required": ["file_path"],
        "additionalProperties": False,
    }

    def __init__(self, project_root: str | Path = ".") -> None:
        self._project_root = Path(project_root).resolve()

    async def execute(self, file_path: str, **_: Any) -> ToolResult:
        started = time.monotonic()
        try:
            if "\x00" in file_path:
                raise ValueError("Path contains a null byte")
            candidate = Path(file_path)
            target = (
                candidate.resolve()
                if candidate.is_absolute()
                else (self._project_root / candidate).resolve()
            )
            target.relative_to(self._project_root)
            if target == self._project_root:
                raise ValueError("The project root cannot be deleted")
            if not target.exists() and not target.is_symlink():
                raise FileNotFoundError(f"Path does not exist: {file_path}")
            if target.is_dir() and not target.is_symlink():
                target.rmdir()
            else:
                target.unlink()
            return ToolResult(
                success=True,
                data={"deleted": str(target.relative_to(self._project_root)).replace("\\", "/")},
                duration_ms=(time.monotonic() - started) * 1000,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            return ToolResult(
                success=False,
                error_message=str(exc),
                error_code="DELETE_FAILED",
                duration_ms=(time.monotonic() - started) * 1000,
            )
