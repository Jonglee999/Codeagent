"""Bounded, workspace-confined file tree listing."""

from __future__ import annotations

import fnmatch
import os
import time
from pathlib import Path
from collections.abc import Iterator
from typing import Any

from codeagent.tools.base import BaseTool, ToolResult

_DEFAULT_EXCLUDES = frozenset(
    {
        ".git",
        ".codeagent",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "node_modules",
    }
)


class ListFilesTool(BaseTool):
    """List files and directories without requiring a terminal command."""

    name = "list_files"
    category = "exploration"
    read_only = True
    description = "列出项目内的文件和目录；支持 glob、递归和有界输出，适合在读取前了解仓库结构"
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "default": ".",
                "description": "相对项目根目录的起始目录",
            },
            "pattern": {
                "type": "string",
                "default": "*",
                "description": "匹配文件名或相对路径的 glob，例如 *.py 或 tests/**/test_*.py",
            },
            "recursive": {
                "type": "boolean",
                "default": False,
                "description": "是否递归列出",
            },
            "include_hidden": {
                "type": "boolean",
                "default": False,
                "description": "是否包含隐藏项；.git 和 .codeagent 始终排除",
            },
            "max_results": {
                "type": "integer",
                "minimum": 1,
                "maximum": 1000,
                "default": 200,
                "description": "最大结果数",
            },
        },
        "additionalProperties": False,
    }

    def __init__(self, project_root: str | Path = ".") -> None:
        self._project_root = Path(project_root).resolve()

    async def execute(
        self,
        path: str = ".",
        pattern: str = "*",
        recursive: bool = False,
        include_hidden: bool = False,
        max_results: int = 200,
        **_: Any,
    ) -> ToolResult:
        started = time.monotonic()
        try:
            if "\x00" in path or "\x00" in pattern:
                raise ValueError("Path and pattern cannot contain null bytes")
            raw = Path(path)
            if raw.is_absolute() or raw.root:
                raise ValueError("Only project-relative paths are allowed")
            root = (self._project_root / raw).resolve()
            root.relative_to(self._project_root)
            if not root.exists():
                raise FileNotFoundError(f"Path does not exist: {path}")
            if not root.is_dir():
                raise NotADirectoryError(f"Not a directory: {path}")

            entries: list[dict[str, Any]] = []
            truncated = False

            def excluded(candidate: Path) -> bool:
                relative = candidate.relative_to(self._project_root)
                if any(part in {".git", ".codeagent"} for part in relative.parts):
                    return True
                if not include_hidden and any(
                    part.startswith(".") or part in _DEFAULT_EXCLUDES for part in relative.parts
                ):
                    return True
                return False

            def walk_candidates() -> Iterator[Path]:
                for current, directories, files in os.walk(root, followlinks=False):
                    current_path = Path(current)
                    directories[:] = sorted(
                        name for name in directories if not excluded(current_path / name)
                    )
                    yield from (current_path / name for name in directories)
                    yield from (current_path / name for name in sorted(files))

            candidates = (
                walk_candidates()
                if recursive
                else iter(sorted(root.iterdir(), key=lambda item: item.name.lower()))
            )

            for candidate in candidates:
                if excluded(candidate):
                    continue
                relative_to_start = candidate.relative_to(root).as_posix()
                if not (
                    fnmatch.fnmatch(candidate.name, pattern)
                    or fnmatch.fnmatch(relative_to_start, pattern)
                ):
                    continue
                if len(entries) >= max_results:
                    truncated = True
                    break
                project_relative = candidate.relative_to(self._project_root).as_posix()
                is_directory = candidate.is_dir()
                entry: dict[str, Any] = {
                    "path": project_relative,
                    "type": "directory" if is_directory else "file",
                }
                if not is_directory:
                    try:
                        entry["size_bytes"] = candidate.stat().st_size
                    except OSError:
                        entry["size_bytes"] = None
                entries.append(entry)

            return ToolResult(
                success=True,
                data={
                    "path": root.relative_to(self._project_root).as_posix() or ".",
                    "pattern": pattern,
                    "recursive": recursive,
                    "entries": entries,
                    "count": len(entries),
                    "truncated": truncated,
                },
                duration_ms=(time.monotonic() - started) * 1000,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            return ToolResult(
                success=False,
                error_message=str(exc),
                error_code="LIST_FILES_FAILED",
                duration_ms=(time.monotonic() - started) * 1000,
            )
