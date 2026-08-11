"""Conflict-aware exact text replacement for existing workspace files."""

from __future__ import annotations

from difflib import SequenceMatcher
import re
import time
from pathlib import Path, PurePosixPath
from typing import Any

from codeagent.tools.base import BaseTool, ToolResult
from codeagent.tools.file.write_file import WriteFileTool


def _normalize_declared_patch_path(value: str) -> str:
    candidate = value.strip().replace("\\", "/")
    if candidate.startswith(("a/", "b/")):
        candidate = candidate[2:]
    path = PurePosixPath(candidate)
    if (
        not candidate
        or candidate == "/dev/null"
        or candidate.startswith("/")
        or re.match(r"^[A-Za-z]:", candidate)
        or any(part in {"", ".", ".."} for part in path.parts)
        or ".git" in path.parts
    ):
        raise ValueError(f"Patch declares an unsafe file path: {value!r}")
    return path.as_posix()


def _infer_single_patch_path(patch: str) -> str:
    """Infer exactly one safe update target from a patch envelope."""

    declared: set[str] = set()
    normalized_patch = patch.replace("\r\n", "\n").replace("\r", "\n")
    for raw_line in normalized_patch.splitlines():
        candidate: str | None = None
        if raw_line.startswith("*** Update File: "):
            candidate = raw_line.removeprefix("*** Update File: ")
        elif raw_line.startswith("+++ "):
            candidate = raw_line[4:].split("\t", 1)[0]
        elif raw_line.startswith("diff --git "):
            fields = raw_line.split()
            if (
                len(fields) == 4
                and fields[2].removeprefix("a/")
                == fields[3].removeprefix("b/")
            ):
                candidate = fields[3]
        if candidate and candidate.strip() != "/dev/null":
            declared.add(_normalize_declared_patch_path(candidate))
    if len(declared) != 1:
        raise ValueError(
            "Patch without file_path must declare exactly one safe update target; "
            f"found {len(declared)}"
        )
    return next(iter(declared))


def _apply_unified_patch(original: str, patch: str, expected_path: str) -> tuple[str, int]:
    """Apply a bounded single-file unified/Codex patch by exact hunk matching."""
    lines = patch.replace("\r\n", "\n").replace("\r", "\n").splitlines(keepends=True)
    hunks: list[list[str]] = []
    current: list[str] | None = None
    declared_path: str | None = None

    for line in lines:
        marker = line.rstrip("\n")
        if marker in {"*** Begin Patch", "*** End Patch"}:
            continue
        if marker.startswith("*** Update File: "):
            declared_path = marker.removeprefix("*** Update File: ").strip()
            continue
        if marker.startswith("*** Add File:") or marker.startswith("*** Delete File:"):
            raise ValueError("apply_patch only updates the supplied existing file")
        if line.startswith("--- "):
            continue
        if line.startswith("+++ "):
            candidate = marker[4:].strip()
            declared_path = candidate[2:] if candidate.startswith("b/") else candidate
            continue
        if line.startswith("@@"):
            current = []
            hunks.append(current)
            continue
        if marker == r"\ No newline at end of file":
            continue
        if current is not None:
            if line[:1] not in {" ", "+", "-"}:
                raise ValueError("Malformed unified patch hunk")
            current.append(line)
        elif marker.strip():
            raise ValueError("Unified patch content must be inside an @@ hunk")

    normalized_expected = expected_path.replace("\\", "/")
    if declared_path and declared_path.replace("\\", "/") != normalized_expected:
        raise ValueError(
            f"Patch targets '{declared_path}', not supplied file '{normalized_expected}'"
        )
    if not hunks or any(not hunk for hunk in hunks):
        raise ValueError("Unified patch must contain at least one non-empty @@ hunk")

    updated = original
    for hunk in hunks:
        old_text = "".join(line[1:] for line in hunk if line[:1] in {" ", "-"})
        new_text = "".join(line[1:] for line in hunk if line[:1] in {" ", "+"})
        if not old_text:
            raise ValueError("Unified patch hunk must include context or removed text")
        matches = updated.count(old_text)
        if matches != 1:
            raise ValueError(
                f"Patch conflict: hunk expected 1 exact match, found {matches}"
            )
        updated = updated.replace(old_text, new_text, 1)
    return updated, len(hunks)


def _closest_patch_context(original: str, patch: str) -> dict[str, Any]:
    """Return a bounded exact source excerpt for a conflicting patch hunk.

    This is diagnostic only: exact matching and all mutation safety checks stay
    unchanged. It helps a provider correct whitespace or line-wrapping guesses.
    """

    expected_lines: list[str] = []
    in_hunk = False
    for line in patch.replace("\r\n", "\n").replace("\r", "\n").splitlines():
        if line.startswith("@@"):
            in_hunk = True
            continue
        if not in_hunk or not line or line.startswith("+"):
            continue
        if line[0] in {" ", "-"}:
            expected_lines.append(line[1:])
    source_lines = original.splitlines()
    if not expected_lines or not source_lines:
        return {}

    expected = "\n".join(expected_lines)
    expected_count = len(expected_lines)
    best_ratio = -1.0
    best_start = 0
    best_end = 0
    min_size = max(1, expected_count - 2)
    max_size = min(len(source_lines), expected_count + 3)
    for size in range(min_size, max_size + 1):
        for start in range(0, len(source_lines) - size + 1):
            candidate = "\n".join(source_lines[start : start + size])
            ratio = SequenceMatcher(None, expected, candidate, autojunk=False).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_start = start
                best_end = start + size

    return {
        "closest_context_start_line": best_start + 1,
        "closest_context": "\n".join(source_lines[best_start:best_end])[:4000],
        "similarity": round(max(0.0, best_ratio), 3),
    }


class ApplyPatchTool(BaseTool):
    """Apply an exact replacement only when the expected file state still matches."""

    name = "apply_patch"
    category = "mutation"
    risk_level = "medium"
    description = "精确替换已有文件中的文本；旧文本和期望匹配次数不一致时拒绝写入，避免覆盖并发修改"
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "minLength": 1,
                "description": "相对项目根目录的文件路径",
            },
            "old_text": {
                "type": "string",
                "minLength": 1,
                "description": "必须精确匹配的原文本",
            },
            "new_text": {
                "type": "string",
                "description": "替换后的文本；可为空以删除匹配内容",
            },
            "expected_replacements": {
                "type": "integer",
                "minimum": 1,
                "maximum": 100,
                "default": 1,
                "description": "期望匹配和替换次数，默认 1",
            },
        },
        "required": ["file_path", "old_text", "new_text"],
        "additionalProperties": False,
    }

    description = (
        "Safely update one existing UTF-8 file. Provide either old_text/new_text "
        "for an exact replacement, or patch for a single-file unified/Codex patch."
    )
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "minLength": 1,
                "description": "Path relative to the workspace root",
            },
            "old_text": {
                "type": "string",
                "minLength": 1,
                "description": "Existing text that must match exactly",
            },
            "new_text": {
                "type": "string",
                "description": "Replacement text; may be empty",
            },
            "patch": {
                "type": "string",
                "minLength": 1,
                "description": (
                    "Single-file unified patch, optionally wrapped in "
                    "*** Begin Patch / *** Update File markers"
                ),
            },
            "expected_replacements": {
                "type": "integer",
                "minimum": 1,
                "maximum": 100,
                "default": 1,
            },
        },
        "oneOf": [
            {"required": ["file_path", "old_text", "new_text"]},
            {"required": ["patch"]},
        ],
        "additionalProperties": False,
    }

    def __init__(self, project_root: str | Path = ".") -> None:
        self._project_root = Path(project_root).resolve()
        self._writer = WriteFileTool(project_root=self._project_root)

    async def execute(
        self,
        file_path: str | None = None,
        old_text: str | None = None,
        new_text: str | None = None,
        expected_replacements: int = 1,
        patch: str | None = None,
        **_: Any,
    ) -> ToolResult:
        started = time.monotonic()
        try:
            if file_path is None:
                if patch is None:
                    raise ValueError("file_path is required for exact text replacement")
                file_path = _infer_single_patch_path(patch)
            if "\x00" in file_path:
                raise ValueError("Path cannot contain a null byte")
            raw = Path(file_path)
            if raw.is_absolute() or raw.root:
                raise ValueError("Only project-relative paths are allowed")
            target = (self._project_root / raw).resolve()
            target.relative_to(self._project_root)
            if ".git" in target.relative_to(self._project_root).parts:
                raise ValueError("Cannot modify files inside .git")
            if not target.is_file():
                raise FileNotFoundError(f"File not found: {file_path}")
            original = target.read_text(encoding="utf-8")
            if patch is not None:
                if old_text is not None or new_text is not None:
                    raise ValueError("Provide patch or old_text/new_text, not both")
                try:
                    updated, replacements = _apply_unified_patch(
                        original, patch, file_path
                    )
                except ValueError as exc:
                    if str(exc).startswith("Patch conflict:"):
                        return ToolResult(
                            success=False,
                            data={
                                "file_path": raw.as_posix(),
                                **_closest_patch_context(original, patch),
                            },
                            error_message=str(exc),
                            error_code="PATCH_CONFLICT",
                            retryable=True,
                            suggested_recovery=(
                                "Re-read the reported file near the closest context, "
                                "then regenerate the smallest exact replacement."
                            ),
                            duration_ms=(time.monotonic() - started) * 1000,
                        )
                    raise
                result = await self._writer.execute(
                    file_path=raw.as_posix(), content=updated, mode="modify"
                )
                result.duration_ms = (time.monotonic() - started) * 1000
                if result.success and isinstance(result.data, dict):
                    result.data["replacements"] = replacements
                return result
            if old_text is None or new_text is None:
                raise ValueError("Provide patch or both old_text and new_text")
            matches = original.count(old_text)
            if matches != expected_replacements:
                return ToolResult(
                    success=False,
                    data={
                        "file_path": raw.as_posix(),
                        "expected_replacements": expected_replacements,
                        "actual_matches": matches,
                    },
                    error_message=(
                        f"Patch conflict: expected {expected_replacements} exact match(es), "
                        f"found {matches}; file was not changed"
                    ),
                    error_code="PATCH_CONFLICT",
                    retryable=True,
                    suggested_recovery=(
                        "Re-read the target text and retry with an exact current match."
                    ),
                    duration_ms=(time.monotonic() - started) * 1000,
                )
            updated = original.replace(old_text, new_text, expected_replacements)
            result = await self._writer.execute(
                file_path=raw.as_posix(), content=updated, mode="modify"
            )
            result.duration_ms = (time.monotonic() - started) * 1000
            if result.success and isinstance(result.data, dict):
                result.data["replacements"] = expected_replacements
            return result
        except UnicodeDecodeError as exc:
            return ToolResult(
                success=False,
                error_message=f"File is not UTF-8 text: {exc}",
                error_code="PATCH_DECODE_ERROR",
                duration_ms=(time.monotonic() - started) * 1000,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            return ToolResult(
                success=False,
                error_message=str(exc),
                error_code="PATCH_FAILED",
                duration_ms=(time.monotonic() - started) * 1000,
            )
