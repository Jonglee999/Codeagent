"""WriteFileTool — 带安全限制和自动备份的文件写入工具。"""

from __future__ import annotations

import difflib
import time
from datetime import datetime, timezone
from pathlib import Path

from codeagent.tools.base import BaseTool, ToolResult


def _generate_diff(
    original: str, updated: str, file_path: str
) -> tuple[str, int, int]:
    """生成 unified diff 并统计增删行数。

    Returns:
        (diff_string, lines_added, lines_removed)
    """
    orig_lines = original.splitlines(keepends=True)
    new_lines = updated.splitlines(keepends=True)

    diff_lines = list(
        difflib.unified_diff(
            orig_lines,
            new_lines,
            fromfile=f"a/{file_path}",
            tofile=f"b/{file_path}",
        )
    )

    diff_str = "".join(diff_lines)

    # 统计增删行数
    lines_added = sum(
        1 for line in diff_lines if line.startswith("+") and not line.startswith("+++")
    )
    lines_removed = sum(
        1 for line in diff_lines if line.startswith("-") and not line.startswith("---")
    )

    return diff_str, lines_added, lines_removed


def _is_path_traversal(project_root: Path, resolved: Path) -> bool:
    """检查解析后的路径是否在项目根目录内。"""
    try:
        resolved.relative_to(project_root)
        return False
    except ValueError:
        return True


def _is_within_git(resolved: Path) -> bool:
    """检查路径是否在 .git 目录内。"""
    parts = resolved.parts
    return ".git" in parts


class WriteFileTool(BaseTool):
    """创建或修改文件，写入前自动备份、写入后生成 diff。"""

    name = "write_file"
    category = "mutation"
    risk_level = "medium"
    description = "创建或修改文件，写入前自动备份、写入后生成 diff"
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "文件路径（相对项目根目录）",
            },
            "content": {
                "type": "string",
                "description": "文件内容",
            },
            "mode": {
                "type": "string",
                "enum": ["create", "modify"],
                "description": "create: 创建新文件（父目录自动创建）；modify: 修改已有文件",
            },
        },
        "required": ["file_path", "content"],
    }

    def __init__(self, project_root: str | Path = ".") -> None:
        super().__init__()
        self._project_root = Path(project_root).resolve()
        self._backup_root = self._project_root / ".codeagent" / "backups"

    async def execute(  # type: ignore[override]
        self,
        file_path: str,
        content: str,
        mode: str | None = None,
    ) -> ToolResult:
        """创建或修改文件。

        Args:
            file_path: 文件路径（相对项目根目录）
            content: 文件内容
            mode: create（创建新文件）或 modify（修改已有文件）

        Returns:
            ToolResult: 包含写入结果、diff、备份路径等信息
        """
        start_time = time.monotonic()

        if "\0" in file_path:
            return ToolResult(
                success=False,
                error_message="Invalid file path: path contains a null character",
                error_code="INVALID_PATH",
                duration_ms=(time.monotonic() - start_time) * 1000,
            )

        # ── 参数校验 ──────────────────────────────────────
        if mode not in (None, "create", "modify"):
            return ToolResult(
                success=False,
                error_message=f"Invalid mode '{mode}'. Use 'create' or 'modify'.",
                error_code="INVALID_MODE",
                duration_ms=(time.monotonic() - start_time) * 1000,
            )

        # ── 路径安全校验 ──────────────────────────────────
        raw = Path(file_path)
        # 拒绝绝对路径：is_absolute() 覆盖 POSIX 和带驱动器的 Windows 路径，
        # raw.root 覆盖仅以 / 或 \ 开头的路径（在 Windows 上 is_absolute 可能为 False）
        if raw.is_absolute() or raw.root:
            return ToolResult(
                success=False,
                error_message=(
                    f"Absolute paths are not allowed: {file_path}. "
                    "Use paths relative to the project root."
                ),
                error_code="PERMISSION_DENIED",
                duration_ms=(time.monotonic() - start_time) * 1000,
            )

        try:
            target = (self._project_root / raw).resolve()
        except (OSError, RuntimeError, ValueError):
            return ToolResult(
                success=False,
                error_message=f"Invalid file path: {file_path}",
                error_code="INVALID_PATH",
                duration_ms=(time.monotonic() - start_time) * 1000,
            )

        if _is_path_traversal(self._project_root, target):
            return ToolResult(
                success=False,
                error_message=f"Path traversal denied: {file_path}",
                error_code="PATH_TRAVERSAL",
                duration_ms=(time.monotonic() - start_time) * 1000,
            )

        # ── .git 目录保护 ─────────────────────────────────
        if _is_within_git(target):
            return ToolResult(
                success=False,
                error_message="Cannot modify files inside .git directory",
                error_code="GIT_PROTECTED",
                duration_ms=(time.monotonic() - start_time) * 1000,
            )

        # ── 模式检查 ──────────────────────────────────────
        target_exists = target.exists()

        if mode is None:
            mode = "modify" if target_exists else "create"

        if mode == "create" and target_exists:
            return ToolResult(
                success=False,
                error_message=f"File already exists: {file_path}. Use mode='modify' to update.",
                error_code="FILE_EXISTS",
                duration_ms=(time.monotonic() - start_time) * 1000,
            )

        if mode == "modify" and not target_exists:
            return ToolResult(
                success=False,
                error_message=f"File not found: {file_path}. Use mode='create' to create new file.",
                error_code="FILE_NOT_FOUND",
                duration_ms=(time.monotonic() - start_time) * 1000,
            )

        # ── 备份（修改模式） ──────────────────────────────
        backup_path: str | None = None
        original_content = ""

        if mode == "modify" and target_exists:
            try:
                original_content = target.read_text(encoding="utf-8")
                backup_path = self._create_backup(target, original_content)
            except Exception as e:
                return ToolResult(
                    success=False,
                    error_message=f"Backup failed: {e}",
                    error_code="BACKUP_ERROR",
                    duration_ms=(time.monotonic() - start_time) * 1000,
                )

        # ── 写入文件 ──────────────────────────────────────
        try:
            if mode == "create":
                target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        except PermissionError:
            return ToolResult(
                success=False,
                error_message=f"Permission denied: {file_path}",
                error_code="PERMISSION_DENIED",
                duration_ms=(time.monotonic() - start_time) * 1000,
            )
        except OSError as e:
            return ToolResult(
                success=False,
                error_message=f"Error writing file: {e}",
                error_code="WRITE_ERROR",
                duration_ms=(time.monotonic() - start_time) * 1000,
            )

        # ── 生成 diff ──────────────────────────────────────
        rel_path = str(target.relative_to(self._project_root))
        diff_str, added, removed = _generate_diff(original_content, content, rel_path)

        duration = (time.monotonic() - start_time) * 1000

        return ToolResult(
            success=True,
            data={
                "file_path": rel_path,
                "mode": mode,
                "diff": diff_str,
                "backup_path": backup_path,
                "lines_added": added,
                "lines_removed": removed,
            },
            duration_ms=duration,
        )

    def _create_backup(self, target: Path, content: str) -> str:
        """创建文件的备份副本。

        Returns:
            备份文件的相对路径
        """
        self._backup_root.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup_name = f"{timestamp}_{target.name}"
        backup_file = self._backup_root / backup_name

        backup_file.write_text(content, encoding="utf-8")

        return str(backup_file.relative_to(self._project_root))
