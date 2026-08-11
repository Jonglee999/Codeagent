"""ReadFileTool — 带安全限制的文件读取工具。"""

from __future__ import annotations

from pathlib import Path

from codeagent.tools.base import BaseTool, ToolResult

# 敏感文件名模式（fnmatch 模式字符串）
_SENSITIVE_PATTERNS = [
    ".env",
    ".env.*",
    "*.key",
    "*.pem",
    "*.p12",
    "credentials*",
    "secrets*",
    "*.secret",
    ".token",
    "*password*",
    "*.cert",
]

# 最大行数限制
_MAX_LINES = 2000
_DEFAULT_LINES = 500

# 扩展名 → 语言映射
_EXTENSION_MAP: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".jsx": "jsx",
    ".rs": "rust",
    ".go": "go",
    ".java": "java",
    ".rb": "ruby",
    ".php": "php",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".swift": "swift",
    ".kt": "kotlin",
    ".scala": "scala",
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "bash",
    ".ps1": "powershell",
    ".sql": "sql",
    ".html": "html",
    ".css": "css",
    ".scss": "scss",
    ".less": "less",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".xml": "xml",
    ".md": "markdown",
    ".rst": "rst",
    ".tex": "latex",
    ".dockerfile": "dockerfile",
    ".txt": "text",
    ".csv": "csv",
    ".ini": "ini",
    ".cfg": "ini",
    ".conf": "ini",
}

# fnmatch 风格的全局匹配（不使用额外的库，简单实现）
import fnmatch  # noqa: E402


def _is_sensitive(file_name: str) -> bool:
    """检查文件名是否匹配敏感文件模式。"""
    return any(fnmatch.fnmatch(file_name, pat) for pat in _SENSITIVE_PATTERNS)


def _is_path_traversal(project_root: Path, resolved: Path) -> bool:
    """检查解析后的路径是否在项目根目录内。"""
    try:
        resolved.relative_to(project_root)
        return False
    except ValueError:
        return True


def _detect_language(file_path: str) -> str:
    """根据文件扩展名推断编程语言。"""
    lower = file_path.lower()
    for ext, lang in _EXTENSION_MAP.items():
        if lower.endswith(ext):
            return lang
    return "text"


class ReadFileTool(BaseTool):
    """读取文件内容，支持指定行范围和安全限制。"""

    name = "read_file"
    category = "exploration"
    read_only = True
    description = "读取文件内容，支持指定行范围"
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "文件路径（相对项目根目录）",
            },
            "start_line": {
                "type": "integer",
                "description": "起始行号（1-indexed，可选）",
            },
            "end_line": {
                "type": "integer",
                "description": "结束行号（含，可选）",
            },
            "encoding": {
                "type": "string",
                "description": "文件编码，默认 utf-8",
                "default": "utf-8",
            },
        },
        "required": ["file_path"],
    }

    def __init__(self, project_root: str | Path = ".") -> None:
        super().__init__()
        self._project_root = Path(project_root).resolve()

    async def execute(  # type: ignore[override]
        self,
        file_path: str,
        start_line: int | None = None,
        end_line: int | None = None,
        encoding: str = "utf-8",
    ) -> ToolResult:
        """读取文件内容。

        Args:
            file_path: 文件路径（相对项目根目录）
            start_line: 起始行号（1-indexed），从第一行开始
            end_line: 结束行号（含），读取到最后一行
            encoding: 文件编码

        Returns:
            ToolResult: 包含文件内容和元数据
        """
        import time

        start_time = time.monotonic()

        if "\0" in file_path:
            return ToolResult(
                success=False,
                error_message="Invalid file path: path contains a null character",
                error_code="INVALID_PATH",
                duration_ms=(time.monotonic() - start_time) * 1000,
            )

        # ── 路径安全校验 ──────────────────────────────────
        try:
            target = (self._project_root / file_path).resolve()
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

        # ── 敏感文件检查 ──────────────────────────────────
        if _is_sensitive(target.name):
            return ToolResult(
                success=False,
                error_message=f"Cannot read sensitive file: {file_path}",
                error_code="SENSITIVE_FILE",
                duration_ms=(time.monotonic() - start_time) * 1000,
            )

        # ── 文件存在性检查 ────────────────────────────────
        if not target.exists():
            return ToolResult(
                success=False,
                error_message=f"File not found: {file_path}",
                error_code="FILE_NOT_FOUND",
                duration_ms=(time.monotonic() - start_time) * 1000,
            )

        if not target.is_file():
            return ToolResult(
                success=False,
                error_message=f"Not a file: {file_path}",
                error_code="NOT_A_FILE",
                duration_ms=(time.monotonic() - start_time) * 1000,
            )

        # ── 读取文件 ──────────────────────────────────────
        try:
            text = target.read_text(encoding=encoding)
        except UnicodeDecodeError:
            return ToolResult(
                success=False,
                error_message=f"Cannot decode file with encoding {encoding}: {file_path}",
                error_code="DECODE_ERROR",
                duration_ms=(time.monotonic() - start_time) * 1000,
            )
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
                error_message=f"Error reading file: {e}",
                error_code="READ_ERROR",
                duration_ms=(time.monotonic() - start_time) * 1000,
            )

        lines = text.splitlines(keepends=True)
        total_lines = len(lines)

        # ── 行范围处理 ────────────────────────────────────
        implicit_end = end_line is None
        if start_line is None:
            start_line = 1
        if end_line is None:
            end_line = min(total_lines, start_line + _DEFAULT_LINES - 1)
        elif end_line > total_lines:
            end_line = total_lines

        if start_line < 1:
            start_line = 1
        if start_line > total_lines:
            # start_line 超出文件范围 → 返回空内容，仍报告文件实际行数
            content = ""
            end_line = total_lines
        elif end_line < start_line:
            return ToolResult(
                success=False,
                error_message="start_line must be <= end_line",
                error_code="INVALID_RANGE",
                duration_ms=(time.monotonic() - start_time) * 1000,
            )

        # 行数限制
        hard_truncated = end_line - start_line + 1 > _MAX_LINES
        if hard_truncated:
            end_line = start_line + _MAX_LINES - 1
            content = "".join(lines[start_line - 1 : end_line])
        else:
            content = "".join(lines[start_line - 1 : end_line])
        truncated = hard_truncated or (implicit_end and end_line < total_lines)

        language = _detect_language(str(target))

        result_data = {
            "content": content,
            "total_lines": total_lines,
            "read_range": {"start": start_line, "end": end_line},
            "file_path": str(target.relative_to(self._project_root)),
            "language": language,
        }

        if truncated:
            result_data["warning"] = (
                f"Partial file read ({start_line}-{end_line} of {total_lines} lines). "
                "Request an explicit line range to continue."
            )
            result_data["next_start_line"] = end_line + 1

        return ToolResult(
            success=True,
            data=result_data,
            duration_ms=(time.monotonic() - start_time) * 1000,
        )
