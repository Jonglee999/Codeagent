"""GetDiagnosticsTool — 获取文件的 LSP 诊断信息（错误、警告、提示）。"""

from __future__ import annotations

import time
from pathlib import Path

from codeagent.tools.base import BaseTool, ToolResult
from codeagent.tools.lsp.lsp_client import LspClientPool, _detect_language

_VALID_SEVERITIES = frozenset({"error", "warning", "info", "all"})


class GetDiagnosticsTool(BaseTool):
    """获取文件的 LSP 诊断信息（错误、警告、提示）。"""

    name = "get_diagnostics"
    description = "获取文件的 LSP 诊断信息（错误、警告、提示）"
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "文件路径（相对项目根目录）",
            },
            "severity_filter": {
                "type": "string",
                "enum": ["error", "warning", "info", "all"],
                "default": "error",
                "description": "严重级别过滤：error（仅错误）、warning（错误+警告）、info（全部）、all（全部）",
            },
        },
        "required": ["file_path"],
    }
    max_timeout_seconds: int = 60

    def __init__(
        self,
        project_root: str | Path = ".",
        client_pool: LspClientPool | None = None,
    ) -> None:
        """初始化 GetDiagnosticsTool。

        Args:
            project_root: 项目根目录路径。
            client_pool: LspClientPool 实例，省略则自动创建。
        """
        super().__init__()
        self._project_root = Path(project_root).resolve()
        self._client_pool = client_pool or LspClientPool()

    async def execute(  # type: ignore[override]
        self,
        file_path: str,
        severity_filter: str = "error",
    ) -> ToolResult:
        """获取文件诊断信息。

        Args:
            file_path: 文件路径（相对项目根目录）。
            severity_filter: 严重级别过滤。

        Returns:
            ToolResult: 诊断结果。
        """
        start_time = time.monotonic()

        # ── 参数校验 ──────────────────────────────────────
        if severity_filter not in _VALID_SEVERITIES:
            return ToolResult(
                success=False,
                error_message=(
                    f"Invalid severity_filter '{severity_filter}'. "
                    f"Use one of: {', '.join(sorted(_VALID_SEVERITIES))}"
                ),
                error_code="INVALID_SEVERITY",
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

        try:
            target.relative_to(self._project_root)
        except ValueError:
            return ToolResult(
                success=False,
                error_message=f"Path traversal denied: {file_path}",
                error_code="PATH_TRAVERSAL",
                duration_ms=(time.monotonic() - start_time) * 1000,
            )

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

        # ── 获取诊断 ──────────────────────────────────────
        language = _detect_language(file_path)
        client = self._client_pool.get_client(language)

        try:
            diagnostics = await client.get_diagnostics(str(target))
        except Exception as e:
            return ToolResult(
                success=False,
                error_message=f"Diagnostics error: {e}",
                error_code="DIAGNOSTICS_ERROR",
                duration_ms=(time.monotonic() - start_time) * 1000,
            )

        # ── 按严重级别过滤 ────────────────────────────────
        if severity_filter != "all":
            # severity_filter 为 "error" 时只保留 error
            # severity_filter 为 "warning" 时保留 error + warning
            valid_severities = _get_filtered_severities(severity_filter)
            diagnostics = [d for d in diagnostics if d.severity in valid_severities]

        # ── 按类型统计 ────────────────────────────────────
        error_count = sum(1 for d in diagnostics if d.severity == "error")
        warning_count = sum(1 for d in diagnostics if d.severity == "warning")
        info_count = sum(1 for d in diagnostics if d.severity == "info")

        rel_path = str(target.relative_to(self._project_root))

        return ToolResult(
            success=True,
            data={
                "file_path": rel_path,
                "language": language,
                "diagnostics": [d.to_dict() for d in diagnostics],
                "total": len(diagnostics),
                "error_count": error_count,
                "warning_count": warning_count,
                "info_count": info_count,
            },
            duration_ms=(time.monotonic() - start_time) * 1000,
        )


def _get_filtered_severities(severity_filter: str) -> frozenset[str]:
    """根据过滤条件返回包含的严重级别集合。

    - "error" → {"error"}
    - "warning" → {"error", "warning"}
    - "info" → {"error", "warning", "info"}
    - "all" → {"error", "warning", "info"}
    """
    if severity_filter == "error":
        return frozenset({"error"})
    elif severity_filter == "warning":
        return frozenset({"error", "warning"})
    else:  # "info" or "all"
        return frozenset({"error", "warning", "info"})
