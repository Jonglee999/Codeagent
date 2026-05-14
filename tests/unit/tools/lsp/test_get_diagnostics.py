"""GetDiagnosticsTool 单元测试。

Mock LspClientPool，测试参数校验、路径安全、严重级别过滤。
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from codeagent.tools.lsp.get_diagnostics import GetDiagnosticsTool
from codeagent.tools.lsp.lsp_client import Diagnostic


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_pool() -> MagicMock:
    """创建 mock LspClientPool 实例。"""
    pool = MagicMock()
    mock_client = MagicMock()
    mock_client.get_diagnostics = AsyncMock()
    pool.get_client.return_value = mock_client
    return pool


@pytest.fixture
def tool(mock_pool: MagicMock, tmp_path) -> GetDiagnosticsTool:
    """创建带 mock pool 的 GetDiagnosticsTool 实例。"""
    return GetDiagnosticsTool(project_root=tmp_path, client_pool=mock_pool)


@pytest.fixture
def py_file(tmp_path) -> str:
    """创建示例 Python 文件。"""
    f = tmp_path / "main.py"
    f.write_text("x = 1\n")
    return "main.py"


# ── 正常场景 ──────────────────────────────────────────────────────────────────


class TestGetDiagnosticsNormal:
    """正常获取诊断场景。"""

    @pytest.mark.asyncio
    async def test_basic_diagnostics(
        self, tool: GetDiagnosticsTool, mock_pool: MagicMock, py_file: str,
    ) -> None:
        """基本诊断获取。"""
        mock_client = mock_pool.get_client.return_value
        mock_client.get_diagnostics.return_value = [
            Diagnostic("main.py", 1, 1, "unused import", "error", "ruff", "F401"),
        ]

        result = await tool.execute(file_path=py_file, severity_filter="all")
        assert result.success is True
        assert result.data["total"] == 1
        assert result.data["error_count"] == 1
        assert result.data["diagnostics"][0]["code"] == "F401"

    @pytest.mark.asyncio
    async def test_severity_filter_error(
        self, tool: GetDiagnosticsTool, mock_pool: MagicMock, py_file: str,
    ) -> None:
        """severity_filter=error 只返回 error。"""
        mock_client = mock_pool.get_client.return_value
        mock_client.get_diagnostics.return_value = [
            Diagnostic("main.py", 1, 1, "error msg", "error", "ruff", "F401"),
            Diagnostic("main.py", 2, 1, "warning msg", "warning", "ruff", "W292"),
            Diagnostic("main.py", 3, 1, "info msg", "info", "ruff", "D100"),
        ]

        result = await tool.execute(file_path=py_file, severity_filter="error")
        assert result.success is True
        assert result.data["total"] == 1
        assert result.data["error_count"] == 1
        assert result.data["warning_count"] == 0
        assert result.data["info_count"] == 0

    @pytest.mark.asyncio
    async def test_severity_filter_warning(
        self, tool: GetDiagnosticsTool, mock_pool: MagicMock, py_file: str,
    ) -> None:
        """severity_filter=warning 返回 error + warning。"""
        mock_client = mock_pool.get_client.return_value
        mock_client.get_diagnostics.return_value = [
            Diagnostic("main.py", 1, 1, "error msg", "error", "ruff", "F401"),
            Diagnostic("main.py", 2, 1, "warning msg", "warning", "ruff", "W292"),
            Diagnostic("main.py", 3, 1, "info msg", "info", "ruff", "D100"),
        ]

        result = await tool.execute(file_path=py_file, severity_filter="warning")
        assert result.success is True
        assert result.data["total"] == 2
        assert result.data["error_count"] == 1
        assert result.data["warning_count"] == 1

    @pytest.mark.asyncio
    async def test_severity_filter_all(
        self, tool: GetDiagnosticsTool, mock_pool: MagicMock, py_file: str,
    ) -> None:
        """severity_filter=all 返回全部诊断。"""
        mock_client = mock_pool.get_client.return_value
        mock_client.get_diagnostics.return_value = [
            Diagnostic("main.py", 1, 1, "error", "error", "ruff", "F401"),
            Diagnostic("main.py", 2, 1, "info", "info", "ruff", "D100"),
        ]

        result = await tool.execute(file_path=py_file, severity_filter="all")
        assert result.success is True
        assert result.data["total"] == 2
        assert result.data["error_count"] == 1
        assert result.data["info_count"] == 1

    @pytest.mark.asyncio
    async def test_no_diagnostics(
        self, tool: GetDiagnosticsTool, mock_pool: MagicMock, py_file: str,
    ) -> None:
        """无诊断时返回空列表。"""
        mock_client = mock_pool.get_client.return_value
        mock_client.get_diagnostics.return_value = []

        result = await tool.execute(file_path=py_file, severity_filter="all")
        assert result.success is True
        assert result.data["total"] == 0
        assert result.data["error_count"] == 0
        assert result.data["warning_count"] == 0
        assert result.data["info_count"] == 0

    @pytest.mark.asyncio
    async def test_language_detected(
        self, tool: GetDiagnosticsTool, mock_pool: MagicMock, tmp_path,
    ) -> None:
        """语言应自动检测并传递给 client_pool。"""
        f = tmp_path / "app.ts"
        f.write_text("const x = 1;\n")
        mock_client = mock_pool.get_client.return_value
        mock_client.get_diagnostics.return_value = []

        result = await tool.execute(file_path="app.ts", severity_filter="all")
        assert result.success is True
        assert result.data["language"] == "typescript"

    @pytest.mark.asyncio
    async def test_diagnostics_error(
        self, tool: GetDiagnosticsTool, mock_pool: MagicMock, py_file: str,
    ) -> None:
        """诊断异常时应返回错误。"""
        mock_client = mock_pool.get_client.return_value
        mock_client.get_diagnostics.side_effect = RuntimeError("LSP crashed")

        result = await tool.execute(file_path=py_file, severity_filter="all")
        assert result.success is False
        assert result.error_code == "DIAGNOSTICS_ERROR"


# ── 边界情况 ──────────────────────────────────────────────────────────────────


class TestGetDiagnosticsEdgeCases:
    """边界情况测试。"""

    @pytest.mark.asyncio
    async def test_invalid_severity_filter(
        self, tool: GetDiagnosticsTool, py_file: str,
    ) -> None:
        """无效 severity_filter 应返回错误。"""
        result = await tool.execute(file_path=py_file, severity_filter="critical")
        assert result.success is False
        assert result.error_code == "INVALID_SEVERITY"

    @pytest.mark.asyncio
    async def test_file_not_found(self, tool: GetDiagnosticsTool) -> None:
        """不存在的文件应返回错误。"""
        result = await tool.execute(file_path="nonexistent.py")
        assert result.success is False
        assert result.error_code == "FILE_NOT_FOUND"

    @pytest.mark.asyncio
    async def test_path_traversal(self, tool: GetDiagnosticsTool) -> None:
        """路径遍历应被拒绝。"""
        result = await tool.execute(file_path="../outside.py")
        assert result.success is False
        assert result.error_code == "PATH_TRAVERSAL"

    @pytest.mark.asyncio
    async def test_directory_path(self, tool: GetDiagnosticsTool, tmp_path) -> None:
        """目录路径应返回错误。"""
        result = await tool.execute(file_path=".")
        assert result.success is False
        assert result.error_code == "NOT_A_FILE"

    @pytest.mark.asyncio
    async def test_invalid_path(self, tool: GetDiagnosticsTool) -> None:
        """无效路径应返回错误。"""
        result = await tool.execute(file_path="\0bad")
        assert result.success is False
        assert result.error_code in ("INVALID_PATH", "FILE_NOT_FOUND")
