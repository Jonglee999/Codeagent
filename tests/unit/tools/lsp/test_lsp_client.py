"""LspClient 单元测试。

Mock ruff 子进程，测试诊断解析、ast.parse 降级、边界情况。
"""

from __future__ import annotations

import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from codeagent.tools.lsp.lsp_client import LspClient, LspClientPool


class _MockProcess:
    """Mock asyncio subprocess 返回值。"""

    def __init__(
        self,
        stdout_bytes: bytes = b"",
        stderr_bytes: bytes = b"",
        returncode: int = 0,
    ) -> None:
        self.stdout_bytes = stdout_bytes
        self.stderr_bytes = stderr_bytes
        self.returncode = returncode

    async def communicate(self) -> tuple[bytes, bytes]:
        return (self.stdout_bytes, self.stderr_bytes)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def py_lsp() -> LspClient:
    """创建 Python LspClient 实例。"""
    return LspClient("python")


@pytest.fixture
def ts_lsp() -> LspClient:
    """创建 TypeScript LspClient 实例。"""
    return LspClient("typescript")


@pytest.fixture
def valid_py_file(tmp_path) -> str:
    """创建无问题的 Python 文件。"""
    f = tmp_path / "valid.py"
    f.write_text("x = 1\n")
    return str(f)


@pytest.fixture
def invalid_py_file(tmp_path) -> str:
    """创建有语法错误的 Python 文件。"""
    f = tmp_path / "invalid.py"
    f.write_text("def foo(:\n    pass\n")
    return str(f)


# ── Ruff 诊断测试 ─────────────────────────────────────────────────────────────


class TestRuffDiagnostics:
    """Ruff CLI 诊断测试。"""

    @pytest.mark.asyncio
    async def test_ruff_returns_diagnostics(self, py_lsp: LspClient, tmp_path) -> None:
        """ruff 返回诊断时应正确解析。"""
        f = tmp_path / "app.py"
        f.write_text("import os\nimport sys\n\nx = 1\n")
        ruff_output = json.dumps([
            {
                "code": "F401",
                "column": 1,
                "end_column": 9,
                "end_line": 1,
                "filename": str(f),
                "location": {"row": 1, "column": 1},
                "message": "`os` imported but unused",
                "noqa_row": 1,
            },
            {
                "code": "F401",
                "column": 1,
                "end_column": 10,
                "end_line": 2,
                "filename": str(f),
                "location": {"row": 2, "column": 1},
                "message": "`sys` imported but unused",
                "noqa_row": 2,
            },
        ])

        with patch("asyncio.create_subprocess_exec") as mock_sub:
            mock_sub.return_value = _MockProcess(
                stdout_bytes=ruff_output.encode(),
                returncode=1,
            )
            results = await py_lsp.get_diagnostics(str(f))

        assert len(results) == 2
        assert results[0].code == "F401"
        assert results[0].severity == "error"  # F prefix → error
        assert results[0].source == "ruff"
        assert results[0].message == "`os` imported but unused"
        assert results[1].code == "F401"

    @pytest.mark.asyncio
    async def test_ruff_no_issues(self, py_lsp: LspClient, valid_py_file: str) -> None:
        """ruff 无问题时返回空列表。"""
        with patch("asyncio.create_subprocess_exec") as mock_sub:
            mock_sub.return_value = _MockProcess(
                stdout_bytes=b"[]",
                returncode=0,
            )
            results = await py_lsp.get_diagnostics(valid_py_file)

        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_ruff_code_severity_mapping(self, py_lsp: LspClient, tmp_path) -> None:
        """ruff 代码前缀应正确映射到严重级别。"""
        f = tmp_path / "app.py"
        f.write_text("import os\n")
        ruff_output = json.dumps([
            {"code": "F401", "filename": str(f), "location": {"row": 1, "column": 1},
             "message": "test error"},
            {"code": "W292", "filename": str(f), "location": {"row": 1, "column": 1},
             "message": "test warning"},
            {"code": "D100", "filename": str(f), "location": {"row": 1, "column": 1},
             "message": "test info"},
        ])

        with patch("asyncio.create_subprocess_exec") as mock_sub:
            mock_sub.return_value = _MockProcess(
                stdout_bytes=ruff_output.encode(),
                returncode=1,
            )
            results = await py_lsp.get_diagnostics(str(f))

        assert len(results) == 3
        assert results[0].severity == "error"    # F401
        assert results[1].severity == "warning"  # W292
        assert results[2].severity == "info"     # D100

    @pytest.mark.asyncio
    async def test_diagnostic_to_dict(self, py_lsp: LspClient, tmp_path) -> None:
        """Diagnostic.to_dict() 应返回正确结构。"""
        import json

        from codeagent.tools.lsp.lsp_client import Diagnostic

        d = Diagnostic(
            file_path="test.py",
            line=10,
            column=5,
            message="test",
            severity="error",
            source="ruff",
            code="F401",
        )
        result = d.to_dict()
        assert result["file_path"] == "test.py"
        assert result["line"] == 10
        assert result["column"] == 5
        assert result["severity"] == "error"
        assert result["code"] == "F401"

        # 无 code 时不应包含 code 字段
        d2 = Diagnostic(
            file_path="test.py",
            line=1, column=1, message="test",
            severity="error", source="ast",
        )
        result2 = d2.to_dict()
        assert "code" not in result2


# ── AST 降级测试 ──────────────────────────────────────────────────────────────


class TestAstFallback:
    """当 ruff 不可用时，ast.parse 降级测试。"""

    @pytest.mark.asyncio
    async def test_ruff_not_found_fallback_to_ast(self, py_lsp: LspClient, invalid_py_file: str) -> None:
        """ruff 未安装时自动降级到 ast.parse。"""
        with patch("asyncio.create_subprocess_exec") as mock_sub:
            mock_sub.side_effect = FileNotFoundError("ruff not found")
            results = await py_lsp.get_diagnostics(invalid_py_file)

        assert len(results) >= 1
        assert results[0].severity == "error"
        assert results[0].source == "ast"

    @pytest.mark.asyncio
    async def test_ast_catches_syntax_error(self, py_lsp: LspClient, invalid_py_file: str) -> None:
        """ast.parse 应捕获语法错误。"""
        with patch("asyncio.create_subprocess_exec") as mock_sub:
            mock_sub.side_effect = FileNotFoundError("ruff not found")
            results = await py_lsp.get_diagnostics(invalid_py_file)

        assert len(results) == 1
        assert results[0].severity == "error"
        assert results[0].source == "ast"
        assert results[0].code == "syntax-error"

    @pytest.mark.asyncio
    async def test_ast_valid_file_no_diagnostics(self, py_lsp: LspClient, valid_py_file: str) -> None:
        """ast.parse 通过时无诊断。"""
        with patch("asyncio.create_subprocess_exec") as mock_sub:
            mock_sub.side_effect = FileNotFoundError("ruff not found")
            results = await py_lsp.get_diagnostics(valid_py_file)

        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_ast_file_not_found(self, py_lsp: LspClient) -> None:
        """ast.parse 遇不到文件时返回文件不存在诊断。"""
        with patch("asyncio.create_subprocess_exec") as mock_sub:
            mock_sub.side_effect = FileNotFoundError("ruff not found")
            results = await py_lsp.get_diagnostics("/nonexistent/file.py")

        assert len(results) == 1
        assert "not found" in results[0].message.lower()


# ── 非 Python 语言测试 ─────────────────────────────────────────────────────────


class TestOtherLanguages:
    """非 Python 语言诊断测试。"""

    @pytest.mark.asyncio
    async def test_typescript_returns_empty(self, ts_lsp: LspClient, tmp_path) -> None:
        """TypeScript 文件暂时返回空列表。"""
        f = tmp_path / "app.ts"
        f.write_text("const x: number = 1;\n")
        results = await ts_lsp.get_diagnostics(str(f))
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_unknown_language_returns_empty(self, tmp_path) -> None:
        """未知语言返回空列表。"""
        client = LspClient("unknown")
        f = tmp_path / "file.xyz"
        f.write_text("content")
        results = await client.get_diagnostics(str(f))
        assert len(results) == 0


# ── LspClient 生命周期测试 ────────────────────────────────────────────────────


class TestLspClientLifecycle:
    """LspClient 生命周期测试。"""

    def test_idle_timeout(self) -> None:
        """空闲超时应正确报告。"""
        client = LspClient("python")
        assert client.is_idle is False  # 刚创建，未超时

        # 模拟过去的时间
        client._last_used = time.monotonic() - 700  # > 600s
        assert client.is_idle is True

    @pytest.mark.asyncio
    async def test_shutdown(self) -> None:
        """shutdown 后应标记为已关闭。"""
        client = LspClient("python")
        assert client.is_closed is False
        await client.shutdown()
        assert client.is_closed is True

    @pytest.mark.asyncio
    async def test_open_change_close_noop(self, py_lsp: LspClient) -> None:
        """open_file/change_file/close_file 不应抛出异常。"""
        await py_lsp.open_file("test.py")
        await py_lsp.change_file("test.py", "content")
        await py_lsp.close_file("test.py")
        # 不应抛出任何异常


# ── LspClientPool 测试 ────────────────────────────────────────────────────────


class TestLspClientPool:
    """LspClientPool 测试。"""

    def test_get_client_creates_new(self) -> None:
        """get_client 应创建新客户端。"""
        pool = LspClientPool()
        client = pool.get_client("python")
        assert client is not None
        assert client.language == "python"

    def test_get_client_reuses_existing(self) -> None:
        """get_client 应复用已有客户端。"""
        pool = LspClientPool()
        c1 = pool.get_client("python")
        c2 = pool.get_client("python")
        assert c1 is c2

    def test_get_client_different_languages(self) -> None:
        """不同语言应返回不同客户端。"""
        pool = LspClientPool()
        py = pool.get_client("python")
        ts = pool.get_client("typescript")
        assert py is not ts
        assert py.language == "python"
        assert ts.language == "typescript"

    @pytest.mark.asyncio
    async def test_shutdown_all(self) -> None:
        """shutdown_all 应关闭所有客户端。"""
        pool = LspClientPool()
        pool.get_client("python")
        pool.get_client("typescript")
        assert len(pool._clients) == 2

        await pool.shutdown_all()
        assert len(pool._clients) == 0

    def test_cleanup_idle(self) -> None:
        """cleanup_idle 应移除超时客户端。"""
        pool = LspClientPool()
        client = pool.get_client("python")
        client._last_used = time.monotonic() - 700  # idle

        cleaned = pool.cleanup_idle()
        assert cleaned == 1
        assert "python" not in pool._clients

    def test_cleanup_idle_skip_active(self) -> None:
        """cleanup_idle 不应移除活跃客户端。"""
        pool = LspClientPool()
        pool.get_client("python")
        cleaned = pool.cleanup_idle()
        assert cleaned == 0

    def test_get_stats(self) -> None:
        """get_stats 应返回正确的统计信息。"""
        pool = LspClientPool()
        pool.get_client("python")
        pool.get_client("typescript")

        stats = pool.get_stats()
        assert stats["total_clients"] == 2
        assert "python" in stats["languages"]
        assert "typescript" in stats["languages"]



