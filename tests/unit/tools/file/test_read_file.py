"""ReadFileTool 单元测试。"""

from __future__ import annotations

import os

import pytest

from codeagent.tools.file.read_file import ReadFileTool


@pytest.fixture
def tool(tmp_path):
    """在临时目录创建 ReadFileTool 实例。"""
    return ReadFileTool(project_root=tmp_path)


@pytest.fixture
def sample_file(tmp_path):
    """创建一个包含多行内容的示例文件。"""
    lines = [f"Line {i+1}\n" for i in range(20)]
    file_path = tmp_path / "sample.txt"
    file_path.write_text("".join(lines), encoding="utf-8")
    return file_path


class TestReadNormal:
    """正常读取场景。"""

    @pytest.mark.asyncio
    async def test_read_full_file(self, tool, sample_file):
        result = await tool.execute(file_path=str(sample_file.name))
        assert result.success is True
        assert result.data["total_lines"] == 20
        assert result.data["read_range"] == {"start": 1, "end": 20}
        assert "Line 1" in result.data["content"]
        assert "Line 20" in result.data["content"]

    @pytest.mark.asyncio
    async def test_read_with_range(self, tool, sample_file):
        result = await tool.execute(
            file_path=str(sample_file.name),
            start_line=5,
            end_line=10,
        )
        assert result.success is True
        assert result.data["read_range"] == {"start": 5, "end": 10}
        assert "Line 5" in result.data["content"]
        assert "Line 10" in result.data["content"]
        assert "Line 4" not in result.data["content"]
        assert "Line 11" not in result.data["content"]

    @pytest.mark.asyncio
    async def test_read_start_only(self, tool, sample_file):
        """不传 end_line 应读取到文件末尾。"""
        result = await tool.execute(
            file_path=str(sample_file.name), start_line=18
        )
        assert result.success is True
        assert result.data["read_range"] == {"start": 18, "end": 20}

    @pytest.mark.asyncio
    async def test_read_auto_language_detection(self, tool, tmp_path):
        py_file = tmp_path / "hello.py"
        py_file.write_text("print('hello')")
        result = await tool.execute(file_path="hello.py")
        assert result.data["language"] == "python"

    @pytest.mark.asyncio
    async def test_read_unknown_extension(self, tool, tmp_path):
        f = tmp_path / "data.xyz"
        f.write_text("unknown")
        result = await tool.execute(file_path="data.xyz")
        assert result.data["language"] == "text"


class TestReadSensitiveFiles:
    """敏感文件拒绝场景。"""

    @pytest.fixture(autouse=True)
    def _create_sensitive_files(self, tmp_path):
        for name in [".env", "credentials.txt", "secret.key", "id_rsa.pem"]:
            (tmp_path / name).write_text("secret")

    @pytest.mark.asyncio
    async def test_reject_env_file(self, tool):
        result = await tool.execute(file_path=".env")
        assert result.success is False
        assert result.error_code == "SENSITIVE_FILE"

    @pytest.mark.asyncio
    async def test_reject_key_file(self, tool):
        result = await tool.execute(file_path="secret.key")
        assert result.success is False
        assert result.error_code == "SENSITIVE_FILE"

    @pytest.mark.asyncio
    async def test_reject_pem_file(self, tool):
        result = await tool.execute(file_path="id_rsa.pem")
        assert result.success is False
        assert result.error_code == "SENSITIVE_FILE"

    @pytest.mark.asyncio
    async def test_reject_credentials_file(self, tool):
        result = await tool.execute(file_path="credentials.txt")
        assert result.success is False
        assert result.error_code == "SENSITIVE_FILE"


class TestReadPathTraversal:
    """路径遍历攻击拒绝场景。"""

    @pytest.mark.asyncio
    async def test_reject_simple_traversal(self, tool, tmp_path):
        result = await tool.execute(file_path="../outside.txt")
        assert result.success is False
        assert result.error_code == "PATH_TRAVERSAL"

    @pytest.mark.asyncio
    async def test_reject_deep_traversal(self, tool):
        result = await tool.execute(file_path="a/../../../../etc/passwd")
        assert result.success is False
        assert result.error_code == "PATH_TRAVERSAL"

    @pytest.mark.asyncio
    async def test_reject_absolute_path(self, tool):
        result = await tool.execute(file_path="/etc/passwd")
        assert result.success is False
        assert result.error_code == "PATH_TRAVERSAL"


class TestReadEdgeCases:
    """边缘场景。"""

    @pytest.mark.asyncio
    async def test_file_not_found(self, tool):
        result = await tool.execute(file_path="nonexistent.py")
        assert result.success is False
        assert result.error_code == "FILE_NOT_FOUND"

    @pytest.mark.asyncio
    async def test_empty_file(self, tool, tmp_path):
        empty = tmp_path / "empty.txt"
        empty.write_text("")
        result = await tool.execute(file_path="empty.txt")
        assert result.success is True
        assert result.data["total_lines"] == 0
        assert result.data["content"] == ""

    @pytest.mark.asyncio
    async def test_invalid_start_line_large(self, tool, sample_file):
        """start_line 超过总行数，end_line 自动收缩到最大行。"""
        result = await tool.execute(
            file_path=str(sample_file.name),
            start_line=100,
            end_line=200,
        )
        assert result.success is True
        # end_line 会被限制到 total_lines
        assert result.data["read_range"]["end"] == 20
        assert result.data["content"] == ""

    @pytest.mark.asyncio
    async def test_start_line_less_than_one(self, tool, sample_file):
        """start_line < 1 应被修正为 1。"""
        result = await tool.execute(
            file_path=str(sample_file.name), start_line=-5, end_line=5
        )
        assert result.success is True
        assert result.data["read_range"]["start"] == 1

    @pytest.mark.asyncio
    async def test_start_gt_end(self, tool, sample_file):
        """start_line > end_line 应返回错误。"""
        result = await tool.execute(
            file_path=str(sample_file.name), start_line=10, end_line=5
        )
        assert result.success is False
        assert result.error_code == "INVALID_RANGE"

    @pytest.mark.asyncio
    async def test_binary_file_decode_error(self, tool, tmp_path):
        """二进制文件应返回解码错误。"""
        bin_file = tmp_path / "data.bin"
        bin_file.write_bytes(bytes(range(256)))
        result = await tool.execute(file_path="data.bin")
        assert result.success is False
        assert result.error_code == "DECODE_ERROR"

    @pytest.mark.asyncio
    async def test_read_directory(self, tool, tmp_path):
        """读取目录应返回错误。"""
        result = await tool.execute(file_path=".")
        assert result.success is False
        assert result.error_code == "NOT_A_FILE"

    @pytest.mark.asyncio
    async def test_invalid_path_returns_error(self, tool):
        """无效路径应返回 INVALID_PATH。"""
        import os
        # 使用 NUL 设备路径（Windows）或 /dev/null（Unix）来触发路径解析异常
        result = await tool.execute(file_path="\0invalid")
        assert result.success is False
        assert result.error_code == "INVALID_PATH"

    @pytest.mark.asyncio
    async def test_custom_encoding(self, tool, tmp_path):
        """指定编码读取文件。"""
        f = tmp_path / "latin1.txt"
        f.write_bytes("café".encode("latin-1"))
        result = await tool.execute(file_path="latin1.txt", encoding="latin-1")
        assert result.success is True
        assert "café" in result.data["content"]

    @pytest.mark.asyncio
    async def test_wrong_encoding_returns_decode_error(self, tool, tmp_path):
        """使用错误编码读取应返回 DECODE_ERROR。"""
        f = tmp_path / "utf8.txt"
        f.write_text("hello 世界", encoding="utf-8")
        result = await tool.execute(file_path="utf8.txt", encoding="ascii")
        assert result.success is False
        assert result.error_code == "DECODE_ERROR"


class TestReadLargeFile:
    """大文件读取限制。"""

    @pytest.mark.asyncio
    async def test_file_truncated_above_2000_lines(self, tool, tmp_path):
        lines = [f"Line {i+1}\n" for i in range(2500)]
        big_file = tmp_path / "big.txt"
        big_file.write_text("".join(lines))
        result = await tool.execute(file_path="big.txt")
        assert result.success is True
        # 应只读取 2000 行
        content_lines = result.data["content"].splitlines()
        assert len(content_lines) <= 2000
        assert "warning" in result.data
        assert result.data["total_lines"] == 2500
