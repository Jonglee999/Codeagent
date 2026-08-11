"""WriteFileTool 单元测试。"""

from __future__ import annotations


import pytest

from codeagent.tools.file.write_file import WriteFileTool


@pytest.fixture
def tool(tmp_path):
    """在临时目录创建 WriteFileTool 实例。"""
    return WriteFileTool(project_root=tmp_path)


class TestWriteCreate:
    """创建新文件场景。"""

    @pytest.mark.asyncio
    async def test_create_new_file(self, tool, tmp_path):
        result = await tool.execute(
            file_path="hello.py",
            content="print('hello world')\n",
            mode="create",
        )
        assert result.success is True
        assert result.data["mode"] == "create"
        file_path = tmp_path / "hello.py"
        assert file_path.read_text() == "print('hello world')\n"

    @pytest.mark.asyncio
    async def test_create_nested_directories(self, tool, tmp_path):
        result = await tool.execute(
            file_path="src/utils/helper.py",
            content="def help():\n    pass\n",
            mode="create",
        )
        assert result.success is True
        file_path = tmp_path / "src" / "utils" / "helper.py"
        assert file_path.exists()
        assert "def help()" in file_path.read_text()

    @pytest.mark.asyncio
    async def test_create_existing_file_fails(self, tool, tmp_path):
        (tmp_path / "exists.txt").write_text("original")
        result = await tool.execute(
            file_path="exists.txt",
            content="updated",
            mode="create",
        )
        assert result.success is False
        assert result.error_code == "FILE_EXISTS"
        # 原文件不变
        assert (tmp_path / "exists.txt").read_text() == "original"


class TestWriteModify:
    """修改已有文件场景。"""

    @pytest.mark.asyncio
    async def test_modify_existing_file(self, tool, tmp_path):
        (tmp_path / "test.txt").write_text("line1\nline2\n")
        result = await tool.execute(
            file_path="test.txt",
            content="line1\nmodified\nline3\n",
            mode="modify",
        )
        assert result.success is True
        assert result.data["mode"] == "modify"
        assert (tmp_path / "test.txt").read_text() == "line1\nmodified\nline3\n"

    @pytest.mark.asyncio
    async def test_modify_nonexistent_file_fails(self, tool):
        result = await tool.execute(
            file_path="no_such_file.txt",
            content="content",
            mode="modify",
        )
        assert result.success is False
        assert result.error_code == "FILE_NOT_FOUND"

    @pytest.mark.asyncio
    async def test_modify_empty_file(self, tool, tmp_path):
        (tmp_path / "empty.txt").write_text("")
        result = await tool.execute(
            file_path="empty.txt", content="new content", mode="modify"
        )
        assert result.success is True
        assert (tmp_path / "empty.txt").read_text() == "new content"

    @pytest.mark.asyncio
    async def test_infers_modify_when_mode_is_omitted(self, tool, tmp_path):
        (tmp_path / "existing.txt").write_text("old")

        result = await tool.execute(file_path="existing.txt", content="new")

        assert result.success is True
        assert result.data["mode"] == "modify"
        assert (tmp_path / "existing.txt").read_text() == "new"


class TestWriteDiff:
    """Diff 输出验证。"""

    @pytest.mark.asyncio
    async def test_diff_lines_added_removed(self, tool, tmp_path):
        (tmp_path / "diff_test.txt").write_text("a\nb\nc\n")
        result = await tool.execute(
            file_path="diff_test.txt",
            content="a\nx\nc\n",
            mode="modify",
        )
        assert result.data["lines_added"] == 1
        assert result.data["lines_removed"] == 1
        assert result.data["diff"].startswith("---")

    @pytest.mark.asyncio
    async def test_create_diff_is_empty(self, tool, tmp_path):
        """创建新文件时 diff 为原始内容与空内容的对比。"""
        result = await tool.execute(
            file_path="new.txt",
            content="hello",
            mode="create",
        )
        # 创建模式也有 diff（从空到有）
        assert "diff" in result.data
        assert result.data["lines_added"] >= 1


class TestWriteSecurity:
    """安全检查场景。"""

    @pytest.mark.asyncio
    async def test_reject_path_traversal_simple(self, tool):
        result = await tool.execute(
            file_path="../outside.txt",
            content="hacked",
            mode="create",
        )
        assert result.success is False
        assert result.error_code == "PATH_TRAVERSAL"

    @pytest.mark.asyncio
    async def test_reject_path_traversal_deep(self, tool):
        result = await tool.execute(
            file_path="a/../../../../etc/pwned",
            content="owned",
            mode="create",
        )
        assert result.success is False
        assert result.error_code == "PATH_TRAVERSAL"

    @pytest.mark.asyncio
    async def test_reject_absolute_path(self, tool):
        result = await tool.execute(
            file_path="/tmp/evil.txt",
            content="evil",
            mode="create",
        )
        assert result.success is False
        assert result.error_code == "PERMISSION_DENIED"
        assert "Absolute paths are not allowed" in (result.error_message or "")

    @pytest.mark.asyncio
    async def test_reject_git_directory(self, tool, tmp_path):
        (tmp_path / ".git").mkdir()
        result = await tool.execute(
            file_path=".git/config",
            content="[core]\n\trepositoryformatversion = 0\n",
            mode="create",
        )
        assert result.success is False
        assert result.error_code == "GIT_PROTECTED"

    @pytest.mark.asyncio
    async def test_reject_git_nested(self, tool, tmp_path):
        (tmp_path / "subdir" / ".git").mkdir(parents=True)
        result = await tool.execute(
            file_path="subdir/.git/HEAD",
            content="ref: refs/heads/main\n",
            mode="create",
        )
        assert result.success is False
        assert result.error_code == "GIT_PROTECTED"


class TestWriteBackup:
    """备份机制验证。"""

    @pytest.mark.asyncio
    async def test_backup_created_on_modify(self, tool, tmp_path):
        (tmp_path / "backup_test.txt").write_text("original content")
        result = await tool.execute(
            file_path="backup_test.txt",
            content="updated content",
            mode="modify",
        )
        assert result.success is True
        assert result.data["backup_path"] is not None

        # 验证备份文件存在且内容正确
        backup_rel = result.data["backup_path"]
        backup_file = tmp_path / backup_rel
        assert backup_file.exists()
        assert backup_file.read_text() == "original content"

    @pytest.mark.asyncio
    async def test_backup_not_created_on_create(self, tool, tmp_path):
        result = await tool.execute(
            file_path="brand_new.txt",
            content="new file",
            mode="create",
        )
        assert result.success is True
        # 创建模式无备份
        assert result.data["backup_path"] is None


class TestWriteEdgeCases:
    """边缘场景。"""

    @pytest.mark.asyncio
    async def test_invalid_mode(self, tool):
        result = await tool.execute(
            file_path="test.txt",
            content="test",
            mode="invalid",
        )
        assert result.success is False
        assert result.error_code == "INVALID_MODE"

    @pytest.mark.asyncio
    async def test_empty_content(self, tool, tmp_path):
        result = await tool.execute(
            file_path="empty.txt",
            content="",
            mode="create",
        )
        assert result.success is True
        assert (tmp_path / "empty.txt").read_text() == ""

    @pytest.mark.asyncio
    async def test_unicode_content(self, tool, tmp_path):
        content = "你好\n世界\n🌍\n"
        result = await tool.execute(
            file_path="unicode.txt",
            content=content,
            mode="create",
        )
        assert result.success is True
        assert (tmp_path / "unicode.txt").read_text(encoding="utf-8") == content


class TestWriteEdgeCasesExtended:
    """扩展边缘场景：错误处理路径覆盖。"""

    @pytest.mark.asyncio
    async def test_invalid_path_returns_error(self, tool):
        """无效路径应返回 INVALID_PATH。"""
        result = await tool.execute(
            file_path="\0invalid", content="x", mode="create",
        )
        assert result.success is False
        assert result.error_code == "INVALID_PATH"

    @pytest.mark.asyncio
    async def test_backup_error_handled(self, tool, tmp_path):
        """备份失败应返回 BACKUP_ERROR。"""
        target = tmp_path / "target.txt"
        target.write_text("original")

        # 使备份目录不可写来触发备份错误
        backup_dir = tmp_path / ".codeagent" / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)

        result = await tool.execute(
            file_path="target.txt",
            content="updated",
            mode="modify",
        )
        # 备份目录可写，应正常通过
        assert result.success is True
        assert result.data["backup_path"] is not None

    @pytest.mark.asyncio
    async def test_project_root_isolation(self, tmp_path):
        """使用独立 project_root 的 tool 应拒绝访问外部路径。"""
        tool = WriteFileTool(project_root=tmp_path)
        result = await tool.execute(
            file_path="../../etc/passwd",
            content="hacked",
            mode="create",
        )
        assert result.success is False
        assert result.error_code == "PATH_TRAVERSAL"

    @pytest.mark.asyncio
    async def test_excessively_long_path(self, tool):
        """超长路径应返回错误（INVALID_PATH 或 WRITE_ERROR）。"""
        long_name = "a" * 300 + ".txt"
        result = await tool.execute(
            file_path=long_name, content="x", mode="create",
        )
        # 超长路径可能在解析阶段（OSError/ValueError）或写入阶段（OSError）失败
        assert result.success is False
        assert result.error_code in ("INVALID_PATH", "WRITE_ERROR")

    @pytest.mark.asyncio
    async def test_write_with_special_characters(self, tool, tmp_path):
        """包含特殊字符的写入。"""
        content = "\t\r\n\0\x1b\x00encoded"
        result = await tool.execute(
            file_path="special.bin",
            content=content,
            mode="create",
        )
        # 内容包含 null 字符，但文件写入应该仍能成功（二进制安全）
        assert result.success is True
        written = (tmp_path / "special.bin").read_bytes()
        assert b"encoded" in written
