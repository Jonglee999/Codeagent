"""GitTool 单元测试。

Mock git 子进程，测试各种操作、安全规则和边界情况。
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from codeagent.tools.git.git_tool import GitTool, _split_args


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
def git_repo(tmp_path) -> str:
    """创建包含 .git 目录的模拟 Git 仓库。"""
    (tmp_path / ".git").mkdir()
    return str(tmp_path)


@pytest.fixture
def tool(git_repo: str) -> GitTool:
    """创建指向 Git 仓库的 GitTool 实例。"""
    return GitTool(project_root=git_repo)


@pytest.fixture
def tool_no_git(tmp_path) -> GitTool:
    """创建指向非 Git 目录的 GitTool 实例。"""
    return GitTool(project_root=tmp_path)


# ── 只读操作测试 ──────────────────────────────────────────────────────────────


class TestReadOnlyActions:
    """只读操作（status, diff, log, show, branch）。"""

    @pytest.mark.asyncio
    async def test_status(self, tool: GitTool) -> None:
        """status 应返回工作区状态。"""
        with patch("asyncio.create_subprocess_exec") as mock_sub:
            mock_sub.return_value = _MockProcess(
                stdout_bytes=b"On branch main\nnothing to commit\n",
            )
            result = await tool.execute(action="status")

        assert result.success is True
        assert "On branch main" in result.data["stdout"]
        assert result.data["readonly"] is True

    @pytest.mark.asyncio
    async def test_diff(self, tool: GitTool) -> None:
        """diff 应返回变更内容。"""
        with patch("asyncio.create_subprocess_exec") as mock_sub:
            mock_sub.return_value = _MockProcess(
                stdout_bytes=b"diff --git a/file.py b/file.py\n-index...\n+new line\n",
            )
            result = await tool.execute(action="diff")

        assert result.success is True
        assert "diff --git" in result.data["stdout"]
        assert result.data["readonly"] is True

    @pytest.mark.asyncio
    async def test_log(self, tool: GitTool) -> None:
        """log 应返回提交历史。"""
        with patch("asyncio.create_subprocess_exec") as mock_sub:
            mock_sub.return_value = _MockProcess(
                stdout_bytes=b"abc1234 feat: add new feature\ndef5678 fix: resolve bug\n",
            )
            result = await tool.execute(action="log", args="--oneline -5")

        assert result.success is True
        assert "abc1234" in result.data["stdout"]

    @pytest.mark.asyncio
    async def test_log_with_count(self, tool: GitTool) -> None:
        """log 支持 args 传递计数参数。"""
        with patch("asyncio.create_subprocess_exec") as mock_sub:
            mock_sub.return_value = _MockProcess(
                stdout_bytes=b"abc1234 feat\n",
            )
            result = await tool.execute(action="log", args="-1")

        assert result.success is True
        assert result.data["args"] == "-1"

    @pytest.mark.asyncio
    async def test_show(self, tool: GitTool) -> None:
        """show 应返回提交详情。"""
        with patch("asyncio.create_subprocess_exec") as mock_sub:
            mock_sub.return_value = _MockProcess(
                stdout_bytes=b"commit abc1234\nAuthor: test\n",
            )
            result = await tool.execute(action="show", args="abc1234")

        assert result.success is True
        assert "commit abc1234" in result.data["stdout"]

    @pytest.mark.asyncio
    async def test_branch(self, tool: GitTool) -> None:
        """branch 应返回分支列表。"""
        with patch("asyncio.create_subprocess_exec") as mock_sub:
            mock_sub.return_value = _MockProcess(
                stdout_bytes=b"* main\n  feature\n",
            )
            result = await tool.execute(action="branch")

        assert result.success is True
        assert "* main" in result.data["stdout"]

    @pytest.mark.asyncio
    async def test_diff_with_path(self, tool: GitTool) -> None:
        """diff 支持文件路径参数。"""
        with patch("asyncio.create_subprocess_exec") as mock_sub:
            mock_sub.return_value = _MockProcess(
                stdout_bytes=b"diff --git a/src/main.py b/src/main.py\n",
            )
            result = await tool.execute(action="diff", args="src/main.py")

        assert result.success is True
        assert result.data["readonly"] is True


# ── 可写操作测试 ──────────────────────────────────────────────────────────────


class TestWriteActions:
    """可写操作（add, commit, checkout, stash）。"""

    @pytest.mark.asyncio
    async def test_add(self, tool: GitTool) -> None:
        """add 应暂存文件。"""
        with patch("asyncio.create_subprocess_exec") as mock_sub:
            mock_sub.return_value = _MockProcess(stdout_bytes=b"")
            result = await tool.execute(action="add", args="src/main.py")

        assert result.success is True
        assert result.data["action"] == "add"
        assert result.data["readonly"] is False

    @pytest.mark.asyncio
    async def test_add_all(self, tool: GitTool) -> None:
        """add 支持通配符。"""
        with patch("asyncio.create_subprocess_exec") as mock_sub:
            mock_sub.return_value = _MockProcess(stdout_bytes=b"")
            result = await tool.execute(action="add", args=".")

        assert result.success is True

    @pytest.mark.asyncio
    async def test_commit(self, tool: GitTool) -> None:
        """commit 应提交暂存区。"""
        with patch("asyncio.create_subprocess_exec") as mock_sub:
            mock_sub.return_value = _MockProcess(
                stdout_bytes=b"[main abc1234] feat: add feature\n",
            )
            result = await tool.execute(
                action="commit", args='-m "feat: add feature"',
            )

        assert result.success is True
        assert "abc1234" in result.data["stdout"]

    @pytest.mark.asyncio
    async def test_commit_empty_message(self, tool: GitTool) -> None:
        """commit 无消息时应返回错误。"""
        result = await tool.execute(action="commit", args="")
        assert result.success is False
        assert result.error_code == "COMMIT_MESSAGE_REQUIRED"

    @pytest.mark.asyncio
    async def test_checkout_branch(self, tool: GitTool) -> None:
        """checkout 应切换分支。"""
        with patch("asyncio.create_subprocess_exec") as mock_sub:
            mock_sub.return_value = _MockProcess(
                stdout_bytes=b"Switched to branch 'feature'\n",
            )
            result = await tool.execute(action="checkout", args="feature")

        assert result.success is True

    @pytest.mark.asyncio
    async def test_stash(self, tool: GitTool) -> None:
        """stash 应暂存工作进度。"""
        with patch("asyncio.create_subprocess_exec") as mock_sub:
            mock_sub.return_value = _MockProcess(
                stdout_bytes=b"Saved working directory and index state WIP on main\n",
            )
            result = await tool.execute(action="stash")

        assert result.success is True
        assert result.data["readonly"] is False


# ── 安全检查测试 ──────────────────────────────────────────────────────────────


class TestSafetyChecks:
    """安全规则测试。"""

    @pytest.mark.asyncio
    async def test_checkout_dot_blocked(self, tool: GitTool) -> None:
        """checkout -- . 应被阻断。"""
        result = await tool.execute(action="checkout", args="-- .")
        assert result.success is False
        assert result.error_code == "CHECKOUT_DOT_BLOCKED"

    @pytest.mark.asyncio
    async def test_checkout_dot_alt_blocked(self, tool: GitTool) -> None:
        """checkout --. 也应被阻断。"""
        result = await tool.execute(action="checkout", args="--.")
        assert result.success is False
        assert result.error_code == "CHECKOUT_DOT_BLOCKED"

    @pytest.mark.asyncio
    async def test_invalid_action(self, tool: GitTool) -> None:
        """无效 action 应返回错误。"""
        result = await tool.execute(action="invalid_op")
        assert result.success is False
        assert result.error_code == "INVALID_ACTION"

    @pytest.mark.asyncio
    async def test_not_git_directory(self, tool_no_git: GitTool) -> None:
        """非 Git 目录应返回错误。"""
        result = await tool_no_git.execute(action="status")
        assert result.success is False
        assert result.error_code == "NOT_A_GIT_REPO"


# ── 禁止操作测试 ──────────────────────────────────────────────────────────────


class TestBlockedActions:
    """明确禁止的操作测试。"""

    @pytest.mark.asyncio
    async def test_push_blocked(self, tool: GitTool) -> None:
        """push 应被阻断。"""
        result = await tool.execute(action="push")
        assert result.success is False
        assert result.error_code == "BLOCKED_ACTION"

    @pytest.mark.asyncio
    async def test_reset_hard_blocked(self, tool: GitTool) -> None:
        """reset 应被阻断。"""
        result = await tool.execute(action="reset")
        assert result.success is False
        assert result.error_code == "BLOCKED_ACTION"

    @pytest.mark.asyncio
    async def test_rebase_blocked(self, tool: GitTool) -> None:
        """rebase 应被阻断。"""
        result = await tool.execute(action="rebase")
        assert result.success is False
        assert result.error_code == "BLOCKED_ACTION"

    @pytest.mark.asyncio
    async def test_merge_blocked(self, tool: GitTool) -> None:
        """merge 应被阻断。"""
        result = await tool.execute(action="merge")
        assert result.success is False
        assert result.error_code == "BLOCKED_ACTION"

    @pytest.mark.asyncio
    async def test_cherry_pick_blocked(self, tool: GitTool) -> None:
        """cherry-pick 应被阻断。"""
        result = await tool.execute(action="cherry-pick")
        assert result.success is False
        assert result.error_code == "BLOCKED_ACTION"


# ── 错误处理测试 ──────────────────────────────────────────────────────────────


class TestErrorHandling:
    """错误处理测试。"""

    @pytest.mark.asyncio
    async def test_git_not_found(self, tool: GitTool) -> None:
        """Git 未安装时应返回友好错误。"""
        with patch("asyncio.create_subprocess_exec") as mock_sub:
            mock_sub.side_effect = FileNotFoundError("git not found")
            result = await tool.execute(action="status")

        assert result.success is False
        assert result.error_code == "GIT_NOT_FOUND"

    @pytest.mark.asyncio
    async def test_git_command_error(self, tool: GitTool) -> None:
        """Git 命令失败时应返回错误。"""
        with patch("asyncio.create_subprocess_exec") as mock_sub:
            mock_sub.return_value = _MockProcess(
                stderr_bytes=b"fatal: unknown option",
                returncode=128,
            )
            result = await tool.execute(action="log", args="--bad-flag")

        assert result.success is False
        assert result.error_code == "GIT_ERROR"
        assert "fatal: unknown option" in result.error_message


# ── 辅助函数测试 ──────────────────────────────────────────────────────────────


class TestSplitArgs:
    """_split_args 辅助函数测试。"""

    def test_simple_args(self) -> None:
        """简单空格分隔参数。"""
        assert _split_args("--oneline -5") == ["--oneline", "-5"]

    def test_quoted_args(self) -> None:
        """带引号参数。"""
        assert _split_args('-m "fix: bug"') == ["-m", "fix: bug"]

    def test_single_quoted_args(self) -> None:
        """单引号参数。"""
        assert _split_args("-m 'fix: bug'") == ["-m", "fix: bug"]

    def test_empty_string(self) -> None:
        """空字符串返回空列表。"""
        assert _split_args("") == []

    def test_mixed_args(self) -> None:
        """混合参数。"""
        result = _split_args('-m "feat: add" --oneline')
        assert result == ["-m", "feat: add", "--oneline"]
