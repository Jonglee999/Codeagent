"""RunTerminalTool 单元测试。

使用 mock SafetyChecker + TerminalSandbox 进行测试。
覆盖场景：
- 正常命令执行（安全检查通过 → Docker 执行 → 返回结果）
- 安全检查失败（Layer 1 拦截）
- Docker 不可用（Layer 2 不可用）
- Docker 守护进程错误
- 空命令、超时等边界情况
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from codeagent.gateway.tool_gateway import ToolResult
from codeagent.tools.terminal.run_terminal import RunTerminalTool
from codeagent.tools.terminal.safety_checker import SafetyResult


@pytest.fixture
def mock_safety_checker() -> MagicMock:
    """创建 mock SafetyChecker 实例。"""
    checker = MagicMock()
    # 默认所有命令安全
    checker.check_command.return_value = SafetyResult(
        safe=True,
        reason="Command passed all safety checks",
        risk_level="safe",
    )
    return checker


@pytest.fixture
def mock_sandbox() -> MagicMock:
    """创建 mock TerminalSandbox 实例。"""
    sandbox = MagicMock()
    sandbox.acquire_container.return_value = "sandbox_container_123"
    sandbox.exec_command.return_value = ("hello stdout", "", 0)
    return sandbox


@pytest.fixture
def tool(
    mock_safety_checker: MagicMock,
    mock_sandbox: MagicMock,
) -> RunTerminalTool:
    """创建 RunTerminalTool 实例（全部已 mock）。"""
    return RunTerminalTool(
        sandbox=mock_sandbox,
        safety_checker=mock_safety_checker,
        project_root="/test/project",
    )


class TestRunTerminalToolSuccess:
    """正常执行场景。"""

    @pytest.mark.asyncio
    async def test_simple_command(self, tool: RunTerminalTool) -> None:
        result = await tool.execute(command="python --version")
        assert result.success is True
        assert result.data["stdout"] == "hello stdout"
        assert result.data["exit_code"] == 0
        assert result.data["sandbox_id"] == "sandbox_container_123"
        assert result.data["duration_ms"] >= 0

    @pytest.mark.asyncio
    async def test_command_with_options(self, tool: RunTerminalTool) -> None:
        result = await tool.execute(
            command="pip install requests",
            work_dir="/workspace/project",
            timeout=120,
            network="limited",
        )
        assert result.success is True
        assert result.data["exit_code"] == 0

    @pytest.mark.asyncio
    async def test_non_zero_exit(self, tool: RunTerminalTool, mock_sandbox: MagicMock) -> None:
        mock_sandbox.exec_command.return_value = ("", "file not found", 1)
        result = await tool.execute(command="python nonexistent.py")
        assert result.success is True  # 非零退出码不算工具失败，交由 LLM 判断
        assert result.data["exit_code"] == 1
        assert result.data["stderr"] == "file not found"

    @pytest.mark.asyncio
    async def test_project_root_mounted(self, tool: RunTerminalTool, mock_sandbox: MagicMock) -> None:
        await tool.execute(command="ls -la")
        # 验证 project_root 被正确传递为 volume
        call_kwargs = mock_sandbox.acquire_container.call_args.kwargs
        assert call_kwargs["volumes"] == {"/test/project": {"bind": "/project", "mode": "ro"}}


class TestRunTerminalToolSafetyBlock:
    """安全检查拦截场景。"""

    @pytest.mark.asyncio
    async def test_blocked_command(self, tool: RunTerminalTool, mock_safety_checker: MagicMock) -> None:
        mock_safety_checker.check_command.return_value = SafetyResult(
            safe=False,
            reason="Blacklisted command detected: sudo",
            risk_level="dangerous",
            matched_patterns=["sudo"],
        )
        result = await tool.execute(command="sudo rm -rf /")
        assert result.success is False
        assert "Safety check failed" in (result.error_message or "")
        assert result.error_code == "SAFETY_BLOCKED"

    @pytest.mark.asyncio
    async def test_safety_block_prevents_docker_call(
        self, tool: RunTerminalTool, mock_sandbox: MagicMock, mock_safety_checker: MagicMock
    ) -> None:
        """安全检查失败时不应调用 Docker。"""
        mock_safety_checker.check_command.return_value = SafetyResult(
            safe=False,
            reason="Dangerous pattern",
            risk_level="dangerous",
        )
        await tool.execute(command="rm -rf /")
        mock_sandbox.acquire_container.assert_not_called()
        mock_sandbox.exec_command.assert_not_called()

    @pytest.mark.asyncio
    async def test_safety_block_data(self, tool: RunTerminalTool, mock_safety_checker: MagicMock) -> None:
        mock_safety_checker.check_command.return_value = SafetyResult(
            safe=False,
            reason="Blacklisted command detected: sudo",
            risk_level="dangerous",
            matched_patterns=["sudo"],
        )
        result = await tool.execute(command="sudo something")
        assert result.data["risk_level"] == "dangerous"
        assert "sudo" in result.data["matched_patterns"]


class TestRunTerminalToolSandboxErrors:
    """Docker 沙箱错误处理。"""

    @pytest.mark.asyncio
    async def test_docker_daemon_error(self, tool: RunTerminalTool, mock_sandbox: MagicMock) -> None:
        from codeagent.tools.terminal.sandbox import DockerDaemonError

        mock_sandbox.acquire_container.side_effect = DockerDaemonError("Cannot connect")
        result = await tool.execute(command="echo test")
        assert result.success is False
        assert result.error_code == "DOCKER_DAEMON_ERROR"

    @pytest.mark.asyncio
    async def test_docker_not_available(self, mock_safety_checker: MagicMock) -> None:
        from codeagent.tools.terminal.sandbox import DockerNotAvailableError

        mock_sandbox = MagicMock()
        mock_sandbox.acquire_container.side_effect = DockerNotAvailableError()
        tool = RunTerminalTool(sandbox=mock_sandbox, safety_checker=mock_safety_checker)
        result = await tool.execute(command="echo test")
        assert result.success is False
        assert result.error_code == "DOCKER_NOT_AVAILABLE"

    @pytest.mark.asyncio
    async def test_sandbox_exec_failure(self, tool: RunTerminalTool, mock_sandbox: MagicMock) -> None:
        mock_sandbox.exec_command.side_effect = Exception("Container crashed")
        result = await tool.execute(command="python test.py")
        assert result.success is False
        assert result.error_code == "UNEXPECTED_ERROR"

    @pytest.mark.asyncio
    async def test_release_called_after_exec(
        self, tool: RunTerminalTool, mock_sandbox: MagicMock
    ) -> None:
        """验证执行后容器被正确释放到温池。"""
        await tool.execute(command="echo test")
        mock_sandbox.release_container.assert_called_once_with("sandbox_container_123")

    @pytest.mark.asyncio
    async def test_release_called_after_exec_failure(
        self, tool: RunTerminalTool, mock_sandbox: MagicMock
    ) -> None:
        """验证执行失败后容器仍被释放。"""
        mock_sandbox.exec_command.side_effect = Exception("crash")
        await tool.execute(command="echo test")
        mock_sandbox.release_container.assert_called_once()


class TestRunTerminalToolEdgeCases:
    """边界情况测试。"""

    @pytest.mark.asyncio
    async def test_empty_command(self, tool: RunTerminalTool) -> None:
        result = await tool.execute(command="")
        assert result.success is False
        assert result.error_code == "EMPTY_COMMAND"

    @pytest.mark.asyncio
    async def test_whitespace_command(self, tool: RunTerminalTool) -> None:
        result = await tool.execute(command="   ")
        assert result.success is False
        assert result.error_code == "EMPTY_COMMAND"

    @pytest.mark.asyncio
    async def test_missing_command_param(self, tool: RunTerminalTool) -> None:
        result = await tool.execute()
        assert result.success is False

    @pytest.mark.asyncio
    async def test_validate_params(self, tool: RunTerminalTool) -> None:
        assert tool.validate_params(command="echo test") is True
        # command 是必须的
        assert tool.validate_params() is False
        # network 只接受枚举值
        assert tool.validate_params(command="echo test", network="none") is True
        assert tool.validate_params(command="echo test", network="invalid") is False

    @pytest.mark.asyncio
    async def test_sandbox_release_on_cleanup(
        self, tool: RunTerminalTool, mock_sandbox: MagicMock
    ) -> None:
        """即使在异常路径也确保 release 被调用。"""
        mock_sandbox.exec_command.side_effect = Exception("unexpected")
        result = await tool.execute(command="echo hello")
        assert result.success is False
        mock_sandbox.release_container.assert_called_once()
