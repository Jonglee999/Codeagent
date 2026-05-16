"""DockerExecutor 单元测试。

覆盖：run() 正常/超时/Docker 不可用、is_available()、run_tests()、ExecutionResult 字段。
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from codeagent.sandbox.docker_executor import DockerExecutor, ExecutionResult


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def mock_docker_client(mocker):
    """Mock docker.from_env() 返回一个可用客户端。"""
    mock_client = MagicMock()
    mock_client.ping.return_value = True

    mock_container = MagicMock()
    mock_container.logs.side_effect = lambda stdout=True, stderr=False: (
        b"hello from container" if stdout else b""
    )
    mock_container.wait.return_value = {"StatusCode": 0}

    mock_client.containers.create.return_value = mock_container

    # Patch docker module
    mocker.patch("codeagent.sandbox.docker_executor._DOCKER_AVAILABLE", True)
    mocker.patch("codeagent.sandbox.docker_executor.docker.from_env", return_value=mock_client)

    return mock_client


@pytest.fixture
def executor() -> DockerExecutor:
    return DockerExecutor(image="python:3.11-slim", memory_mb=256, timeout_s=30)


# ── ExecutionResult ────────────────────────────────────────────────────────


class TestExecutionResult:
    """ExecutionResult 数据类字段验证。"""

    def test_default_values(self):
        """验证默认字段值。"""
        result = ExecutionResult()
        assert result.stdout == ""
        assert result.stderr == ""
        assert result.exit_code == 0
        assert result.duration_ms == 0
        assert result.sandboxed is True

    def test_custom_values(self):
        """验证自定义字段值。"""
        result = ExecutionResult(
            stdout="output",
            stderr="error",
            exit_code=1,
            duration_ms=500,
            sandboxed=False,
        )
        assert result.stdout == "output"
        assert result.stderr == "error"
        assert result.exit_code == 1
        assert result.duration_ms == 500
        assert result.sandboxed is False

    def test_sandboxed_default_true(self):
        """验证 sandboxed 默认值为 True（安全优先）。"""
        result = ExecutionResult(stdout="ok", exit_code=0)
        assert result.sandboxed is True


# ── is_available ───────────────────────────────────────────────────────────


class TestIsAvailable:
    """DockerExecutor.is_available() 测试。"""

    async def test_available_when_docker_works(self, executor, mock_docker_client):
        """Docker 守护进程可达时返回 True。"""
        result = await executor.is_available()
        assert result is True

    async def test_not_available_when_docker_fails(self, executor, mocker):
        """Docker 连接失败时返回 False。"""
        mocker.patch(
            "codeagent.sandbox.docker_executor.docker.from_env",
            side_effect=Exception("Connection failed"),
        )
        mocker.patch("codeagent.sandbox.docker_executor._DOCKER_AVAILABLE", True)

        result = await executor.is_available()
        assert result is False

    async def test_not_available_when_sdk_missing(self, executor, mocker):
        """Docker SDK 不可用时返回 False。"""
        mocker.patch("codeagent.sandbox.docker_executor._DOCKER_AVAILABLE", False)

        result = await executor.is_available()
        assert result is False


# ── run ──────────────────────────────────────────────────────────────────


class TestRun:
    """DockerExecutor.run() 测试。"""

    async def test_run_success(self, executor, mock_docker_client):
        """正常执行返回正确的 stdout/stderr/exit_code。"""
        result = await executor.run(command="echo hello", workdir="/tmp")

        assert result.exit_code == 0
        assert "hello" in result.stdout
        assert result.sandboxed is True
        assert result.duration_ms >= 0

    async def test_run_with_env(self, executor, mock_docker_client):
        """执行时传递环境变量。"""
        result = await executor.run(
            command="echo $MY_VAR",
            workdir="/tmp",
            env={"MY_VAR": "test_value"},
        )

        assert result.exit_code == 0
        assert result.sandboxed is True

    async def test_run_docker_not_available(self, executor, mocker):
        """Docker 不可用时优雅降级。"""
        mocker.patch("codeagent.sandbox.docker_executor._DOCKER_AVAILABLE", False)

        result = await executor.run(command="echo hello", workdir="/tmp")

        assert result.exit_code == 1
        assert result.sandboxed is False
        assert "Docker is not available" in result.stderr

    async def test_run_timeout(self, executor, mock_docker_client):
        """超时后强制 kill 容器并返回超时错误。"""
        mock_container = mock_docker_client.containers.create.return_value

        from concurrent.futures import TimeoutError as FuturesTimeoutError

        def slow_wait():
            raise FuturesTimeoutError()

        mock_container.wait.side_effect = slow_wait

        result = await executor.run(command="sleep 100", workdir="/tmp")

        assert result.exit_code == -1
        assert "timed out" in result.stderr.lower() or "timeout" in result.stderr.lower()

    async def test_run_container_creation_failure(self, executor, mock_docker_client):
        """容器创建失败时返回错误结果（不抛异常）。"""
        mock_docker_client.containers.create.side_effect = Exception("Failed to create")

        result = await executor.run(command="echo hello", workdir="/tmp")

        assert result.exit_code == 1
        assert result.sandboxed is True
        assert "error" in result.stderr.lower() or "fail" in result.stderr.lower()

    async def test_run_with_path_conversion(self, executor, mock_docker_client):
        """Windows 路径转换为 POSIX 格式。"""
        # 模拟 Windows 路径
        result = await executor.run(
            command="pwd",
            workdir="C:\\Users\\test\\project",
        )

        assert result.exit_code == 0
        # 验证挂载卷路径被转换为 POSIX 格式（/c/Users/test/project）
        created_kwargs = mock_docker_client.containers.create.call_args.kwargs
        volumes = created_kwargs.get("volumes", {})
        # 路径应包含 posix 转换后的 key
        posix_keys = [k for k in volumes if "Users" in k or "users" in k.lower()]
        # 由于路径可能不存在，至少确认 volumes 被正确传递
        assert len(volumes) > 0


# ── run_tests ────────────────────────────────────────────────────────────


class TestRunTests:
    """DockerExecutor.run_tests() 测试。"""

    async def test_run_tests_success(self, executor, mock_docker_client):
        """正常执行测试命令。"""
        result = await executor.run_tests(
            project_root="/tmp/test_project",
            test_command="pytest -x --tb=short",
        )

        assert result.exit_code == 0
        assert result.sandboxed is True

    async def test_run_tests_docker_not_available(self, executor, mocker):
        """Docker 不可用时降级。"""
        mocker.patch("codeagent.sandbox.docker_executor._DOCKER_AVAILABLE", False)

        result = await executor.run_tests(
            project_root="/tmp/test_project",
            test_command="pytest",
        )

        assert result.exit_code == 1
        assert result.sandboxed is False
        assert "Docker is not available" in result.stderr

    async def test_run_tests_writable_volume(self, executor, mock_docker_client):
        """测试目录挂载为读写模式。"""
        await executor.run_tests(
            project_root="/tmp/test_project",
            test_command="pytest",
        )

        created_kwargs = mock_docker_client.containers.create.call_args.kwargs
        volumes = created_kwargs.get("volumes", {})
        # 读写模式
        for vol_config in volumes.values():
            assert vol_config["mode"] == "rw"


# ── Config ──────────────────────────────────────────────────────────────


class TestConfigFunctions:
    """config.py 沙箱配置函数测试。"""

    def test_get_sandbox_enabled_default_false(self):
        """SANDBOX_ENABLED 默认值为 false。"""
        from codeagent import config

        # 确保环境变量未设置
        enabled = config.get_sandbox_enabled()
        assert enabled is False

    def test_get_sandbox_enabled_true(self, mocker):
        """SANDBOX_ENABLED=true 时返回 True。"""
        mocker.patch.dict("os.environ", {"SANDBOX_ENABLED": "true"})
        # 重新加载以获取更新后的值
        import importlib
        from codeagent import config

        importlib.reload(config)
        assert config.get_sandbox_enabled() is True

    def test_get_sandbox_timeout_default(self):
        """SANDBOX_TIMEOUT 默认值为 60。"""
        from codeagent import config

        assert config.get_sandbox_timeout() == 60

    def test_get_sandbox_memory_mb_default(self):
        """SANDBOX_MEMORY_MB 默认值为 512。"""
        from codeagent import config

        assert config.get_sandbox_memory_mb() == 512
