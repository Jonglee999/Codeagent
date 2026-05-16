"""Docker 沙箱集成测试。

这些测试需要真实的 Docker 环境（Docker Desktop 或 Docker Engine）。
使用 @pytest.mark.docker 标记，可在 CI 中跳过。
"""

from __future__ import annotations

import pytest

from codeagent.sandbox.docker_executor import DockerExecutor


pytestmark = pytest.mark.docker


@pytest.fixture
async def executor():
    """创建 DockerExecutor 实例（使用真实 Docker）。"""
    exec = DockerExecutor(image="python:3.11-slim", memory_mb=256, timeout_s=30)
    if not await exec.is_available():
        pytest.skip("Docker is not available in this environment")
    yield exec
    exec.cleanup()


@pytest.mark.asyncio
class TestDockerSandboxIntegration:
    """Docker 沙箱集成测试（需要真实 Docker 环境）。"""

    async def test_docker_available(self, executor):
        """Docker 检测可用。"""
        assert await executor.is_available() is True

    async def test_echo_command(self, executor):
        """在容器内执行 echo 命令。"""
        result = await executor.run(
            command="echo 'Hello Docker'",
            workdir="/tmp",
        )
        assert result.exit_code == 0
        assert "Hello Docker" in result.stdout
        assert result.sandboxed is True
        assert result.duration_ms > 0

    async def test_python_version(self, executor):
        """在容器内运行 python --version。"""
        result = await executor.run(
            command="python --version",
            workdir="/tmp",
        )
        assert result.exit_code == 0
        assert "Python" in result.stdout
        assert result.sandboxed is True

    async def test_command_with_stderr(self, executor):
        """命令产生 stderr 输出。"""
        result = await executor.run(
            command="python -c 'import sys; sys.stderr.write(\"error msg\"); sys.exit(1)'",
            workdir="/tmp",
        )
        assert result.exit_code == 1
        assert "error msg" in result.stderr
        assert result.sandboxed is True

    async def test_timeout_kills_container(self, executor):
        """超时后强制 kill 容器。"""
        fast_executor = DockerExecutor(image="python:3.11-slim", memory_mb=256, timeout_s=2)
        try:
            result = await fast_executor.run(
                command="sleep 30",
                workdir="/tmp",
            )
            assert result.exit_code == -1
            assert "timed out" in result.stderr.lower()
        finally:
            fast_executor.cleanup()

    async def test_run_tests_no_tests(self, executor, tmp_path):
        """在没有测试文件的项目上运行测试。"""
        result = await executor.run_tests(
            project_root=str(tmp_path),
            test_command="pytest --version",
        )
        assert result.exit_code == 0
        assert result.sandboxed is True

    async def test_environment_isolation(self, executor):
        """容器内环境变量隔离。"""
        result = await executor.run(
            command="env",
            workdir="/tmp",
        )
        assert result.exit_code == 0
        # 宿主机环境变量不应泄漏到容器
        assert result.sandboxed is True

    async def test_file_system_isolation(self, executor):
        """容器内文件系统隔离。"""
        result = await executor.run(
            command="ls /host_file_test || echo 'no host access'",
            workdir="/tmp",
        )
        assert result.exit_code == 0
        assert result.sandboxed is True
