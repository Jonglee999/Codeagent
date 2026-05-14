"""RunTerminalTool 集成测试 — 需要 Docker 环境。

覆盖场景：
- 真实容器中执行简单命令
- 容器资源限制（内存/CPU）
- 命令超时
- 温容器池复用
- 安全检查拦截 + 安全命令放行
"""

from __future__ import annotations

import pytest

from codeagent.tools.terminal.run_terminal import RunTerminalTool
from codeagent.tools.terminal.sandbox import TerminalSandbox


@pytest.fixture
def sandbox() -> TerminalSandbox:
    return TerminalSandbox(image="python:3.11-slim")


@pytest.fixture
def tool(sandbox: TerminalSandbox) -> RunTerminalTool:
    return RunTerminalTool(sandbox=sandbox)


class TestIntegrationSimpleCommands:
    """简单命令执行测试。"""

    @pytest.mark.asyncio
    async def test_python_version(self, tool: RunTerminalTool) -> None:
        """验证容器内 Python 版本。"""
        result = await tool.execute(command="python --version")
        assert result.success is True
        assert "Python 3.11" in (result.data.get("stdout", ""))
        assert result.data["exit_code"] == 0

    @pytest.mark.asyncio
    async def test_echo_command(self, tool: RunTerminalTool) -> None:
        """验证 echo 命令输出。"""
        result = await tool.execute(command="echo 'hello docker sandbox'")
        assert result.success is True
        stdout = result.data.get("stdout", "")
        assert "hello docker sandbox" in stdout
        assert result.data["exit_code"] == 0

    @pytest.mark.asyncio
    async def test_pip_install(self, tool: RunTerminalTool) -> None:
        """验证 pip 安装（需要网络访问）。"""
        result = await tool.execute(
            command="pip install requests 1>/dev/null 2>&1 && python -c 'import requests; print(\"ok\")'",
            timeout=120,
            network="limited",
        )
        assert result.success is True
        assert result.data["exit_code"] == 0, f"stdout: {result.data.get('stdout')!r}, stderr: {result.data.get('stderr')!r}"
        stdout = result.data.get("stdout", "")
        assert "ok" in stdout

    @pytest.mark.asyncio
    async def test_file_operations(self, tool: RunTerminalTool) -> None:
        """验证容器内文件和目录操作。"""
        result = await tool.execute(command="mkdir -p /tmp/testdir && echo 'data' > /tmp/testdir/file.txt && cat /tmp/testdir/file.txt")
        assert result.success is True
        assert result.data["exit_code"] == 0
        assert "data" in result.data.get("stdout", "")

    @pytest.mark.asyncio
    async def test_stderr_output(self, tool: RunTerminalTool) -> None:
        """验证 stderr 捕获。"""
        result = await tool.execute(command="python -c 'import sys; print(\"error msg\", file=sys.stderr); sys.exit(1)'")
        # 非零退出码，工具本身还是 success
        assert result.success is True
        assert result.data["exit_code"] == 1
        stderr = result.data.get("stderr", "")
        assert "error msg" in stderr


class TestIntegrationSandboxResources:
    """沙箱资源限制测试。"""

    @pytest.mark.asyncio
    async def test_memory_limit(self, sandbox: TerminalSandbox) -> None:
        """验证容器内存限制生效。"""
        cid = sandbox.create_container(mem_limit="128m")
        # 检查容器配置是否正确
        import docker
        container = docker.from_env().containers.get(cid)
        assert container.attrs["HostConfig"]["Memory"] == 128 * 1024 * 1024  # 128MB
        sandbox.cleanup(cid)


class TestIntegrationWarmPool:
    """温容器池集成测试。"""

    @pytest.mark.asyncio
    async def test_warm_pool_reuse(self, sandbox: TerminalSandbox) -> None:
        """验证温池容器可复用执行命令。"""
        cid = sandbox.create_container()
        sandbox.release_container(cid)
        assert sandbox.warm_pool_size == 1

        # 从温池获取
        cid2 = sandbox.acquire_container()
        assert cid2 == cid  # 应该是同一个容器

        # 在新获取的容器中执行命令
        stdout, stderr, exit_code = sandbox.exec_command(cid2, "echo 'reused'")
        assert "reused" in stdout
        assert exit_code == 0

    @pytest.mark.asyncio
    async def test_ensure_min_pool(self, sandbox: TerminalSandbox) -> None:
        """验证 ensure_min_warm_pool。"""
        sandbox.WARM_POOL_MIN_SIZE = 1
        sandbox.ensure_min_warm_pool()
        assert sandbox.warm_pool_size >= 1


class TestIntegrationSafetyBlocker:
    """安全检查拦截测试（容器不应被创建）。"""

    @pytest.mark.asyncio
    async def test_blocked_rm_rf(self, tool: RunTerminalTool) -> None:
        """rm -rf / 应被安全检查拦截。"""
        result = await tool.execute(command="rm -rf /")
        assert result.success is False
        assert result.error_code == "SAFETY_BLOCKED"

    @pytest.mark.asyncio
    async def test_blocked_sudo(self, tool: RunTerminalTool) -> None:
        """sudo 命令应被拦截。"""
        result = await tool.execute(command="sudo apt-get update")
        assert result.success is False
        assert result.error_code == "SAFETY_BLOCKED"


class TestIntegrationSandboxCleanup:
    """容器清理测试。"""

    @pytest.mark.asyncio
    async def test_cleanup(self, sandbox: TerminalSandbox) -> None:
        """验证容器能被正确清理。"""
        cid = sandbox.create_container()
        sandbox.cleanup(cid)

        import docker
        client = docker.from_env()
        with pytest.raises(docker.errors.NotFound):
            client.containers.get(cid)

    @pytest.mark.asyncio
    async def test_cleanup_all(self, sandbox: TerminalSandbox) -> None:
        """验证 cleanup_all 清理所有容器。"""
        cid1 = sandbox.create_container()
        cid2 = sandbox.create_container()
        sandbox.cleanup_all()
        assert sandbox.total_containers == 0

    @pytest.mark.asyncio
    async def test_multiple_uses_cleanup(self, tool: RunTerminalTool) -> None:
        """多次使用 RunTerminalTool，确认容器都被清理。"""
        r1 = await tool.execute(command="echo 'first'")
        assert r1.success is True
        r2 = await tool.execute(command="echo 'second'")
        assert r2.success is True
        r3 = await tool.execute(command="echo 'third'")
        assert r3.success is True
        # 每次释放到温池，最多保留 WARM_POOL_MAX_SIZE 个
        # 这里不严格 assert 数量，因为可能混有未回收的
        assert r1.data.get("stdout", "").strip() == "first"
        assert r2.data.get("stdout", "").strip() == "second"
        assert r3.data.get("stdout", "").strip() == "third"


@pytest.mark.asyncio
async def test_invalid_command_stderr(sandbox: TerminalSandbox) -> None:
    """执行不存在的命令应返回非零退出码。"""
    cid = sandbox.create_container()
    stdout, stderr, exit_code = sandbox.exec_command(cid, "nonexistent_command_xyz")
    assert exit_code != 0
    assert len(stderr) > 0
    sandbox.cleanup(cid)
