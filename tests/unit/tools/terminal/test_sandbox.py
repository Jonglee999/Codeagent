"""TerminalSandbox 单元测试。

使用 mock Docker SDK 进行测试。
覆盖场景：
- 容器创建/执行/清理
- 温容器池的获取/释放/回收
- 错误处理（Docker 不可用、守护进程连接失败等）
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import TimeoutError as FuturesTimeoutError
from unittest.mock import MagicMock, patch

import pytest

from codeagent.tools.terminal.sandbox import (
    DockerDaemonError,
    DockerNotAvailableError,
    TerminalSandbox,
)


@pytest.fixture
def mock_docker_client() -> MagicMock:
    """创建 mock Docker 客户端（每次返回不同容器 ID）。"""
    client = MagicMock()

    # 使用计数器生成不同 ID
    container_counter: list[int] = [0]

    def create_container_side_effect(**kwargs: object) -> MagicMock:
        container_counter[0] += 1
        mock_container = MagicMock()
        mock_container.id = f"container_id_{container_counter[0]:03d}"
        mock_container.status = "running"
        mock_container.exec_run.return_value = (0, (b"hello world", b""))
        return mock_container

    client.containers.create.side_effect = create_container_side_effect

    # get 默认返回一个通用容器
    default_container = MagicMock()
    default_container.id = "default_container"
    default_container.status = "running"
    default_container.exec_run.return_value = (0, (b"hello world", b""))
    client.containers.get.return_value = default_container

    client.ping.return_value = True
    return client


@pytest.fixture
def sandbox(mock_docker_client: MagicMock) -> TerminalSandbox:
    """创建 TerminalSandbox 实例（docker 已 mock）。"""
    with patch("codeagent.tools.terminal.sandbox.docker") as mock_docker:
        mock_docker.from_env.return_value = mock_docker_client
        sb = TerminalSandbox(image="python:3.11-slim")
        # 替换 client 属性为 mock 版本
        sb._client = mock_docker_client
        return sb


class TestSandboxCreateContainer:
    """容器创建测试。"""

    def test_create_container_success(self, sandbox: TerminalSandbox) -> None:
        cid = sandbox.create_container()
        assert cid == "container_id_001"
        assert cid in sandbox._containers
        assert sandbox.total_containers == 1
        kwargs = sandbox.client.containers.create.call_args.kwargs
        assert kwargs["name"].startswith("codeagent-sandbox-")
        assert kwargs["labels"] == {
            "com.codeagent.managed": "true",
            "com.codeagent.kind": "sandbox",
            "com.codeagent.scope": "task",
        }

    def test_create_container_with_volumes(self, sandbox: TerminalSandbox) -> None:
        volumes = {"/project": {"bind": "/workspace", "mode": "ro"}}
        cid = sandbox.create_container(volumes=volumes)
        assert cid == "container_id_001"
        sandbox.client.containers.create.assert_called_once()
        args, kwargs = sandbox.client.containers.create.call_args
        assert kwargs["volumes"] == volumes
        assert kwargs["network"] == "none"

    def test_create_container_with_network_bridge(self, sandbox: TerminalSandbox) -> None:
        cid = sandbox.create_container(network="bridge")
        assert cid == "container_id_001"
        args, kwargs = sandbox.client.containers.create.call_args
        assert kwargs["network"] == "bridge"

    def test_create_container_with_unknown_network_falls_back_to_none(self, sandbox: TerminalSandbox, caplog) -> None:
        """未知网络模式应回退 'none' 并记录 warning。"""
        with caplog.at_level(logging.WARNING):
            cid = sandbox.create_container(network="limited")
        assert cid == "container_id_001"
        args, kwargs = sandbox.client.containers.create.call_args
        assert kwargs["network"] == "none"
        assert "Unknown network mode" in caplog.text

    def test_create_container_resource_limits(self, sandbox: TerminalSandbox) -> None:
        cid = sandbox.create_container(mem_limit="256m", cpu_count=0.5)
        assert cid == "container_id_001"
        args, kwargs = sandbox.client.containers.create.call_args
        assert kwargs["mem_limit"] == "256m"
        assert kwargs["nano_cpus"] == int(0.5 * 1e9)

    def test_create_container_failure(self, sandbox: TerminalSandbox) -> None:
        sandbox.client.containers.create.side_effect = Exception("Cannot create container")
        with pytest.raises(DockerDaemonError, match="Failed to create container"):
            sandbox.create_container()


class TestSandboxExecCommand:
    """命令执行测试。"""

    def test_exec_command_success(self, sandbox: TerminalSandbox) -> None:
        stdout, stderr, exit_code = sandbox.exec_command("abc123", "python --version")
        assert stdout == "hello world"
        assert stderr == ""
        assert exit_code == 0

    def test_exec_command_with_stderr(self, sandbox: TerminalSandbox) -> None:
        mock_container = sandbox.client.containers.get.return_value
        mock_container.exec_run.return_value = (1, (b"", b"some error"))
        stdout, stderr, exit_code = sandbox.exec_command("abc123", "python nonexistent.py")
        assert stdout == ""
        assert stderr == "some error"
        assert exit_code == 1

    def test_exec_command_timeout(self, sandbox: TerminalSandbox) -> None:
        mock_container = sandbox.client.containers.get.return_value
        mock_container.exec_run.side_effect = Exception("Timeout")
        with pytest.raises(DockerDaemonError, match="Execution failed"):
            sandbox.exec_command("abc123", "sleep 100", timeout=5)

    def test_real_timeout_kills_container(self, sandbox: TerminalSandbox, monkeypatch) -> None:
        mock_container = sandbox.client.containers.get.return_value

        class FakeFuture:
            def result(self, timeout=None):
                raise FuturesTimeoutError()

        class FakePool:
            def __init__(self, **_kwargs):
                pass

            def submit(self, *_args, **_kwargs):
                return FakeFuture()

            def shutdown(self, **_kwargs):
                pass

        monkeypatch.setattr(
            "codeagent.tools.terminal.sandbox.ThreadPoolExecutor", FakePool
        )

        with pytest.raises(DockerDaemonError, match="timed out"):
            sandbox.exec_command("abc123", "sleep 100", timeout=1)

        mock_container.kill.assert_called_once_with()

    def test_exec_command_updates_last_used(self, sandbox: TerminalSandbox) -> None:
        cid = sandbox.create_container()
        old_time = sandbox._containers[cid].last_used_at
        time.sleep(0.001)
        sandbox.exec_command(cid, "echo test")
        new_time = sandbox._containers[cid].last_used_at
        assert new_time > old_time

    def test_exec_container_not_found(self, sandbox: TerminalSandbox) -> None:
        import docker as docker_module
        sandbox.client.containers.get.side_effect = docker_module.errors.NotFound("Not found")
        with pytest.raises(DockerDaemonError, match="not found"):
            sandbox.exec_command("nonexistent", "echo test")


class TestSandboxCleanup:
    """容器清理测试。"""

    def test_cleanup_existing_container(self, sandbox: TerminalSandbox) -> None:
        cid = sandbox.create_container()
        sandbox.cleanup(cid)
        sandbox.client.containers.get.return_value.remove.assert_called_once_with(force=True, v=True)
        assert cid not in sandbox._containers

    def test_cleanup_retries_transient_failure(self, sandbox: TerminalSandbox) -> None:
        cid = sandbox.create_container()
        container = sandbox.client.containers.get.return_value
        container.remove.side_effect = [RuntimeError("busy"), None]

        sandbox.cleanup(cid)

        assert container.remove.call_count == 2
        assert cid not in sandbox._containers

    def test_cleanup_nonexistent_container(self, sandbox: TerminalSandbox) -> None:
        import docker as docker_module
        sandbox.client.containers.get.side_effect = docker_module.errors.NotFound("Not found")
        # 不应该抛出异常
        sandbox.cleanup("nonexistent")

    def test_cleanup_all(self, sandbox: TerminalSandbox) -> None:
        sandbox.create_container()
        sandbox.create_container()
        sandbox.cleanup_all()
        assert sandbox.total_containers == 0
        assert sandbox.warm_pool_size == 0


class TestSandboxWarmPool:
    """温容器池测试。"""

    def test_acquire_from_warm_pool(self, sandbox: TerminalSandbox) -> None:
        cid = sandbox.create_container()
        sandbox._warm_pool.append(cid)

        acquired = sandbox.acquire_container()
        assert acquired == cid
        assert sandbox.warm_pool_size == 0  # 从温池取出后池大小减1

    def test_acquire_creates_new_when_pool_empty(self, sandbox: TerminalSandbox) -> None:
        cid = sandbox.acquire_container()
        assert cid == "container_id_001"

    def test_release_to_warm_pool(self, sandbox: TerminalSandbox) -> None:
        cid = sandbox.create_container()
        sandbox.release_container(cid)
        assert sandbox.warm_pool_size == 1
        assert cid in sandbox._warm_pool

    def test_default_pool_is_one_task_container(self, sandbox: TerminalSandbox) -> None:
        assert sandbox.WARM_POOL_MIN_SIZE == 0
        assert sandbox.WARM_POOL_MAX_SIZE == 1

    def test_release_when_pool_full(self, sandbox: TerminalSandbox) -> None:
        sandbox.WARM_POOL_MAX_SIZE = 2
        cid1 = sandbox.create_container()
        cid2 = sandbox.create_container()
        cid3 = sandbox.create_container()

        sandbox.release_container(cid1)
        sandbox.release_container(cid2)
        sandbox.release_container(cid3)  # 第三个应该被清理而不是加入温池

        assert sandbox.warm_pool_size == 2
        # cid3 不应在温池中
        assert cid3 not in sandbox._warm_pool

    def test_recycle_expired_containers(self, sandbox: TerminalSandbox) -> None:
        cid = sandbox.create_container()
        sandbox._containers[cid].created_at = time.time() - 400  # 6 分 40 秒前
        sandbox._warm_pool.append(cid)

        sandbox._recycle_expired()
        assert cid not in sandbox._warm_pool
        assert cid not in sandbox._containers

    def test_warm_container_not_expired(self, sandbox: TerminalSandbox) -> None:
        cid = sandbox.create_container()
        sandbox._containers[cid].created_at = time.time() - 60  # 1 分钟前
        sandbox._warm_pool.append(cid)

        sandbox._recycle_expired()
        assert cid in sandbox._warm_pool  # 未过期，应该保留

    def test_ensure_min_warm_pool(self, sandbox: TerminalSandbox) -> None:
        sandbox.WARM_POOL_MIN_SIZE = 2
        sandbox.ensure_min_warm_pool()
        assert sandbox.warm_pool_size == 2


class TestSandboxEdgeCases:
    """边界情况测试。"""

    def test_acquire_cleans_stale_pool_entries(self, sandbox: TerminalSandbox) -> None:
        """温池中有无效记录（_containers 中已不存在）时，获取时跳过。"""
        sandbox._warm_pool.append("stale_cid")
        cid = sandbox.acquire_container()
        assert cid == "container_id_001"  # 新创建的，跳过了 stale_cid
        assert "stale_cid" not in sandbox._warm_pool


class TestSandboxDockerNotAvailable:
    """Docker SDK 不可用时的测试。"""

    def test_create_without_docker(self) -> None:
        with patch("codeagent.tools.terminal.sandbox._DOCKER_AVAILABLE", False):
            with pytest.raises(DockerNotAvailableError):
                TerminalSandbox()
