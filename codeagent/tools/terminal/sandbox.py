"""Docker 容器生命周期管理 — Layer 2 + Layer 3 安全防护。

管理隔离容器的创建、命令执行、清理，以及温容器池优化。
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import docker

logger = logging.getLogger(__name__)

# Docker SDK 是否可用的标记
_DOCKER_AVAILABLE: bool = True
try:
    import docker
except ImportError:  # pragma: no cover
    _DOCKER_AVAILABLE = False
    docker = None  # type: ignore[assignment]


@dataclass
class ContainerInfo:
    """容器信息。"""

    container_id: str
    image: str
    created_at: float
    last_used_at: float
    is_warm: bool = False


class DockerNotAvailableError(Exception):
    """Docker SDK 不可用时抛出。"""

    def __init__(self) -> None:
        super().__init__(
            "Docker SDK is not available. "
            "Install it with: pip install docker>=7.0.0"
        )


class DockerDaemonError(Exception):
    """Docker 守护进程连接失败时抛出。"""

    def __init__(self, message: str = "") -> None:
        msg = f"Docker daemon connection failed: {message}" if message else "Docker daemon is not running or not accessible"
        super().__init__(msg)


class SandboxExecutionError(Exception):
    """容器内命令执行失败时抛出。"""

    def __init__(self, message: str, exit_code: int, stdout: str, stderr: str) -> None:
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr
        super().__init__(message)


class TerminalSandbox:
    """Docker 沙箱 — 容器生命周期管理。

    功能：
    - 创建隔离容器执行命令
    - 维护温容器池（2-4 个，最大存活 5 分钟）
    - 自动清理过期容器
    - 资源限制（CPU/内存/网络/磁盘）

    使用示例：
        sandbox = TerminalSandbox()
        cid = sandbox.create_container(
            volumes={"/project": {"bind": "/workspace", "mode": "ro"}},
            network="none",
        )
        stdout, stderr, exit_code = sandbox.exec_command(cid, "python --version")
        sandbox.cleanup(cid)
    """

    # 默认容器配置
    DEFAULT_IMAGE = "python:3.11-slim"
    DEFAULT_MEM_LIMIT = "512m"
    DEFAULT_CPU_COUNT = 1.0
    DEFAULT_TIMEOUT = 60

    # 温容器池配置
    WARM_POOL_MIN_SIZE = 2
    WARM_POOL_MAX_SIZE = 4
    WARM_POOL_MAX_LIFETIME = 300  # 5 分钟

    def __init__(self, image: str = DEFAULT_IMAGE) -> None:
        """初始化 Sandbox。

        Args:
            image: Docker 镜像名称

        Raises:
            DockerNotAvailableError: Docker SDK 未安装
        """
        if not _DOCKER_AVAILABLE:
            raise DockerNotAvailableError()

        self._image = image
        self._client: docker.DockerClient | None = None
        self._containers: dict[str, ContainerInfo] = {}
        self._warm_pool: list[str] = []  # 容器 ID 列表

    # ── 客户端连接 ────────────────────────────────────────────

    @property
    def client(self) -> docker.DockerClient:
        """获取或创建 Docker 客户端连接。"""
        if self._client is None:
            try:
                self._client = docker.from_env()
                # 测试连接
                self._client.ping()
            except Exception as exc:
                self._client = None
                raise DockerDaemonError(str(exc)) from exc
        return self._client

    # ── 容器管理 ──────────────────────────────────────────────

    def _resolve_network(self, network: str) -> str:
        """将逻辑网络模式映射为 Docker 网络模式。

        支持的模式：
        - "none"   → 完全隔离，容器内无法访问任何网络（推荐用于代码执行）
        - "bridge" → 使用宿主机默认 bridge 网络，容器内可访问外网

        注意：当前不支持细粒度的网络访问控制（如白名单域名）。
        如需限制网络访问，请使用 "none" 模式。
        """
        if network not in ("none", "bridge"):
            logger.warning(
                "Unknown network mode '%s', falling back to 'none' for safety.",
                network,
            )
            return "none"
        return network

    def create_container(
        self,
        volumes: dict | None = None,
        network: str = "none",
        mem_limit: str = DEFAULT_MEM_LIMIT,
        cpu_count: float = DEFAULT_CPU_COUNT,
        working_dir: str = "/workspace",
    ) -> str:
        """创建隔离的 Docker 容器。

        Args:
            volumes: 挂载卷映射
            network: 网络模式 — "none"（完全隔离，默认）或 "bridge"（完整网络访问）
            mem_limit: 内存限制
            cpu_count: CPU 核心数
            working_dir: 容器内工作目录

        Returns:
            str: 容器 ID

        Raises:
            DockerDaemonError: Docker 守护进程连接失败
        """
        try:
            container = self.client.containers.create(
                image=self._image,
                working_dir=working_dir,
                volumes=volumes or {},
                network=self._resolve_network(network),
                mem_limit=mem_limit,
                nano_cpus=int(cpu_count * 1e9),
                detach=True,
                stdin_open=True,
                tty=False,
                auto_remove=False,  # 手动控制清理
            )
            info = ContainerInfo(
                container_id=container.id,
                image=self._image,
                created_at=time.time(),
                last_used_at=time.time(),
            )
            self._containers[container.id] = info
            logger.debug("Created container %s", container.id[:12])
            return container.id
        except Exception as exc:
            raise DockerDaemonError(f"Failed to create container: {exc}") from exc

    def exec_command(
        self,
        container_id: str,
        command: str,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> tuple[str, str, int]:
        """在容器内执行命令。

        Args:
            container_id: 容器 ID
            command: 要执行的命令
            timeout: 超时秒数

        Returns:
            tuple[str, str, int]: (stdout, stderr, exit_code)

        Raises:
            DockerDaemonError: 容器连接或执行失败
        """
        try:
            container = self.client.containers.get(container_id)
            # 启动容器（如果是停止状态）
            if container.status != "running":
                container.start()

            # 更新最后使用时间
            if container_id in self._containers:
                self._containers[container_id].last_used_at = time.time()

            # exec_run — demux=True 分别返回 stdout 和 stderr
            # docker SDK 7.x 不支持 timeout 参数，用线程池实现超时
            _exec_result: list[tuple[int, tuple[bytes, bytes]]] = []

            def _exec() -> None:
                _exec_result.append(container.exec_run(
                    cmd=["sh", "-c", command],
                    demux=True,
                ))

            pool = ThreadPoolExecutor(max_workers=1)
            future = pool.submit(_exec)
            try:
                future.result(timeout=timeout)
            except FuturesTimeoutError:
                pool.shutdown(wait=False, cancel_futures=True)
                raise TimeoutError(f"Command timed out after {timeout}s")
            except Exception as exc:
                pool.shutdown(wait=False)
                raise exc
            finally:
                pool.shutdown(wait=False)

            exit_code, output = _exec_result[0]

            stdout_bytes, stderr_bytes = output if output else (b"", b"")
            stdout = stdout_bytes.decode("utf-8", errors="replace") if stdout_bytes else ""
            stderr = stderr_bytes.decode("utf-8", errors="replace") if stderr_bytes else ""

            return stdout, stderr, exit_code
        except docker.errors.NotFound:
            raise DockerDaemonError(f"Container {container_id[:12]} not found")
        except docker.errors.APIError as exc:
            raise DockerDaemonError(f"Docker API error: {exc}") from exc
        except TimeoutError as exc:
            raise DockerDaemonError(str(exc)) from exc
        except Exception as exc:
            raise DockerDaemonError(f"Execution failed: {exc}") from exc

    def cleanup(self, container_id: str, force: bool = True) -> None:
        """清理指定容器。

        Args:
            container_id: 容器 ID
            force: 是否强制删除
        """
        try:
            container = self.client.containers.get(container_id)
            container.remove(force=force)
        except docker.errors.NotFound:
            pass  # 容器已经不存在
        except Exception as exc:
            logger.warning("Failed to cleanup container %s: %s", container_id[:12], exc)

        # 从记录中移除
        self._containers.pop(container_id, None)
        if container_id in self._warm_pool:
            self._warm_pool.remove(container_id)

    # ── 温容器池 ──────────────────────────────────────────────

    def acquire_container(
        self,
        volumes: dict | None = None,
        network: str = "none",
    ) -> str:
        """从容器的温池中获取一个容器。

        优先从温池获取，温池不足时创建新容器。
        自动回收过期的温容器。

        Returns:
            str: 容器 ID
        """
        self._recycle_expired()

        # 温池有可用容器
        while self._warm_pool:
            cid = self._warm_pool.pop(0)
            if cid in self._containers:
                self._containers[cid].last_used_at = time.time()
                logger.debug("Reused warm container %s", cid[:12])
                return cid
            # 容器记录不存在，跳过

        return self.create_container(volumes=volumes, network=network)

    def release_container(self, container_id: str) -> None:
        """将容器释放回温池（如果未超过池大小限制）。

        Args:
            container_id: 容器 ID
        """
        if len(self._warm_pool) >= self.WARM_POOL_MAX_SIZE:
            # 池已满，直接清理
            self.cleanup(container_id)
            return

        info = self._containers.get(container_id)
        if info:
            info.last_used_at = time.time()
            info.is_warm = True
            self._warm_pool.append(container_id)
            logger.debug("Released container %s to warm pool", container_id[:12])

    def _recycle_expired(self) -> None:
        """回收所有超过最大存活时间的温容器。"""
        now = time.time()
        expired: list[str] = []
        for cid in self._warm_pool:
            info = self._containers.get(cid)
            if info and (now - info.created_at) > self.WARM_POOL_MAX_LIFETIME:
                expired.append(cid)

        for cid in expired:
            self._warm_pool.remove(cid)
            self.cleanup(cid)
            logger.debug("Recycled expired warm container %s", cid[:12])

    def cleanup_all(self) -> None:
        """清理所有容器。"""
        for cid in list(self._containers.keys()):
            self.cleanup(cid)
        self._warm_pool.clear()

    @property
    def warm_pool_size(self) -> int:
        """当前温容器池大小。"""
        return len(self._warm_pool)

    @property
    def total_containers(self) -> int:
        """当前管理的容器总数。"""
        return len(self._containers)

    def ensure_min_warm_pool(self) -> None:
        """确保温池至少有最小数量的容器。"""
        self._recycle_expired()
        current = len(self._warm_pool)
        if current < self.WARM_POOL_MIN_SIZE:
            for _ in range(self.WARM_POOL_MIN_SIZE - current):
                cid = self.create_container()
                self._warm_pool.append(cid)
                if cid in self._containers:
                    self._containers[cid].is_warm = True
