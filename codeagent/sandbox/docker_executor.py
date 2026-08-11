"""Docker 沙箱执行器 — 在隔离容器中执行命令和测试。

提供安全的命令执行环境，将宿主机文件系统挂载到容器中。
Docker 不可用时优雅降级，不阻断主流程。
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_DOCKER_AVAILABLE: bool = True
try:
    import docker
except ImportError:  # pragma: no cover
    _DOCKER_AVAILABLE = False
    docker = None  # type: ignore[assignment]


@dataclass
class ExecutionResult:
    """Docker 执行结果。

    Attributes:
        stdout: 标准输出
        stderr: 标准错误
        exit_code: 退出码
        duration_ms: 执行耗时（毫秒）
        sandboxed: 是否在沙箱中执行
    """

    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    duration_ms: int = 0
    sandboxed: bool = True


class DockerExecutor:
    """Docker 沙箱执行器。

    在隔离的 Docker 容器中执行命令和测试。
    Docker 不可用时优雅降级，不抛出异常。

    用法:
        executor = DockerExecutor()
        if await executor.is_available():
            result = await executor.run("python --version", "/my/project")
    """

    def __init__(
        self,
        image: str = "python:3.11-slim",
        memory_mb: int = 512,
        timeout_s: int = 60,
    ) -> None:
        """初始化 DockerExecutor。

        Args:
            image: Docker 镜像名称
            memory_mb: 内存限制（MB）
            timeout_s: 默认超时时间（秒）
        """
        self._image = image
        self._memory_mb = memory_mb
        self._timeout_s = timeout_s
        self._client: Optional[Any] = None
        self._sync_pool = ThreadPoolExecutor(max_workers=4)
        self._active_lock = threading.RLock()
        self._active_containers: dict[str, Any] = {}

    # ── 客户端连接 ────────────────────────────────────────────

    def _get_client(self) -> Optional[Any]:
        """获取或创建 Docker 客户端连接。"""
        if self._client is not None:
            return self._client
        if not _DOCKER_AVAILABLE:
            return None
        try:
            client = docker.from_env()
            client.ping()
            self._client = client
        except Exception as exc:
            logger.debug("Docker not available: %s", exc)
            self._client = None
        return self._client

    async def is_available(self) -> bool:
        """检测 Docker 是否可用。

        Returns:
            True 表示 Docker 守护进程可达
        """
        client = await asyncio.to_thread(self._get_client)
        return client is not None

    def close(self) -> None:
        """Release client-side resources owned by the executor."""
        with self._active_lock:
            active = list(self._active_containers.values())
        for container in active:
            try:
                container.kill()
            except Exception:
                logger.debug("Could not kill active Docker container during close", exc_info=True)
        self._sync_pool.shutdown(wait=False, cancel_futures=True)
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                logger.debug("Could not close Docker client", exc_info=True)
            self._client = None

    # ── 命令执行 ──────────────────────────────────────────────

    async def run(
        self,
        command: str,
        workdir: str,
        env: Optional[dict] = None,
    ) -> ExecutionResult:
        """在容器内执行命令。

        将 workdir 以读写卷挂载到容器 /workspace 后执行命令。
        捕获 stdout/stderr，超时后强制 kill 容器。

        Args:
            command: 要执行的命令
            workdir: 工作目录路径（挂载到容器 /workspace）
            env: 额外环境变量

        Returns:
            ExecutionResult: 执行结果
        """
        start = time.monotonic()
        client = self._get_client()

        if not client:
            return ExecutionResult(
                stderr="Docker is not available",
                exit_code=1,
                duration_ms=int((time.monotonic() - start) * 1000),
                sandboxed=False,
            )

        workdir_path = Path(workdir).resolve()
        container_workdir = "/workspace"

        volumes = {
            workdir_path.as_posix(): {
                "bind": container_workdir,
                "mode": "rw",
            }
        }

        env_vars = {}
        if env:
            env_vars.update(env)

        run_id = uuid.uuid4().hex
        cancel_event = threading.Event()
        try:
            result = await asyncio.to_thread(
                self._run_in_container_sync,
                client,
                command,
                container_workdir,
                volumes,
                env_vars,
                run_id,
                cancel_event,
            )
            result.duration_ms = int((time.monotonic() - start) * 1000)
            return result
        except asyncio.CancelledError:
            cancel_event.set()
            await asyncio.to_thread(self._cancel_active_run, run_id)
            raise
        except Exception as exc:
            logger.exception("Unexpected error in Docker run")
            return ExecutionResult(
                stderr=f"Unexpected Docker execution error: {exc}",
                exit_code=1,
                duration_ms=int((time.monotonic() - start) * 1000),
                sandboxed=True,
            )

    def _run_in_container_sync(
        self,
        client: Any,
        command: str,
        container_workdir: str,
        volumes: dict,
        env_vars: dict,
        run_id: str | None = None,
        cancel_event: threading.Event | None = None,
    ) -> ExecutionResult:
        """同步执行容器内命令（在线程池中运行）。"""
        container = None
        try:
            container = client.containers.create(
                image=self._image,
                name=f"codeagent-sandbox-{uuid.uuid4().hex[:12]}",
                command=["sh", "-c", command],
                working_dir=container_workdir,
                volumes=volumes,
                mem_limit=f"{self._memory_mb}m",
                environment=env_vars or None,
                detach=True,
                auto_remove=False,
                labels={
                    "com.codeagent.managed": "true",
                    "com.codeagent.kind": "sandbox",
                },
            )

            if run_id is not None:
                with self._active_lock:
                    self._active_containers[run_id] = container
            if cancel_event is not None and cancel_event.is_set():
                return ExecutionResult(
                    stderr="Container execution cancelled before start",
                    exit_code=-2,
                    sandboxed=True,
                )

            container.start()

            # 使用 ThreadPoolExecutor 实现超时
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(container.wait)
                try:
                    response = future.result(timeout=self._timeout_s)
                except FuturesTimeoutError:
                    container.kill()
                    return ExecutionResult(
                        stdout="",
                        stderr=f"Command timed out after {self._timeout_s}s",
                        exit_code=-1,
                        sandboxed=True,
                    )

            exit_code = response.get("StatusCode", -1)
            stdout = (container.logs(stdout=True, stderr=False) or b"").decode(
                "utf-8", errors="replace"
            )
            stderr = (container.logs(stdout=False, stderr=True) or b"").decode(
                "utf-8", errors="replace"
            )

            return ExecutionResult(
                stdout=stdout,
                stderr=stderr,
                exit_code=exit_code,
                sandboxed=True,
            )
        except Exception as exc:
            logger.exception("Container execution failed")
            return ExecutionResult(
                stderr=f"Container execution failed: {exc}",
                exit_code=1,
                sandboxed=True,
            )
        finally:
            if run_id is not None:
                with self._active_lock:
                    self._active_containers.pop(run_id, None)
            if container is not None:
                self._remove_container(container)

    def _cancel_active_run(self, run_id: str) -> None:
        with self._active_lock:
            container = self._active_containers.get(run_id)
        if container is None:
            return
        try:
            container.kill()
        except Exception:
            logger.debug("Could not kill cancelled Docker run %s", run_id, exc_info=True)

    @staticmethod
    def _remove_container(container: Any) -> None:
        """Remove a sandbox container, retrying transient Docker Desktop races."""
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                container.remove(force=True, v=True)
                return
            except Exception as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(0.1 * (attempt + 1))
        logger.warning(
            "Failed to remove CodeAgent sandbox container %s after 3 attempts: %s",
            getattr(container, "id", "unknown"),
            last_error,
        )

    # ── 测试执行 ──────────────────────────────────────────────

    async def run_tests(
        self,
        project_root: str,
        test_command: str = "pytest",
        timeout_s: int | None = None,
    ) -> ExecutionResult:
        """在容器内执行测试命令，自动安装 pytest（如未预装）。

        挂载项目目录为读写卷（测试可能需要写临时文件）。

        Args:
            project_root: 项目根目录路径
            test_command: 测试命令（如 "pytest -x --tb=short"），默认 "pytest"
            timeout_s: 超时秒数，默认使用 self._timeout_s

        Returns:
            ExecutionResult: 执行结果
        """
        # 先确保 pytest 可用
        install_cmd = (
            "python -m pytest --version >/dev/null 2>&1 || "
            "(pip install pytest -q --break-system-packages 2>/dev/null || pip install pytest -q)"
        )
        full_cmd = f"({install_cmd}) && {test_command}"

        start = time.monotonic()
        client = self._get_client()

        if not client:
            return ExecutionResult(
                stderr="Docker is not available",
                exit_code=1,
                duration_ms=int((time.monotonic() - start) * 1000),
                sandboxed=False,
            )

        project_path = Path(project_root).resolve()
        container_workdir = "/workspace"

        volumes = {
            project_path.as_posix(): {
                "bind": container_workdir,
                "mode": "rw",
            }
        }

        try:
            result = await asyncio.to_thread(
                self._run_in_container_sync,
                client,
                full_cmd,
                container_workdir,
                volumes,
                {},
            )
            result.duration_ms = int((time.monotonic() - start) * 1000)
            return result
        except Exception as exc:
            logger.exception("Unexpected error in Docker test execution")
            return ExecutionResult(
                stderr=f"Unexpected Docker test execution error: {exc}",
                exit_code=1,
                duration_ms=int((time.monotonic() - start) * 1000),
                sandboxed=True,
            )

    def cleanup(self) -> None:
        """清理线程池资源。"""
        self._sync_pool.shutdown(wait=False)
