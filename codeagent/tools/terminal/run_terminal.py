"""RunTerminalTool — 在隔离的 Docker 容器中执行终端命令。

安全设计（三层防护）：
- Layer 1: SafetyChecker 命令静态分析（黑名单 + 注入检测 + 文件系统危险操作）
- Layer 2: Docker 容器隔离
- Layer 3: 资源限制（CPU/内存/网络）
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Optional

from codeagent import config as codeagent_config
from codeagent.gateway.tool_gateway import ToolResult
from codeagent.sandbox.docker_executor import DockerExecutor
from codeagent.tools.base import BaseTool
from codeagent.tools.terminal.safety_checker import SafetyChecker, SafetyResult
from codeagent.tools.terminal.sandbox import (
    DockerDaemonError,
    DockerNotAvailableError,
    TerminalSandbox,
)

logger = logging.getLogger(__name__)


class RunTerminalTool(BaseTool):
    """在隔离的 Docker 容器中执行终端命令。

    使用三层安全防护确保命令执行的安全性：
    1. SafetyChecker 静态扫描命令
    2. Docker 容器隔离执行
    3. 资源限制（内存 512MB, CPU 1核, 网络白名单）

    用法:
        tool = RunTerminalTool(sandbox=sandbox)
        result = await tool.execute(command="python --version")
    """

    name = "run_terminal"
    description = "在隔离的 Docker 容器中执行终端命令，返回命令输出"
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "要执行的终端命令",
            },
            "work_dir": {
                "type": "string",
                "description": "容器内的工作目录（可选）",
            },
            "timeout": {
                "type": "integer",
                "description": "命令执行超时秒数（默认 60）",
                "default": 60,
            },
            "network": {
                "type": "string",
                "enum": ["none", "limited"],
                "description": "网络模式：none（默认，无网络）、limited（仅白名单域名）",
                "default": "none",
            },
        },
        "required": ["command"],
    }

    # 终端命令可能耗时较长
    max_timeout_seconds: int = 300

    # 项目根目录（用于挂载卷）
    _project_root: str = ""

    def __init__(
        self,
        sandbox: TerminalSandbox | None = None,
        safety_checker: SafetyChecker | None = None,
        project_root: str = "",
        executor: Optional[DockerExecutor] = None,
    ) -> None:
        """初始化 RunTerminalTool。

        Args:
            sandbox: TerminalSandbox 实例（如不提供则自动创建）
            safety_checker: SafetyChecker 实例（如不提供则自动创建）
            project_root: 项目根目录（用于挂载到容器）
            executor: DockerExecutor 实例（SANDBOX_ENABLED=true 时使用）
        """
        super().__init__()
        self._safety_checker = safety_checker or SafetyChecker()
        self._sandbox = sandbox
        self._project_root = project_root
        self._executor = executor

    @property
    def sandbox(self) -> TerminalSandbox:
        """获取或创建 TerminalSandbox 实例。"""
        if self._sandbox is None:
            self._sandbox = TerminalSandbox()
        return self._sandbox

    async def execute(self, **kwargs: Any) -> ToolResult:
        """执行终端命令。

        Args:
            **kwargs: 命令参数（command, work_dir, timeout, network）

        Returns:
            ToolResult: 执行结果
                - success: 命令是否成功执行（非零退出码也算 success=True，
                  具体查看 exit_code 字段）
                - data: {
                    "stdout": "...",
                    "stderr": "...",
                    "exit_code": 0,
                    "duration_ms": 1234.5,
                    "sandbox_id": "abc123"
                }
                - error_message: 安全检查失败或 Docker 错误的描述
        """
        start = time.monotonic()
        command = kwargs.get("command", "")
        work_dir = kwargs.get("work_dir", "")
        timeout = kwargs.get("timeout", 60)
        network = kwargs.get("network", "none")

        if not command or not command.strip():
            duration = (time.monotonic() - start) * 1000
            return ToolResult(
                success=False,
                error_message="Command cannot be empty",
                error_code="EMPTY_COMMAND",
                duration_ms=duration,
            )

        # ── Layer 1: Safety Check ─────────────────────────────
        safety_result: SafetyResult = self._safety_checker.check_command(command)
        if not safety_result.safe:
            duration = (time.monotonic() - start) * 1000
            return ToolResult(
                success=False,
                error_message=f"Safety check failed: {safety_result.reason}",
                error_code="SAFETY_BLOCKED",
                data={
                    "risk_level": safety_result.risk_level,
                    "matched_patterns": safety_result.matched_patterns,
                },
                duration_ms=duration,
            )

        # ── Layer 2: DockerExecutor 沙箱模式 ──────────────────
        if self._executor and codeagent_config.get_sandbox_enabled():
            try:
                exec_result = await self._executor.run(
                    command=command,
                    workdir=self._project_root or ".",
                )
                duration = (time.monotonic() - start) * 1000
                return ToolResult(
                    success=exec_result.exit_code == 0,
                    data={
                        "stdout": exec_result.stdout,
                        "stderr": exec_result.stderr,
                        "exit_code": exec_result.exit_code,
                        "duration_ms": duration,
                        "sandboxed": exec_result.sandboxed,
                    },
                    duration_ms=duration,
                )
            except Exception as exc:
                duration = (time.monotonic() - start) * 1000
                logger.exception("DockerExecutor execution failed")
                return ToolResult(
                    success=False,
                    error_message=f"Docker execution error: {exc}",
                    error_code="DOCKER_EXECUTOR_ERROR",
                    duration_ms=duration,
                )

        # ── Layer 2+3: TerminalSandbox (现有 Docker 沙箱) ─────
        try:
            # 构建卷映射
            volumes: dict = {}
            if self._project_root:
                volumes[self._project_root] = {
                    "bind": "/project",
                    "mode": "ro",
                }

            # 获取容器
            cid = self.sandbox.acquire_container(
                volumes=volumes if volumes else None,
                network=network,
            )

            try:
                stdout, stderr, exit_code = self.sandbox.exec_command(
                    container_id=cid,
                    command=command,
                    timeout=timeout,
                )
            finally:
                # 释放容器回温池（或清理）
                self.sandbox.release_container(cid)

            duration = (time.monotonic() - start) * 1000
            return ToolResult(
                success=True,
                data={
                    "stdout": stdout,
                    "stderr": stderr,
                    "exit_code": exit_code,
                    "duration_ms": duration,
                    "sandbox_id": cid,
                },
                duration_ms=duration,
            )

        except DockerNotAvailableError as exc:
            duration = (time.monotonic() - start) * 1000
            return ToolResult(
                success=False,
                error_message=str(exc),
                error_code="DOCKER_NOT_AVAILABLE",
                duration_ms=duration,
            )
        except DockerDaemonError as exc:
            duration = (time.monotonic() - start) * 1000
            return ToolResult(
                success=False,
                error_message=str(exc),
                error_code="DOCKER_DAEMON_ERROR",
                duration_ms=duration,
            )
        except Exception as exc:
            duration = (time.monotonic() - start) * 1000
            logger.exception("Unexpected error during command execution")
            return ToolResult(
                success=False,
                error_message=f"Unexpected error: {exc}",
                error_code="UNEXPECTED_ERROR",
                duration_ms=duration,
            )
