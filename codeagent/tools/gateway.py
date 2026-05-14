"""ToolGateway — 工具系统的统一调用入口。

实现 IToolGateway 接口，负责：
- 通过 ToolRegistry 调度工具调用
- 参数校验
- 超时控制
- 执行日志记录
- Metrics 收集
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from codeagent.gateway.tool_gateway import IToolGateway, ToolDefinition, ToolResult
from codeagent.tools.registry import ToolNotFoundError, ToolRegistry

logger = logging.getLogger(__name__)


@dataclass
class ToolMetrics:
    """工具 Metrics — 记录每个工具的调用统计信息。

    Attributes:
        call_count: 调用次数
        success_count: 成功次数
        total_duration_ms: 总耗时（毫秒）
    """

    call_count: int = 0
    success_count: int = 0
    total_duration_ms: float = 0.0

    @property
    def success_rate(self) -> float:
        """成功率（0.0 ~ 1.0）。"""
        if self.call_count == 0:
            return 0.0
        return self.success_count / self.call_count

    @property
    def avg_duration_ms(self) -> float:
        """平均耗时（毫秒）。"""
        if self.call_count == 0:
            return 0.0
        return self.total_duration_ms / self.call_count


@dataclass
class ExecutionLogEntry:
    """单次工具调用的执行日志条目。"""

    tool_name: str
    params: dict
    success: bool
    duration_ms: float
    error_code: str | None = None
    timestamp: float = field(default_factory=time.time)


class ToolGateway(IToolGateway):
    """工具系统统一调用入口。

    通过 ToolRegistry 获取工具实例，执行参数校验、超时控制、日志记录和 Metrics 收集。
    """

    def __init__(self, registry: ToolRegistry) -> None:
        """初始化 ToolGateway。

        Args:
            registry: 工具注册中心实例
        """
        self._registry = registry
        self._metrics: dict[str, ToolMetrics] = {}
        self._execution_log: list[ExecutionLogEntry] = []

    async def execute_tool(self, tool_name: str, params: dict[str, Any]) -> ToolResult:
        """执行指定工具。

        执行流程：
        1. 从 registry 获取工具
        2. 调用 tool.validate_params()
        3. 调用 tool.execute()（带超时保护）
        4. 记录执行日志
        5. 更新 metrics
        6. 返回 ToolResult

        Args:
            tool_name: 工具名称
            params: 工具参数

        Returns:
            ToolResult: 执行结果
        """
        start_time = time.monotonic()

        # ── 获取工具 ──────────────────────────────────────
        try:
            tool = self._registry.get(tool_name)
        except ToolNotFoundError:
            elapsed = (time.monotonic() - start_time) * 1000
            self._record_execution(tool_name, params, False, elapsed, "TOOL_NOT_FOUND")
            return ToolResult(
                success=False,
                error_message=f"Tool not found: {tool_name}",
                error_code="TOOL_NOT_FOUND",
                duration_ms=elapsed,
            )

        # ── 参数校验 ──────────────────────────────────────
        try:
            valid = tool.validate_params(**params)
        except Exception as e:
            self._record_execution(tool_name, params, False, 0.0, "VALIDATION_ERROR")
            return ToolResult(
                success=False,
                error_message=f"Parameter validation error: {e}",
                error_code="VALIDATION_ERROR",
                duration_ms=(time.monotonic() - start_time) * 1000,
            )

        if not valid:
            self._record_execution(tool_name, params, False, 0.0, "INVALID_PARAMS")
            return ToolResult(
                success=False,
                error_message=f"Invalid parameters for tool '{tool_name}'",
                error_code="INVALID_PARAMS",
                duration_ms=(time.monotonic() - start_time) * 1000,
            )

        # ── 执行（带超时） ────────────────────────────────
        try:
            timeout = tool.max_timeout_seconds
            result = await asyncio.wait_for(
                tool.execute(**params),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            duration = (time.monotonic() - start_time) * 1000
            self._record_execution(tool_name, params, False, duration, "TIMEOUT")
            logger.warning("Tool '%s' timed out after %ds", tool_name, timeout)
            return ToolResult(
                success=False,
                error_message=f"Tool '{tool_name}' timed out after {timeout}s",
                error_code="TIMEOUT",
                duration_ms=duration,
            )
        except Exception as e:
            duration = (time.monotonic() - start_time) * 1000
            self._record_execution(tool_name, params, False, duration, "EXECUTION_ERROR")
            logger.error("Tool '%s' execution failed: %s", tool_name, e)
            return ToolResult(
                success=False,
                error_message=f"Tool '{tool_name}' execution error: {e}",
                error_code="EXECUTION_ERROR",
                duration_ms=duration,
            )

        elapsed = (time.monotonic() - start_time) * 1000

        # ── 记录日志与 metrics ────────────────────────────
        self._record_execution(tool_name, params, result.success, elapsed)

        logger.info(
            "Tool '%s' executed in %.1fms (success=%s)",
            tool_name,
            elapsed,
            result.success,
        )

        return result

    def list_tools(self) -> list[ToolDefinition]:
        """列出所有已注册的工具。

        Returns:
            list[ToolDefinition]: 所有可用工具的定义列表
        """
        return self._registry.list_tools()

    async def validate_tool_params(self, tool_name: str, params: dict[str, Any]) -> bool:
        """验证工具参数是否合法。

        Args:
            tool_name: 工具名称
            params: 待验证的参数

        Returns:
            bool: 参数是否有效
        """
        try:
            tool = self._registry.get(tool_name)
        except ToolNotFoundError:
            return False

        try:
            return tool.validate_params(**params)
        except Exception:
            return False

    def get_metrics(self) -> dict[str, ToolMetrics]:
        """获取所有工具的统计信息。

        Returns:
            dict[str, ToolMetrics]: 工具名称到 Metrics 的映射
        """
        return dict(self._metrics)

    def get_execution_log(self) -> list[ExecutionLogEntry]:
        """获取执行日志。

        Returns:
            list[ExecutionLogEntry]: 执行日志条目列表
        """
        return list(self._execution_log)

    def clear_metrics(self) -> None:
        """清空所有 metrics 和执行日志。"""
        self._metrics.clear()
        self._execution_log.clear()

    def _record_execution(
        self,
        tool_name: str,
        params: dict[str, Any],
        success: bool,
        duration_ms: float,
        error_code: str | None = None,
    ) -> None:
        """记录一次工具执行到日志和 metrics。"""
        self._execution_log.append(
            ExecutionLogEntry(
                tool_name=tool_name,
                params=params,
                success=success,
                duration_ms=duration_ms,
                error_code=error_code,
            )
        )

        if tool_name not in self._metrics:
            self._metrics[tool_name] = ToolMetrics()

        metric = self._metrics[tool_name]
        metric.call_count += 1
        if success:
            metric.success_count += 1
        metric.total_duration_ms += duration_ms
