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
import inspect
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from codeagent.gateway.tool_gateway import IToolGateway, ToolDefinition, ToolResult
from codeagent.interaction.api.metrics import observe_tool_call
from codeagent.tools.registry import ToolNotFoundError, ToolRegistry
from codeagent.tools.working_set import TaskWorkingSet
from codeagent.tracing import get_tracer

logger = logging.getLogger(__name__)

_SENSITIVE_PARAM_PATTERN = re.compile(
    r"(?:authorization|api[_-]?key|password|secret|token|cookie)", re.IGNORECASE
)
_MAX_LOG_VALUE_CHARS = 1000


def _coerce_safe_schema_scalars(
    params: dict[str, Any], schema: dict[str, Any]
) -> None:
    """Normalize only lossless, unambiguous provider scalar encodings."""

    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return
    for name, property_schema in properties.items():
        value = params.get(name)
        if (
            isinstance(property_schema, dict)
            and property_schema.get("type") == "integer"
            and isinstance(value, str)
            and re.fullmatch(r"[+-]?\d+", value.strip())
        ):
            params[name] = int(value.strip())


def _safe_log_params(params: dict[str, Any]) -> dict[str, Any]:
    """Return bounded parameters with common credential fields redacted."""
    def scrub(value: Any, *, key: str = "", depth: int = 0) -> Any:
        if key and _SENSITIVE_PARAM_PATTERN.search(key):
            return "[REDACTED]"
        if isinstance(value, str):
            if len(value) > _MAX_LOG_VALUE_CHARS:
                return value[:_MAX_LOG_VALUE_CHARS] + "…[TRUNCATED]"
            return value
        if depth >= 5:
            return "[MAX_DEPTH]"
        if isinstance(value, dict):
            return {
                str(child_key): scrub(
                    child_value, key=str(child_key), depth=depth + 1
                )
                for child_key, child_value in list(value.items())[:100]
            }
        if isinstance(value, (list, tuple)):
            return [scrub(item, depth=depth + 1) for item in value[:100]]
        return value

    return scrub(params)


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

    def __init__(
        self,
        registry: ToolRegistry | None = None,
        project_root: str = ".",
        allow_high_risk_extensions: bool = False,
        context_engine: Any | None = None,
    ) -> None:
        """初始化 ToolGateway。

        Args:
            registry: 工具注册中心实例（如不提供则自动创建并注册默认工具）
            project_root: 项目根目录（用于传递给读写文件工具进行路径安全校验）
        """
        self._project_root = project_root
        self._context_engine = context_engine
        self._allow_high_risk_extensions = allow_high_risk_extensions
        if registry is not None:
            self._registry = registry
        else:
            self._registry = ToolRegistry()
            self._register_default_tools()
        self._metrics: dict[str, ToolMetrics] = {}
        self._execution_log: list[ExecutionLogEntry] = []
        self._extension_status: dict[str, Any] = {
            "mcp": {
                "enabled": True,
                "configured": False,
                "available": False,
                "servers": [],
                "warnings": [],
            }
        }
        self._mcp_runtimes: list[Any] = []
        self._mcp_deferred_servers: list[dict[str, Any]] = []
        self._working_set = TaskWorkingSet(self._project_root)

    def _register_default_tools(self) -> None:
        """注册默认工具集（ReadFileTool、WriteFileTool）并传入 project_root。"""
        from codeagent.tools.file.apply_patch import ApplyPatchTool
        from codeagent.tools.file.delete_file import DeleteFileTool
        from codeagent.tools.file.list_files import ListFilesTool
        from codeagent.tools.file.read_file import ReadFileTool
        from codeagent.tools.file.write_file import WriteFileTool
        from codeagent.tools.git.git_tool import GitTool
        from codeagent.tools.lsp.get_diagnostics import GetDiagnosticsTool
        from codeagent.tools.search.search_code import SearchCodeTool
        from codeagent.tools.terminal.run_terminal import RunTerminalTool

        self._registry.register(ReadFileTool(project_root=self._project_root))
        self._registry.register(ListFilesTool(project_root=self._project_root))
        self._registry.register(DeleteFileTool(project_root=self._project_root))
        self._registry.register(WriteFileTool(project_root=self._project_root))
        self._registry.register(ApplyPatchTool(project_root=self._project_root))
        self._registry.register(SearchCodeTool(
            project_root=self._project_root,
            context_engine=self._context_engine,
        ))
        self._registry.register(GitTool(project_root=self._project_root))
        self._registry.register(GetDiagnosticsTool(project_root=self._project_root))
        self._registry.register(RunTerminalTool(project_root=self._project_root))

    async def initialize_extensions(
        self,
        progress_callback: Any | None = None,
        *,
        query: str | None = None,
        external_enabled: bool = True,
    ) -> dict[str, Any]:
        """Handshake only MCP servers relevant to this task's request.

        A missing query preserves the diagnostic/backward-compatible behavior
        of inspecting every configured server.
        """
        from codeagent.extensions.mcp import resolve_mcp_config
        from codeagent.config import get_mcp_enabled
        from codeagent.orchestration.capabilities import select_mcp_server_names
        from codeagent.tools.mcp import discover_mcp_tools

        resolution = resolve_mcp_config(self._project_root)
        if not get_mcp_enabled():
            warning = "MCP disabled by MCP_ENABLED=false"
            self._extension_status["mcp"] = {
                "enabled": False,
                "configured": bool(resolution.servers or resolution.config_paths),
                "available": False,
                "servers": [config.public_metadata() for config in resolution.servers.values()],
                "tools": [],
                "blocked_tools": [],
                "warnings": [*resolution.warnings, warning],
                "detail": warning,
            }
            if progress_callback is not None:
                emitted = progress_callback({
                    "type": "capability_degraded",
                    "summary": warning,
                    "data": {"capability": "mcp", "reason": "MCP_ENABLED=false"},
                })
                if inspect.isawaitable(emitted):
                    await emitted
            return self.extension_status()
        selected_server_names = (
            set()
            if not external_enabled
            else None
            if query is None
            else select_mcp_server_names(query, set(resolution.servers))
        )
        if selected_server_names == set():
            self._mcp_deferred_servers = [
                {
                    **config.public_metadata(),
                    "available": False,
                    "deferred": True,
                    "tools": [],
                    "tool_count": 0,
                    "last_error": None,
                    "last_latency_ms": None,
                    "last_attempt_at": None,
                    "last_success_at": None,
                    "call_count": 0,
                    "failure_count": 0,
                }
                for config in resolution.servers.values()
            ]
            detail = (
                f"Deferred {len(self._mcp_deferred_servers)} unrelated MCP server(s)"
                if self._mcp_deferred_servers
                else "No MCP configuration resolved for this workspace"
            )
            self._extension_status["mcp"] = {
                "enabled": True,
                "configured": bool(resolution.servers or resolution.config_paths),
                "available": False,
                "servers": list(self._mcp_deferred_servers),
                "tools": [],
                "blocked_tools": [],
                "warnings": list(resolution.warnings),
                "detail": detail,
            }
            # Do not invoke discovery at all when no server matches. This keeps
            # ordinary local turns free of MCP transport/list-tools work and
            # avoids presenting deferral as an executed tool-discovery step.
            return self.extension_status()
        discovery = await discover_mcp_tools(
            self._project_root,
            event_callback=progress_callback,
            server_names=selected_server_names,
            resolution=resolution,
        )
        self._mcp_runtimes = list(discovery.runtimes)
        selected_names = (
            set(resolution.servers)
            if selected_server_names is None
            else selected_server_names
        )
        self._mcp_deferred_servers = [
            {
                **config.public_metadata(),
                "available": False,
                "deferred": True,
                "tools": [],
                "tool_count": 0,
                "last_error": None,
                "last_latency_ms": None,
                "last_attempt_at": None,
                "last_success_at": None,
                "call_count": 0,
                "failure_count": 0,
            }
            for name, config in resolution.servers.items()
            if name not in selected_names
        ]
        if self._mcp_deferred_servers and progress_callback is not None:
            emitted = progress_callback({
                "type": "mcp_discovery_deferred",
                "summary": f"Deferred {len(self._mcp_deferred_servers)} unrelated MCP server(s)",
                "data": {
                    "servers": [item["name"] for item in self._mcp_deferred_servers],
                    "reason": "request did not match server capabilities",
                },
            })
            if inspect.isawaitable(emitted):
                await emitted
        registered: list[str] = []
        blocked: list[str] = []
        for tool in discovery.tools:
            if getattr(tool, "risk_level", "low") == "high" and not self._allow_high_risk_extensions:
                blocked.append(tool.name)
                discovery.warnings.append(
                    f"High-risk MCP tool requires explicit auto_mode authorization: {tool.name}"
                )
                continue
            try:
                self._registry.register(tool)
                registered.append(tool.name)
            except Exception as exc:
                discovery.warnings.append(f"Could not register {tool.name}: {exc}")
        self._extension_status["mcp"] = {
            "enabled": True,
            "configured": bool(resolution.servers or resolution.config_paths),
            "available": bool(registered),
            "servers": [*discovery.servers, *self._mcp_deferred_servers],
            "tools": registered,
            "blocked_tools": blocked,
            "warnings": discovery.warnings,
            "detail": (
                f"{len(registered)} MCP tool(s) available from "
                f"{sum(bool(item.get('available')) for item in discovery.servers)} server(s)"
                if registered
                else f"Deferred {len(self._mcp_deferred_servers)} MCP server(s); not needed by this request"
                if self._mcp_deferred_servers and not selected_names
                else "MCP configured but no server passed initialize/list_tools"
                if resolution.servers or resolution.config_paths
                else "No MCP configuration resolved for this workspace"
            ),
        }
        return self.extension_status()

    def extension_status(self) -> dict[str, Any]:
        if self._mcp_runtimes:
            current = [
                *[runtime.public_status() for runtime in self._mcp_runtimes],
                *self._mcp_deferred_servers,
            ]
            mcp = self._extension_status["mcp"]
            mcp["servers"] = current
            mcp["available"] = bool(mcp.get("tools")) and any(
                bool(item.get("available")) for item in current
            )
        return {
            key: dict(value) if isinstance(value, dict) else value
            for key, value in self._extension_status.items()
        }

    async def aclose(self) -> None:
        """Close task-scoped tools and their external resources."""
        try:
            terminal = self._registry.get("run_terminal")
        except ToolNotFoundError:
            return
        close = getattr(terminal, "close", None)
        if close is None:
            return
        result = close()
        if inspect.isawaitable(result):
            await result

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
        # Several OpenAI-compatible providers encode integer tool arguments as
        # strings. Normalize this lossless shape before strict schema validation
        # and mutate the caller-owned dict so orchestration logs the actual call.
        _coerce_safe_schema_scalars(params, tool.parameters)
        try:
            validation = tool.validate_params_detailed(**params)
        except Exception as e:
            self._record_execution(tool_name, params, False, 0.0, "VALIDATION_ERROR")
            return ToolResult(
                success=False,
                error_message=f"Parameter validation error: {e}",
                error_code="VALIDATION_ERROR",
                duration_ms=(time.monotonic() - start_time) * 1000,
            )

        if not validation.valid:
            self._record_execution(tool_name, params, False, 0.0, "INVALID_PARAMS")
            supplied = ", ".join(sorted(params)) or "none"
            details = [
                {
                    "path": error.path,
                    "message": error.message,
                    "validator": error.validator,
                }
                for error in validation.errors
            ]
            error_summary = "; ".join(
                f"{error.path}: {error.message}" for error in validation.errors
            )
            return ToolResult(
                success=False,
                data={"validation_errors": details},
                error_message=(
                    f"Invalid parameters for tool '{tool_name}' (supplied keys: "
                    f"{supplied}). {error_summary or 'Follow the JSON schema exactly.'}"
                ),
                error_code="INVALID_PARAMS",
                duration_ms=(time.monotonic() - start_time) * 1000,
            )

        if tool_name == "read_file":
            cached = self._working_set.get_read(params)
            if cached is not None:
                cached.duration_ms = (time.monotonic() - start_time) * 1000
                self._record_execution(tool_name, params, True, cached.duration_ms)
                return cached

        # ── 执行（带超时 + 追踪 Span） ─────────────────────
        try:
            timeout = tool.max_timeout_seconds
            with get_tracer().start_as_current_span("tool.call") as span:
                span.set_attribute("tool_name", tool_name)
                span.set_attribute("span.kind", "client")
                result = await asyncio.wait_for(
                    tool.execute(**params),
                    timeout=timeout,
                )
                span.set_attribute("success", result.success)
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

        if result.success:
            if tool_name == "read_file":
                self._working_set.remember_read(params, result)
            elif tool_name in {"write_file", "apply_patch", "delete_file"}:
                self._working_set.invalidate_file(
                    params.get("file_path"), reason=f"{tool_name}_success"
                )
                if self._context_engine is not None and params.get("file_path"):
                    try:
                        changed_path = str(
                            (Path(self._project_root) / str(params["file_path"])).resolve()
                        )
                        await self._context_engine.notify_workspace_changed(
                            changed_path,
                            deleted=tool_name == "delete_file",
                        )
                    except Exception as exc:
                        # Index refresh is a progressive enhancement and must not
                        # convert an already-successful file mutation into failure.
                        logger.warning("Workspace index refresh degraded: %s", exc)
            elif tool_name in {"run_terminal", "git"}:
                # These tools may change an unknown set of files. Preserve
                # correctness by dropping only this task's read observations.
                self._working_set.invalidate_all(reason=f"{tool_name}_success")

        # ── 记录日志与 metrics ────────────────────────────
        self._record_execution(tool_name, params, result.success, elapsed, result.error_code)

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

    def resilience_status(self) -> dict[str, Any]:
        """Return bounded, secret-free tool failure evidence for task reports."""
        error_counts: dict[str, int] = {}
        failed_tools: set[str] = set()
        for entry in self._execution_log:
            if entry.success:
                continue
            failed_tools.add(entry.tool_name)
            code = entry.error_code or "UNKNOWN"
            error_counts[code] = error_counts.get(code, 0) + 1
        return {
            "call_count": len(self._execution_log),
            "failure_count": sum(error_counts.values()),
            "timeout_count": sum(
                count for code, count in error_counts.items() if "TIMEOUT" in code
            ),
            "failed_tools": sorted(failed_tools),
            "error_counts": error_counts,
            "working_set": self._working_set.snapshot(),
        }

    def working_set_status(self) -> dict[str, Any]:
        """Expose bounded task-local cache evidence for diagnostics."""
        return self._working_set.snapshot()

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
                params=_safe_log_params(params),
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

        # Prometheus metrics (non-blocking)
        observe_tool_call(tool_name, duration_ms / 1000.0, success)
