"""Dynamic, policy-bounded MCP stdio and Streamable HTTP tools."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import re
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, AsyncIterator, Callable

import httpx
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client

from codeagent.extensions.mcp import MCPServerConfig, resolve_mcp_config
from codeagent.tools.base import BaseTool, ToolResult
from codeagent.tools.mcp.resilience import CircuitBreaker, CircuitOpenError, CircuitTransition

EventCallback = Callable[[dict[str, Any]], Any]


def _dynamic_name(server: str, tool: str) -> str:
    raw = f"mcp__{server}__{tool}"
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", raw)
    if len(safe) <= 64:
        return safe
    digest = hashlib.sha256(safe.encode("utf-8")).hexdigest()[:8]
    return f"{safe[:55]}_{digest}"


@dataclass
class MCPDiscovery:
    tools: list["MCPDynamicTool"] = field(default_factory=list)
    runtimes: list["MCPServerRuntime"] = field(default_factory=list)
    servers: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class MCPServerRuntime:
    """One server's bulkhead, circuit and transport lifecycle."""

    def __init__(
        self,
        config: MCPServerConfig,
        event_callback: EventCallback | None = None,
    ) -> None:
        self.config = config
        self._event_callback = event_callback
        self._semaphore = asyncio.Semaphore(config.max_concurrency)
        self._circuit = CircuitBreaker(
            failure_threshold=config.failure_threshold,
            cooldown_seconds=config.cooldown_seconds,
            label=f"MCP server {config.name}",
        )
        self.discovered_tools: list[str] = []
        self.available = False
        self.last_error: str | None = None
        self.last_latency_ms: float | None = None
        self.last_attempt_at: float | None = None
        self.last_success_at: float | None = None
        self.call_count = 0
        self.failure_count = 0

    async def _emit(self, event: dict[str, Any]) -> None:
        if self._event_callback is None:
            return
        result = self._event_callback(event)
        if inspect.isawaitable(result):
            await result

    def safe_error(self, exc: BaseException) -> str:
        """Format an error after removing configured secret values."""
        message = str(exc)
        secret_values = [
            *self.config.environment.values(),
            *self.config.headers.values(),
        ]
        for secret in secret_values:
            if secret:
                message = message.replace(secret, "[REDACTED]")
        return message

    async def _emit_transition(self, transition: CircuitTransition | None) -> None:
        if transition is None:
            return
        await self._emit({
            "type": "circuit_state_changed",
            "summary": (
                f"MCP server {self.config.name}: "
                f"{transition.previous} -> {transition.current}"
            ),
            "data": {
                "capability": "mcp",
                "server": self.config.name,
                "previous": transition.previous,
                "current": transition.current,
                "reason": transition.reason,
            },
        })

    @asynccontextmanager
    async def _session(self) -> AsyncIterator[ClientSession]:
        if self.config.transport == "stdio":
            params = StdioServerParameters(
                command=self.config.command or "",
                args=list(self.config.args),
                env=self.config.environment or None,
                cwd=self.config.source_root,
            )
            async with stdio_client(params) as (read, write):
                async with ClientSession(
                    read,
                    write,
                    read_timeout_seconds=timedelta(seconds=self.config.timeout_seconds),
                ) as session:
                    yield session
            return

        async with httpx.AsyncClient(
            headers=self.config.headers,
            timeout=self.config.timeout_seconds,
            follow_redirects=False,
        ) as http_client:
            async with streamable_http_client(
                self.config.url or "",
                http_client=http_client,
            ) as (read, write, _session_id):
                async with ClientSession(
                    read,
                    write,
                    read_timeout_seconds=timedelta(seconds=self.config.timeout_seconds),
                ) as session:
                    yield session

    async def discover(self) -> list[dict[str, Any]]:
        return await self._invoke("__list_tools__", {}, discovery=True)

    async def call(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if self.config.allowed_tools and tool_name not in self.config.allowed_tools:
            await self._emit({
                "type": "permission_denied",
                "summary": f"MCP tool {self.config.name}/{tool_name} denied by allowlist",
                "data": {"capability": "mcp", "server": self.config.name, "tool": tool_name},
            })
            raise PermissionError(f"MCP tool is not allowed: {self.config.name}/{tool_name}")
        return await self._invoke(tool_name, arguments, discovery=False)

    async def _invoke(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        discovery: bool,
    ) -> Any:
        started = time.monotonic()
        self.last_attempt_at = time.time()
        self.call_count += 1
        try:
            await self._emit_transition(await self._circuit.before_call())
        except CircuitOpenError:
            self.last_latency_ms = (time.monotonic() - started) * 1000
            raise

        try:
            async with self._semaphore:
                async with asyncio.timeout(self.config.timeout_seconds):
                    async with self._session() as session:
                        await session.initialize()
                        if discovery:
                            tools: list[dict[str, Any]] = []
                            cursor: str | None = None
                            seen_cursors: set[str] = set()
                            while True:
                                response = await session.list_tools(cursor=cursor)
                                tools.extend(
                                    {
                                        "name": item.name,
                                        "description": item.description or "",
                                        "input_schema": item.inputSchema,
                                    }
                                    for item in response.tools
                                    if not self.config.allowed_tools
                                    or item.name in self.config.allowed_tools
                                )
                                next_cursor = response.nextCursor
                                if not next_cursor:
                                    break
                                if next_cursor in seen_cursors or len(seen_cursors) >= 99:
                                    raise RuntimeError("MCP list_tools pagination did not terminate")
                                seen_cursors.add(next_cursor)
                                cursor = next_cursor
                            self.discovered_tools = [item["name"] for item in tools]
                            self.available = True
                            self.last_error = None
                            self.last_latency_ms = (time.monotonic() - started) * 1000
                            self.last_success_at = time.time()
                            await self._emit_transition(await self._circuit.record_success())
                            return tools

                        response = await session.call_tool(
                            tool_name,
                            arguments=arguments,
                            read_timeout_seconds=timedelta(seconds=self.config.timeout_seconds),
                        )
                        data = {
                            "server": self.config.name,
                            "tool": tool_name,
                            "content": [item.model_dump(mode="json") for item in response.content],
                            "structured_content": response.structuredContent,
                        }
                        if response.isError:
                            raise RuntimeError("MCP server reported a tool error")
                        self.available = True
                        self.last_error = None
                        self.last_latency_ms = (time.monotonic() - started) * 1000
                        self.last_success_at = time.time()
                        await self._emit_transition(await self._circuit.record_success())
                        return self._bound_output(data)
        except PermissionError:
            raise
        except Exception as exc:
            self.available = False
            self.failure_count += 1
            self.last_latency_ms = (time.monotonic() - started) * 1000
            self.last_error = f"{type(exc).__name__}: {self.safe_error(exc)}"
            await self._emit_transition(await self._circuit.record_failure(self.last_error))
            raise

    def _bound_output(self, data: dict[str, Any]) -> dict[str, Any]:
        encoded = json.dumps(data, ensure_ascii=False, default=str)
        if len(encoded) <= self.config.max_output_chars:
            return data
        return {
            "server": self.config.name,
            "tool": data.get("tool"),
            "truncated": True,
            "original_chars": len(encoded),
            "preview": encoded[: self.config.max_output_chars],
        }

    def public_status(self) -> dict[str, Any]:
        return {
            **self.config.public_metadata(),
            "available": self.available,
            "deferred": False,
            "tools": list(self.discovered_tools),
            "tool_count": len(self.discovered_tools),
            "last_error": self.last_error,
            "last_latency_ms": self.last_latency_ms,
            "last_attempt_at": self.last_attempt_at,
            "last_success_at": self.last_success_at,
            "call_count": self.call_count,
            "failure_count": self.failure_count,
            "circuit": self._circuit.snapshot(),
        }


class MCPDynamicTool(BaseTool):
    """One independently registered LLM tool backed by one MCP server tool."""

    def __init__(
        self,
        runtime: MCPServerRuntime,
        remote_name: str,
        description: str,
        parameters: dict[str, Any],
    ) -> None:
        self._runtime = runtime
        self.remote_name = remote_name
        self.name = _dynamic_name(runtime.config.name, remote_name)
        self.description = (
            f"MCP {runtime.config.name}/{remote_name}: {description}".strip()
        )
        self.parameters = parameters or {"type": "object", "properties": {}}
        self.max_timeout_seconds = runtime.config.timeout_seconds + 2
        self.risk_level = runtime.config.risk_level
        self.mcp_server = runtime.config.name
        self.category = "external"
        self.source = f"mcp:{runtime.config.name}"
        self.external = True
        self.read_only = False
        self.latency_hint = "network"
        self.cost_hint = "unknown"
        self.idempotent = False
        self.reversible = False

    async def execute(self, **kwargs: Any) -> ToolResult:
        started = time.monotonic()
        try:
            data = await self._runtime.call(self.remote_name, kwargs)
            return ToolResult(
                success=True,
                data=data,
                duration_ms=(time.monotonic() - started) * 1000,
            )
        except PermissionError as caught:
            code = "MCP_PERMISSION_DENIED"
            error_message = str(caught)
        except CircuitOpenError as caught:
            code = "MCP_CIRCUIT_OPEN"
            error_message = str(caught)
        except asyncio.TimeoutError as caught:
            code = "MCP_TIMEOUT"
            error_message = str(caught)
        except Exception as caught:
            code = "MCP_ERROR"
            error_message = self._runtime.safe_error(caught)
        return ToolResult(
            success=False,
            error_message=error_message,
            error_code=code,
            duration_ms=(time.monotonic() - started) * 1000,
        )


async def discover_mcp_tools(
    project_root: str | Path,
    *,
    product_root: str | Path | None = None,
    event_callback: EventCallback | None = None,
    server_names: set[str] | None = None,
    resolution: Any | None = None,
) -> MCPDiscovery:
    resolution = resolution or resolve_mcp_config(project_root, product_root=product_root)
    discovery = MCPDiscovery(warnings=list(resolution.warnings))
    selected_configs = [
        config
        for name, config in resolution.servers.items()
        if server_names is None or name in server_names
    ]
    if server_names:
        missing = sorted(server_names.difference(resolution.servers))
        discovery.warnings.extend(
            f"Unknown MCP server: {name}" for name in missing
        )

    async def inspect_server(config: MCPServerConfig) -> tuple[MCPServerRuntime, list[dict[str, Any]] | Exception]:
        runtime = MCPServerRuntime(config, event_callback=event_callback)
        missing_env = config.missing_required_env()
        if missing_env:
            error = RuntimeError(
                "missing required environment variable(s): "
                + ", ".join(missing_env)
            )
            runtime.last_error = str(error)
            return runtime, error
        try:
            return runtime, await runtime.discover()
        except Exception as exc:
            return runtime, exc

    inspected = await asyncio.gather(
        *(inspect_server(config) for config in selected_configs)
    ) if selected_configs else []

    for runtime, result in inspected:
        discovery.runtimes.append(runtime)
        if isinstance(result, Exception):
            warning = f"MCP server {runtime.config.name} unavailable: {type(result).__name__}: {result}"
            discovery.warnings.append(warning)
            if event_callback is not None:
                emitted = event_callback({
                    "type": "capability_degraded",
                    "summary": warning,
                    "data": {"capability": "mcp", "server": runtime.config.name},
                })
                if inspect.isawaitable(emitted):
                    await emitted
        else:
            dynamic_tools = [
                MCPDynamicTool(
                    runtime,
                    remote_name=item["name"],
                    description=item["description"],
                    parameters=item["input_schema"],
                )
                for item in result
            ]
            discovery.tools.extend(dynamic_tools)
            if event_callback is not None:
                for tool in dynamic_tools:
                    emitted = event_callback({
                        "type": "mcp_tool_discovered",
                        "summary": f"Registered {tool.name}",
                        "data": {
                            "server": runtime.config.name,
                            "tool": tool.name,
                            "remote_tool": tool.remote_name,
                            "risk_level": tool.risk_level,
                        },
                    })
                    if inspect.isawaitable(emitted):
                        await emitted
        public_status = runtime.public_status()
        discovery.servers.append(public_status)
        if event_callback is not None:
            emitted = event_callback({
                "type": "mcp_server_status",
                "summary": (
                    f"MCP server {runtime.config.name} available"
                    if runtime.available
                    else f"MCP server {runtime.config.name} unavailable"
                ),
                "data": public_status,
            })
            if inspect.isawaitable(emitted):
                await emitted
    return discovery


class MCPTool(BaseTool):
    """Compatibility adapter for direct diagnostics; not exposed to the LLM registry."""

    name = "mcp"
    description = "Diagnostic MCP adapter"
    parameters = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["list", "call"]},
            "server": {"type": "string"},
            "tool": {"type": "string"},
            "arguments": {"type": "object"},
        },
        "required": ["action", "server"],
    }

    def __init__(self, project_root: str | Path = ".") -> None:
        self._project_root = Path(project_root).resolve()

    async def execute(
        self,
        action: str,
        server: str,
        tool: str | None = None,
        arguments: dict[str, Any] | None = None,
    ) -> ToolResult:
        started = time.monotonic()
        resolution = resolve_mcp_config(self._project_root)
        config = resolution.servers.get(server)
        if config is None:
            return ToolResult(
                success=False,
                error_message=(
                    f"Unknown MCP server {server}; configure .codeagent/mcp.json. "
                    + "; ".join(resolution.warnings)
                ),
                error_code="MCP_ERROR",
                duration_ms=(time.monotonic() - started) * 1000,
            )
        runtime = MCPServerRuntime(config)
        try:
            if action == "list":
                data = {"server": server, "tools": await runtime.discover()}
            elif action == "call" and tool:
                data = await runtime.call(tool, arguments or {})
            else:
                raise ValueError("tool is required when action=call")
            return ToolResult(
                success=True,
                data=data,
                duration_ms=(time.monotonic() - started) * 1000,
            )
        except Exception as exc:
            return ToolResult(
                success=False,
                error_message=runtime.safe_error(exc),
                error_code="MCP_ERROR",
                duration_ms=(time.monotonic() - started) * 1000,
            )
