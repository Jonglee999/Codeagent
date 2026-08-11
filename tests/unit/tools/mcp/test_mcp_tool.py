from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from codeagent.extensions.mcp import resolve_mcp_config
from codeagent.orchestration.nodes.execution_node import ExecutionNode
from codeagent.orchestration.state import AgentState
from codeagent.tools.gateway import ToolGateway
from codeagent.tools.mcp import MCPTool
from codeagent.tools.mcp.mcp_tool import MCPServerRuntime
from codeagent.tools.mcp.resilience import CircuitOpenError


@pytest.mark.asyncio
async def test_missing_config_returns_actionable_error(tmp_path):
    result = await MCPTool(tmp_path).execute(action="list", server="local")
    assert not result.success
    assert result.error_code == "MCP_ERROR"
    assert ".codeagent/mcp.json" in (result.error_message or "")


@pytest.mark.asyncio
async def test_mcp_capability_can_be_disabled_without_local_tool_loss(tmp_path, monkeypatch):
    events = []
    monkeypatch.setattr("codeagent.config.get_mcp_enabled", lambda: False)
    gateway = ToolGateway(project_root=str(tmp_path))

    status = await gateway.initialize_extensions(events.append)

    assert status["mcp"]["enabled"] is False
    assert status["mcp"]["available"] is False
    assert "read_file" in [tool.name for tool in gateway.list_tools()]
    assert events[0]["type"] == "capability_degraded"


@pytest.mark.asyncio
async def test_missing_required_env_degrades_without_starting_server(tmp_path):
    config_dir = tmp_path / ".codeagent"
    config_dir.mkdir()
    (config_dir / "mcp.json").write_text(
        json.dumps({"servers": {"secured": {
            "command": "must-not-be-started",
            "env_allowlist": ["MISSING_TEST_TOKEN"],
            "required_env": ["MISSING_TEST_TOKEN"],
        }}}),
        encoding="utf-8",
    )
    gateway = ToolGateway(project_root=str(tmp_path))

    status = await gateway.initialize_extensions()
    secured = next(
        server for server in status["mcp"]["servers"]
        if server["name"] == "secured"
    )

    assert secured["available"] is False
    assert "MISSING_TEST_TOKEN" in (secured["last_error"] or "")
    assert any("MISSING_TEST_TOKEN" in item for item in status["mcp"]["warnings"])
    assert "read_file" in [tool.name for tool in gateway.list_tools()]


@pytest.mark.asyncio
async def test_unrelated_task_defers_mcp_server_without_starting_it(tmp_path):
    config_dir = tmp_path / ".codeagent"
    config_dir.mkdir()
    (config_dir / "mcp.json").write_text(
        json.dumps({"servers": {"github": {"command": "must-not-be-started"}}}),
        encoding="utf-8",
    )
    events = []
    gateway = ToolGateway(project_root=str(tmp_path))

    status = await gateway.initialize_extensions(
        events.append, query="Fix the local parser"
    )
    github = next(server for server in status["mcp"]["servers"] if server["name"] == "github")

    assert github["deferred"] is True
    assert github["last_attempt_at"] is None
    assert status["mcp"]["warnings"] == []
    assert "mcp__github" not in " ".join(tool.name for tool in gateway.list_tools())
    # No matching server means discovery is skipped entirely; deferral remains
    # available in status metadata without creating a noisy execution event.
    assert events == []


@pytest.mark.asyncio
async def test_stdio_server_lists_and_executes_tools(tmp_path):
    server_file = tmp_path / "server.py"
    server_file.write_text(
        """from mcp.server.fastmcp import FastMCP
mcp = FastMCP('codeagent-test')
@mcp.tool()
def add(a: int, b: int) -> int:
    return a + b
if __name__ == '__main__':
    mcp.run(transport='stdio')
""",
        encoding="utf-8",
    )
    config_dir = tmp_path / ".codeagent"
    config_dir.mkdir()
    (config_dir / "mcp.json").write_text(
        json.dumps({
            "servers": {
                "local": {
                    "transport": "stdio",
                    "command": sys.executable,
                    "args": [str(server_file)],
                },
            },
        }),
        encoding="utf-8",
    )
    adapter = MCPTool(tmp_path)

    listed = await adapter.execute(action="list", server="local")
    called = await adapter.execute(
        action="call",
        server="local",
        tool="add",
        arguments={"a": 2, "b": 5},
    )

    assert listed.success
    assert [item["name"] for item in listed.data["tools"]] == ["add"]
    assert called.success
    assert called.data["structured_content"]["result"] == 7


@pytest.mark.asyncio
async def test_stdio_tools_are_registered_as_independent_definitions(tmp_path):
    server_file = tmp_path / "server.py"
    server_file.write_text(
        """from mcp.server.fastmcp import FastMCP
mcp = FastMCP('dynamic-test')
@mcp.tool()
def add(a: int, b: int) -> int:
    return a + b
if __name__ == '__main__':
    mcp.run(transport='stdio')
""",
        encoding="utf-8",
    )
    config_dir = tmp_path / ".codeagent"
    config_dir.mkdir()
    (config_dir / "mcp.json").write_text(
        json.dumps({"servers": {"local": {
            "transport": "stdio",
            "command": sys.executable,
            "args": [str(server_file)],
            "allowed_tools": ["add"],
        }}}),
        encoding="utf-8",
    )
    gateway = ToolGateway(project_root=str(tmp_path))

    status = await gateway.initialize_extensions()
    names = [tool.name for tool in gateway.list_tools()]
    result = await gateway.execute_tool("mcp__local__add", {"a": 4, "b": 6})

    assert "mcp" not in names
    assert "mcp__local__add" in names
    assert status["mcp"]["available"] is True
    assert result.success
    assert result.data["structured_content"]["result"] == 10


@pytest.mark.asyncio
async def test_agent_can_autonomously_select_and_call_discovered_mcp_tool(tmp_path):
    server_file = tmp_path / "server.py"
    server_file.write_text(
        """from mcp.server.fastmcp import FastMCP
mcp = FastMCP('agent-mcp-test')
@mcp.tool()
def add(a: int, b: int) -> int:
    return a + b
if __name__ == '__main__':
    mcp.run(transport='stdio')
""",
        encoding="utf-8",
    )
    config_dir = tmp_path / ".codeagent"
    config_dir.mkdir()
    (config_dir / "mcp.json").write_text(
        json.dumps({
            "servers": {
                "local": {
                    "command": sys.executable,
                    "args": [str(server_file)],
                    "allowed_tools": ["add"],
                }
            }
        }),
        encoding="utf-8",
    )
    gateway = ToolGateway(project_root=str(tmp_path))
    await gateway.initialize_extensions()
    llm = AsyncMock(side_effect=[
        SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(
                content=None,
                tool_calls=[SimpleNamespace(
                    id="call-mcp-1",
                    function=SimpleNamespace(
                        name="mcp__local__add",
                        arguments='{"a": 20, "b": 22}',
                    ),
                )],
            ))],
            usage=None,
        ),
        SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(
                content="The result is 42.", tool_calls=None,
            ))],
            usage=None,
        ),
    ])
    node = ExecutionNode(
        llm=llm,
        tool_gateway=gateway,
        validation_gateway=AsyncMock(),
    )
    state = AgentState(
        user_request="Use the available MCP calculator to add 20 and 22",
        project_root=str(tmp_path),
        allowed_tools=["mcp:*"],
    )

    try:
        result = await node(state)
    finally:
        await gateway.aclose()

    advertised = llm.await_args_list[0].kwargs["tools"]
    assert any(
        item["function"]["name"] == "mcp__local__add" for item in advertised
    )
    assert llm.await_args_list[0].kwargs["tool_choice"] == "auto"
    calls = [
        item for item in result["execution_log"]
        if item.get("type") == "tool_call"
    ]
    assert calls[0]["tool_name"] == "mcp__local__add"
    assert calls[0]["success"] is True
    assert calls[0]["result"]["structured_content"]["result"] == 42


@pytest.mark.asyncio
async def test_permission_denial_happens_before_server_call(tmp_path):
    config_dir = tmp_path / ".codeagent"
    config_dir.mkdir()
    (config_dir / "mcp.json").write_text(
        json.dumps({"servers": {"locked": {
            "command": "does-not-need-to-exist",
            "allowed_tools": ["safe_tool"],
        }}}),
        encoding="utf-8",
    )
    config = resolve_mcp_config(tmp_path).servers["locked"]
    events = []
    runtime = MCPServerRuntime(config, event_callback=events.append)

    with pytest.raises(PermissionError, match="not allowed"):
        await runtime.call("dangerous_tool", {})

    assert runtime.public_status()["circuit"]["failure_count"] == 0
    assert events[0]["type"] == "permission_denied"
    assert "dangerous_tool" in events[0]["summary"]


@pytest.mark.asyncio
async def test_server_failure_opens_circuit_without_affecting_local_tools(tmp_path):
    config_dir = tmp_path / ".codeagent"
    config_dir.mkdir()
    (config_dir / "mcp.json").write_text(
        json.dumps({"servers": {"broken": {
            "command": "missing-codeagent-mcp-command",
            "failure_threshold": 1,
            "cooldown_seconds": 60,
        }}}),
        encoding="utf-8",
    )
    runtime = MCPServerRuntime(resolve_mcp_config(tmp_path).servers["broken"])

    with pytest.raises(Exception):
        await runtime.discover()
    with pytest.raises(CircuitOpenError):
        await runtime.discover()

    gateway = ToolGateway(project_root=str(tmp_path))
    assert "read_file" in [tool.name for tool in gateway.list_tools()]


@pytest.mark.asyncio
async def test_real_stdio_tool_failure_opens_circuit_and_local_tool_recovers(tmp_path):
    server_file = Path(__file__).parents[4] / "evals" / "mcp" / "contract_server.py"
    config_dir = tmp_path / ".codeagent"
    config_dir.mkdir()
    (config_dir / "mcp.json").write_text(
        json.dumps({"servers": {"flaky": {
            "command": sys.executable,
            "args": [str(server_file), "stdio"],
            "allowed_tools": ["failure_probe"],
            "failure_threshold": 1,
            "cooldown_seconds": 60,
        }}}),
        encoding="utf-8",
    )
    (tmp_path / "healthy.txt").write_text("local tools remain healthy", encoding="utf-8")
    gateway = ToolGateway(project_root=str(tmp_path))
    await gateway.initialize_extensions()

    first = await gateway.execute_tool("mcp__flaky__failure_probe", {})
    second = await gateway.execute_tool("mcp__flaky__failure_probe", {})
    local = await gateway.execute_tool("read_file", {"file_path": "healthy.txt"})
    status = next(
        server
        for server in gateway.extension_status()["mcp"]["servers"]
        if server["name"] == "flaky"
    )

    assert first.error_code == "MCP_ERROR"
    assert second.error_code == "MCP_CIRCUIT_OPEN"
    assert local.success
    assert "local tools remain healthy" in local.data["content"]
    assert status["circuit"]["state"] == "open"


@pytest.mark.asyncio
async def test_streamable_http_server_lists_and_executes_tools(tmp_path):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    server_file = tmp_path / "http_server.py"
    server_file.write_text(
        f"""from mcp.server.fastmcp import FastMCP
mcp = FastMCP('http-test', host='127.0.0.1', port={port}, stateless_http=True)
@mcp.tool()
def multiply(a: int, b: int) -> int:
    return a * b
if __name__ == '__main__':
    mcp.run(transport='streamable-http')
""",
        encoding="utf-8",
    )
    process = subprocess.Popen(
        [sys.executable, str(server_file)],
        cwd=tmp_path,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                    break
            except OSError:
                time.sleep(0.1)
        else:
            raise AssertionError("Streamable HTTP MCP server did not start")

        config_dir = tmp_path / ".codeagent"
        config_dir.mkdir()
        (config_dir / "mcp.json").write_text(
            json.dumps({"servers": {"remote": {
                "transport": "streamable-http",
                "url": f"http://127.0.0.1:{port}/mcp",
                "allowed_tools": ["multiply"],
            }}}),
            encoding="utf-8",
        )
        gateway = ToolGateway(project_root=str(tmp_path))

        status = await gateway.initialize_extensions()
        result = await gateway.execute_tool("mcp__remote__multiply", {"a": 6, "b": 7})

        remote_status = next(
            server
            for server in status["mcp"]["servers"]
            if server["name"] == "remote"
        )
        assert remote_status["transport"] == "streamable-http"
        assert remote_status["available"] is True
        assert result.success
        assert result.data["structured_content"]["result"] == 42
    finally:
        process.terminate()
        process.wait(timeout=10)


@pytest.mark.asyncio
async def test_discovery_follows_tool_pagination_and_reports_diagnostics(tmp_path):
    config_dir = tmp_path / ".codeagent"
    config_dir.mkdir()
    (config_dir / "mcp.json").write_text(
        json.dumps({"servers": {"paged": {"command": "unused"}}}),
        encoding="utf-8",
    )
    runtime = MCPServerRuntime(resolve_mcp_config(tmp_path).servers["paged"])

    class FakeSession:
        async def initialize(self):
            return None

        async def list_tools(self, cursor=None):
            name = "first" if cursor is None else "second"
            next_cursor = "page-2" if cursor is None else None
            tool = SimpleNamespace(
                name=name, description=f"{name} tool", inputSchema={"type": "object"}
            )
            return SimpleNamespace(tools=[tool], nextCursor=next_cursor)

    @asynccontextmanager
    async def fake_session():
        yield FakeSession()

    runtime._session = fake_session  # type: ignore[method-assign]

    tools = await runtime.discover()
    status = runtime.public_status()

    assert [item["name"] for item in tools] == ["first", "second"]
    assert status["available"] is True
    assert status["call_count"] == 1
    assert status["failure_count"] == 0
    assert status["last_latency_ms"] is not None
    assert status["last_success_at"] is not None


def test_runtime_error_format_redacts_configured_secrets(tmp_path):
    config_dir = tmp_path / ".codeagent"
    config_dir.mkdir()
    (config_dir / "mcp.secrets.json").write_text(
        json.dumps({"servers": {"safe": {"env": {"API_TOKEN": "top-secret"}}}}),
        encoding="utf-8",
    )
    (config_dir / "mcp.json").write_text(
        json.dumps({
            "servers": {
                "safe": {"command": "unused", "env_allowlist": ["API_TOKEN"]}
            }
        }),
        encoding="utf-8",
    )
    runtime = MCPServerRuntime(resolve_mcp_config(tmp_path).servers["safe"])

    message = runtime.safe_error(RuntimeError("request failed with top-secret"))

    assert message == "request failed with [REDACTED]"
