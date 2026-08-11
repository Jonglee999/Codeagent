"""ToolGateway 单元测试 — 验证 Gateway 调度逻辑。"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from codeagent.gateway.tool_gateway import ToolDefinition, ToolResult
from codeagent.tools.base import BaseTool
from codeagent.tools.gateway import ToolGateway, ToolMetrics
from codeagent.tools.registry import ToolRegistry


@pytest.mark.asyncio
async def test_extension_initialization_skips_discovery_without_matching_server(
    tmp_path, monkeypatch,
) -> None:
    from codeagent.extensions.mcp import MCPConfigResolution, MCPServerConfig

    server = MCPServerConfig(
        name="github",
        source="workspace",
        source_root=tmp_path,
        command="unused-command",
    )
    monkeypatch.setattr(
        "codeagent.extensions.mcp.resolve_mcp_config",
        lambda _root: MCPConfigResolution(servers={"github": server}),
    )
    monkeypatch.setattr("codeagent.config.get_mcp_enabled", lambda: True)
    discovery = AsyncMock()
    monkeypatch.setattr("codeagent.tools.mcp.discover_mcp_tools", discovery)
    gateway = ToolGateway(project_root=str(tmp_path))

    status = await gateway.initialize_extensions(query="Run the local script again")

    discovery.assert_not_awaited()
    assert status["mcp"]["available"] is False
    assert status["mcp"]["servers"][0]["deferred"] is True


@pytest.mark.asyncio
async def test_benchmark_extension_initialization_never_starts_matching_server(
    tmp_path, monkeypatch,
) -> None:
    from codeagent.extensions.mcp import MCPConfigResolution, MCPServerConfig

    server = MCPServerConfig(
        name="fetch",
        source="workspace",
        source_root=tmp_path,
        command="must-not-be-started",
    )
    monkeypatch.setattr(
        "codeagent.extensions.mcp.resolve_mcp_config",
        lambda _root: MCPConfigResolution(servers={"fetch": server}),
    )
    monkeypatch.setattr("codeagent.config.get_mcp_enabled", lambda: True)
    discovery = AsyncMock()
    monkeypatch.setattr("codeagent.tools.mcp.discover_mcp_tools", discovery)
    gateway = ToolGateway(project_root=str(tmp_path))

    status = await gateway.initialize_extensions(
        query="See https://example.invalid/spec",
        external_enabled=False,
    )

    discovery.assert_not_awaited()
    assert status["mcp"]["servers"][0]["deferred"] is True
    assert not any(tool.external for tool in gateway.list_tools())


# ── Helper Mock Tools ─────────────────────────────────────────────────────


class SuccessTool(BaseTool):
    """总是成功返回的工具。"""

    name = "success"
    description = "A tool that always succeeds"
    parameters = {
        "type": "object",
        "properties": {
            "msg": {"type": "string", "description": "A message"},
        },
        "required": ["msg"],
    }
    max_timeout_seconds = 30

    async def execute(self, **kwargs: object) -> ToolResult:
        return ToolResult(
            success=True,
            data={"echo": kwargs.get("msg", "")},
            duration_ms=10.0,
        )


class SlowTool(BaseTool):
    """执行很慢的工具，用于测试超时。"""

    name = "slow"
    description = "A slow tool"
    parameters = {
        "type": "object",
        "properties": {"delay": {"type": "number"}},
    }
    max_timeout_seconds = 1

    async def execute(self, **kwargs: object) -> ToolResult:
        delay = float(kwargs.get("delay", 5))
        await asyncio.sleep(delay)
        return ToolResult(success=True, data={"slept": delay})


class FailingTool(BaseTool):
    """执行时抛出异常的工具。"""

    name = "failing"
    description = "A tool that raises during execution"
    parameters = {}
    max_timeout_seconds = 30

    async def execute(self, **kwargs: object) -> ToolResult:
        msg = kwargs.get("message", "runtime failure")
        raise RuntimeError(msg)


class NoParamsTool(BaseTool):
    """无参数的工具。"""

    name = "no_params"
    description = "A tool with no parameters"
    parameters = {}
    max_timeout_seconds = 30

    async def execute(self, **kwargs: object) -> ToolResult:
        return ToolResult(success=True, data={})


class BoundedIntegerTool(BaseTool):
    name = "bounded_integer"
    description = "Accept a bounded integer"
    parameters = {
        "type": "object",
        "properties": {
            "limit": {"type": "integer", "minimum": 1, "maximum": 20},
        },
        "required": ["limit"],
        "additionalProperties": False,
    }

    async def execute(self, **kwargs: object) -> ToolResult:
        return ToolResult(success=True, data={"limit": kwargs["limit"]})


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(SuccessTool())
    reg.register(SlowTool())
    reg.register(FailingTool())
    reg.register(NoParamsTool())
    return reg


@pytest.fixture
def gateway(registry: ToolRegistry) -> ToolGateway:
    return ToolGateway(registry)


# ── Tests: execute_tool ───────────────────────────────────────────────────


class TestExecuteTool:
    """验证 ToolGateway.execute_tool 的各种场景。"""

    @pytest.mark.asyncio
    async def test_execute_success(self, gateway: ToolGateway) -> None:
        result = await gateway.execute_tool("success", {"msg": "hello"})
        assert result.success is True
        assert result.data == {"echo": "hello"}
        assert result.duration_ms >= 0

    @pytest.mark.asyncio
    async def test_execute_nonexistent_tool(self, gateway: ToolGateway) -> None:
        result = await gateway.execute_tool("not_found", {})
        assert result.success is False
        assert result.error_code == "TOOL_NOT_FOUND"
        assert "not found" in (result.error_message or "").lower()

    @pytest.mark.asyncio
    async def test_execute_invalid_params(self, gateway: ToolGateway) -> None:
        """缺少必填参数时应返回错误。"""
        result = await gateway.execute_tool("success", {})
        assert result.success is False
        assert result.error_code == "INVALID_PARAMS"

    @pytest.mark.asyncio
    async def test_execute_wrong_param_type(self, gateway: ToolGateway) -> None:
        """参数类型错误时应返回错误。"""
        result = await gateway.execute_tool("success", {"msg": 123})
        assert result.success is False
        assert result.error_code in ("INVALID_PARAMS", "VALIDATION_ERROR")

    @pytest.mark.asyncio
    async def test_execute_timeout(self, gateway: ToolGateway) -> None:
        """工具超时应返回 TIMEOUT 错误。"""
        result = await gateway.execute_tool("slow", {"delay": 10})
        assert result.success is False
        assert result.error_code == "TIMEOUT"
        assert result.retryable is True
        assert "Retry" in (result.suggested_recovery or "")

    @pytest.mark.asyncio
    async def test_execute_runtime_error(self, gateway: ToolGateway) -> None:
        """工具执行抛出异常时应返回 EXECUTION_ERROR。"""
        result = await gateway.execute_tool("failing", {"message": "boom"})
        assert result.success is False
        assert result.error_code == "EXECUTION_ERROR"
        assert "boom" in (result.error_message or "")
        assert result.retryable is True

    @pytest.mark.asyncio
    async def test_execute_no_params_tool(self, gateway: ToolGateway) -> None:
        """无参工具应能正常执行。"""
        result = await gateway.execute_tool("no_params", {})
        assert result.success is True


# ── Tests: list_tools ─────────────────────────────────────────────────────


class TestListTools:
    """验证 ToolGateway.list_tools。"""

    def test_list_tools(self, gateway: ToolGateway) -> None:
        tools = gateway.list_tools()
        assert len(tools) == 4
        names = {t.name for t in tools}
        assert names == {"success", "slow", "failing", "no_params"}

    def test_list_tools_returns_definitions(self, gateway: ToolGateway) -> None:
        tools = gateway.list_tools()
        for t in tools:
            assert isinstance(t, ToolDefinition)
            assert t.name
            assert t.description
            assert isinstance(t.parameters_schema, dict)

    def test_list_tools_after_registry_change(self, registry: ToolRegistry) -> None:
        """Registry 变化应反映在 gateway.list_tools 中。"""
        gateway = ToolGateway(registry)
        assert len(gateway.list_tools()) == 4
        registry.unregister("success")
        assert len(gateway.list_tools()) == 3


# ── Tests: validate_tool_params ───────────────────────────────────────────


class TestValidateToolParams:
    """验证 ToolGateway.validate_tool_params。"""

    @pytest.mark.asyncio
    async def test_valid_params(self, gateway: ToolGateway) -> None:
        valid = await gateway.validate_tool_params("success", {"msg": "hello"})
        assert valid is True

    @pytest.mark.asyncio
    async def test_invalid_params(self, gateway: ToolGateway) -> None:
        valid = await gateway.validate_tool_params("success", {})
        assert valid is False

    @pytest.mark.asyncio
    async def test_nonexistent_tool(self, gateway: ToolGateway) -> None:
        valid = await gateway.validate_tool_params("not_found", {})
        assert valid is False

    @pytest.mark.asyncio
    async def test_no_params_tool(self, gateway: ToolGateway) -> None:
        valid = await gateway.validate_tool_params("no_params", {})
        assert valid is True


# ── Tests: metrics ────────────────────────────────────────────────────────


class TestMetrics:
    """验证 ToolGateway 的 metrics 收集功能。"""

    @pytest.mark.asyncio
    async def test_metrics_after_success(self, gateway: ToolGateway) -> None:
        await gateway.execute_tool("success", {"msg": "hello"})
        metrics = gateway.get_metrics()
        assert "success" in metrics
        m = metrics["success"]
        assert m.call_count == 1
        assert m.success_count == 1
        assert m.total_duration_ms >= 0

    @pytest.mark.asyncio
    async def test_metrics_after_failure(self, gateway: ToolGateway) -> None:
        await gateway.execute_tool("not_found", {})
        metrics = gateway.get_metrics()
        assert "not_found" in metrics
        m = metrics["not_found"]
        assert m.call_count == 1
        assert m.success_count == 0

    @pytest.mark.asyncio
    async def test_metrics_multiple_calls(self, gateway: ToolGateway) -> None:
        for i in range(5):
            await gateway.execute_tool("success", {"msg": f"hello_{i}"})
        m = gateway.get_metrics()["success"]
        assert m.call_count == 5
        assert m.success_count == 5
        assert m.total_duration_ms >= 0

    @pytest.mark.asyncio
    async def test_metrics_mixed_success_failure(self, gateway: ToolGateway) -> None:
        await gateway.execute_tool("success", {"msg": "ok"})
        await gateway.execute_tool("success", {})  # 缺参 → 失败
        m = gateway.get_metrics()["success"]
        assert m.call_count == 2
        assert m.success_count == 1

    def test_metrics_success_rate(self) -> None:
        m = ToolMetrics(call_count=10, success_count=7, total_duration_ms=500)
        assert m.success_rate == 0.7
        assert m.avg_duration_ms == 50.0

    def test_metrics_empty(self) -> None:
        m = ToolMetrics()
        assert m.success_rate == 0.0
        assert m.avg_duration_ms == 0.0

    def test_metrics_initial_empty(self, gateway: ToolGateway) -> None:
        assert gateway.get_metrics() == {}

    @pytest.mark.asyncio
    async def test_metrics_multiple_tools(self, gateway: ToolGateway) -> None:
        await gateway.execute_tool("success", {"msg": "a"})
        await gateway.execute_tool("no_params", {})
        metrics = gateway.get_metrics()
        assert len(metrics) == 2
        assert metrics["success"].call_count == 1
        assert metrics["no_params"].call_count == 1


# ── Tests: execution log ──────────────────────────────────────────────────


class TestExecutionLog:
    """验证 ToolGateway 的执行日志。"""

    @pytest.mark.asyncio
    async def test_log_after_execution(self, gateway: ToolGateway) -> None:
        await gateway.execute_tool("success", {"msg": "hello"})
        log = gateway.get_execution_log()
        assert len(log) == 1
        entry = log[0]
        assert entry.tool_name == "success"
        assert entry.params == {"msg": "hello"}
        assert entry.success is True
        assert entry.duration_ms >= 0
        assert entry.error_code is None

    @pytest.mark.asyncio
    async def test_log_after_failure(self, gateway: ToolGateway) -> None:
        await gateway.execute_tool("not_found", {})
        log = gateway.get_execution_log()
        assert len(log) == 1
        assert log[0].success is False
        assert log[0].error_code == "TOOL_NOT_FOUND"

    @pytest.mark.asyncio
    async def test_log_multiple_calls(self, gateway: ToolGateway) -> None:
        for i in range(3):
            await gateway.execute_tool("success", {"msg": str(i)})
        log = gateway.get_execution_log()
        assert len(log) == 3

    @pytest.mark.asyncio
    async def test_log_redacts_credentials_and_bounds_large_values(
        self, gateway: ToolGateway
    ) -> None:
        await gateway.execute_tool(
            "success",
            {"msg": "x" * 1200, "api_token": "do-not-log"},
        )

        params = gateway.get_execution_log()[0].params
        assert params["api_token"] == "[REDACTED]"
        assert "do-not-log" not in str(params)
        assert params["msg"].endswith("…[TRUNCATED]")


class TestClearMetrics:
    """验证清空功能。"""

    @pytest.mark.asyncio
    async def test_clear_metrics(self, gateway: ToolGateway) -> None:
        await gateway.execute_tool("success", {"msg": "hello"})
        assert len(gateway.get_metrics()) == 1
        assert len(gateway.get_execution_log()) == 1
        gateway.clear_metrics()
        assert gateway.get_metrics() == {}
        assert gateway.get_execution_log() == []


# ── Tests: edge cases ─────────────────────────────────────────────────────


class TestEdgeCases:
    """边界场景测试。"""

    @pytest.mark.asyncio
    async def test_empty_registry(self) -> None:
        """空 registry 的 gateway 应优雅处理。"""
        empty_registry = ToolRegistry()
        gw = ToolGateway(empty_registry)
        result = await gw.execute_tool("anything", {})
        assert result.success is False
        assert result.error_code == "TOOL_NOT_FOUND"
        assert gw.list_tools() == []


class TestToolGatewayProjectRoot:
    """验证 ToolGateway 的 project_root 参数。"""

    def test_create_with_project_root(self) -> None:
        gw = ToolGateway(project_root="/tmp/test_proj")
        assert gw._project_root == "/tmp/test_proj"

    def test_create_without_registry_has_default_tools(self) -> None:
        gw = ToolGateway(project_root="/tmp/test_proj")
        tools = gw.list_tools()
        names = {t.name for t in tools}
        assert "read_file" in names
        assert "write_file" in names

    @pytest.mark.asyncio
    async def test_execute_with_empty_params(self, gateway: ToolGateway) -> None:
        """空参数调用无参工具应成功。"""
        result = await gateway.execute_tool("no_params", {})
        assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_with_extra_params(self, gateway: ToolGateway) -> None:
        """多余参数默认被 JSON Schema 忽略（无 additionalProperties 约束）。"""
        result = await gateway.execute_tool("success", {"msg": "hi", "extra": 1})
        assert result.success is True
        assert result.data == {"echo": "hi"}

    @pytest.mark.asyncio
    async def test_lossless_integer_string_is_normalized_before_validation(self) -> None:
        registry = ToolRegistry()
        registry.register(BoundedIntegerTool())
        gateway = ToolGateway(registry)
        params = {"limit": "5"}

        result = await gateway.execute_tool("bounded_integer", params)

        assert result.success is True
        assert result.data == {"limit": 5}
        assert params == {"limit": 5}

    @pytest.mark.asyncio
    async def test_non_integer_string_remains_a_schema_error(self) -> None:
        registry = ToolRegistry()
        registry.register(BoundedIntegerTool())
        gateway = ToolGateway(registry)

        result = await gateway.execute_tool("bounded_integer", {"limit": "5.5"})

        assert result.success is False
        assert result.error_code == "INVALID_PARAMS"

    @pytest.mark.asyncio
    async def test_aclose_closes_default_terminal_tool(self) -> None:
        gw = ToolGateway(project_root="/tmp/test_proj")
        terminal = gw._registry.get("run_terminal")
        terminal.close = MagicMock()

        await gw.aclose()

        terminal.close.assert_called_once_with()


@pytest.mark.asyncio
async def test_resilience_status_preserves_tool_timeout_code(registry: ToolRegistry) -> None:
    class TimeoutResultTool(BaseTool):
        name = "timeout_result"
        description = "returns a timeout result"
        parameters = {}

        async def execute(self, **kwargs: object) -> ToolResult:
            return ToolResult(success=False, error_code="DOCKER_TIMEOUT", error_message="timed out")

    registry.register(TimeoutResultTool())
    gateway = ToolGateway(registry)

    result = await gateway.execute_tool("timeout_result", {})
    status = gateway.resilience_status()

    assert result.error_code == "DOCKER_TIMEOUT"
    assert status["timeout_count"] == 1
    assert status["error_counts"] == {"DOCKER_TIMEOUT": 1}


@pytest.mark.asyncio
async def test_task_working_set_reuses_unchanged_file_reads(tmp_path) -> None:
    (tmp_path / "module.py").write_text("value = 1\n", encoding="utf-8")
    gateway = ToolGateway(project_root=str(tmp_path))

    first = await gateway.execute_tool("read_file", {"file_path": "module.py"})
    second = await gateway.execute_tool("read_file", {"file_path": "module.py"})
    status = gateway.working_set_status()

    assert first.success and second.success
    assert "working_set" not in first.data
    assert second.data["working_set"] == {"cache_hit": True}
    assert status["read_requests"] == 2
    assert status["cache_hits"] == 1
    assert status["cache_misses"] == 1
    assert status["tracked_files"] == 1


@pytest.mark.asyncio
async def test_task_working_set_precisely_invalidates_changed_file(tmp_path) -> None:
    (tmp_path / "a.py").write_text("a = 1\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("b = 1\n", encoding="utf-8")
    gateway = ToolGateway(project_root=str(tmp_path))
    await gateway.execute_tool("read_file", {"file_path": "a.py"})
    await gateway.execute_tool("read_file", {"file_path": "b.py"})

    changed = await gateway.execute_tool(
        "write_file",
        {"file_path": "a.py", "content": "a = 2\n", "mode": "modify"},
    )
    reread_a = await gateway.execute_tool("read_file", {"file_path": "a.py"})
    reread_b = await gateway.execute_tool("read_file", {"file_path": "b.py"})
    status = gateway.working_set_status()

    assert changed.success
    assert reread_a.data["content"] == "a = 2\n"
    assert "working_set" not in reread_a.data
    assert reread_b.data["working_set"] == {"cache_hit": True}
    assert status["invalidation_count"] == 1
    assert status["broad_invalidation_count"] == 0


@pytest.mark.asyncio
async def test_task_working_set_detects_external_file_changes(tmp_path) -> None:
    target = tmp_path / "module.py"
    target.write_text("old\n", encoding="utf-8")
    gateway = ToolGateway(project_root=str(tmp_path))
    await gateway.execute_tool("read_file", {"file_path": "module.py"})

    target.write_text("new content\n", encoding="utf-8")
    reread = await gateway.execute_tool("read_file", {"file_path": "module.py"})

    assert reread.data["content"] == "new content\n"
    assert "working_set" not in reread.data
    assert gateway.working_set_status()["recent_invalidations"][-1]["reason"] == "external_change"
