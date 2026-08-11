"""BaseTool 抽象基类单元测试。"""

from __future__ import annotations

import pytest

from codeagent.tools.base import BaseTool, ConcreteTool, ToolResult


class TestBaseToolAbstraction:
    """验证抽象基类不能直接实例化。"""

    def test_cannot_instantiate_abstract(self) -> None:
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BaseTool()  # type: ignore[abstract]

    def test_concrete_tool_can_instantiate(self) -> None:
        tool = ConcreteTool()
        assert isinstance(tool, BaseTool)

    def test_concrete_tool_has_default_metadata(self) -> None:
        tool = ConcreteTool()
        assert tool.name == "concrete_tool"
        assert tool.description == "A concrete tool for testing"
        assert "msg" in tool.parameters.get("properties", {})
        assert not tool.requires_sandbox
        assert tool.allowed_paths == []
        assert tool.max_timeout_seconds == 30


class TestToolResult:
    """验证 ToolResult 数据类。"""

    def test_default_values(self) -> None:
        r = ToolResult(success=True)
        assert r.success is True
        assert r.data is None
        assert r.error_message is None
        assert r.error_code is None
        assert r.retryable is False
        assert r.duration_ms == 0.0
        assert r.tokens_consumed == 0

    def test_full_construction(self) -> None:
        r = ToolResult(
            success=False,
            data={"partial": "out"},
            error_message="fail",
            error_code="ERR_1",
            duration_ms=50.0,
            tokens_consumed=10,
        )
        assert r.success is False
        assert r.data == {"partial": "out"}
        assert r.error_message == "fail"
        assert r.error_code == "ERR_1"
        assert r.retryable is False
        assert r.suggested_recovery
        assert r.duration_ms == 50.0
        assert r.tokens_consumed == 10

    def test_success_result(self) -> None:
        r = ToolResult(success=True, data="ok")
        assert r.success is True
        assert r.data == "ok"

    def test_error_result(self) -> None:
        r = ToolResult(
            success=False,
            error_message="permission denied",
            error_code="E_PERM",
        )
        assert r.success is False
        assert r.data is None
        assert r.error_message == "permission denied"


class TestValidateParams:
    """验证参数 JSON Schema 校验。"""

    def test_valid_params(self) -> None:
        tool = ConcreteTool()
        assert tool.validate_params(msg="hello") is True

    def test_invalid_params_missing_required(self) -> None:
        tool = ConcreteTool()
        # msg is required but not provided
        assert tool.validate_params() is False

    def test_invalid_params_wrong_type(self) -> None:
        tool = ConcreteTool()
        assert tool.validate_params(msg=123) is False

    def test_detailed_validation_includes_exact_json_path(self) -> None:
        result = ConcreteTool().validate_params_detailed(msg=123)

        assert result.valid is False
        assert result.errors[0].path == "$.msg"
        assert "string" in result.errors[0].message

    def test_no_schema_returns_true(self) -> None:
        """如果 parameters 为空，validate_params 默认返回 True。"""

        class NoSchemaTool(BaseTool):
            name = "no_schema"
            description = "no schema"
            parameters = {}

            async def execute(self, **kwargs: object) -> ToolResult:
                return ToolResult(success=True)

        tool = NoSchemaTool()
        assert tool.validate_params(foo="bar") is True


class TestConcreteToolExecution:
    """验证 ConcreteTool 的执行行为。"""

    @pytest.mark.asyncio
    async def test_execute_success(self) -> None:
        tool = ConcreteTool()
        result = await tool.execute(msg="hello")
        assert result.success is True
        assert result.data == {"echo": "hello"}
        assert result.duration_ms >= 0


class TestGetLangchainTool:
    """验证 get_langchain_tool 返回 LangChain 兼容对象。"""

    def test_returns_structured_tool(self) -> None:
        from langchain_core.tools import StructuredTool

        tool = ConcreteTool()
        lc_tool = tool.get_langchain_tool()
        assert isinstance(lc_tool, StructuredTool)
        assert lc_tool.name == "concrete_tool"
        assert lc_tool.description == "A concrete tool for testing"

    def test_sync_execute_raises(self) -> None:
        """同步调用应该明确报错。"""
        tool = ConcreteTool()
        lc_tool = tool.get_langchain_tool()
        with pytest.raises(RuntimeError, match="async-only"):
            lc_tool.func(msg="hello")
