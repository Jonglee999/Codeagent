"""Gateway 接口契约单元测试。

验证三个抽象接口的定义、Mock 实现的兼容性、以及 DTO 数据类的正确性。
"""

from __future__ import annotations

import inspect
from collections.abc import AsyncIterator
from typing import Any, get_type_hints

import pytest

from codeagent.gateway.tool_gateway import (
    IToolGateway,
    ToolDefinition,
    ToolResult,
)
from codeagent.gateway.context_gateway import (
    CodeSnippet,
    ContextPackage,
    IContextGateway,
)
from codeagent.gateway.validation_gateway import (
    IValidationGateway,
    ValidationError,
    ValidationResult,
)


# =============================================================================
# 测试辅助 — Mock 实现
# =============================================================================


class MockToolGateway(IToolGateway):
    """IToolGateway 的 Mock 实现，用于 isinstance 验证。"""

    async def execute_tool(self, tool_name: str, params: dict) -> ToolResult:
        return ToolResult(success=True, data="mock")

    def list_tools(self) -> list[ToolDefinition]:
        return []

    async def validate_tool_params(self, tool_name: str, params: dict) -> bool:
        return True


class MockContextGateway(IContextGateway):
    """IContextGateway 的 Mock 实现，用于 isinstance 验证。"""

    async def build_context(
        self, project_root: str, query: str
    ) -> ContextPackage:
        return ContextPackage()

    async def update_index(self, project_root: str) -> None:
        return None

    async def search_semantic(
        self, query: str, top_k: int = 5
    ) -> list[CodeSnippet]:
        return []


class MockValidationGateway(IValidationGateway):
    """IValidationGateway 的 Mock 实现，用于 isinstance 验证。"""

    async def run_syntax_check(self, file_path: str) -> ValidationResult:
        return ValidationResult()

    async def run_lint(self, files: list[str]) -> ValidationResult:
        return ValidationResult()

    async def run_tests(self, project_root: str) -> ValidationResult:
        return ValidationResult()

    async def run_runtime_check(self, file_path: str) -> ValidationResult:
        return ValidationResult()


# =============================================================================
# 测试：抽象接口不能直接实例化
# =============================================================================


class TestInterfaceInstantiation:
    """验证每个抽象接口不能直接实例化。"""

    @pytest.mark.parametrize(
        "interface",
        [
            IToolGateway,
            IContextGateway,
            IValidationGateway,
        ],
    )
    def test_cannot_instantiate_abstract(self, interface: type) -> None:
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            interface()  # type: ignore[abstract]


# =============================================================================
# 测试：Mock 实现通过 isinstance 检查
# =============================================================================


class TestMockImplementation:
    """验证 Mock 实现可以通过 isinstance 检查。"""

    @pytest.mark.parametrize(
        ("mock", "interface"),
        [
            (MockToolGateway(), IToolGateway),
            (MockContextGateway(), IContextGateway),
            (MockValidationGateway(), IValidationGateway),
        ],
    )
    def test_mock_isinstance(
        self, mock: object, interface: type
    ) -> None:
        assert isinstance(mock, interface), (
            f"{type(mock).__name__} should satisfy {interface.__name__}"
        )


# =============================================================================
# 测试：接口方法签名一致性
# =============================================================================


class TestInterfaceSignatures:
    """验证接口方法签名（参数名、类型注解、返回值）符合预期。"""

    # ── IToolGateway ──────────────────────────────────────────────────────

    def test_tool_gateway_execute_tool_signature(self) -> None:
        sig = inspect.signature(IToolGateway.execute_tool)
        params = sig.parameters

        assert "tool_name" in params
        assert params["tool_name"].annotation is str

        assert "params" in params
        assert params["params"].annotation is dict

        hints = get_type_hints(IToolGateway.execute_tool)
        assert hints["return"] is ToolResult

    def test_tool_gateway_list_tools_signature(self) -> None:
        sig = inspect.signature(IToolGateway.list_tools)
        hints = get_type_hints(IToolGateway.list_tools)
        assert hints["return"] == list[ToolDefinition]

    def test_tool_gateway_validate_tool_params_signature(self) -> None:
        sig = inspect.signature(IToolGateway.validate_tool_params)
        hints = get_type_hints(IToolGateway.validate_tool_params)
        assert hints["return"] is bool

    # ── IContextGateway ───────────────────────────────────────────────────

    def test_context_gateway_build_context_signature(self) -> None:
        sig = inspect.signature(IContextGateway.build_context)
        params = sig.parameters

        assert "project_root" in params
        assert params["project_root"].annotation is str

        assert "query" in params
        assert params["query"].annotation is str

        hints = get_type_hints(IContextGateway.build_context)
        assert hints["return"] is ContextPackage

    def test_context_gateway_update_index_signature(self) -> None:
        sig = inspect.signature(IContextGateway.update_index)
        params = sig.parameters
        assert "project_root" in params
        assert params["project_root"].annotation is str

        hints = get_type_hints(IContextGateway.update_index)
        assert hints["return"] is type(None)

    def test_context_gateway_search_semantic_signature(self) -> None:
        sig = inspect.signature(IContextGateway.search_semantic)
        params = sig.parameters

        assert "query" in params
        assert params["query"].annotation is str

        assert "top_k" in params
        # 默认值检查
        assert params["top_k"].default == 5

        hints = get_type_hints(IContextGateway.search_semantic)
        assert hints["return"] == list[CodeSnippet]

    # ── IValidationGateway ────────────────────────────────────────────────

    def test_validation_gateway_run_syntax_check_signature(self) -> None:
        sig = inspect.signature(IValidationGateway.run_syntax_check)
        params = sig.parameters
        assert "file_path" in params
        assert params["file_path"].annotation is str

        hints = get_type_hints(IValidationGateway.run_syntax_check)
        assert hints["return"] is ValidationResult

    def test_validation_gateway_run_lint_signature(self) -> None:
        sig = inspect.signature(IValidationGateway.run_lint)
        params = sig.parameters
        assert "files" in params
        assert params["files"].annotation == list[str]

        hints = get_type_hints(IValidationGateway.run_lint)
        assert hints["return"] is ValidationResult

    def test_validation_gateway_run_tests_signature(self) -> None:
        sig = inspect.signature(IValidationGateway.run_tests)
        params = sig.parameters
        assert "project_root" in params
        assert params["project_root"].annotation is str

        hints = get_type_hints(IValidationGateway.run_tests)
        assert hints["return"] is ValidationResult

    def test_validation_gateway_run_runtime_check_signature(self) -> None:
        sig = inspect.signature(IValidationGateway.run_runtime_check)
        params = sig.parameters
        assert "file_path" in params
        assert params["file_path"].annotation is str

        hints = get_type_hints(IValidationGateway.run_runtime_check)
        assert hints["return"] is ValidationResult


# =============================================================================
# 测试：DTO 数据类字段
# =============================================================================


class TestToolResult:
    """验证 ToolResult dataclass 字段类型和默认值。"""

    def test_fields(self) -> None:
        fields = {f.name: f for f in ToolResult.__dataclass_fields__.values()}

        assert fields["success"].type is bool
        assert fields["data"].type is Any
        assert fields["error_message"].type == str | None
        assert fields["error_code"].type == str | None
        assert fields["duration_ms"].type is float
        assert fields["tokens_consumed"].type is int

    def test_defaults(self) -> None:
        result = ToolResult(success=True)
        assert result.data is None
        assert result.error_message is None
        assert result.error_code is None
        assert result.duration_ms == 0.0
        assert result.tokens_consumed == 0

    def test_full_construction(self) -> None:
        result = ToolResult(
            success=False,
            data={"partial": "output"},
            error_message="Something went wrong",
            error_code="ERR_001",
            duration_ms=123.4,
            tokens_consumed=50,
        )
        assert result.success is False
        assert result.data == {"partial": "output"}
        assert result.error_message == "Something went wrong"


class TestToolDefinition:
    """验证 ToolDefinition dataclass 字段类型和默认值。"""

    def test_fields(self) -> None:
        fields = {
            f.name: f for f in ToolDefinition.__dataclass_fields__.values()
        }

        assert fields["name"].type is str
        assert fields["description"].type is str
        assert fields["parameters_schema"].type is dict

    def test_default_parameters_schema(self) -> None:
        td = ToolDefinition(name="test", description="A test tool")
        assert td.parameters_schema == {}

    def test_full_construction(self) -> None:
        schema = {"type": "object", "properties": {"path": {"type": "string"}}}
        td = ToolDefinition(
            name="read_file",
            description="Read a file",
            parameters_schema=schema,
        )
        assert td.name == "read_file"
        assert td.parameters_schema == schema


class TestCodeSnippet:
    """验证 CodeSnippet dataclass 字段。"""

    def test_fields(self) -> None:
        fields = {
            f.name: f for f in CodeSnippet.__dataclass_fields__.values()
        }

        assert fields["file_path"].type is str
        assert fields["start_line"].type is int
        assert fields["end_line"].type is int
        assert fields["code"].type is str
        assert fields["score"].type is float

    def test_default_score(self) -> None:
        snippet = CodeSnippet(
            file_path="main.py", start_line=1, end_line=10, code="..."
        )
        assert snippet.score == 0.0


class TestContextPackage:
    """验证 ContextPackage dataclass 字段。"""

    def test_fields(self) -> None:
        fields = {
            f.name: f for f in ContextPackage.__dataclass_fields__.values()
        }

        assert fields["related_code"].type == list[CodeSnippet]
        assert fields["dependency_info"].type is dict

    def test_defaults(self) -> None:
        pkg = ContextPackage()
        assert pkg.file_tree is None
        assert pkg.related_code == []
        assert pkg.dependency_info == {}


class TestValidationError:
    """验证 ValidationError dataclass 字段。"""

    def test_fields(self) -> None:
        fields = {
            f.name: f for f in ValidationError.__dataclass_fields__.values()
        }

        assert fields["file_path"].type is str
        assert fields["line"].type is int
        assert fields["column"].type is int
        assert fields["message"].type is str
        assert fields["code"].type is str
        assert fields["severity"].type is str

    def test_default_severity(self) -> None:
        err = ValidationError(file_path="main.py")
        assert err.severity == "error"


class TestValidationResult:
    """验证 ValidationResult dataclass 字段。"""

    def test_fields(self) -> None:
        fields = {
            f.name: f for f in ValidationResult.__dataclass_fields__.values()
        }

        assert fields["passed"].type is bool
        assert fields["errors"].type == list[ValidationError]
        assert fields["warnings"].type == list[ValidationError]
        assert fields["duration_ms"].type is float

    def test_defaults(self) -> None:
        result = ValidationResult()
        assert result.passed is True
        assert result.errors == []
        assert result.warnings == []
        assert result.duration_ms == 0.0


# =============================================================================
# 测试：Mock 行为验证
# =============================================================================


class TestMockBehavior:
    """验证 Mock 实现的基本行为。"""

    @pytest.mark.asyncio
    async def test_mock_tool_gateway_execute(self) -> None:
        gw = MockToolGateway()
        result = await gw.execute_tool("read_file", {"path": "test.py"})
        assert result.success is True
        assert result.data == "mock"

    @pytest.mark.asyncio
    async def test_mock_context_gateway_build(self) -> None:
        gw = MockContextGateway()
        pkg = await gw.build_context("/project", "query")
        assert isinstance(pkg, ContextPackage)

    @pytest.mark.asyncio
    async def test_mock_validation_gateway_syntax_check(self) -> None:
        gw = MockValidationGateway()
        result = await gw.run_syntax_check("test.py")
        assert result.passed is True
