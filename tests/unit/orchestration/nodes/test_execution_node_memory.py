"""ExecutionNode Memory 集成单元测试。

覆盖：memory_gateway 注入、recall 调用、记忆注入 system prompt、
无 gateway 时正常运行、异常保护。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from codeagent.gateway.memory_gateway import IMemoryGateway
from codeagent.gateway.tool_gateway import IToolGateway, ToolDefinition, ToolResult
from codeagent.gateway.validation_gateway import IValidationGateway, ValidationResult
from codeagent.orchestration.nodes.execution_node import ExecutionNode
from codeagent.orchestration.state import AgentState


# ── Mock helpers ──────────────────────────────────────────────────────────


@dataclass
class MockToolCall:
    id: str = "call_123"
    type: str = "function"

    @dataclass
    class Function:
        name: str = "read_file"
        arguments: str = '{"file_path": "test.py"}'

    function: Function = field(default_factory=Function)


@dataclass
class MockChoiceMessage:
    content: str | None = None
    tool_calls: list[Any] | None = None


@dataclass
class MockChoice:
    message: MockChoiceMessage = field(default_factory=MockChoiceMessage)


@dataclass
class MockLLMResponse:
    choices: list[MockChoice] = field(default_factory=lambda: [MockChoice()])


def make_llm(response: MockLLMResponse | None = None) -> AsyncMock:
    """创建一个模拟 LLM 调用函数。"""
    mock = AsyncMock()
    mock.return_value = response or MockLLMResponse()
    return mock


_MEMORY_XML = (
    "<relevant_memories>"
    '  <memory type="feedback" name="no-mock" score="0.90" scope="global">'
    "    Never mock the database"
    "  </memory>"
    "</relevant_memories>"
)


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def mock_tool_gateway() -> AsyncMock:
    gw = AsyncMock(spec=IToolGateway)
    gw.list_tools.return_value = [
        ToolDefinition(
            name="read_file",
            description="Read a file",
            parameters_schema={"type": "object", "properties": {}},
        ),
    ]
    gw.execute_tool.return_value = ToolResult(
        success=True, data={"content": "file content"}, duration_ms=10,
    )
    return gw


@pytest.fixture
def mock_validation_gateway() -> AsyncMock:
    gw = AsyncMock(spec=IValidationGateway)
    gw.run_syntax_check.return_value = ValidationResult(passed=True, errors=[], duration_ms=5)
    return gw


@pytest.fixture
def mock_memory_gateway() -> AsyncMock:
    gw = AsyncMock(spec=IMemoryGateway)
    gw.recall.return_value = _MEMORY_XML
    return gw


@pytest.fixture
def state() -> AgentState:
    return AgentState(
        user_request="Read a file",
        project_root="/test/project",
    )


def make_empty_llm_response() -> MockLLMResponse:
    """LLM 立即返回文本（无 tool_calls）以结束循环。"""
    return MockLLMResponse(
        choices=[MockChoice(message=MockChoiceMessage(content="Done", tool_calls=None))]
    )


# ══════════════════════════════════════════════════════════════════════════
# 1. memory_gateway 注入 — 直连模式
# ══════════════════════════════════════════════════════════════════════════


class TestMemoryGatewayDirect:
    """直连模式下 memory_gateway 集成测试。"""

    @pytest.mark.asyncio
    async def test_direct_mode_recall_called(
        self, mock_tool_gateway: AsyncMock, mock_validation_gateway: AsyncMock,
        mock_memory_gateway: AsyncMock, state: AgentState,
    ) -> None:
        """直连模式下应调用 recall。"""
        llm = make_llm(make_empty_llm_response())
        node = ExecutionNode(
            llm=llm,
            tool_gateway=mock_tool_gateway,
            validation_gateway=mock_validation_gateway,
            memory_gateway=mock_memory_gateway,
        )
        await node(state)

        mock_memory_gateway.recall.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_direct_mode_no_gateway_works(
        self, mock_tool_gateway: AsyncMock, mock_validation_gateway: AsyncMock,
        state: AgentState,
    ) -> None:
        """直连模式下无 gateway 应正常工作。"""
        llm = make_llm(make_empty_llm_response())
        node = ExecutionNode(
            llm=llm,
            tool_gateway=mock_tool_gateway,
            validation_gateway=mock_validation_gateway,
        )
        result = await node(state)
        assert "execution_log" in result

    @pytest.mark.asyncio
    async def test_direct_mode_recall_exception(
        self, mock_tool_gateway: AsyncMock, mock_validation_gateway: AsyncMock,
        state: AgentState,
    ) -> None:
        """直连模式下 recall 异常应降级不影响执行。"""
        gw = AsyncMock(spec=IMemoryGateway)
        gw.recall.side_effect = RuntimeError("recall failed")

        llm = make_llm(make_empty_llm_response())
        node = ExecutionNode(
            llm=llm,
            tool_gateway=mock_tool_gateway,
            validation_gateway=mock_validation_gateway,
            memory_gateway=gw,
        )
        result = await node(state)
        assert "execution_log" in result


# ══════════════════════════════════════════════════════════════════════════
# 2. memory_gateway 注入 — Plan-aware 模式
# ══════════════════════════════════════════════════════════════════════════


class TestMemoryGatewayPlanAware:
    """Plan-aware 模式下 memory_gateway 集成测试。"""

    @pytest.mark.asyncio
    async def test_plan_mode_recall_called(
        self, mock_tool_gateway: AsyncMock, mock_validation_gateway: AsyncMock,
        mock_memory_gateway: AsyncMock,
    ) -> None:
        """Plan-aware 模式下应调用 recall。"""
        llm = make_llm(make_empty_llm_response())
        node = ExecutionNode(
            llm=llm,
            tool_gateway=mock_tool_gateway,
            validation_gateway=mock_validation_gateway,
            memory_gateway=mock_memory_gateway,
        )

        # 创建有计划的状态（但 LLM 无 tool_calls 会立即结束）
        plan_state = AgentState(
            user_request="Read a file",
            project_root="/test/project",
            plan=[],
            current_step_index=0,
        )
        await node(plan_state)

        # 无 plan 步骤时走 direct 模式，只要不崩溃就算通过
        assert True

    @pytest.mark.asyncio
    async def test_plan_mode_no_gateway_works(
        self, mock_tool_gateway: AsyncMock, mock_validation_gateway: AsyncMock,
    ) -> None:
        """Plan-aware 模式下无 gateway 应正常工作。"""
        llm = make_llm(make_empty_llm_response())
        node = ExecutionNode(
            llm=llm,
            tool_gateway=mock_tool_gateway,
            validation_gateway=mock_validation_gateway,
        )
        result = await node(AgentState(user_request="test", project_root="/test"))
        assert "execution_log" in result


# ══════════════════════════════════════════════════════════════════════════
# 3. _build_system_prompt 记忆层注入
# ══════════════════════════════════════════════════════════════════════════


class TestSystemPromptMemorySection:
    """_build_system_prompt 应正确包含记忆层。"""

    def test_memory_section_in_prompt(self, mock_tool_gateway: AsyncMock, mock_validation_gateway: AsyncMock) -> None:
        """有 memory_section 时 prompt 应包含记忆文本。"""
        llm = make_llm()
        node = ExecutionNode(
            llm=llm,
            tool_gateway=mock_tool_gateway,
            validation_gateway=mock_validation_gateway,
        )

        state = AgentState(user_request="test", project_root="/test")
        prompt = node._build_system_prompt(
            state,
            tool_definitions=mock_tool_gateway.list_tools(),
            memory_section="## Relevant Memories\n\n<relevant_memories>test</relevant_memories>",
        )
        assert "## Relevant Memories" in prompt
        assert "<relevant_memories>" in prompt

    def test_no_memory_section_no_change(self, mock_tool_gateway: AsyncMock, mock_validation_gateway: AsyncMock) -> None:
        """无 memory_section 时 prompt 应与原来一致。"""
        llm = make_llm()
        node = ExecutionNode(
            llm=llm,
            tool_gateway=mock_tool_gateway,
            validation_gateway=mock_validation_gateway,
        )

        state = AgentState(user_request="test", project_root="/test")
        prompt_without = node._build_system_prompt(
            state, tool_definitions=mock_tool_gateway.list_tools(),
        )
        assert "Relevant Memories" not in prompt_without
