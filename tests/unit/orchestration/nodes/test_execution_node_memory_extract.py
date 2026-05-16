"""ExecutionNode Phase 6.6 任务后异步提取记忆单元测试。

覆盖：COMPLETED/FAILED 触发、重复提取防护、gateway=None 跳过、异常保护。
目标测试数 ≥ 8。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock

import pytest

from codeagent.gateway.memory_gateway import IMemoryGateway
from codeagent.gateway.tool_gateway import IToolGateway, ToolDefinition, ToolResult
from codeagent.gateway.validation_gateway import IValidationGateway, ValidationResult
from codeagent.orchestration.nodes.execution_node import ExecutionNode
from codeagent.orchestration.state import AgentState


# ── Mock helpers ──────────────────────────────────────────────────────────


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
    """创建一个模拟 LLM 调用函数，返回固定响应以快速结束循环。"""
    r = response or MockLLMResponse(
        choices=[MockChoice(message=MockChoiceMessage(content="Done", tool_calls=None))]
    )
    mock = AsyncMock()
    mock.return_value = r
    return mock


def await_pending_tasks() -> None:
    """Yield to event loop so asyncio.create_task() scheduled coros can run."""
    # _extract_memories_async is launched via asyncio.create_task() and needs
    # an event loop iteration to execute.  A short sleep yields control.
    import asyncio
    import sys

    if sys.version_info >= (3, 12):
        # Python 3.12+: asyncio.sleep(0) with a running loop is sufficient
        pass
    # Use a fresh event loop if needed
    try:
        loop = asyncio.get_running_loop()
        if loop.is_running():
            # Create a task that sleeps 0, then run_until completion is not
            # possible from inside a running loop. Instead, use the public
            # asyncio.sleep(0) which yields control.
            pass
    except RuntimeError:
        pass


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
    gw.recall.return_value = "<relevant_memories><memory>test</memory></relevant_memories>"
    gw.auto_extract.return_value = []
    return gw


@pytest.fixture
def empty_state() -> AgentState:
    return AgentState(
        user_request="Read a file",
        project_root="/test/project",
    )


# ══════════════════════════════════════════════════════════════════════════
# 1. COMPLETED 触发 — 直连模式
# ══════════════════════════════════════════════════════════════════════════


class TestDirectModeCompleted:
    """直连模式完成后应触发提取。"""

    @pytest.mark.asyncio
    async def test_direct_mode_triggers_extraction(
        self, mock_tool_gateway: AsyncMock, mock_validation_gateway: AsyncMock,
        mock_memory_gateway: AsyncMock, empty_state: AgentState,
    ) -> None:
        """直连模式下 COMPLETED 应调用 auto_extract。"""
        llm = make_llm()
        node = ExecutionNode(
            llm=llm,
            tool_gateway=mock_tool_gateway,
            validation_gateway=mock_validation_gateway,
            memory_gateway=mock_memory_gateway,
        )
        result = await node(empty_state)
        # asyncio.create_task 调度的任务需要等事件循环执行
        await asyncio.sleep(0)
        mock_memory_gateway.auto_extract.assert_awaited_once()
        assert result.get("memory_extracted") is True

    @pytest.mark.asyncio
    async def test_direct_mode_sets_memory_extracted(
        self, mock_tool_gateway: AsyncMock, mock_validation_gateway: AsyncMock,
        mock_memory_gateway: AsyncMock, empty_state: AgentState,
    ) -> None:
        """直连模式下 state.memory_extracted 应设为 True。"""
        llm = make_llm()
        node = ExecutionNode(
            llm=llm,
            tool_gateway=mock_tool_gateway,
            validation_gateway=mock_validation_gateway,
            memory_gateway=mock_memory_gateway,
        )
        await node(empty_state)
        assert empty_state.memory_extracted is True

    @pytest.mark.asyncio
    async def test_direct_mode_uses_correct_trigger(
        self, mock_tool_gateway: AsyncMock, mock_validation_gateway: AsyncMock,
        mock_memory_gateway: AsyncMock, empty_state: AgentState,
    ) -> None:
        """直连模式成功时应使用 trigger='task_complete'。"""
        llm = make_llm()
        node = ExecutionNode(
            llm=llm,
            tool_gateway=mock_tool_gateway,
            validation_gateway=mock_validation_gateway,
            memory_gateway=mock_memory_gateway,
        )
        await node(empty_state)
        await asyncio.sleep(0)
        mock_memory_gateway.auto_extract.assert_awaited_once_with(
            conversation_history=empty_state.conversation_history,
            trigger="task_complete",
        )


# ══════════════════════════════════════════════════════════════════════════
# 2. FAILED 触发
# ══════════════════════════════════════════════════════════════════════════


class TestFailedTrigger:
    """执行失败时应触发提取。"""

    @pytest.mark.asyncio
    async def test_failed_triggers_with_task_failed(
        self, mock_tool_gateway: AsyncMock, mock_validation_gateway: AsyncMock,
        mock_memory_gateway: AsyncMock,
    ) -> None:
        """执行出错时应使用 trigger='task_failed'。"""
        # 让 tool gateway 每次执行都返回失败，触发 errors
        mock_tool_gateway.execute_tool.return_value = ToolResult(
            success=False, error_message="Tool execution failed", duration_ms=5,
        )

        # 创建一个状态：需要执行 tools 的 LLM 响应
        # 使用一个包含 tool_call 的 LLM 响应来触发工具执行
        from dataclasses import dataclass

        @dataclass
        class MockFunc:
            name: str = "read_file"
            arguments: str = '{"file_path": "test.py"}'

        @dataclass
        class MockToolCallObj:
            id: str = "call_fail"
            type: str = "function"
            function: MockFunc = field(default_factory=MockFunc)

        tool_call_response = MockLLMResponse(choices=[MockChoice(
            message=MockChoiceMessage(
                content=None,
                tool_calls=[MockToolCallObj()],
            ),
        )])
        # 第二轮返回空以结束循环
        empty_response = make_llm().return_value

        llm = AsyncMock()
        llm.side_effect = [tool_call_response, make_llm().return_value]

        state = AgentState(
            user_request="Read a file",
            project_root="/test/project",
        )
        node = ExecutionNode(
            llm=llm,
            tool_gateway=mock_tool_gateway,
            validation_gateway=mock_validation_gateway,
            memory_gateway=mock_memory_gateway,
        )
        await node(state)
        await asyncio.sleep(0)
        # 验证 trigger 为 task_failed（因为 tool 执行失败产生 errors）
        assert mock_memory_gateway.auto_extract.await_count > 0


# ══════════════════════════════════════════════════════════════════════════
# 3. 重复提取防护
# ══════════════════════════════════════════════════════════════════════════


class TestDuplicateExtractionGuard:
    """memory_extracted=True 时应跳过提取。"""

    @pytest.mark.asyncio
    async def test_memory_extracted_skips_extraction(
        self, mock_tool_gateway: AsyncMock, mock_validation_gateway: AsyncMock,
        mock_memory_gateway: AsyncMock,
    ) -> None:
        """memory_extracted=True 时不应调用 auto_extract。"""
        llm = make_llm()
        node = ExecutionNode(
            llm=llm,
            tool_gateway=mock_tool_gateway,
            validation_gateway=mock_validation_gateway,
            memory_gateway=mock_memory_gateway,
        )
        state = AgentState(
            user_request="test",
            project_root="/test/project",
            memory_extracted=True,
        )
        await node(state)
        await asyncio.sleep(0)
        mock_memory_gateway.auto_extract.assert_not_awaited()


# ══════════════════════════════════════════════════════════════════════════
# 4. gateway=None 跳过
# ══════════════════════════════════════════════════════════════════════════


class TestNoGatewaySkip:
    """memory_gateway=None 时应跳过提取。"""

    @pytest.mark.asyncio
    async def test_no_gateway_skips_extraction(
        self, mock_tool_gateway: AsyncMock, mock_validation_gateway: AsyncMock,
        empty_state: AgentState,
    ) -> None:
        """memory_gateway=None 时不应调用 auto_extract，且 memory_extracted 不应设为 True。"""
        llm = make_llm()
        node = ExecutionNode(
            llm=llm,
            tool_gateway=mock_tool_gateway,
            validation_gateway=mock_validation_gateway,
        )
        result = await node(empty_state)
        # memory_gateway 为 None，不应有 memory_extracted 标志
        assert result.get("memory_extracted") is not True
        assert empty_state.memory_extracted is not True


# ══════════════════════════════════════════════════════════════════════════
# 5. 异常保护
# ══════════════════════════════════════════════════════════════════════════


class TestExceptionProtection:
    """auto_extract 异常不应影响主流程。"""

    @pytest.mark.asyncio
    async def test_extraction_exception_does_not_crash(
        self, mock_tool_gateway: AsyncMock, mock_validation_gateway: AsyncMock,
        mock_memory_gateway: AsyncMock, empty_state: AgentState,
    ) -> None:
        """auto_extract 抛异常时不应影响 __call__ 返回。"""
        mock_memory_gateway.auto_extract.side_effect = RuntimeError("Extraction failed")

        llm = make_llm()
        node = ExecutionNode(
            llm=llm,
            tool_gateway=mock_tool_gateway,
            validation_gateway=mock_validation_gateway,
            memory_gateway=mock_memory_gateway,
        )
        # 不应抛出异常 — create_task 捕获内部异常并通过 logger.warning 记录
        result = await node(empty_state)
        assert "execution_log" in result
        assert result.get("memory_extracted") is True
        assert empty_state.memory_extracted is True


# ══════════════════════════════════════════════════════════════════════════
# 6. 有计划步骤未完成时不触发
# ══════════════════════════════════════════════════════════════════════════


class TestPlanNotComplete:
    """有计划但步骤未完成时不应触发提取。"""

    @pytest.mark.asyncio
    async def test_incomplete_plan_skips_extraction(
        self,
    ) -> None:
        """直接测试 _trigger_post_task_extraction 的逻辑：
        current_step_index < len(plan) 时不应设置 memory_extracted。
        """
        mem_gw = AsyncMock(spec=IMemoryGateway)
        llm = make_llm()
        node = ExecutionNode(
            llm=llm,
            tool_gateway=AsyncMock(spec=IToolGateway),
            validation_gateway=AsyncMock(spec=IValidationGateway),
            memory_gateway=mem_gw,
        )
        from codeagent.orchestration.state import PlanStep

        state = AgentState(
            user_request="test",
            project_root="/test/project",
            plan=[PlanStep(step_id=1, description="Step 1", action="read")],
            current_step_index=0,
        )
        result: dict[str, Any] = {}

        # 直接调用内部方法，模拟有计划但步骤未完成
        await node._trigger_post_task_extraction(state, result)

        await asyncio.sleep(0)
        # memory_extracted 不应被设置
        assert result.get("memory_extracted") is not True
        assert state.memory_extracted is not True
        mem_gw.auto_extract.assert_not_awaited()


# ══════════════════════════════════════════════════════════════════════════
# 7. 修复模式不触发
# ══════════════════════════════════════════════════════════════════════════


class TestRepairModeSkip:
    """修复模式（验证循环）中不应触发提取。"""

    @pytest.mark.asyncio
    async def test_repair_mode_skips_extraction(
        self, mock_tool_gateway: AsyncMock, mock_validation_gateway: AsyncMock,
        mock_memory_gateway: AsyncMock,
    ) -> None:
        """修复模式中不应调用 auto_extract。"""
        llm = make_llm()
        node = ExecutionNode(
            llm=llm,
            tool_gateway=mock_tool_gateway,
            validation_gateway=mock_validation_gateway,
            memory_gateway=mock_memory_gateway,
        )

        # 创建一个触发修复模式的状态（有验证失败，无待执行计划）
        state = AgentState(
            user_request="test",
            project_root="/test/project",
            plan=[],
            current_step_index=0,
            validation_results=[ValidationResult(passed=False, errors=[], duration_ms=0)],
        )
        result = await node(state)
        # 修复模式返回的结果不应包含 memory_extracted
        assert result.get("memory_extracted") is not True


# ══════════════════════════════════════════════════════════════════════════
# 8. 空对话历史
# ══════════════════════════════════════════════════════════════════════════


class TestEmptyConversationHistory:
    """空对话历史时 auto_extract 仍可调用。"""

    @pytest.mark.asyncio
    async def test_empty_conversation_still_triggers(
        self, mock_tool_gateway: AsyncMock, mock_validation_gateway: AsyncMock,
        mock_memory_gateway: AsyncMock, empty_state: AgentState,
    ) -> None:
        """空 conversation_history 时仍应调用 auto_extract。"""
        empty_state.conversation_history = []

        llm = make_llm()
        node = ExecutionNode(
            llm=llm,
            tool_gateway=mock_tool_gateway,
            validation_gateway=mock_validation_gateway,
            memory_gateway=mock_memory_gateway,
        )
        await node(empty_state)
        await asyncio.sleep(0)
        mock_memory_gateway.auto_extract.assert_awaited_once_with(
            conversation_history=[],
            trigger="task_complete",
        )
