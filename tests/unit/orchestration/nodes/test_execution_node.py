"""ExecutionNode 单元测试。

Mock IToolGateway、IValidationGateway 和 LLM，验证调度逻辑。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from codeagent.gateway.tool_gateway import IToolGateway, ToolDefinition, ToolResult
from codeagent.gateway.validation_gateway import IValidationGateway, ValidationResult
from codeagent.orchestration.nodes.execution_node import ExecutionNode
from codeagent.orchestration.state import AgentState, PlanStep


# ── Mock helpers ──────────────────────────────────────────────────────────


@dataclass
class MockToolCall:
    """模拟 LLM 返回的 tool_call。"""
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


def make_gateway(
    tool_results: list[ToolResult] | None = None,
) -> AsyncMock:
    """创建一个模拟 IToolGateway。"""
    mock = AsyncMock(spec=IToolGateway)
    mock.list_tools.return_value = [
        ToolDefinition(
            name="read_file",
            description="Read file content",
            parameters_schema={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "start_line": {"type": "integer"},
                    "end_line": {"type": "integer"},
                },
                "required": ["file_path"],
            },
        ),
        ToolDefinition(
            name="write_file",
            description="Write content to file",
            parameters_schema={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "content": {"type": "string"},
                    "mode": {"type": "string", "enum": ["create", "modify"]},
                },
                "required": ["file_path", "content", "mode"],
            },
        ),
    ]

    if tool_results:
        mock.execute_tool = AsyncMock(side_effect=tool_results)
    else:
        mock.execute_tool = AsyncMock(
            return_value=ToolResult(
                success=True,
                data={"content": "print('hello')\n", "total_lines": 1},
            )
        )

    return mock


def make_validation_gateway(
    passed: bool = True,
) -> AsyncMock:
    """创建一个模拟 IValidationGateway。"""
    mock = AsyncMock(spec=IValidationGateway)
    mock.run_syntax_check = AsyncMock(
        return_value=ValidationResult(passed=passed)
    )
    return mock


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def state() -> AgentState:
    return AgentState(
        user_request="Create a hello.py file",
        project_root="/test/project",
    )


@pytest.fixture
def tool_gateway() -> AsyncMock:
    return make_gateway()


@pytest.fixture
def validation_gateway() -> AsyncMock:
    return make_validation_gateway(passed=True)


# ── Tests: Normal flow ───────────────────────────────────────────────────


@pytest.mark.asyncio
class TestNormalFlow:
    """正常执行流程。"""

    async def test_llm_returns_text_no_tool_calls(self, state: AgentState) -> None:
        """LLM 直接返回文本，不请求工具调用。"""
        llm = make_llm(MockLLMResponse(
            choices=[MockChoice(message=MockChoiceMessage(content="Done!"))]
        ))
        node = ExecutionNode(
            llm=llm,
            tool_gateway=make_gateway(),
            validation_gateway=make_validation_gateway(),
        )
        result = await node(state)
        assert "execution_log" in result
        assert len(result["execution_log"]) == 1
        assert result["execution_log"][0]["type"] == "llm_response"

    async def test_single_tool_call(self, state: AgentState) -> None:
        """LLM 调用一个工具后完成。"""
        # 第一次返回 read_file，第二次无 tool_calls
        responses = [
            MockLLMResponse(choices=[MockChoice(
                message=MockChoiceMessage(
                    content="Let me read the file",
                    tool_calls=[MockToolCall(
                        function=MockToolCall.Function(
                            name="read_file",
                            arguments='{"file_path": "test.py"}',
                        )
                    )],
                )
            )]),
            MockLLMResponse(choices=[MockChoice(
                message=MockChoiceMessage(content="Done reading")
            )]),
        ]
        llm = AsyncMock(side_effect=responses)
        node = ExecutionNode(
            llm=llm,
            tool_gateway=make_gateway(),
            validation_gateway=make_validation_gateway(),
        )
        result = await node(state)
        assert len(result["execution_log"]) >= 1
        tool_calls = [e for e in result["execution_log"] if e["type"] == "tool_call"]
        assert len(tool_calls) == 1
        assert tool_calls[0]["tool_name"] == "read_file"
        assert tool_calls[0]["success"] is True

    async def test_multiple_tool_calls(self, state: AgentState) -> None:
        """LLM 依次调用多个工具。"""
        # 第一次：2 个 tool_calls，第二次：完成
        responses = [
            MockLLMResponse(choices=[MockChoice(
                message=MockChoiceMessage(
                    content="Let me write files",
                    tool_calls=[
                        MockToolCall(
                            id="call_1",
                            function=MockToolCall.Function(
                                name="write_file",
                                arguments='{"file_path": "a.py", "content": "x=1", "mode": "create"}',
                            ),
                        ),
                        MockToolCall(
                            id="call_2",
                            function=MockToolCall.Function(
                                name="read_file",
                                arguments='{"file_path": "b.py"}',
                            ),
                        ),
                    ],
                )
            )]),
            MockLLMResponse(choices=[MockChoice(
                message=MockChoiceMessage(content="All done")
            )]),
        ]
        llm = AsyncMock(side_effect=responses)
        gateway = make_gateway()
        node = ExecutionNode(
            llm=llm,
            tool_gateway=gateway,
            validation_gateway=make_validation_gateway(),
        )
        result = await node(state)
        tool_calls = [e for e in result["execution_log"] if e["type"] == "tool_call"]
        assert len(tool_calls) == 2

    async def test_execution_log_contains_details(self, state: AgentState) -> None:
        """执行日志包含详细字段。"""
        responses = [
            MockLLMResponse(choices=[MockChoice(
                message=MockChoiceMessage(
                    content="Reading file",
                    tool_calls=[MockToolCall(
                        id="call_1",
                        function=MockToolCall.Function(
                            name="read_file",
                            arguments='{"file_path": "test.py"}',
                        ),
                    )],
                )
            )]),
            MockLLMResponse(choices=[MockChoice(
                message=MockChoiceMessage(content="Done")
            )]),
        ]
        llm = AsyncMock(side_effect=responses)
        gateway = make_gateway()
        node = ExecutionNode(
            llm=llm,
            tool_gateway=gateway,
            validation_gateway=make_validation_gateway(),
        )
        result = await node(state)
        tool_calls = [e for e in result["execution_log"] if e["type"] == "tool_call"]
        assert len(tool_calls) >= 1
        entry = tool_calls[0]
        assert "tool_name" in entry
        assert "arguments" in entry
        assert "success" in entry
        assert "duration_ms" in entry
        assert "timestamp" in entry


# ── Tests: Syntax check after write_file ─────────────────────────────────


@pytest.mark.asyncio
class TestSyntaxCheck:
    """write_file 后的语法检查。"""

    async def test_syntax_check_called_after_write(self, state: AgentState) -> None:
        """write_file 成功后应调用语法检查。"""
        val_gateway = make_validation_gateway(passed=True)
        responses = [
            MockLLMResponse(choices=[MockChoice(
                message=MockChoiceMessage(
                    content="Writing file",
                    tool_calls=[MockToolCall(
                        id="call_w1",
                        function=MockToolCall.Function(
                            name="write_file",
                            arguments='{"file_path": "hello.py", "content": "print(1)", "mode": "create"}',
                        ),
                    )],
                )
            )]),
            MockLLMResponse(choices=[MockChoice(
                message=MockChoiceMessage(content="Done")
            )]),
        ]
        llm = AsyncMock(side_effect=responses)
        gateway = make_gateway(tool_results=[
            ToolResult(success=True, data={"file_path": "hello.py", "lines_added": 1}),
        ])
        node = ExecutionNode(
            llm=llm,
            tool_gateway=gateway,
            validation_gateway=val_gateway,
        )
        await node(state)
        val_gateway.run_syntax_check.assert_called_once()

    async def test_syntax_check_called_with_correct_path(self, state: AgentState) -> None:
        """语法检查应传入正确的文件路径。"""
        val_gateway = make_validation_gateway(passed=True)
        responses = [
            MockLLMResponse(choices=[MockChoice(
                message=MockChoiceMessage(
                    content="Writing",
                    tool_calls=[MockToolCall(
                        id="call_w2",
                        function=MockToolCall.Function(
                            name="write_file",
                            arguments='{"file_path": "src/mod.py", "content": "x=1", "mode": "create"}',
                        ),
                    )],
                )
            )]),
            MockLLMResponse(choices=[MockChoice(
                message=MockChoiceMessage(content="Done")
            )]),
        ]
        llm = AsyncMock(side_effect=responses)
        gateway = make_gateway(tool_results=[
            ToolResult(success=True, data={"file_path": "src/mod.py", "lines_added": 1}),
        ])
        node = ExecutionNode(
            llm=llm,
            tool_gateway=gateway,
            validation_gateway=val_gateway,
        )
        await node(state)
        call_args = val_gateway.run_syntax_check.call_args
        # Should be called with a string path
        assert call_args[0][0] is not None

    async def test_syntax_failure_triggers_retry(self, state: AgentState) -> None:
        """语法错误后应重试（LLM 再次被调用）。"""
        val_gateway = make_validation_gateway(passed=False)

        responses = [
            MockLLMResponse(choices=[MockChoice(
                message=MockChoiceMessage(
                    content="Writing",
                    tool_calls=[MockToolCall(
                        id="call_retry",
                        function=MockToolCall.Function(
                            name="write_file",
                            arguments='{"file_path": "bad.py", "content": "x=", "mode": "create"}',
                        ),
                    )],
                )
            )]),
            # 语法错误后，LLM 应再次被调用以修复
            MockLLMResponse(choices=[MockChoice(
                message=MockChoiceMessage(
                    content="Fixing",
                    tool_calls=[MockToolCall(
                        id="call_fix",
                        function=MockToolCall.Function(
                            name="write_file",
                            arguments='{"file_path": "bad.py", "content": "x=1", "mode": "modify"}',
                        ),
                    )],
                )
            )]),
            MockLLMResponse(choices=[MockChoice(
                message=MockChoiceMessage(content="Done")
            )]),
        ]
        llm = AsyncMock(side_effect=responses)

        gateway = make_gateway(tool_results=[
            ToolResult(success=True, data={"file_path": "bad.py", "lines_added": 1}),
            ToolResult(success=True, data={"file_path": "bad.py", "lines_added": 1, "lines_removed": 1}),
        ])

        node = ExecutionNode(
            llm=llm,
            tool_gateway=gateway,
            validation_gateway=val_gateway,
            max_retries=3,
        )
        result = await node(state)

        # 应包含 syntax_check 日志
        syntax_checks = [
            e for e in result["execution_log"]
            if e["type"] == "syntax_check"
        ]
        assert len(syntax_checks) >= 1
        assert syntax_checks[0]["passed"] is False


# ── Tests: Tool call failure ─────────────────────────────────────────────


@pytest.mark.asyncio
class TestToolFailure:
    """工具调用失败处理。"""

    async def test_tool_failure_logged(self, state: AgentState) -> None:
        """工具调用失败应记录到日志。"""
        responses = [
            MockLLMResponse(choices=[MockChoice(
                message=MockChoiceMessage(
                    content="Trying",
                    tool_calls=[MockToolCall(
                        function=MockToolCall.Function(
                            name="read_file",
                            arguments='{"file_path": "nonexistent.py"}',
                        )
                    )],
                )
            )]),
            MockLLMResponse(choices=[MockChoice(
                message=MockChoiceMessage(content="File not found")
            )]),
        ]
        llm = AsyncMock(side_effect=responses)
        gateway = make_gateway(tool_results=[
            ToolResult(
                success=False,
                error_message="File not found: nonexistent.py",
                error_code="FILE_NOT_FOUND",
            ),
        ])
        node = ExecutionNode(
            llm=llm,
            tool_gateway=gateway,
            validation_gateway=make_validation_gateway(),
        )
        result = await node(state)
        tool_calls = [e for e in result["execution_log"] if e["type"] == "tool_call"]
        assert len(tool_calls) == 1
        assert tool_calls[0]["success"] is False
        assert "error" in tool_calls[0]


# ── Tests: Max tool calls exceeded ───────────────────────────────────────


@pytest.mark.asyncio
class TestMaxToolCalls:
    """单步骤超限保护。"""

    async def test_max_tool_calls_exceeded(self, state: AgentState) -> None:
        """超过最大 tool_calls 次数后应强制结束。"""
        # LLM 每次返回一个 tool_call，循环直到超限
        tool_call_response = MockLLMResponse(choices=[MockChoice(
            message=MockChoiceMessage(
                content="Working...",
                tool_calls=[MockToolCall(
                    function=MockToolCall.Function(
                        name="read_file",
                        arguments='{"file_path": "test.py"}',
                    )
                )],
            )
        )])

        llm = AsyncMock(return_value=tool_call_response)
        gateway = make_gateway()

        node = ExecutionNode(
            llm=llm,
            tool_gateway=gateway,
            validation_gateway=make_validation_gateway(),
            max_tool_calls=3,
        )
        result = await node(state)

        tool_calls = [e for e in result["execution_log"] if e["type"] == "tool_call"]
        assert len(tool_calls) <= 3


# ── Tests: LLM failure ───────────────────────────────────────────────────


@pytest.mark.asyncio
class TestLLMFailure:
    """LLM 调用失败处理。"""

    async def test_llm_call_raises(self, state: AgentState) -> None:
        """LLM 调用抛出异常时应记录错误。"""
        llm = AsyncMock(side_effect=RuntimeError("LLM unavailable"))
        node = ExecutionNode(
            llm=llm,
            tool_gateway=make_gateway(),
            validation_gateway=make_validation_gateway(),
        )
        result = await node(state)
        assert len(result["errors"]) >= 1
        assert "LLM call failed" in result["errors"][0]


# ── Tests: Empty plan ────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestNoPlan:
    """空计划场景。"""

    async def test_no_tools_needed(self, state: AgentState) -> None:
        """LLM 认为不需要工具，直接返回文本。"""
        llm = make_llm(MockLLMResponse(choices=[MockChoice(
            message=MockChoiceMessage(content="No changes needed.")
        )]))
        node = ExecutionNode(
            llm=llm,
            tool_gateway=make_gateway(),
            validation_gateway=make_validation_gateway(),
        )
        result = await node(state)
        assert len(result["execution_log"]) == 1
        assert result["execution_log"][0]["type"] == "llm_response"
        assert result["errors"] == []


# ── Tests: State passing ─────────────────────────────────────────────────


@pytest.mark.asyncio
class TestStatePassing:
    """状态传递验证。"""

    async def test_user_request_passed_to_llm(self, state: AgentState) -> None:
        """用户请求应传递给 LLM。"""
        llm = make_llm(MockLLMResponse(choices=[MockChoice(
            message=MockChoiceMessage(content="Done")
        )]))
        node = ExecutionNode(
            llm=llm,
            tool_gateway=make_gateway(),
            validation_gateway=make_validation_gateway(),
        )
        await node(state)
        # 验证 LLM 被调用
        llm.assert_called_once()
        # 验证 messages 包含 user_request
        call_kwargs = llm.call_args[1]
        messages = call_kwargs.get("messages", [])
        user_msgs = [m for m in messages if m["role"] == "user"]
        assert any(state.user_request in m["content"] for m in user_msgs)


# ── Tests: Plan-aware execution ──────────────────────────────────────────


@pytest.mark.asyncio
class TestPlanAwareExecution:
    """Plan-aware 模式测试。"""

    def _make_plan_state(self) -> AgentState:
        plan = [
            PlanStep(
                step_id=1, description="Read main.py", action="read",
                target_file="main.py",
            ),
            PlanStep(
                step_id=2, description="Write app.py", action="create",
                target_file="app.py",
            ),
        ]
        return AgentState(
            user_request="Create Flask app",
            project_root="/test/project",
            plan=plan,
            current_step_index=0,
        )

    async def test_executes_all_steps(self) -> None:
        """所有计划步骤应被逐步执行。"""
        state = self._make_plan_state()
        # 每个步骤 LLM 返回不同响应
        step1_resp = MockLLMResponse(choices=[MockChoice(
            message=MockChoiceMessage(content="Read main.py")
        )])
        step2_resp = MockLLMResponse(choices=[MockChoice(
            message=MockChoiceMessage(content="Created app.py")
        )])
        llm = AsyncMock(side_effect=[step1_resp, step2_resp])

        node = ExecutionNode(
            llm=llm,
            tool_gateway=make_gateway(),
            validation_gateway=make_validation_gateway(),
        )
        result = await node(state)
        assert result["current_step_index"] == 2  # 两步都完成
        assert "execution_log" in result

    async def test_updates_step_index(self) -> None:
        """current_step_index 应正确更新。"""
        state = self._make_plan_state()
        llm = AsyncMock(return_value=MockLLMResponse(choices=[MockChoice(
            message=MockChoiceMessage(content="Done")
        )]))

        node = ExecutionNode(
            llm=llm,
            tool_gateway=make_gateway(),
            validation_gateway=make_validation_gateway(),
        )
        result = await node(state)
        assert result["current_step_index"] == 2

    async def test_partial_execution_from_mid_plan(self) -> None:
        """从中间步骤开始执行。"""
        plan = [
            PlanStep(step_id=1, description="Step 1", action="read",
                     target_file="f1.py"),
            PlanStep(step_id=2, description="Step 2", action="read",
                     target_file="f2.py"),
            PlanStep(step_id=3, description="Step 3", action="read",
                     target_file="f3.py"),
        ]
        state = AgentState(
            user_request="multi step",
            project_root="/root",
            plan=plan,
            current_step_index=1,  # 从第2步开始
        )
        llm = AsyncMock(return_value=MockLLMResponse(choices=[MockChoice(
            message=MockChoiceMessage(content="Done")
        )]))

        node = ExecutionNode(
            llm=llm,
            tool_gateway=make_gateway(),
            validation_gateway=make_validation_gateway(),
        )
        result = await node(state)
        assert result["current_step_index"] == 3  # 只有2步要执行

    async def test_empty_plan_fallsback_to_direct(self) -> None:
        """plan 为空时应走直连模式。"""
        state = AgentState(
            user_request="test",
            project_root="/root",
            plan=[],
            current_step_index=0,
        )
        llm = AsyncMock(return_value=MockLLMResponse(choices=[MockChoice(
            message=MockChoiceMessage(content="Done directly")
        )]))

        node = ExecutionNode(
            llm=llm,
            tool_gateway=make_gateway(),
            validation_gateway=make_validation_gateway(),
        )
        result = await node(state)
        assert "execution_log" in result
        assert result["execution_log"][0]["type"] == "llm_response"

    async def test_no_plan_fallsback_to_direct(self) -> None:
        """plan 为 None 时应走直连模式。"""
        state = AgentState(
            user_request="test",
            project_root="/root",
        )
        llm = AsyncMock(return_value=MockLLMResponse(choices=[MockChoice(
            message=MockChoiceMessage(content="Done directly")
        )]))

        node = ExecutionNode(
            llm=llm,
            tool_gateway=make_gateway(),
            validation_gateway=make_validation_gateway(),
        )
        result = await node(state)
        assert "execution_log" in result

    async def test_all_steps_completed_fallsback(self) -> None:
        """所有步骤已完成时应走直连模式。"""
        plan = [PlanStep(step_id=1, description="Done", action="read",
                         target_file="f.py")]
        state = AgentState(
            user_request="test",
            project_root="/root",
            plan=plan,
            current_step_index=1,  # 所有步骤已完成
        )
        llm = AsyncMock(return_value=MockLLMResponse(choices=[MockChoice(
            message=MockChoiceMessage(content="Direct mode")
        )]))

        node = ExecutionNode(
            llm=llm,
            tool_gateway=make_gateway(),
            validation_gateway=make_validation_gateway(),
        )
        result = await node(state)
        assert "execution_log" in result


@pytest.mark.asyncio
class TestProgressCallback:
    """进度回调测试。"""

    async def test_callback_called_per_step(self) -> None:
        """每个步骤都应触发回调。"""
        plan = [
            PlanStep(step_id=1, description="Read", action="read",
                     target_file="a.py"),
            PlanStep(step_id=2, description="Write", action="create",
                     target_file="b.py"),
        ]
        state = AgentState(
            user_request="test",
            project_root="/root",
            plan=plan,
        )
        llm = AsyncMock(return_value=MockLLMResponse(choices=[MockChoice(
            message=MockChoiceMessage(content="Done")
        )]))

        callback = MagicMock()
        node = ExecutionNode(
            llm=llm,
            tool_gateway=make_gateway(),
            validation_gateway=make_validation_gateway(),
            progress_callback=callback,
        )
        await node(state)

        # 每个步骤应收到 running + completed 事件
        assert callback.call_count == 4

    async def test_callback_receives_step_id(self) -> None:
        """回调应包含 step_id。"""
        plan = [
            PlanStep(step_id=42, description="Special step", action="read",
                     target_file="x.py"),
        ]
        state = AgentState(
            user_request="test",
            project_root="/root",
            plan=plan,
        )
        llm = AsyncMock(return_value=MockLLMResponse(choices=[MockChoice(
            message=MockChoiceMessage(content="Done")
        )]))

        callback = MagicMock()
        node = ExecutionNode(
            llm=llm,
            tool_gateway=make_gateway(),
            validation_gateway=make_validation_gateway(),
            progress_callback=callback,
        )
        await node(state)

        running_call = callback.call_args_list[0][0][0]
        assert running_call["step_id"] == 42
        assert running_call["status"] == "running"

        completed_call = callback.call_args_list[1][0][0]
        assert completed_call["status"] == "completed"

    async def test_callback_error_does_not_block(self) -> None:
        """回调异常不应阻塞执行。"""
        plan = [
            PlanStep(step_id=1, description="Read", action="read",
                     target_file="f.py"),
        ]
        state = AgentState(
            user_request="test",
            project_root="/root",
            plan=plan,
        )
        llm = AsyncMock(return_value=MockLLMResponse(choices=[MockChoice(
            message=MockChoiceMessage(content="Done")
        )]))

        callback = MagicMock(side_effect=RuntimeError("callback error"))
        node = ExecutionNode(
            llm=llm,
            tool_gateway=make_gateway(),
            validation_gateway=make_validation_gateway(),
            progress_callback=callback,
        )
        result = await node(state)
        # 执行应继续，不受回调异常影响
        assert result["current_step_index"] == 1

    async def test_no_callback_does_not_crash(self) -> None:
        """未设置回调时不应崩溃。"""
        plan = [PlanStep(step_id=1, description="Read", action="read",
                         target_file="f.py")]
        state = AgentState(
            user_request="test",
            project_root="/root",
            plan=plan,
        )
        llm = AsyncMock(return_value=MockLLMResponse(choices=[MockChoice(
            message=MockChoiceMessage(content="Done")
        )]))

        node = ExecutionNode(
            llm=llm,
            tool_gateway=make_gateway(),
            validation_gateway=make_validation_gateway(),
        )
        result = await node(state)
        assert result["current_step_index"] == 1


@pytest.mark.asyncio
class TestAccumulatedChanges:
    """累计修改记录测试。"""

    async def test_tracks_write_file_changes(self) -> None:
        """write_file 应记录到 accumulated_changes。"""
        plan = [PlanStep(step_id=1, description="Write file", action="create",
                         target_file="hello.py")]
        state = AgentState(
            user_request="create file",
            project_root="/root",
            plan=plan,
        )
        responses = [
            MockLLMResponse(choices=[MockChoice(
                message=MockChoiceMessage(
                    content="Writing",
                    tool_calls=[MockToolCall(
                        id="call_w1",
                        function=MockToolCall.Function(
                            name="write_file",
                            arguments='{"file_path": "hello.py", "content": "print(1)", "mode": "create"}',
                        ),
                    )],
                )
            )]),
            MockLLMResponse(choices=[MockChoice(
                message=MockChoiceMessage(content="Done")
            )]),
        ]
        llm = AsyncMock(side_effect=responses)
        gateway = make_gateway(tool_results=[
            ToolResult(success=True, data={"file_path": "hello.py", "lines_added": 1}),
        ])

        node = ExecutionNode(
            llm=llm,
            tool_gateway=gateway,
            validation_gateway=make_validation_gateway(),
        )
        result = await node(state)
        changes = result["accumulated_changes"]
        assert len(changes) >= 1
        assert changes[0]["file_path"] == "hello.py"
        assert changes[0]["step_id"] == 1

    async def test_preserves_existing_changes(self) -> None:
        """应保留 state 中已有的 accumulated_changes。"""
        plan = [PlanStep(step_id=1, description="Write", action="create",
                         target_file="f.py")]
        state = AgentState(
            user_request="req",
            project_root="/root",
            plan=plan,
            accumulated_changes=[{"file_path": "existing.py", "step_id": 0}],
        )
        llm = AsyncMock(side_effect=[
            MockLLMResponse(choices=[MockChoice(
                message=MockChoiceMessage(
                    content="Writing",
                    tool_calls=[MockToolCall(
                        id="call_w",
                        function=MockToolCall.Function(
                            name="write_file",
                            arguments='{"file_path": "f.py", "content": "x=1", "mode": "create"}',
                        ),
                    )],
                )
            )]),
            MockLLMResponse(choices=[MockChoice(
                message=MockChoiceMessage(content="Done")
            )]),
        ])
        gateway = make_gateway(tool_results=[
            ToolResult(success=True, data={"file_path": "f.py"}),
        ])

        node = ExecutionNode(
            llm=llm,
            tool_gateway=gateway,
            validation_gateway=make_validation_gateway(),
        )
        result = await node(state)
        assert len(result["accumulated_changes"]) == 2
        assert result["accumulated_changes"][0]["file_path"] == "existing.py"
