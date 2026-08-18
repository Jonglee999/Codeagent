"""ExecutionNode 单元测试。

Mock IToolGateway、IValidationGateway 和 LLM，验证调度逻辑。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from codeagent.gateway.tool_gateway import IToolGateway, ToolDefinition, ToolResult
from codeagent.gateway.validation_gateway import IValidationGateway, ValidationError, ValidationResult
from codeagent.orchestration.nodes.execution_node import (
    _ALLOWED_TOOLS_BY_ACTION,
    _MAX_PRESERVED_REASONING_CHARS,
    _MAX_TOOL_OBSERVATION_CHARS,
    ExecutionNode,
    _assistant_retry_message,
    _provider_response_diagnostic,
    _recover_dsml_tool_calls,
    _is_test_command,
    _mentioned_source_paths,
    _tool_result_message,
)
from codeagent.orchestration.state import AgentState, PlanStep, RepairContext, StructuredError


def test_provider_diagnostic_does_not_persist_private_reasoning_text() -> None:
    diagnostic = _provider_response_diagnostic(
        type(
            "Message",
            (),
            {"content": "visible", "reasoning_content": "private chain of thought"},
        )(),
        finish_reason="length",
    )

    assert diagnostic["content_present"] is True
    assert diagnostic["reasoning_present"] is True
    assert diagnostic["reasoning_chars"] == len("private chain of thought")
    assert diagnostic["finish_reason"] == "length"
    assert "private chain of thought" not in str(diagnostic)


def test_tool_result_message_preserves_failure_recovery_details() -> None:
    result = ToolResult(
        success=False,
        error_message="unsafe command",
        error_code="SAFETY_BLOCKED",
        data={"retry_guidance": "run a direct relative command"},
    )

    message = _tool_result_message(result)

    assert "SAFETY_BLOCKED" in message
    assert "run a direct relative command" in message


def test_tool_result_message_bounds_large_observations_and_keeps_tail() -> None:
    result = ToolResult(
        success=True,
        data={"stdout": "A" * 20_000 + "FINAL_TEST_FAILURE"},
    )

    message = _tool_result_message(result)

    assert len(message) == _MAX_TOOL_OBSERVATION_CHARS
    assert "TOOL_OBSERVATION_TRUNCATED" in message
    assert "FINAL_TEST_FAILURE" in message


def test_recovers_provider_dsml_tool_call_from_plain_text() -> None:
    message = MockChoiceMessage(content="""<｜｜DSML｜｜tool_calls>
<｜｜DSML｜｜invoke name="apply_patch">
<｜｜DSML｜｜parameter name="file_path" string="true">django/admin.py</｜｜DSML｜｜parameter>
<｜｜DSML｜｜parameter name="patch" string="true">*** Begin Patch
+value = &quot;fixed&quot;
*** End Patch</｜｜DSML｜｜parameter>
</｜｜DSML｜｜invoke>
</｜｜DSML｜｜tool_calls>""")

    calls = _recover_dsml_tool_calls(message)

    assert len(calls) == 1
    assert calls[0].function.name == "apply_patch"
    assert json.loads(calls[0].function.arguments) == {
        "file_path": "django/admin.py",
        "patch": '*** Begin Patch\n+value = "fixed"\n*** End Patch',
    }


def test_does_not_parse_dsml_example_embedded_in_prose() -> None:
    message = MockChoiceMessage(
        content="For example: <｜｜DSML｜｜tool_calls></｜｜DSML｜｜tool_calls>"
    )

    assert _recover_dsml_tool_calls(message) == []


def test_recovers_dsml_tool_call_from_reasoning_content() -> None:
    message = MockChoiceMessage(content="")
    message.reasoning_content = """<|DSML|tool_calls>
<|DSML|invoke name="write_file">
<|DSML|parameter name="file_path">fix.py</|DSML|parameter>
<|DSML|parameter name="content">value = 2</|DSML|parameter>
</|DSML|invoke>
</|DSML|tool_calls>"""

    calls = _recover_dsml_tool_calls(message)

    assert len(calls) == 1
    assert calls[0].function.name == "write_file"
    assert json.loads(calls[0].function.arguments) == {
        "file_path": "fix.py",
        "content": "value = 2",
    }


def test_retry_message_preserves_reasoning_only_provider_response() -> None:
    message = MockChoiceMessage(content="")
    message.reasoning_content = "I found the target and will patch it next."

    retry = _assistant_retry_message(message, "fallback")

    assert retry == {
        "role": "assistant",
        "content": "",
        "reasoning_content": "I found the target and will patch it next.",
    }


def test_retry_message_bounds_preserved_reasoning() -> None:
    message = MockChoiceMessage(content="")
    message.reasoning_content = "A" * (_MAX_PRESERVED_REASONING_CHARS + 500)

    retry = _assistant_retry_message(message, "fallback")

    assert len(retry["reasoning_content"]) == _MAX_PRESERVED_REASONING_CHARS


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("pytest -q tests/test_case.py::test_it", True),
        ("python -m pytest -q", True),
        ("python tests/runtests.py admin_views", True),
        ("python manage.py test app.tests", True),
        ('grep -rn "has_add_permission" tests | head -30', False),
        ("find tests -name '*.py'", False),
    ],
)
def test_test_command_evidence_requires_a_test_runner(
    command: str, expected: bool
) -> None:
    assert _is_test_command(command) is expected


def test_extracts_explicit_source_paths_from_issue() -> None:
    assert _mentioned_source_paths(
        'At "django/contrib/admin/templatetags/admin_modify.py" add a check.'
    ) == {"django/contrib/admin/templatetags/admin_modify.py"}


@pytest.mark.asyncio
async def test_token_budget_stops_llm_before_another_call(mocker) -> None:
    mocker.patch(
        "codeagent.orchestration.nodes.execution_node.codeagent_config.get_max_llm_calls_per_task",
        return_value=50,
    )
    mocker.patch(
        "codeagent.orchestration.nodes.execution_node.codeagent_config.get_max_tokens_per_task",
        return_value=100,
    )
    state = AgentState(
        user_request="test",
        project_root="/root",
        estimated_tokens=100,
    )
    llm = AsyncMock()
    node = ExecutionNode(
        llm=llm,
        tool_gateway=make_gateway(),
        validation_gateway=make_validation_gateway(),
    )

    result = await node._call_llm_with_limit(
        state=state,
        messages=[{"role": "user", "content": "hello"}],
        tools=None,
        tool_choice=None,
    )

    assert result is None
    assert state.review_type == "token_budget_exhausted"
    llm.assert_not_called()


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

    async def test_planned_mutation_requires_provider_tool_call(self) -> None:
        state = AgentState(
            user_request="Fix benchmark issue",
            project_root="/test/project",
            benchmark_instance_id="owner__repo-1",
            plan=[PlanStep(
                step_id=1,
                description="Implement fix",
                action="modify",
                acceptance_criteria="Apply a source patch",
            )],
        )
        llm = AsyncMock(return_value=MockLLMResponse(choices=[MockChoice(
            message=MockChoiceMessage(
                tool_calls=[MockToolCall(function=MockToolCall.Function(
                    name="write_file",
                    arguments=(
                        '{"file_path": "fix.py", "content": "fixed = True", '
                        '"mode": "modify"}'
                    ),
                ))],
            )
        )]))
        node = ExecutionNode(
            llm=llm,
            tool_gateway=make_gateway(tool_results=[
                ToolResult(success=True, data={"file_path": "fix.py"}),
            ]),
            validation_gateway=make_validation_gateway(),
        )

        result = await node(state)

        assert result["accumulated_changes"][0]["file_path"] == "fix.py"
        assert llm.await_args.kwargs["tool_choice"] == "auto"

    async def test_direct_benchmark_requires_test_attempt_after_mutation(self) -> None:
        state = AgentState(
            user_request="Fix benchmark issue",
            project_root="/test/project",
            benchmark_instance_id="owner__repo-1",
            benchmark_fail_to_pass=["tests/test_regression.py::test_case"],
        )
        write_call = MockToolCall(function=MockToolCall.Function(
            name="write_file",
            arguments=(
                '{"file_path": "fix.py", "content": "fixed = True", '
                '"mode": "modify"}'
            ),
        ))
        test_call = MockToolCall(
            id="run_target_test",
            function=MockToolCall.Function(
                name="run_terminal",
                arguments=(
                    '{"command": "pytest -q '
                    'tests/test_regression.py::test_case"}'
                ),
            ),
        )
        llm = AsyncMock(side_effect=[
            MockLLMResponse(choices=[MockChoice(
                message=MockChoiceMessage(tool_calls=[write_call])
            )]),
            MockLLMResponse(choices=[MockChoice(
                message=MockChoiceMessage(content="Implemented")
            )]),
            MockLLMResponse(choices=[MockChoice(
                message=MockChoiceMessage(tool_calls=[test_call])
            )]),
            MockLLMResponse(choices=[MockChoice(
                message=MockChoiceMessage(content="Tested")
            )]),
        ])
        gateway = make_gateway(tool_results=[
            ToolResult(success=True, data={"file_path": "fix.py"}),
            ToolResult(success=True, data={"exit_code": 0, "stdout": "1 passed"}),
        ])
        gateway.list_tools.return_value.append(ToolDefinition(
            name="run_terminal",
            description="Run a command",
            parameters_schema={
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        ))
        node = ExecutionNode(
            llm=llm,
            tool_gateway=gateway,
            validation_gateway=make_validation_gateway(),
        )

        result = await node(state)

        assert gateway.execute_tool.await_count == 2
        assert any(
            "has not been tested" in str(message.get("content", ""))
            for message in llm.await_args_list[2].kwargs["messages"]
        )
        stages = [
            entry["stage"]
            for entry in result["execution_log"]
            if entry["type"] == "benchmark_stage"
        ]
        assert stages == ["locate", "target_test", "regression_test"]

    async def test_direct_benchmark_forces_mutation_after_explicit_target_stagnates(
        self,
    ) -> None:
        target = "pkg/target.py"
        state = AgentState(
            user_request=f'At "{target}" add the missing permission check.',
            project_root="/test/project",
            benchmark_instance_id="owner__repo-2",
        )

        def read_call(index: int, path: str) -> MockLLMResponse:
            return MockLLMResponse(choices=[MockChoice(message=MockChoiceMessage(
                tool_calls=[MockToolCall(
                    id=f"read_{index}",
                    function=MockToolCall.Function(
                        name="read_file",
                        arguments=json.dumps({"file_path": path}),
                    ),
                )]
            ))])

        write_call = MockLLMResponse(choices=[MockChoice(message=MockChoiceMessage(
            tool_calls=[MockToolCall(
                id="write_fix",
                function=MockToolCall.Function(
                    name="write_file",
                    arguments=json.dumps({
                        "file_path": target,
                        "content": "fixed = True\n",
                        "mode": "modify",
                    }),
                ),
            )]
        ))])
        disallowed_read = read_call(5, "pkg/another_helper.py")
        reasoning_only = MockLLMResponse(choices=[MockChoice(
            message=MockChoiceMessage(content="I should make the change now.")
        )])
        recovery_read = read_call(6, target)
        test_call = MockLLMResponse(choices=[MockChoice(message=MockChoiceMessage(
            tool_calls=[MockToolCall(
                id="test_fix",
                function=MockToolCall.Function(
                    name="run_terminal",
                    arguments='{"command": "pytest -q tests/test_target.py"}',
                ),
            )]
        ))])
        llm = AsyncMock(side_effect=[
            read_call(1, target),
            read_call(2, "tests/test_target.py"),
            read_call(3, "pkg/helper.py"),
            read_call(4, "pkg/config.py"),
            disallowed_read,
            reasoning_only,
            write_call,
            recovery_read,
            write_call,
            test_call,
            MockLLMResponse(choices=[MockChoice(
                message=MockChoiceMessage(content="Fixed and tested")
            )]),
        ])
        gateway = make_gateway(tool_results=[
            *[
                ToolResult(success=True, data={"content": "source\n"})
                for _ in range(5)
            ],
            ToolResult(
                success=False,
                error_message="Patch conflict",
                error_code="PATCH_CONFLICT",
                retryable=True,
            ),
            ToolResult(success=True, data={"content": "current source\n"}),
            ToolResult(success=True, data={"file_path": target}),
            ToolResult(success=True, data={"exit_code": 0, "stdout": "1 passed"}),
        ])
        gateway.list_tools.return_value.append(ToolDefinition(
            name="run_terminal",
            description="Run a command",
            parameters_schema={"type": "object"},
        ))
        node = ExecutionNode(
            llm=llm,
            tool_gateway=gateway,
            validation_gateway=make_validation_gateway(),
        )

        result = await node(state)

        fifth_tool_names = {
            tool["function"]["name"]
            for tool in llm.await_args_list[4].kwargs["tools"]
        }
        assert fifth_tool_names == {"read_file", "write_file"}
        assert llm.await_args_list[4].kwargs["tool_choice"] == "auto"
        assert llm.await_args_list[5].kwargs["tool_choice"] == "auto"
        recovery_tool_names = {
            tool["function"]["name"]
            for tool in llm.await_args_list[7].kwargs["tools"]
        }
        assert recovery_tool_names == {"read_file", "write_file"}
        assert gateway.execute_tool.await_count == 9
        rejected = [
            entry
            for entry in result["execution_log"]
            if entry.get("error_code") == "CAPABILITY_NOT_EXPOSED"
        ]
        assert rejected == []
        assert not any("Exceeded max tool calls" in error for error in result["errors"])
        guards = [
            entry
            for entry in result["execution_log"]
            if entry["type"] == "benchmark_stagnation_guard"
        ]
        assert guards == [
            {
                "type": "benchmark_stagnation_guard",
                "discovery_calls": 4,
                "explicit_target_inspected": True,
                "timestamp": guards[0]["timestamp"],
            }
        ]
        assert any(
            entry["type"] == "benchmark_patch_conflict_recovery"
            for entry in result["execution_log"]
        )

    async def test_action_follow_up_reprompts_instead_of_repeating_prior_answer(self) -> None:
        state = AgentState(
            user_request="Run it again and return the result",
            project_root="/test/project",
            conversation_history=[
                {"role": "user", "content": "Create hello.py"},
                {"role": "assistant", "content": "Created it. Output: hello"},
                {"role": "user", "content": "Run it again and return the result"},
            ],
        )
        responses = [
            MockLLMResponse(choices=[MockChoice(
                message=MockChoiceMessage(content="Created it. Output: hello")
            )]),
            MockLLMResponse(choices=[MockChoice(
                message=MockChoiceMessage(
                    content="I will verify it now",
                    tool_calls=[MockToolCall()],
                )
            )]),
            MockLLMResponse(choices=[MockChoice(
                message=MockChoiceMessage(content="Fresh result: hello")
            )]),
        ]
        llm = AsyncMock(side_effect=responses)
        node = ExecutionNode(
            llm=llm,
            tool_gateway=make_gateway(),
            validation_gateway=make_validation_gateway(),
        )

        result = await node(state)

        assert llm.await_count == 3
        assert any(entry["type"] == "tool_call" for entry in result["execution_log"])
        second_messages = llm.await_args_list[1].kwargs["messages"]
        assert any(
            "Do not repeat the previous answer" in str(message.get("content", ""))
            for message in second_messages
        )

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
        assert tool_calls[0]["error_code"] == "FILE_NOT_FOUND"
        retry_messages = llm.call_args_list[1].kwargs["messages"]
        tool_message = next(item for item in retry_messages if item["role"] == "tool")
        assert '"error_code": "FILE_NOT_FOUND"' in tool_message["content"]
        assert '"retryable": false' in tool_message["content"]
        assert '"suggested_recovery"' in tool_message["content"]
        assert '"bounded_output": true' in tool_message["content"]

    async def test_benchmark_direct_mode_reprompts_until_file_change(self) -> None:
        state = AgentState(
            user_request="fix benchmark issue",
            project_root="/root",
            direct_execution=True,
            benchmark_instance_id="owner__repo-1",
        )
        llm = AsyncMock(side_effect=[
            MockLLMResponse(choices=[MockChoice(
                message=MockChoiceMessage(content="The fix should be small")
            )]),
            MockLLMResponse(choices=[MockChoice(
                message=MockChoiceMessage(
                    content="Applying fix",
                    tool_calls=[MockToolCall(
                        id="call_fix",
                        function=MockToolCall.Function(
                            name="write_file",
                            arguments=(
                                '{"file_path": "fix.py", "content": "fixed = True", '
                                '"mode": "create"}'
                            ),
                        ),
                    )],
                )
            )]),
            MockLLMResponse(choices=[MockChoice(
                message=MockChoiceMessage(content="Implemented")
            )]),
            MockLLMResponse(choices=[MockChoice(
                message=MockChoiceMessage(tool_calls=[MockToolCall(
                    id="call_test",
                    function=MockToolCall.Function(
                        name="run_terminal",
                        arguments='{"command": "pytest -q"}',
                    ),
                )])
            )]),
            MockLLMResponse(choices=[MockChoice(
                message=MockChoiceMessage(content="Implemented and tested")
            )]),
        ])
        gateway = make_gateway(tool_results=[
            ToolResult(success=True, data={"file_path": "fix.py"}),
            ToolResult(success=True, data={"exit_code": 0, "stdout": "1 passed"}),
        ])
        gateway.list_tools.return_value.append(ToolDefinition(
            name="run_terminal",
            description="Run a command",
            parameters_schema={"type": "object"},
        ))
        node = ExecutionNode(
            llm=llm,
            tool_gateway=gateway,
            validation_gateway=make_validation_gateway(),
        )

        result = await node(state)

        assert llm.await_count == 5
        assert result["accumulated_changes"][0]["file_path"] == "fix.py"
        second_messages = llm.call_args_list[1].kwargs["messages"]
        assert any(
            "no successful file mutation" in item.get("content", "")
            for item in second_messages
            if item["role"] == "user"
        )


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
        step2_tool_resp = MockLLMResponse(choices=[MockChoice(
            message=MockChoiceMessage(
                content="Creating app.py",
                tool_calls=[MockToolCall(function=MockToolCall.Function(
                    name="write_file",
                    arguments='{"file_path": "app.py", "content": "app = 1", "mode": "create"}',
                ))],
            )
        )])
        step2_done_resp = MockLLMResponse(choices=[MockChoice(
            message=MockChoiceMessage(content="Created app.py")
        )])
        llm = AsyncMock(side_effect=[step1_resp, step2_tool_resp, step2_done_resp])

        node = ExecutionNode(
            llm=llm,
            tool_gateway=make_gateway(),
            validation_gateway=make_validation_gateway(),
        )
        result = await node(state)
        assert result["current_step_index"] == 2  # 两步都完成
        assert "execution_log" in result

    async def test_missing_mutation_evidence_does_not_advance_step(self) -> None:
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
        assert result["current_step_index"] == 1
        assert any("Step evidence missing" in error for error in result["errors"])

    async def test_read_with_acceptance_requires_tool_evidence(self) -> None:
        state = AgentState(
            user_request="inspect file",
            project_root="/root",
            plan=[PlanStep(
                step_id=1,
                description="Read main.py",
                action="read",
                target_file="main.py",
                acceptance_criteria="read_file returns the relevant source",
            )],
        )
        node = ExecutionNode(
            llm=AsyncMock(return_value=MockLLMResponse(choices=[MockChoice(
                message=MockChoiceMessage(content="Done without reading")
            )])),
            tool_gateway=make_gateway(),
            validation_gateway=make_validation_gateway(),
        )

        result = await node(state)

        assert result["current_step_index"] == 0
        assert any("Step evidence missing" in error for error in result["errors"])

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

    async def test_tracks_apply_patch_as_modify_evidence(self, tmp_path) -> None:
        target = tmp_path / "hello.py"
        target.write_text("value = 1\n", encoding="utf-8")
        plan = [PlanStep(
            step_id=1,
            description="Patch file",
            action="modify",
            target_file="hello.py",
        )]
        state = AgentState(
            user_request="modify file",
            project_root=str(tmp_path),
            plan=plan,
        )
        responses = [MockLLMResponse(choices=[MockChoice(
            message=MockChoiceMessage(
                content="Patching",
                tool_calls=[MockToolCall(
                    id="call_patch",
                    function=MockToolCall.Function(
                        name="apply_patch",
                        arguments=(
                            '{"file_path": "hello.py", "old_text": "value = 1", '
                            '"new_text": "value = 2"}'
                        ),
                    ),
                )],
            )
        )])]
        gateway = make_gateway(tool_results=[
            ToolResult(success=True, data={"file_path": "hello.py", "replacements": 1}),
        ])
        node = ExecutionNode(
            llm=AsyncMock(side_effect=responses),
            tool_gateway=gateway,
            validation_gateway=make_validation_gateway(),
        )

        result = await node(state)

        assert result["current_step_index"] == 1
        assert result["accumulated_changes"][0]["file_path"] == "hello.py"
        assert result["accumulated_changes"][0]["action"] == "modify"
        assert state.plan[0].evidence[0]["tool_name"] == "apply_patch"

    async def test_modify_step_forces_mutation_tools_after_discovery_budget(
        self, tmp_path
    ) -> None:
        target = tmp_path / "main.py"
        target.write_text("value = 1\n", encoding="utf-8")
        state = AgentState(
            user_request="modify file",
            project_root=str(tmp_path),
            plan=[PlanStep(
                step_id=1,
                description="Modify main.py",
                action="modify",
                target_file="main.py",
            )],
        )
        read_responses = [
            MockLLMResponse(choices=[MockChoice(message=MockChoiceMessage(
                tool_calls=[MockToolCall(
                    id=f"read_{index}",
                    function=MockToolCall.Function(
                        name="read_file",
                        arguments='{"file_path": "main.py"}',
                    ),
                )]
            ))])
            for index in range(3)
        ]
        write_response = MockLLMResponse(choices=[MockChoice(
            message=MockChoiceMessage(tool_calls=[MockToolCall(
                id="write_1",
                function=MockToolCall.Function(
                    name="write_file",
                    arguments=(
                        '{"file_path": "main.py", "content": "value = 2\\n", '
                        '"mode": "modify"}'
                    ),
                ),
            )])
        )])
        llm = AsyncMock(side_effect=[*read_responses, write_response])
        gateway = make_gateway(tool_results=[
            ToolResult(success=True, data={"content": "value = 1\n"}),
            ToolResult(success=True, data={"content": "value = 1\n"}),
            ToolResult(success=True, data={"content": "value = 1\n"}),
            ToolResult(success=True, data={"file_path": "main.py"}),
        ])
        node = ExecutionNode(
            llm=llm,
            tool_gateway=gateway,
            validation_gateway=make_validation_gateway(),
        )

        result = await node(state)

        fourth_tools = llm.await_args_list[3].kwargs["tools"]
        fourth_names = {tool["function"]["name"] for tool in fourth_tools}
        assert fourth_names == {"write_file"}
        assert result["current_step_index"] == 1

    async def test_modify_retry_keeps_reasoning_before_forcing_mutation(
        self, tmp_path
    ) -> None:
        target = tmp_path / "main.py"
        target.write_text("value = 1\n", encoding="utf-8")
        state = AgentState(
            user_request="modify file",
            project_root=str(tmp_path),
            plan=[PlanStep(
                step_id=1,
                description="Modify main.py",
                action="modify",
                target_file="main.py",
            )],
        )
        thinking = MockChoiceMessage(content="")
        thinking.reasoning_content = "The exact replacement is value = 2."
        patch_response = MockLLMResponse(choices=[MockChoice(
            message=MockChoiceMessage(tool_calls=[MockToolCall(
                id="write_after_reasoning",
                function=MockToolCall.Function(
                    name="write_file",
                    arguments=(
                        '{"file_path": "main.py", "content": "value = 2\\n", '
                        '"mode": "modify"}'
                    ),
                ),
            )])
        )])
        llm = AsyncMock(side_effect=[
            MockLLMResponse(choices=[MockChoice(message=thinking)]),
            patch_response,
        ])
        node = ExecutionNode(
            llm=llm,
            tool_gateway=make_gateway(tool_results=[
                ToolResult(success=True, data={"file_path": "main.py"}),
            ]),
            validation_gateway=make_validation_gateway(),
        )

        result = await node(state)

        retry_messages = llm.await_args_list[1].kwargs["messages"]
        preserved = next(
            message for message in retry_messages
            if message.get("reasoning_content")
        )
        reprompt = next(
            message for message in retry_messages
            if message.get("role") == "user"
            and "do not restart" in message.get("content", "")
        )
        assert preserved["reasoning_content"] == (
            "The exact replacement is value = 2."
        )
        assert "do not restart" in reprompt["content"]
        assert result["current_step_index"] == 1

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


# ── Phase 3.5: TaskFocus — Deviation Detection ─────────────────────────────


@pytest.mark.asyncio
class TestDeviationDetection:
    """偏离检测测试。"""

    def _make_plan_state(self, step: PlanStep | None = None) -> AgentState:
        if step is None:
            step = PlanStep(
                step_id=1, description="Modify main.py", action="modify",
                target_file="main.py",
            )
        return AgentState(
            user_request="Modify main.py",
            project_root="/test/project",
            plan=[step],
            current_step_index=0,
        )

    async def test_write_to_planned_file_no_deviation(self) -> None:
        """写入计划内的文件不应触发偏离。"""
        step = PlanStep(
            step_id=1, description="Create hello.py", action="create",
            target_file="hello.py",
        )
        state = self._make_plan_state(step)
        responses = [
            MockLLMResponse(choices=[MockChoice(
                message=MockChoiceMessage(
                    content="Writing",
                    tool_calls=[MockToolCall(
                        id="call_1",
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
        # 不应有偏离检测标记
        deviations = [e for e in result["execution_log"] if e["type"] == "deviation_detected"]
        assert len(deviations) == 0

    async def test_list_files_and_apply_patch_are_valid_for_modify(self) -> None:
        step = PlanStep(
            step_id=1,
            description="Inspect and patch main.py",
            action="modify",
            target_file="main.py",
        )
        node = ExecutionNode(
            llm=make_llm(),
            tool_gateway=make_gateway(),
            validation_gateway=make_validation_gateway(),
        )

        assert node._check_deviation(step, "list_files", {"path": "."}) is False
        assert node._check_deviation(
            step, "apply_patch", {"file_path": "main.py"}
        ) is False

    async def test_write_outside_plan_triggers_deviation(self) -> None:
        """写入计划外的文件应触发偏离。"""
        step = PlanStep(
            step_id=1, description="Modify main.py", action="modify",
            target_file="main.py",
        )
        state = self._make_plan_state(step)
        # LLM 尝试写入非目标文件 (other.py vs main.py)
        responses = [
            MockLLMResponse(choices=[MockChoice(
                message=MockChoiceMessage(
                    content="Writing",
                    tool_calls=[MockToolCall(
                        id="call_1",
                        function=MockToolCall.Function(
                            name="write_file",
                            arguments='{"file_path": "other.py", "content": "x=1", "mode": "create"}',
                        ),
                    )],
                )
            )]),
            MockLLMResponse(choices=[MockChoice(
                message=MockChoiceMessage(content="Corrected")
            )]),
        ]
        llm = AsyncMock(side_effect=responses)
        gateway = make_gateway(tool_results=[
            ToolResult(success=True, data={"file_path": "other.py", "lines_added": 1}),
        ])
        node = ExecutionNode(
            llm=llm,
            tool_gateway=gateway,
            validation_gateway=make_validation_gateway(),
        )
        result = await node(state)
        deviations = [e for e in result["execution_log"] if e["type"] == "deviation_detected"]
        assert len(deviations) >= 1
        assert deviations[0]["tool_name"] == "write_file"

    async def test_disallowed_tool_triggers_deviation(self) -> None:
        """使用当前步骤不允许的工具应触发偏离。"""
        step = PlanStep(
            step_id=1, description="Read config", action="read",
            target_file="config.py",
        )
        state = self._make_plan_state(step)
        # read 步骤不应该调用 write_file
        responses = [
            MockLLMResponse(choices=[MockChoice(
                message=MockChoiceMessage(
                    content="Writing unexpectedly",
                    tool_calls=[MockToolCall(
                        id="call_1",
                        function=MockToolCall.Function(
                            name="write_file",
                            arguments='{"file_path": "config.py", "content": "x=1", "mode": "modify"}',
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
            ToolResult(success=True, data={"file_path": "config.py", "lines_added": 1}),
        ])
        node = ExecutionNode(
            llm=llm,
            tool_gateway=gateway,
            validation_gateway=make_validation_gateway(),
        )
        result = await node(state)
        deviations = [e for e in result["execution_log"] if e["type"] == "deviation_detected"]
        assert len(deviations) >= 1

    async def test_command_step_no_file_check(self) -> None:
        """command 类型步骤不检查文件路径偏离。"""
        step = PlanStep(
            step_id=1, description="Run tests", action="command",
            target_file=None,
        )
        state = self._make_plan_state(step)
        responses = [
            MockLLMResponse(choices=[MockChoice(
                message=MockChoiceMessage(
                    content="Running tests",
                    tool_calls=[MockToolCall(
                        id="call_1",
                        function=MockToolCall.Function(
                            name="write_file",
                            arguments='{"file_path": "test_output.log", "content": "results", "mode": "create"}',
                        ),
                    )],
                )
            )]),
            MockLLMResponse(choices=[MockChoice(
                message=MockChoiceMessage(content="Done")
            )]),
        ]
        llm = AsyncMock(side_effect=responses)
        node = ExecutionNode(
            llm=llm,
            tool_gateway=make_gateway(),
            validation_gateway=make_validation_gateway(),
        )
        result = await node(state)
        # command 步骤的 target_file 为 None，写入文件不触发偏离（除非工具名不在允许列表）
        deviations = [e for e in result["execution_log"] if e["type"] == "deviation_detected"]
        # write_file 不在 command 的允许工具列表中，所以应触发
        assert len(deviations) >= 1

    async def test_same_file_via_different_path_no_deviation(self) -> None:
        """使用不同但等效的路径写入目标文件不触发偏离。"""
        step = PlanStep(
            step_id=1, description="Modify app/main.py", action="modify",
            target_file="app/main.py",
        )
        state = self._make_plan_state(step)
        responses = [
            MockLLMResponse(choices=[MockChoice(
                message=MockChoiceMessage(
                    content="Writing",
                    tool_calls=[MockToolCall(
                        id="call_1",
                        function=MockToolCall.Function(
                            name="write_file",
                            arguments='{"file_path": "/root/app/main.py", "content": "x=1", "mode": "modify"}',
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
            ToolResult(success=True, data={"file_path": "/root/app/main.py", "lines_added": 1}),
        ])
        node = ExecutionNode(
            llm=llm,
            tool_gateway=gateway,
            validation_gateway=make_validation_gateway(),
        )
        result = await node(state)
        deviations = [e for e in result["execution_log"] if e["type"] == "deviation_detected"]
        assert len(deviations) == 0


# ── Phase 3.5: TaskFocus — Deviation Escalation ────────────────────────────


@pytest.mark.asyncio
class TestDeviationEscalation:
    """偏离计数升级测试。"""

    async def test_first_deviation_triggers_warning(self) -> None:
        """首次偏离应触发警告级别。"""
        step = PlanStep(
            step_id=1, description="Modify main.py", action="modify",
            target_file="main.py",
        )
        plan = [step]
        state = AgentState(
            user_request="Modify main.py",
            project_root="/test/project",
            plan=plan,
            current_step_index=0,
            original_goal_summary="Modify main.py to add logging",
        )
        # LLM 每次连续偏离，先写 plan2.py（偏离），第二次才写 main.py（纠正）
        responses = [
            MockLLMResponse(choices=[MockChoice(
                message=MockChoiceMessage(
                    content="Writing to wrong file",
                    tool_calls=[MockToolCall(
                        id="call_1",
                        function=MockToolCall.Function(
                            name="write_file",
                            arguments='{"file_path": "plan2.py", "content": "x=1", "mode": "create"}',
                        ),
                    )],
                )
            )]),
            MockLLMResponse(choices=[MockChoice(
                message=MockChoiceMessage(
                    content="Now writing correct file",
                    tool_calls=[MockToolCall(
                        id="call_2",
                        function=MockToolCall.Function(
                            name="write_file",
                            arguments='{"file_path": "main.py", "content": "x=1", "mode": "modify"}',
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
            ToolResult(success=True, data={}),
            ToolResult(success=True, data={}),
        ])
        node = ExecutionNode(
            llm=llm,
            tool_gateway=gateway,
            validation_gateway=make_validation_gateway(),
        )
        result = await node(state)
        deviations = [e for e in result["execution_log"] if e["type"] == "deviation_detected"]
        assert len(deviations) == 1
        assert deviations[0]["deviation_count"] == 1

    async def test_two_deviations_escalates_warning(self) -> None:
        """连续2次偏离应升级警告。"""
        step = PlanStep(
            step_id=1, description="Modify main.py", action="modify",
            target_file="main.py",
        )
        plan = [step]
        state = AgentState(
            user_request="Modify main.py",
            project_root="/test/project",
            plan=plan,
            current_step_index=0,
            original_goal_summary="Modify main.py",
        )
        # LLM 两次都写 wrong.py（连续偏离）
        responses = [
            MockLLMResponse(choices=[MockChoice(
                message=MockChoiceMessage(
                    content="Writing",
                    tool_calls=[MockToolCall(
                        id="call_1",
                        function=MockToolCall.Function(
                            name="write_file",
                            arguments='{"file_path": "wrong.py", "content": "x=1", "mode": "create"}',
                        ),
                    )],
                )
            )]),
            MockLLMResponse(choices=[MockChoice(
                message=MockChoiceMessage(
                    content="Still writing wrong",
                    tool_calls=[MockToolCall(
                        id="call_2",
                        function=MockToolCall.Function(
                            name="write_file",
                            arguments='{"file_path": "wrong.py", "content": "y=2", "mode": "create"}',
                        ),
                    )],
                )
            )]),
            MockLLMResponse(choices=[MockChoice(
                message=MockChoiceMessage(content="Corrected")
            )]),
        ]
        llm = AsyncMock(side_effect=responses)
        gateway = make_gateway(tool_results=[
            ToolResult(success=True, data={}),
            ToolResult(success=True, data={}),
        ])
        node = ExecutionNode(
            llm=llm,
            tool_gateway=gateway,
            validation_gateway=make_validation_gateway(),
        )
        result = await node(state)
        deviations = [e for e in result["execution_log"] if e["type"] == "deviation_detected"]
        assert len(deviations) == 2
        # deviation_count 应递增
        assert deviations[0]["deviation_count"] == 1
        assert deviations[1]["deviation_count"] == 2

    async def test_three_deviations_triggers_human_review(self) -> None:
        """连续3次偏离应触发 Human Review。"""
        step = PlanStep(
            step_id=1, description="Modify main.py", action="modify",
            target_file="main.py",
        )
        plan = [step]
        state = AgentState(
            user_request="Modify main.py",
            project_root="/test/project",
            plan=plan,
            current_step_index=0,
            original_goal_summary="Modify main.py",
        )
        # LLM 三次都写 wrong.py（连续偏离 → Human Review）
        responses = [
            MockLLMResponse(choices=[MockChoice(
                message=MockChoiceMessage(
                    content="Writing 1",
                    tool_calls=[MockToolCall(
                        id="call_1",
                        function=MockToolCall.Function(
                            name="write_file",
                            arguments='{"file_path": "wrong.py", "content": "a", "mode": "create"}',
                        ),
                    )],
                )
            )]),
            MockLLMResponse(choices=[MockChoice(
                message=MockChoiceMessage(
                    content="Writing 2",
                    tool_calls=[MockToolCall(
                        id="call_2",
                        function=MockToolCall.Function(
                            name="write_file",
                            arguments='{"file_path": "wrong.py", "content": "b", "mode": "create"}',
                        ),
                    )],
                )
            )]),
            MockLLMResponse(choices=[MockChoice(
                message=MockChoiceMessage(
                    content="Writing 3",
                    tool_calls=[MockToolCall(
                        id="call_3",
                        function=MockToolCall.Function(
                            name="write_file",
                            arguments='{"file_path": "wrong.py", "content": "c", "mode": "create"}',
                        ),
                    )],
                )
            )]),
        ]
        llm = AsyncMock(side_effect=responses)
        gateway = make_gateway(tool_results=[
            ToolResult(success=True, data={}),
            ToolResult(success=True, data={}),
            ToolResult(success=True, data={}),
        ])
        node = ExecutionNode(
            llm=llm,
            tool_gateway=gateway,
            validation_gateway=make_validation_gateway(),
        )
        result = await node(state)
        # 应设置 human_review_required
        assert result["human_review_required"] is True
        # 应包含 human_review_required 日志
        hr_events = [e for e in result["execution_log"] if e["type"] == "human_review_required"]
        assert len(hr_events) >= 1
        # review_request 应包含详细信息
        assert result["review_request"] is not None
        assert result["review_request"]["review_type"] == "deviation_detected"
        assert result["review_request"]["details"]["consecutive_deviations"] == 3


# ── Phase 3.5: TaskFocus — Goal Summary ────────────────────────────────────


@pytest.mark.asyncio
class TestGoalSummary:
    """目标摘要测试。"""

    async def test_goal_summary_injected_in_prompt(self) -> None:
        """original_goal_summary 应注入到步骤提示词中。"""
        plan = [PlanStep(step_id=1, description="Step 1", action="read",
                         target_file="f.py")]
        state = AgentState(
            user_request="test",
            project_root="/root",
            plan=plan,
            original_goal_summary="Create a Flask app with login",
        )
        llm = AsyncMock(return_value=MockLLMResponse(choices=[MockChoice(
            message=MockChoiceMessage(content="Done")
        )]))
        node = ExecutionNode(
            llm=llm,
            tool_gateway=make_gateway(),
            validation_gateway=make_validation_gateway(),
        )
        await node(state)
        # LLM 的 system prompt 应包含 goal summary
        call_kwargs = llm.call_args[1]
        system_msg = call_kwargs["messages"][0]["content"]
        assert "Create a Flask app with login" in system_msg
        assert "原始目标" in system_msg

    async def test_empty_goal_summary_omits_section(self) -> None:
        """空的目标摘要不应添加 TaskFocus 节。"""
        plan = [PlanStep(step_id=1, description="Read", action="read",
                         target_file="f.py")]
        state = AgentState(
            user_request="test",
            project_root="/root",
            plan=plan,
            original_goal_summary="",
        )
        llm = AsyncMock(return_value=MockLLMResponse(choices=[MockChoice(
            message=MockChoiceMessage(content="Done")
        )]))
        node = ExecutionNode(
            llm=llm,
            tool_gateway=make_gateway(),
            validation_gateway=make_validation_gateway(),
        )
        await node(state)
        call_kwargs = llm.call_args[1]
        system_msg = call_kwargs["messages"][0]["content"]
        # 不应包含 TaskFocus 相关节
        assert "原始目标" not in system_msg

    async def test_completed_steps_injected_in_prompt(self) -> None:
        """已完成步骤摘要应注入到提示词。"""
        plan = [
            PlanStep(step_id=1, description="Read config", action="read",
                     target_file="config.py"),
            PlanStep(step_id=2, description="Write app", action="create",
                     target_file="app.py"),
        ]
        state = AgentState(
            user_request="test",
            project_root="/root",
            plan=plan,
            current_step_index=1,  # 第1步已完成
            original_goal_summary="Build app",
            completed_steps_summary="Read config",
        )
        llm = AsyncMock(return_value=MockLLMResponse(choices=[MockChoice(
            message=MockChoiceMessage(content="Done")
        )]))
        node = ExecutionNode(
            llm=llm,
            tool_gateway=make_gateway(),
            validation_gateway=make_validation_gateway(),
        )
        await node(state)
        call_kwargs = llm.call_args[1]
        system_msg = call_kwargs["messages"][0]["content"]
        assert "Read config" in system_msg


# ── Phase 3.5: TaskFocus — Progress Tracking ───────────────────────────────


@pytest.mark.asyncio
class TestProgressTracking:
    """进度追踪测试。"""

    async def test_tracks_completed_steps_summary(self) -> None:
        """执行完步骤应更新 completed_steps_summary。"""
        plan = [
            PlanStep(step_id=1, description="Read main.py", action="read",
                     target_file="main.py"),
            PlanStep(step_id=2, description="Create app.py", action="create",
                     target_file="app.py"),
        ]
        state = AgentState(
            user_request="test",
            project_root="/root",
            plan=plan,
        )
        llm = AsyncMock(side_effect=[
            MockLLMResponse(choices=[MockChoice(
                message=MockChoiceMessage(content="Read complete")
            )]),
            MockLLMResponse(choices=[MockChoice(
                message=MockChoiceMessage(
                    content="Creating app.py",
                    tool_calls=[MockToolCall(function=MockToolCall.Function(
                        name="write_file",
                        arguments='{"file_path": "app.py", "content": "app = 1", "mode": "create"}',
                    ))],
                )
            )]),
            MockLLMResponse(choices=[MockChoice(
                message=MockChoiceMessage(content="Done")
            )]),
        ])
        node = ExecutionNode(
            llm=llm,
            tool_gateway=make_gateway(),
            validation_gateway=make_validation_gateway(),
        )
        result = await node(state)
        assert "completed_steps_summary" in result
        # 两步的描述应以分号分隔
        assert "Read main.py" in result["completed_steps_summary"]
        assert "Create app.py" in result["completed_steps_summary"]

    async def test_tracks_tasks_remaining(self) -> None:
        """应正确计算剩余步骤。"""
        plan = [
            PlanStep(step_id=1, description="Read", action="read",
                     target_file="a.py"),
            PlanStep(step_id=2, description="Read second", action="read",
                     target_file="b.py"),
            PlanStep(step_id=3, description="Read third", action="read",
                     target_file="c.py"),
        ]
        state = AgentState(
            user_request="test",
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
        assert "tasks_remaining" in result
        assert len(result["tasks_remaining"]) == 0  # 所有步骤都执行完了

    async def test_full_cycle_updates_progress(self) -> None:
        """完整执行周期应正确更新所有进度指标。"""
        plan = [
            PlanStep(step_id=1, description="First step", action="read",
                     target_file="f1.py"),
            PlanStep(step_id=2, description="Second step", action="read",
                     target_file="f2.py"),
        ]
        state = AgentState(
            user_request="test",
            project_root="/root",
            plan=plan,
        )
        # 每个步骤 LLM 返回不同内容
        step1_resp = MockLLMResponse(choices=[MockChoice(
            message=MockChoiceMessage(content="Step 1 done")
        )])
        step2_resp = MockLLMResponse(choices=[MockChoice(
            message=MockChoiceMessage(content="Step 2 done")
        )])
        llm = AsyncMock(side_effect=[step1_resp, step2_resp])

        node = ExecutionNode(
            llm=llm,
            tool_gateway=make_gateway(),
            validation_gateway=make_validation_gateway(),
        )
        result = await node(state)
        # current_step_index 应为 2
        assert result["current_step_index"] == 2
        # completed_steps_summary 包含两个步骤
        assert "First step" in result["completed_steps_summary"]
        assert "Second step" in result["completed_steps_summary"]
        # tasks_remaining 应为空（所有步骤完成）
        assert len(result["tasks_remaining"]) == 0


# ── Phase 4.A.5: Repair Mode ───────────────────────────────────────────────


class _MockValidationResult:
    """模拟 ValidationResult（避免导入问题）。"""
    def __init__(self, passed: bool = True,
                 file_path: str = "test.py",
                 line: int = 1,
                 message: str = "test error") -> None:
        self.passed = passed
        self.errors: list[Any] = []
        self.warnings: list[Any] = []
        self.duration_ms: float = 0.0
        if not passed:
            self.errors = [type("Err", (), {
                "file_path": file_path, "line": line,
                "message": message, "severity": "error",
                "code": "E001",
            })()]


@pytest.mark.asyncio
class TestRepairModeEntry:
    """修复模式入口检测测试。"""

    async def test_no_validation_results_no_repair(self) -> None:
        """无验证结果时不应进入修复模式。"""
        state = AgentState(
            user_request="test", project_root="/root",
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
        # 应走直连模式
        assert "execution_log" in result
        repair_entries = [e for e in result["execution_log"] if e["type"] == "repair_mode"]
        assert len(repair_entries) == 0

    async def test_all_passed_no_repair(self) -> None:
        """所有验证通过时不应进入修复模式。"""
        state = AgentState(
            user_request="test", project_root="/root",
            validation_results=[_MockValidationResult(passed=True)],  # type: ignore[arg-type]
            plan=[PlanStep(step_id=1, description="Done", action="read", target_file="f.py")],
            current_step_index=1,
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
        repair_entries = [e for e in result["execution_log"] if e["type"] == "repair_mode"]
        assert len(repair_entries) == 0

    async def test_failure_with_pending_plan_no_repair(self) -> None:
        """验证失败但有未完成的计划步骤时不应进入修复模式。"""
        state = AgentState(
            user_request="test", project_root="/root",
            validation_results=[_MockValidationResult(passed=False)],  # type: ignore[arg-type]
            plan=[PlanStep(step_id=1, description="Do something", action="read", target_file="f.py")],
            current_step_index=0,  # 还有未执行的步骤
        )
        llm = AsyncMock(return_value=MockLLMResponse(choices=[MockChoice(
            message=MockChoiceMessage(content="Execute plan")
        )]))
        node = ExecutionNode(
            llm=llm,
            tool_gateway=make_gateway(),
            validation_gateway=make_validation_gateway(),
        )
        result = await node(state)
        # 应走 plan-aware 模式而非修复模式
        repair_entries = [e for e in result["execution_log"] if e["type"] == "repair_mode"]
        assert len(repair_entries) == 0

    async def test_failure_triggers_repair_mode(self) -> None:
        """验证失败且计划完成时应进入修复模式。"""
        state = AgentState(
            user_request="test", project_root="/root",
            validation_results=[_MockValidationResult(passed=False)],  # type: ignore[arg-type]
            plan=[PlanStep(step_id=1, description="Done", action="read", target_file="f.py")],
            current_step_index=1,  # 所有步骤已完成
        )
        # LLM 在修复模式下直接返回文本（无工具调用）
        llm = AsyncMock(return_value=MockLLMResponse(choices=[MockChoice(
            message=MockChoiceMessage(content="I see the error, let me fix it")
        )]))
        node = ExecutionNode(
            llm=llm,
            tool_gateway=make_gateway(),
            validation_gateway=make_validation_gateway(),
        )
        result = await node(state)
        # 应包含修复模式日志
        repair_entries = [e for e in result["execution_log"] if e["type"] == "repair_mode"]
        assert len(repair_entries) == 1
        assert repair_entries[0]["retry_count"] == 1  # retry_count 应递增

    async def test_repair_mode_no_plan_fallback(self) -> None:
        """无计划时验证失败也应进入修复模式。"""
        state = AgentState(
            user_request="test", project_root="/root",
            validation_results=[_MockValidationResult(passed=False)],  # type: ignore[arg-type]
        )
        llm = AsyncMock(return_value=MockLLMResponse(choices=[MockChoice(
            message=MockChoiceMessage(content="Repairing")
        )]))
        node = ExecutionNode(
            llm=llm,
            tool_gateway=make_gateway(),
            validation_gateway=make_validation_gateway(),
        )
        result = await node(state)
        repair_entries = [e for e in result["execution_log"] if e["type"] == "repair_mode"]
        assert len(repair_entries) == 1


@pytest.mark.asyncio
class TestRepairModeExecution:
    """修复模式执行流程测试。"""

    async def test_repair_success_with_tool_calls(self) -> None:
        """修复模式中 LLM 调用工具成功修复。"""
        state = AgentState(
            user_request="test", project_root="/root",
            validation_results=[_MockValidationResult(passed=False)],  # type: ignore[arg-type]
        )
        # LLM 先调用 write_file 修复，然后返回完成
        responses = [
            MockLLMResponse(choices=[MockChoice(
                message=MockChoiceMessage(
                    content="Fixing error",
                    tool_calls=[MockToolCall(
                        id="call_repair",
                        function=MockToolCall.Function(
                            name="write_file",
                            arguments='{"file_path": "fix.py", "content": "x=1", "mode": "modify"}',
                        ),
                    )],
                )
            )]),
            MockLLMResponse(choices=[MockChoice(
                message=MockChoiceMessage(content="Fixed!")
            )]),
        ]
        llm = AsyncMock(side_effect=responses)
        gateway = make_gateway(tool_results=[
            ToolResult(success=True, data={"file_path": "fix.py", "lines_added": 1}),
        ])
        node = ExecutionNode(
            llm=llm,
            tool_gateway=gateway,
            validation_gateway=make_validation_gateway(),
        )
        result = await node(state)

        # 有修复模式日志
        assert len(result["execution_log"]) >= 2
        repair_entries = [e for e in result["execution_log"] if e["type"] == "repair_mode"]
        assert len(repair_entries) == 1
        assert result["retry_count"] == 1

        # 工具调用包含 repair_mode=True 标记
        tool_entries = [e for e in result["execution_log"]
                        if e.get("type") == "tool_call" and e.get("repair_mode")]
        assert len(tool_entries) >= 1

    async def test_repair_accumulates_changes(self) -> None:
        """修复模式中的 write_file 应记录到 accumulated_changes。"""
        state = AgentState(
            user_request="test", project_root="/root",
            validation_results=[_MockValidationResult(passed=False)],  # type: ignore[arg-type]
        )
        responses = [
            MockLLMResponse(choices=[MockChoice(
                message=MockChoiceMessage(
                    content="Fixing",
                    tool_calls=[MockToolCall(
                        id="call_fix",
                        function=MockToolCall.Function(
                            name="write_file",
                            arguments='{"file_path": "bug.py", "content": "fixed", "mode": "modify"}',
                        ),
                    )],
                )
            )]),
            MockLLMResponse(choices=[MockChoice(
                message=MockChoiceMessage(content="Fixed!")
            )]),
        ]
        llm = AsyncMock(side_effect=responses)
        gateway = make_gateway(tool_results=[
            ToolResult(success=True, data={"file_path": "bug.py", "lines_added": 1}),
        ])
        node = ExecutionNode(
            llm=llm,
            tool_gateway=gateway,
            validation_gateway=make_validation_gateway(),
        )
        result = await node(state)
        assert result["retry_count"] == 1
        changes = result["accumulated_changes"]
        assert len(changes) >= 1
        # repair mode 的 change 应有 repair_mode=True 标记
        repair_changes = [c for c in changes if c.get("repair_mode")]
        assert len(repair_changes) >= 1
        assert repair_changes[0]["file_path"] == "bug.py"

    async def test_repair_llm_failure(self) -> None:
        """修复模式中 LLM 失败应记录错误。"""
        state = AgentState(
            user_request="test", project_root="/root",
            validation_results=[_MockValidationResult(passed=False)],  # type: ignore[arg-type]
        )
        llm = AsyncMock(side_effect=RuntimeError("LLM unavailable in repair"))
        node = ExecutionNode(
            llm=llm,
            tool_gateway=make_gateway(),
            validation_gateway=make_validation_gateway(),
        )
        result = await node(state)
        assert len(result["errors"]) >= 1
        assert "Repair LLM call failed" in result["errors"][0]
        repair_entries = [e for e in result["execution_log"] if e["type"] == "repair_mode"]
        assert len(repair_entries) == 1  # 进入修复模式的日志仍在

    async def test_repair_preserves_existing_errors(self) -> None:
        """修复模式应保留已有的 errors。"""
        state = AgentState(
            user_request="test", project_root="/root",
            validation_results=[_MockValidationResult(passed=False)],  # type: ignore[arg-type]
            errors=["Previous error"],
        )
        llm = AsyncMock(return_value=MockLLMResponse(choices=[MockChoice(
            message=MockChoiceMessage(content="Repairing...")
        )]))
        node = ExecutionNode(
            llm=llm,
            tool_gateway=make_gateway(),
            validation_gateway=make_validation_gateway(),
        )
        result = await node(state)
        assert "Previous error" in result["errors"]


@pytest.mark.asyncio
class TestRetryCounting:
    """重试计数测试。"""

    async def test_first_repair_increments_retry_to_1(self) -> None:
        """首次修复后 retry_count 应为 1。"""
        state = AgentState(
            user_request="test", project_root="/root",
            validation_results=[_MockValidationResult(passed=False)],  # type: ignore[arg-type]
            retry_count=0,
        )
        llm = AsyncMock(return_value=MockLLMResponse(choices=[MockChoice(
            message=MockChoiceMessage(content="Repairing")
        )]))
        node = ExecutionNode(
            llm=llm,
            tool_gateway=make_gateway(),
            validation_gateway=make_validation_gateway(),
        )
        result = await node(state)
        assert result["retry_count"] == 1

    async def test_second_repair_increments_retry_to_2(self) -> None:
        """第二次修复后 retry_count 应为 2。"""
        state = AgentState(
            user_request="test", project_root="/root",
            validation_results=[_MockValidationResult(passed=False)],  # type: ignore[arg-type]
            retry_count=1,
        )
        llm = AsyncMock(return_value=MockLLMResponse(choices=[MockChoice(
            message=MockChoiceMessage(content="Repairing again")
        )]))
        node = ExecutionNode(
            llm=llm,
            tool_gateway=make_gateway(),
            validation_gateway=make_validation_gateway(),
        )
        result = await node(state)
        assert result["retry_count"] == 2

    async def test_third_repair_reaches_limit(self) -> None:
        """第三次修复后 retry_count 应为 3。"""
        state = AgentState(
            user_request="test", project_root="/root",
            validation_results=[_MockValidationResult(passed=False)],  # type: ignore[arg-type]
            retry_count=2,
        )
        llm = AsyncMock(return_value=MockLLMResponse(choices=[MockChoice(
            message=MockChoiceMessage(content="Repairing third time")
        )]))
        node = ExecutionNode(
            llm=llm,
            tool_gateway=make_gateway(),
            validation_gateway=make_validation_gateway(),
        )
        result = await node(state)
        assert result["retry_count"] == 3


@pytest.mark.asyncio
class TestRepairToolCallLimits:
    """修复模式工具调用限制测试。"""

    async def test_repair_has_lower_tool_call_limit(self) -> None:
        """修复模式使用较低的工具调用上限。"""
        state = AgentState(
            user_request="test", project_root="/root",
            validation_results=[_MockValidationResult(passed=False)],  # type: ignore[arg-type]
        )
        # 每次 LLM 返回一个 tool_call，直到超限
        tool_call_resp = MockLLMResponse(choices=[MockChoice(
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
        llm = AsyncMock(return_value=tool_call_resp)
        node = ExecutionNode(
            llm=llm,
            tool_gateway=make_gateway(),
            validation_gateway=make_validation_gateway(),
        )
        result = await node(state)

        tool_calls = [e for e in result["execution_log"]
                      if e.get("type") == "tool_call" and e.get("repair_mode")]
        # 不应超过 5（_MAX_REPAIR_TOOL_CALLS）
        assert len(tool_calls) <= 5

    async def test_repair_exceeding_limit_terminates(self) -> None:
        """超过修复模式工具调用上限应终止。"""
        state = AgentState(
            user_request="test", project_root="/root",
            validation_results=[_MockValidationResult(passed=False)],  # type: ignore[arg-type]
        )
        tool_call_resp = MockLLMResponse(choices=[MockChoice(
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
        llm = AsyncMock(return_value=tool_call_resp)
        node = ExecutionNode(
            llm=llm,
            tool_gateway=make_gateway(),
            validation_gateway=make_validation_gateway(),
        )
        result = await node(state)

        tool_calls = [e for e in result["execution_log"]
                      if e.get("type") == "tool_call" and e.get("repair_mode")]
        assert len(tool_calls) <= 5


@pytest.mark.asyncio
class TestRepairPrompt:
    """修复提示词测试。"""

    async def test_validation_report_in_prompt(self) -> None:
        """修复模式的 LLM prompt 应包含验证报告。"""
        state = AgentState(
            user_request="test", project_root="/root",
            validation_results=[_MockValidationResult(passed=False)],  # type: ignore[arg-type]
        )
        llm = AsyncMock(return_value=MockLLMResponse(choices=[MockChoice(
            message=MockChoiceMessage(content="Repairing based on report")
        )]))
        node = ExecutionNode(
            llm=llm,
            tool_gateway=make_gateway(),
            validation_gateway=make_validation_gateway(),
        )
        await node(state)

        # 检查传递给 LLM 的 user 消息包含结构化错误信息
        call_kwargs = llm.call_args[1]
        messages = call_kwargs.get("messages", [])
        user_msgs = [m for m in messages if m["role"] == "user"]
        assert len(user_msgs) >= 1
        user_content = user_msgs[0]["content"]
        assert "修复尝试 #1" in user_content
        assert "test error" in user_content

    async def test_repair_prompt_structured_instead_of_fix_suggestions(self) -> None:
        """修复模式的 LLM prompt 使用结构化错误格式（替代旧修复建议）。"""
        state = AgentState(
            user_request="test", project_root="/root",
            validation_results=[_MockValidationResult(passed=False)],  # type: ignore[arg-type]
        )
        llm = AsyncMock(return_value=MockLLMResponse(choices=[MockChoice(
            message=MockChoiceMessage(content="Repairing")
        )]))
        node = ExecutionNode(
            llm=llm,
            tool_gateway=make_gateway(),
            validation_gateway=make_validation_gateway(),
        )
        await node(state)

        call_kwargs = llm.call_args[1]
        messages = call_kwargs.get("messages", [])
        user_msgs = [m for m in messages if m["role"] == "user"]
        assert len(user_msgs) >= 1
        user_content = user_msgs[0]["content"]
        # 新的结构化格式包含修复尝试编号和错误详情
        assert "修复尝试 #1" in user_content
        assert "test.py" in user_content

    async def test_repair_prompt_includes_error_details(self) -> None:
        """修复提示词应包含具体错误细节。"""
        state = AgentState(
            user_request="test", project_root="/root",
            validation_results=[_MockValidationResult(
                passed=False, file_path="buggy.py",
                line=42, message="SyntaxError: invalid syntax",
            )],  # type: ignore[arg-type]
        )
        llm = AsyncMock(return_value=MockLLMResponse(choices=[MockChoice(
            message=MockChoiceMessage(content="Fixing syntax")
        )]))
        node = ExecutionNode(
            llm=llm,
            tool_gateway=make_gateway(),
            validation_gateway=make_validation_gateway(),
        )
        await node(state)

        call_kwargs = llm.call_args[1]
        messages = call_kwargs.get("messages", [])
        user_msgs = [m for m in messages if m["role"] == "user"]
        assert len(user_msgs) >= 1
        user_content = user_msgs[0]["content"]
        assert "buggy.py" in user_content or "SyntaxError" in user_content
        assert "invalid syntax" in user_content


# ── Tests: Rollback original_content ────────────────────────────────────


@pytest.mark.asyncio
class TestOriginalContentSaving:
    """_save_original_content_before_tool 测试。"""

    async def test_save_before_write_file(self, tmp_path: Path) -> None:
        """write_file 前保存原始内容。"""
        test_file = tmp_path / "test.py"
        test_file.write_text("# original content")

        node = ExecutionNode(
            llm=make_llm(),
            tool_gateway=make_gateway(),
            validation_gateway=make_validation_gateway(),
        )

        file_path, content = node._save_original_content_before_tool(
            "write_file", {"file_path": str(test_file)},
        )

        assert file_path == str(test_file)
        assert content == "# original content"

    async def test_save_before_write_new_file(self, tmp_path: Path) -> None:
        """新文件（create）write_file 前 original_content 为 None。"""
        new_file = tmp_path / "new.py"

        node = ExecutionNode(
            llm=make_llm(),
            tool_gateway=make_gateway(),
            validation_gateway=make_validation_gateway(),
        )

        file_path, content = node._save_original_content_before_tool(
            "write_file", {"file_path": str(new_file)},
        )

        assert file_path == str(new_file)
        assert content is None  # 文件不存在

    async def test_save_resolves_tool_relative_path_from_workspace(self, tmp_path: Path) -> None:
        test_file = tmp_path / "relative.py"
        test_file.write_text("# existing", encoding="utf-8")
        node = ExecutionNode(
            llm=make_llm(),
            tool_gateway=make_gateway(),
            validation_gateway=make_validation_gateway(),
        )

        file_path, content = node._save_original_content_before_tool(
            "write_file", {"file_path": "relative.py"}, str(tmp_path)
        )

        assert file_path == "relative.py"
        assert content == "# existing"

    async def test_save_before_delete_file(self, tmp_path: Path) -> None:
        """delete_file 前保存原始内容。"""
        test_file = tmp_path / "to_delete.py"
        test_file.write_text("# content to delete")

        node = ExecutionNode(
            llm=make_llm(),
            tool_gateway=make_gateway(),
            validation_gateway=make_validation_gateway(),
        )

        file_path, content = node._save_original_content_before_tool(
            "delete_file", {"file_path": str(test_file)},
        )

        assert file_path == str(test_file)
        assert content == "# content to delete"

    async def test_save_ignores_other_tools(self) -> None:
        """其他工具（如 read_file）不保存原始内容。"""
        node = ExecutionNode(
            llm=make_llm(),
            tool_gateway=make_gateway(),
            validation_gateway=make_validation_gateway(),
        )

        file_path, content = node._save_original_content_before_tool(
            "read_file", {"file_path": "test.py"},
        )

        assert file_path == ""
        assert content is None


# ── Phase 5.4: 修复上下文结构化 ──────────────────────────────────────────


@pytest.mark.asyncio
class TestStructuredRepairContext:
    """修复上下文结构化测试。"""

    async def test_extract_structured_errors_basic(self) -> None:
        """_extract_structured_errors 正确提取单层错误。"""
        node = ExecutionNode(
            llm=make_llm(),
            tool_gateway=make_gateway(),
            validation_gateway=make_validation_gateway(),
        )
        vrs = [
            ValidationResult(passed=False, errors=[
                ValidationError(file_path="test.py", line=5, message="SyntaxError"),
            ]),
            ValidationResult(passed=True),
            ValidationResult(passed=True),
        ]
        new_errors, pre_existing = node._extract_structured_errors(vrs)
        assert len(new_errors) == 1
        assert new_errors[0].file_path == "test.py"
        assert new_errors[0].line_number == 5
        assert new_errors[0].message == "SyntaxError"
        assert new_errors[0].error_type == "syntax"
        assert len(pre_existing) == 0

    async def test_extract_structured_errors_multiple_layers(self) -> None:
        """多层验证错误正确映射 error_type。"""
        node = ExecutionNode(
            llm=make_llm(),
            tool_gateway=make_gateway(),
            validation_gateway=make_validation_gateway(),
        )
        vrs = [
            ValidationResult(passed=False, errors=[
                ValidationError(file_path="a.py", message="E1"),
            ]),
            ValidationResult(passed=False, errors=[
                ValidationError(file_path="b.py", message="E2"),
            ]),
            ValidationResult(passed=False, errors=[
                ValidationError(file_path="c.py", message="E3"),
            ]),
        ]
        new_errors, _ = node._extract_structured_errors(vrs)
        assert len(new_errors) == 3
        assert new_errors[0].error_type == "syntax"
        assert new_errors[1].error_type == "lint"
        assert new_errors[2].error_type == "runtime"

    async def test_extract_structured_errors_empty(self) -> None:
        """空 validation_results 返回空列表。"""
        node = ExecutionNode(
            llm=make_llm(),
            tool_gateway=make_gateway(),
            validation_gateway=make_validation_gateway(),
        )
        new_errors, pre_existing = node._extract_structured_errors([])
        assert len(new_errors) == 0
        assert len(pre_existing) == 0

    async def test_extract_handles_mock_validation_result(self) -> None:
        """兼容 _MockValidationResult 格式。"""
        node = ExecutionNode(
            llm=make_llm(),
            tool_gateway=make_gateway(),
            validation_gateway=make_validation_gateway(),
        )
        vrs = [_MockValidationResult(passed=False, file_path="mock.py", line=3, message="mock error")]  # type: ignore[arg-type]
        new_errors, _ = node._extract_structured_errors(vrs)
        assert len(new_errors) == 1
        assert new_errors[0].file_path == "mock.py"
        assert new_errors[0].line_number == 3
        assert "mock error" in new_errors[0].message

    async def test_build_repair_prompt_with_context(self) -> None:
        """_build_repair_prompt 使用结构化格式。"""
        state = AgentState(
            user_request="test", project_root="/root",
            repair_context=RepairContext(
                attempt_number=2,
                errors=[
                    StructuredError(file_path="bug.py", line_number=10, error_type="syntax", message="invalid syntax"),
                ],
            ),
        )
        node = ExecutionNode(
            llm=make_llm(),
            tool_gateway=make_gateway(),
            validation_gateway=make_validation_gateway(),
        )
        prompt = node._build_repair_prompt(state)
        assert "修复尝试 #2" in prompt
        assert "bug.py:10" in prompt
        assert "invalid syntax" in prompt

    async def test_build_repair_prompt_with_last_fix(self) -> None:
        """包含 last_fix_summary 时显示上次修复内容。"""
        state = AgentState(
            user_request="test", project_root="/root",
            repair_context=RepairContext(
                attempt_number=3,
                errors=[StructuredError(file_path="x.py", message="err")],
                last_fix_summary="Modified x.py",
            ),
        )
        node = ExecutionNode(
            llm=make_llm(),
            tool_gateway=make_gateway(),
            validation_gateway=make_validation_gateway(),
        )
        prompt = node._build_repair_prompt(state)
        assert "上次修复内容" in prompt
        assert "Modified x.py" in prompt

    async def test_build_repair_prompt_with_pre_existing(self) -> None:
        """包含预存在错误时显示请勿修复列表。"""
        state = AgentState(
            user_request="test", project_root="/root",
            repair_context=RepairContext(
                attempt_number=1,
                errors=[StructuredError(file_path="x.py", message="new err")],
                pre_existing_errors=["existing issue in y.py"],
            ),
        )
        node = ExecutionNode(
            llm=make_llm(),
            tool_gateway=make_gateway(),
            validation_gateway=make_validation_gateway(),
        )
        prompt = node._build_repair_prompt(state)
        assert "预存在的错误" in prompt
        assert "existing issue in y.py" in prompt

    async def test_build_repair_prompt_fallback(self) -> None:
        """repair_context 为 None 时回退到传统格式。"""
        state = AgentState(
            user_request="test", project_root="/root",
            validation_results=[_MockValidationResult(passed=False)],  # type: ignore[arg-type]
        )
        node = ExecutionNode(
            llm=make_llm(),
            tool_gateway=make_gateway(),
            validation_gateway=make_validation_gateway(),
        )
        prompt = node._build_repair_prompt(state)
        assert "FAILED" in prompt or "验证报告" in prompt

    async def test_repair_context_populated_after_repair(self) -> None:
        """修复模式执行后 repair_context 被填充。"""
        state = AgentState(
            user_request="test", project_root="/root",
            validation_results=[_MockValidationResult(passed=False)],  # type: ignore[arg-type]
        )
        llm = AsyncMock(return_value=MockLLMResponse(choices=[MockChoice(
            message=MockChoiceMessage(content="Fixed"),
        )]))
        node = ExecutionNode(
            llm=llm,
            tool_gateway=make_gateway(),
            validation_gateway=make_validation_gateway(),
        )
        result = await node(state)
        assert "repair_context" in result
        assert result["repair_context"] is not None
        assert result["repair_context"].attempt_number == 1
        assert len(result["repair_context"].errors) >= 1

    async def test_repair_context_attempt_number_increments(self) -> None:
        """repair_context.attempt_number 继承 retry_count。"""
        state = AgentState(
            user_request="test", project_root="/root",
            validation_results=[_MockValidationResult(passed=False)],  # type: ignore[arg-type]
            retry_count=2,
        )
        llm = AsyncMock(return_value=MockLLMResponse(choices=[MockChoice(
            message=MockChoiceMessage(content="Fixed"),
        )]))
        node = ExecutionNode(
            llm=llm,
            tool_gateway=make_gateway(),
            validation_gateway=make_validation_gateway(),
        )
        result = await node(state)
        assert result["repair_context"].attempt_number == 3

    async def test_repair_context_inherits_last_fix_summary(self) -> None:
        """repair_context 继承上一次的 last_fix_summary。"""
        state = AgentState(
            user_request="test", project_root="/root",
            validation_results=[_MockValidationResult(passed=False)],  # type: ignore[arg-type]
            repair_context=RepairContext(
                attempt_number=1,
                errors=[],
                last_fix_summary="Modified fix.py",
            ),
        )
        llm = AsyncMock(return_value=MockLLMResponse(choices=[MockChoice(
            message=MockChoiceMessage(content="Repairing again"),
        )]))
        node = ExecutionNode(
            llm=llm,
            tool_gateway=make_gateway(),
            validation_gateway=make_validation_gateway(),
        )
        result = await node(state)
        # 新的 repair_context 应继承上次的 last_fix_summary
        assert result["repair_context"].last_fix_summary == "Modified fix.py"

    async def test_last_fix_summary_generated_from_tool_calls(self) -> None:
        """修复模式中的工具调用生成 last_fix_summary。"""
        state = AgentState(
            user_request="test", project_root="/root",
            validation_results=[_MockValidationResult(passed=False)],  # type: ignore[arg-type]
        )
        responses = [
            MockLLMResponse(choices=[MockChoice(
                message=MockChoiceMessage(
                    content="Fixing",
                    tool_calls=[MockToolCall(
                        id="c1",
                        function=MockToolCall.Function(
                            name="write_file",
                            arguments='{"file_path": "fix.py", "content": "x=1", "mode": "modify"}',
                        ),
                    )],
                )
            )]),
            MockLLMResponse(choices=[MockChoice(
                message=MockChoiceMessage(content="Done"),
            )]),
        ]
        llm = AsyncMock(side_effect=responses)
        gateway = make_gateway(tool_results=[
            ToolResult(success=True, data={}),
        ])
        node = ExecutionNode(
            llm=llm,
            tool_gateway=gateway,
            validation_gateway=make_validation_gateway(),
        )
        result = await node(state)
        assert result["repair_context"].last_fix_summary is not None
        assert "fix.py" in result["repair_context"].last_fix_summary


# ── Phase 5.5: LLM 成本控制 ──────────────────────────────────────────────


@pytest.mark.asyncio
class TestLLMCostControl:
    """LLM 成本控制测试。"""

    async def test_limit_reached_sets_human_review(self, mocker) -> None:
        """达到 LLM 调用上限时设置 human_review_required。"""
        mocker.patch(
            "codeagent.orchestration.nodes.execution_node.codeagent_config.get_max_llm_calls_per_task",
            return_value=3,
        )
        state = AgentState(
            user_request="test", project_root="/root",
            llm_call_count=3,
        )
        node = ExecutionNode(
            llm=make_llm(),
            tool_gateway=make_gateway(),
            validation_gateway=make_validation_gateway(),
        )
        result = await node._call_llm_with_limit(
            state=state,
            messages=[{"role": "user", "content": "hello"}],
            tools=None,
            tool_choice=None,
        )
        assert result is None
        assert state.human_review_required is True
        assert state.review_type == "cost_limit_reached"
        assert state.review_request is not None
        assert "LLM 调用次数" in state.review_request.get("title", "")

    async def test_normal_call_increments_counter(self, mocker) -> None:
        """正常 LLM 调用后 llm_call_count 递增。"""
        mocker.patch(
            "codeagent.orchestration.nodes.execution_node.codeagent_config.get_max_llm_calls_per_task",
            return_value=50,
        )
        mocker.patch(
            "codeagent.orchestration.nodes.execution_node.codeagent_config.get_max_completion_tokens_per_call",
            return_value=4096,
        )
        state = AgentState(
            user_request="test", project_root="/root",
            llm_call_count=0,
        )
        llm = AsyncMock(return_value=MockLLMResponse(choices=[MockChoice(
            message=MockChoiceMessage(content="Hello"),
        )]))
        node = ExecutionNode(
            llm=llm,
            tool_gateway=make_gateway(),
            validation_gateway=make_validation_gateway(),
        )
        result = await node._call_llm_with_limit(
            state=state,
            messages=[{"role": "user", "content": "hello"}],
            tools=None,
            tool_choice=None,
        )
        assert result is not None
        assert state.llm_call_count == 1
        llm.assert_called_once()
        assert llm.call_args.kwargs["max_tokens"] == 4096

    async def test_cross_round_accumulation(self, mocker) -> None:
        """llm_call_count 跨多轮累计。"""
        mocker.patch(
            "codeagent.orchestration.nodes.execution_node.codeagent_config.get_max_llm_calls_per_task",
            return_value=50,
        )
        state = AgentState(
            user_request="test", project_root="/root",
        )
        llm = AsyncMock(return_value=MockLLMResponse(choices=[MockChoice(
            message=MockChoiceMessage(content="Hello"),
        )]))
        node = ExecutionNode(
            llm=llm,
            tool_gateway=make_gateway(),
            validation_gateway=make_validation_gateway(),
        )
        # 第一次调用
        await node._call_llm_with_limit(
            state=state,
            messages=[{"role": "user", "content": "hi"}],
            tools=None, tool_choice=None,
        )
        assert state.llm_call_count == 1

        # 第二次调用
        await node._call_llm_with_limit(
            state=state,
            messages=[{"role": "user", "content": "hi"}],
            tools=None, tool_choice=None,
        )
        assert state.llm_call_count == 2

        # 第三次调用
        await node._call_llm_with_limit(
            state=state,
            messages=[{"role": "user", "content": "hi"}],
            tools=None, tool_choice=None,
        )
        assert state.llm_call_count == 3

    async def test_limit_respected_in_direct_mode(self, mocker) -> None:
        """直连模式中达到调用上限时不再调用 LLM。"""
        mocker.patch(
            "codeagent.orchestration.nodes.execution_node.codeagent_config.get_max_llm_calls_per_task",
            return_value=2,
        )
        state = AgentState(
            user_request="test", project_root="/root",
            llm_call_count=2,  # 已达上限
        )
        llm = AsyncMock()
        node = ExecutionNode(
            llm=llm,
            tool_gateway=make_gateway(),
            validation_gateway=make_validation_gateway(),
        )
        result = await node(state)
        # LLM 不应被调用（达到上限后立即 break）
        llm.assert_not_called()
        assert "execution_log" in result

    async def test_zero_max_calls_triggers_limit(self, mocker) -> None:
        """max_llm_calls=0 时立即触发限制。"""
        mocker.patch(
            "codeagent.orchestration.nodes.execution_node.codeagent_config.get_max_llm_calls_per_task",
            return_value=0,
        )
        state = AgentState(
            user_request="test", project_root="/root",
        )
        llm = AsyncMock()
        node = ExecutionNode(
            llm=llm,
            tool_gateway=make_gateway(),
            validation_gateway=make_validation_gateway(),
        )
        result = await node(state)
        assert state.human_review_required is True
        llm.assert_not_called()
        assert "execution_log" in result

    async def test_runtime_steering_is_injected_before_next_llm_call(self) -> None:
        events: list[dict] = []
        provider = AsyncMock(return_value=["Keep the public API backward compatible"])
        llm = AsyncMock(return_value=MagicMock(usage=None))
        node = ExecutionNode(
            llm=llm,
            tool_gateway=make_gateway(),
            validation_gateway=make_validation_gateway(),
            progress_callback=events.append,
            steering_provider=provider,
        )
        state = AgentState(user_request="refactor the API", project_root="/root")
        messages = [{"role": "user", "content": state.user_request}]

        await node._call_llm_with_limit(
            state=state, messages=messages, tools=None, tool_choice=None,
        )

        assert provider.await_count == 1
        assert state.steering_instructions == ["Keep the public API backward compatible"]
        assert "Runtime steering update" in messages[-1]["content"]
        assert llm.await_args.kwargs["messages"][-1] == messages[-1]
        assert events[-1]["type"] == "steering_applied"


# ── P0-1: Action-scoped tool pruning ───────────────────────────────────────


def _make_multi_tool_gateway() -> AsyncMock:
    """A gateway exposing core write/read/explore tools plus MCP and a shell tool."""
    names = [
        "read_file", "list_files", "search_code", "navigate_code", "get_diagnostics",
        "write_file", "apply_patch", "delete_file", "run_terminal", "git",
        "mcp__github",  # MCP tool bypasses _check_deviation by name
    ]
    mock = AsyncMock(spec=IToolGateway)
    mock.list_tools.return_value = [
        ToolDefinition(
            name=name,
            description=f"{name} description",
            parameters_schema={"type": "object", "properties": {}, "required": []},
        )
        for name in names
    ]
    mock.execute_tool = AsyncMock(
        return_value=ToolResult(success=True, data={"content": ""})
    )
    return mock


class TestActionScopedToolPruning:
    """P0-1: plan steps only carry the tools their action allows (+ MCP)."""

    def _node(self, gateway: AsyncMock) -> ExecutionNode:
        return ExecutionNode(
            llm=AsyncMock(),
            tool_gateway=gateway,
            validation_gateway=make_validation_gateway(),
        )

    def test_modify_step_prunes_to_modify_set_plus_mcp(self) -> None:
        node = self._node(_make_multi_tool_gateway())
        state = AgentState(user_request="t", project_root="/root")
        names = {t.name for t in node._available_tools(state, "modify")}
        expected = _ALLOWED_TOOLS_BY_ACTION["modify"] | {"mcp__github"}
        assert names == expected
        # Shell/git/delete are already blocked by _check_deviation at call time.
        assert "run_terminal" not in names
        assert "git" not in names
        assert "delete_file" not in names

    def test_read_step_prunes_write_tools(self) -> None:
        node = self._node(_make_multi_tool_gateway())
        state = AgentState(user_request="t", project_root="/root")
        names = {t.name for t in node._available_tools(state, "read")}
        assert "write_file" not in names
        assert "run_terminal" not in names
        assert "mcp__github" in names

    def test_no_action_returns_all_tools(self) -> None:
        node = self._node(_make_multi_tool_gateway())
        state = AgentState(user_request="t", project_root="/root")
        names = {t.name for t in node._available_tools(state)}
        assert names == {
            "read_file", "list_files", "search_code", "navigate_code", "get_diagnostics",
            "write_file", "apply_patch", "delete_file", "run_terminal", "git",
            "mcp__github",
        }

    def test_unknown_action_defaults_to_no_tools(self) -> None:
        node = self._node(_make_multi_tool_gateway())
        state = AgentState(user_request="t", project_root="/root")
        names = {t.name for t in node._available_tools(state, "no_such_action")}
        assert names == set()

    def test_skill_allowlist_intersects_action_policy(self) -> None:
        node = self._node(_make_multi_tool_gateway())
        state = AgentState(
            user_request="t", project_root="/root", allowed_tools=["git", "read_file"],
        )
        names = {t.name for t in node._available_tools(state, "modify")}
        assert names == {"read_file"}

    def test_skill_allowlist_cannot_restore_manifest_pruned_tool(self) -> None:
        node = self._node(_make_multi_tool_gateway())
        state = AgentState(
            user_request="t",
            project_root="/root",
            allowed_tools=["git", "read_file"],
            tool_manifest={"selected_names": ["read_file"]},
        )
        names = {t.name for t in node._available_tools(state, "read")}
        assert names == {"read_file"}

    async def test_plan_execution_passes_pruned_schema_to_llm(self) -> None:
        """End-to-end: a modify step's tools= payload excludes shell/git/delete."""
        gateway = _make_multi_tool_gateway()
        # Model first calls write_file (valid for modify), then finishes.
        llm = AsyncMock(side_effect=[
            MockLLMResponse(choices=[MockChoice(message=MockChoiceMessage(
                content="modifying",
                tool_calls=[MockToolCall(function=MockToolCall.Function(
                    name="write_file",
                    arguments='{"file_path": "a.py", "content": "x = 1", "mode": "create"}',
                ))],
            ))]),
            MockLLMResponse(choices=[MockChoice(message=MockChoiceMessage(content="done"))]),
        ])
        node = ExecutionNode(
            llm=llm,
            tool_gateway=gateway,
            validation_gateway=make_validation_gateway(),
        )
        state = AgentState(
            user_request="add a file",
            project_root="/root",
            plan=[PlanStep(step_id=1, description="create a.py", action="create", target_file="a.py")],
        )
        await node(state)

        # The first (and only) LLM call that carried tools= must exclude
        # run_terminal / git / delete_file for a create step.
        tool_names = {
            t["function"]["name"]
            for call in llm.await_args_list
            if call.kwargs.get("tools")
            for t in call.kwargs["tools"]
        }
        assert "write_file" in tool_names
        assert "read_file" in tool_names
        assert "run_terminal" not in tool_names
        assert "git" not in tool_names
        assert "delete_file" not in tool_names
        assert "mcp__github" in tool_names


@pytest.mark.asyncio
async def test_planned_execution_preserves_prior_execution_log() -> None:
    prior = {
        "type": "tool_call",
        "tool_name": "apply_patch",
        "success": True,
    }
    node = ExecutionNode(
        llm=AsyncMock(),
        tool_gateway=_make_multi_tool_gateway(),
        validation_gateway=make_validation_gateway(),
    )
    state = AgentState(
        user_request="fix it",
        project_root="/root",
        plan=[PlanStep(step_id=1, description="done", action="modify")],
        current_step_index=1,
        execution_log=[prior],
    )

    result = await node._execute_with_plan(state)

    assert result["execution_log"] == [prior]
