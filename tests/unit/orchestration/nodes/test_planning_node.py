"""PlanningNode 单元测试。

Mock LLM，覆盖正常解析、JSON 错误重试、Schema 校验拒绝、冲突检测、边缘情况。
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from codeagent.orchestration.nodes.planning_node import PlanningNode
from codeagent.orchestration.state import AgentState, PlanStep

# ── 测试数据 ────────────────────────────────────────────────

_VALID_PLAN = [
    {
        "step_id": 1,
        "description": "Read main.py",
        "action": "read",
        "target_file": "main.py",
        "risk": "low",
        "dependencies": [],
    },
    {
        "step_id": 2,
        "description": "Create app.py",
        "action": "create",
        "target_file": "app.py",
        "risk": "low",
        "dependencies": [1],
    },
    {
        "step_id": 3,
        "description": "Run tests",
        "action": "command",
        "target_file": None,
        "risk": "low",
        "dependencies": [2],
    },
]

_CONFLICTING_PLAN = [
    {
        "step_id": 1,
        "description": "Modify main.py",
        "action": "modify",
        "target_file": "main.py",
        "risk": "low",
        "dependencies": [],
    },
    {
        "step_id": 2,
        "description": "Delete main.py",
        "action": "delete",
        "target_file": "main.py",
        "risk": "high",
        "dependencies": [],
    },
]


# ── Fixtures ────────────────────────────────────────────────


def make_llm(response: str | None = None) -> AsyncMock:
    """创建模拟 LLM 调用函数。"""
    mock = AsyncMock()
    message = MagicMock()
    message.content = response or json.dumps(_VALID_PLAN)
    message.reasoning_content = ""
    choice = MagicMock()
    choice.message = message
    mock_response = MagicMock()
    mock_response.choices = [choice]
    mock.return_value = mock_response
    return mock


@pytest.fixture
def state() -> AgentState:
    return AgentState(
        user_request="Create a hello world Flask app",
        project_root="/test/project",
    )


@pytest.fixture
def node() -> PlanningNode:
    return PlanningNode(llm=make_llm())


@pytest.fixture
def node_with_state_context() -> PlanningNode:
    """创建一个有文件树上下文的节点。"""
    state = AgentState(
        user_request="Refactor main.py",
        project_root="/test/project",
        file_tree={
            "name": "project",
            "type": "directory",
            "children": [
                {"name": "main.py", "type": "file"},
                {"name": "utils.py", "type": "file"},
            ],
        },
    )
    node = PlanningNode(llm=make_llm())
    return node, state


# ── Helpers ─────────────────────────────────────────────────


def make_response(content: str | None) -> MagicMock:
    """创建模拟 LLM 响应。"""
    message = MagicMock()
    message.content = content
    message.reasoning_content = ""
    choice = MagicMock()
    choice.message = message
    resp = MagicMock()
    resp.choices = [choice]
    return resp


# ── Tests: Normal flow ──────────────────────────────────────


class TestPlanningNodeNormalFlow:
    @pytest.mark.asyncio
    async def test_uses_reasoning_content_when_provider_content_is_empty(self, state) -> None:
        message = MagicMock()
        message.content = ""
        message.reasoning_content = json.dumps({
            "plan": _VALID_PLAN,
            "original_goal_summary": "Create the requested app",
        })
        response = MagicMock()
        response.choices = [MagicMock(message=message)]
        llm = AsyncMock(return_value=response)

        result = await PlanningNode(llm=llm)(state)

        assert len(result["plan"]) == 3
        assert result["original_goal_summary"] == "Create the requested app"
        assert llm.await_count == 1

    @pytest.mark.asyncio
    async def test_benchmark_uses_deterministic_two_step_plan_without_llm(self) -> None:
        llm = AsyncMock()
        benchmark_state = AgentState(
            user_request="Fix it",
            project_root="/test/project",
            benchmark_instance_id="owner__repo-1",
        )

        result = await PlanningNode(llm=llm)(benchmark_state)

        assert [step.action for step in result["plan"]] == ["modify", "command"]
        assert result["execution_log"][0]["source"] == "deterministic_benchmark_policy"
        assert result["llm_call_count"] == 0
        llm.assert_not_awaited()

    """正常执行流程。"""

    @pytest.mark.asyncio
    async def test_generates_plan(self, node: PlanningNode, state: AgentState) -> None:
        result = await node(state)
        assert "plan" in result
        assert result["plan"] is not None
        assert len(result["plan"]) == 3

    @pytest.mark.asyncio
    async def test_plan_steps_have_correct_types(
        self, node: PlanningNode, state: AgentState
    ) -> None:
        result = await node(state)
        for step in result["plan"]:
            assert isinstance(step, PlanStep)
            assert isinstance(step.step_id, int)
            assert isinstance(step.description, str)
            assert step.action in ("create", "modify", "delete", "read", "command")
            assert step.risk in ("low", "high")
            assert step.acceptance_criteria
            assert step.evidence == []

    @pytest.mark.asyncio
    async def test_returns_execution_log(
        self, node: PlanningNode, state: AgentState
    ) -> None:
        result = await node(state)
        assert "execution_log" in result
        assert result["execution_log"][0]["type"] == "plan_generated"
        assert result["execution_log"][0]["steps"] == 3

    @pytest.mark.asyncio
    async def test_no_errors_on_success(
        self, node: PlanningNode, state: AgentState
    ) -> None:
        result = await node(state)
        assert "errors" not in result or result.get("errors") == []

    @pytest.mark.asyncio
    async def test_prompt_includes_user_request(
        self, state: AgentState
    ) -> None:
        llm = make_llm()
        node = PlanningNode(llm=llm)
        await node(state)
        # LLM 应被调用
        call_kwargs = llm.call_args[1]
        messages = call_kwargs.get("messages", [])
        system_msg = messages[0]
        assert state.user_request in system_msg["content"]
        assert "step_id" in system_msg["content"]
        assert "action" in system_msg["content"]


class TestPlanningNodeWithContext:
    """有上下文时的规划。"""

    @pytest.mark.asyncio
    async def test_context_included_in_prompt(self) -> None:
        """file_tree 和 semantic_context 应包含在提示词中。"""
        state = AgentState(
            user_request="Refactor",
            project_root="/root",
            file_tree={
                "name": "root",
                "type": "directory",
                "children": [
                    {"name": "src", "type": "directory", "children": [
                        {"name": "main.py", "type": "file"},
                    ]},
                ],
            },
            semantic_context="main.py contains Flask app",
        )
        llm = make_llm()
        node = PlanningNode(llm=llm)
        await node(state)

        call_kwargs = llm.call_args[1]
        system_msg = call_kwargs["messages"][0]["content"]
        # The workspace root name is context, not part of tool-relative paths.
        assert "root/" not in system_msg
        assert "src/" in system_msg
        assert "main.py" in system_msg
        assert "Flask app" in system_msg


class TestPlanningNodeJSONParsing:
    """JSON 解析边界情况。"""

    @pytest.mark.asyncio
    async def test_handle_markdown_json(self, state: AgentState) -> None:
        """LLM 用 markdown 代码块包裹 JSON 时也应能解析。"""
        llm = make_llm(
            response=f"```json\n{json.dumps(_VALID_PLAN)}\n```"
        )
        node = PlanningNode(llm=llm)
        result = await node(state)
        assert result["plan"] is not None
        assert len(result["plan"]) == 3

    @pytest.mark.asyncio
    async def test_handle_markdown_with_extra_text(self, state: AgentState) -> None:
        """LLM 在 markdown 前后添加说明文字时也能解析。"""
        content = (
            "Here is the plan I generated:\n\n"
            f"```json\n{json.dumps(_VALID_PLAN)}\n```\n\n"
            "Let me know if you need changes."
        )
        llm = make_llm(response=content)
        node = PlanningNode(llm=llm)
        result = await node(state)
        assert result["plan"] is not None
        assert len(result["plan"]) == 3

    @pytest.mark.asyncio
    async def test_handle_plain_json(self, state: AgentState) -> None:
        """纯 JSON 数组输出。"""
        llm = make_llm(response=json.dumps(_VALID_PLAN))
        node = PlanningNode(llm=llm)
        result = await node(state)
        assert result["plan"] is not None

    @pytest.mark.asyncio
    async def test_handle_extra_whitespace(self, state: AgentState) -> None:
        """JSON 前后有空白字符。"""
        llm = make_llm(response=f"  \n{json.dumps(_VALID_PLAN)}\n  ")
        node = PlanningNode(llm=llm)
        result = await node(state)
        assert result["plan"] is not None
        assert len(result["plan"]) == 3

    @pytest.mark.asyncio
    async def test_single_step_plan(self, state: AgentState) -> None:
        """单步骤计划。"""
        single = [{"step_id": 1, "description": "Do something", "action": "read",
                    "target_file": "f.py", "risk": "low", "dependencies": []}]
        llm = make_llm(response=json.dumps(single))
        node = PlanningNode(llm=llm)
        result = await node(state)
        assert len(result["plan"]) == 1


# ── Tests: Retry logic ──────────────────────────────────────


class TestPlanningNodeRetry:
    """JSON 解析失败重试逻辑。"""

    @pytest.mark.asyncio
    async def test_retry_on_invalid_json(self, state: AgentState) -> None:
        """首次返回无效 JSON，重试后成功。"""
        llm = AsyncMock()
        llm.side_effect = [
            make_response("Not JSON at all"),
            make_response(json.dumps(_VALID_PLAN)),
        ]
        node = PlanningNode(llm=llm, max_retries=2)
        result = await node(state)
        assert result["plan"] is not None
        assert len(result["plan"]) == 3
        assert llm.call_count == 2  # 重试一次后成功

    @pytest.mark.asyncio
    async def test_retry_on_malformed_json(self, state: AgentState) -> None:
        """首次返回部分 JSON，重试后成功。"""
        llm = AsyncMock()
        llm.side_effect = [
            make_response("{invalid json}"),
            make_response(json.dumps(_VALID_PLAN)),
        ]
        node = PlanningNode(llm=llm, max_retries=2)
        result = await node(state)
        assert result["plan"] is not None
        assert llm.call_count == 2

    @pytest.mark.asyncio
    async def test_exhaust_retries(self, state: AgentState) -> None:
        """重试耗尽后应返回错误。"""
        llm = AsyncMock()
        llm.side_effect = [
            make_response("not json"),
            make_response("still not json"),
            make_response("nope"),
        ]
        node = PlanningNode(llm=llm, max_retries=2)
        result = await node(state)
        assert result["plan"] is None
        assert len(result["errors"]) >= 1
        assert "Plan generation failed" in result["errors"][0]
        assert llm.call_count == 3  # 初始 + 2 次重试

    @pytest.mark.asyncio
    async def test_retry_feedback_included(self, state: AgentState) -> None:
        """重试时应包含错误反馈信息。"""
        llm = AsyncMock()
        llm.side_effect = [
            make_response("bad"),
            make_response(json.dumps(_VALID_PLAN)),
        ]
        node = PlanningNode(llm=llm, max_retries=2)
        await node(state)
        # 第二次调用应包含错误反馈
        second_msgs = llm.call_args_list[1][1]["messages"]
        user_msgs = [m for m in second_msgs if m["role"] == "user"]
        assert any("JSON parse error" in m["content"] for m in user_msgs)


# ── Tests: Schema validation ────────────────────────────────


class TestPlanningNodeValidation:
    """Schema 校验。"""

    @pytest.mark.asyncio
    async def test_missing_step_id(self, state: AgentState) -> None:
        data = [{"description": "no id", "action": "read", "target_file": "f.py"}]
        llm = make_llm(response=json.dumps(data))
        node = PlanningNode(llm=llm)
        result = await node(state)
        assert result["plan"] is None
        assert any("step_id" in e for e in result["errors"])

    @pytest.mark.asyncio
    async def test_invalid_action(self, state: AgentState) -> None:
        data = [{"step_id": 1, "description": "bad", "action": "invalid_action",
                 "target_file": "f.py", "risk": "low", "dependencies": []}]
        llm = make_llm(response=json.dumps(data))
        node = PlanningNode(llm=llm)
        result = await node(state)
        assert result["plan"] is None

    @pytest.mark.asyncio
    async def test_invalid_risk(self, state: AgentState) -> None:
        data = [{"step_id": 1, "description": "bad risk", "action": "read",
                 "target_file": "f.py", "risk": "extreme", "dependencies": []}]
        llm = make_llm(response=json.dumps(data))
        node = PlanningNode(llm=llm)
        result = await node(state)
        assert result["plan"] is None

    @pytest.mark.asyncio
    async def test_empty_plan(self, state: AgentState) -> None:
        llm = make_llm(response=json.dumps([]))
        node = PlanningNode(llm=llm)
        result = await node(state)
        assert result["plan"] is None
        assert any("at least one step" in e for e in result["errors"])

    @pytest.mark.asyncio
    async def test_duplicate_step_ids(self, state: AgentState) -> None:
        data = [
            {"step_id": 1, "description": "first", "action": "read",
             "target_file": "a.py", "risk": "low", "dependencies": []},
            {"step_id": 1, "description": "duplicate", "action": "read",
             "target_file": "b.py", "risk": "low", "dependencies": []},
        ]
        llm = make_llm(response=json.dumps(data))
        node = PlanningNode(llm=llm)
        result = await node(state)
        assert result["plan"] is None
        assert any("Duplicate" in e for e in result["errors"])

    @pytest.mark.asyncio
    async def test_non_existent_dependency(self, state: AgentState) -> None:
        data = [
            {"step_id": 1, "description": "first", "action": "read",
             "target_file": "a.py", "risk": "low", "dependencies": []},
            {"step_id": 2, "description": "second", "action": "read",
             "target_file": "b.py", "risk": "low", "dependencies": [5]},
        ]
        llm = make_llm(response=json.dumps(data))
        node = PlanningNode(llm=llm)
        result = await node(state)
        assert result["plan"] is None
        assert any("depends on" in e for e in result["errors"])

    @pytest.mark.asyncio
    async def test_missing_description(self, state: AgentState) -> None:
        data = [{"step_id": 1, "action": "read", "target_file": "f.py"}]
        llm = make_llm(response=json.dumps(data))
        node = PlanningNode(llm=llm)
        result = await node(state)
        assert result["plan"] is None

    @pytest.mark.asyncio
    async def test_not_a_list(self, state: AgentState) -> None:
        llm = make_llm(response=json.dumps({"step_id": 1, "description": "not a list"}))
        node = PlanningNode(llm=llm)
        result = await node(state)
        assert result["plan"] is None


# ── Tests: Conflict detection ───────────────────────────────


class TestPlanningNodeConflicts:
    """冲突检测。"""

    @pytest.mark.asyncio
    async def test_detect_delete_modify_conflict(self, state: AgentState) -> None:
        llm = make_llm(response=json.dumps(_CONFLICTING_PLAN))
        node = PlanningNode(llm=llm)
        result = await node(state)
        assert result["plan"] is not None  # 冲突不阻止计划生成
        log = result["execution_log"][0]
        assert log["type"] == "plan_generated"
        # 冲突应该被记录
        conflicts = log.get("conflicts")
        assert conflicts is not None
        # 应该检测到 delete_and_modify 冲突
        types = [c["type"] for c in conflicts]
        assert "delete_and_modify" in types

    @pytest.mark.asyncio
    async def test_no_conflicts(self, node: PlanningNode, state: AgentState) -> None:
        result = await node(state)
        log = result["execution_log"][0]
        assert log["conflicts"] is None

    @pytest.mark.asyncio
    async def test_delete_create_same_file(self, state: AgentState) -> None:
        plan = [
            {"step_id": 1, "description": "Delete old", "action": "delete",
             "target_file": "main.py", "risk": "high", "dependencies": []},
            {"step_id": 2, "description": "Create new", "action": "create",
             "target_file": "main.py", "risk": "low", "dependencies": []},
        ]
        llm = make_llm(response=json.dumps(plan))
        node = PlanningNode(llm=llm)
        result = await node(state)
        assert result["plan"] is not None
        conflicts = result["execution_log"][0]["conflicts"]
        types = [c["type"] for c in conflicts]
        assert "delete_and_create_same_file" in types

    @pytest.mark.asyncio
    async def test_no_conflict_same_file_different_actions(self, state: AgentState) -> None:
        """同一文件上的 read 和 modify 不应冲突。"""
        plan = [
            {"step_id": 1, "description": "Read main.py", "action": "read",
             "target_file": "main.py", "risk": "low", "dependencies": []},
            {"step_id": 2, "description": "Modify main.py", "action": "modify",
             "target_file": "main.py", "risk": "low", "dependencies": [1]},
        ]
        llm = make_llm(response=json.dumps(plan))
        node = PlanningNode(llm=llm)
        result = await node(state)
        assert result["plan"] is not None
        conflicts = result["execution_log"][0].get("conflicts")
        assert conflicts is None or len(conflicts) == 0


# ── Tests: Edge cases ───────────────────────────────────────


class TestPlanningNodeEdgeCases:
    """边界情况。"""

    @pytest.mark.asyncio
    async def test_empty_response(self, state: AgentState) -> None:
        """LLM 返回空内容。"""
        llm = AsyncMock()
        llm.return_value = make_response("")
        node = PlanningNode(llm=llm)
        result = await node(state)
        assert result["plan"] is None
        assert len(result["errors"]) >= 1

    @pytest.mark.asyncio
    async def test_none_response(self, state: AgentState) -> None:
        """LLM 返回 None。"""
        llm = AsyncMock()
        llm.return_value = make_response(None)
        node = PlanningNode(llm=llm)
        result = await node(state)
        assert result["plan"] is None

    @pytest.mark.asyncio
    async def test_command_action_no_target(self, state: AgentState) -> None:
        """command 类型不需要 target_file。"""
        plan = [{"step_id": 1, "description": "Run lint", "action": "command",
                 "target_file": None, "risk": "low", "dependencies": []}]
        llm = make_llm(response=json.dumps(plan))
        node = PlanningNode(llm=llm)
        result = await node(state)
        assert result["plan"] is not None
        assert result["plan"][0].target_file is None

    @pytest.mark.asyncio
    async def test_long_plan(self, state: AgentState) -> None:
        """多步骤计划。"""
        plan = []
        for i in range(10):
            plan.append({
                "step_id": i + 1,
                "description": f"Step {i + 1}",
                "action": "read",
                "target_file": f"file{i}.py",
                "risk": "low",
                "dependencies": list(range(1, i)) if i > 0 else [],
            })
        llm = make_llm(response=json.dumps(plan))
        node = PlanningNode(llm=llm)
        result = await node(state)
        assert result["plan"] is not None
        assert len(result["plan"]) == 10

    @pytest.mark.asyncio
    async def test_all_high_risk(self, state: AgentState) -> None:
        """所有步骤都是高风险。"""
        plan = [{"step_id": i + 1, "description": f"Delete f{i}",
                 "action": "delete",
                 "target_file": f"f{i}.py", "risk": "high", "dependencies": []}
                for i in range(3)]
        llm = make_llm(response=json.dumps(plan))
        node = PlanningNode(llm=llm)
        result = await node(state)
        assert result["plan"] is not None
        assert all(s.risk == "high" for s in result["plan"])

    @pytest.mark.asyncio
    async def test_clean_json_strips_markdown(self) -> None:
        """_clean_json 去除 markdown 代码块。"""
        node = PlanningNode(llm=make_llm())
        content = "Some text\n```json\n[{\"key\": \"value\"}]\n```\nMore text"
        cleaned = node._clean_json(content)
        assert cleaned.startswith("[")
        assert "key" in cleaned

    @pytest.mark.asyncio
    async def test_clean_json_plain_array(self) -> None:
        node = PlanningNode(llm=make_llm())
        content = '[{"step_id": 1}]'
        cleaned = node._clean_json(content)
        assert cleaned == content

    @pytest.mark.asyncio
    async def test_llm_call_failure(self, state: AgentState) -> None:
        """LLM 调用本身抛出异常。"""
        llm = AsyncMock(side_effect=RuntimeError("LLM unavailable"))
        node = PlanningNode(llm=llm)
        result = await node(state)
        assert result["plan"] is None
        assert len(result["errors"]) >= 1

    @pytest.mark.asyncio
    async def test_command_file_names(self, state: AgentState) -> None:
        """command 类型可以有 null 或省略 target_file。"""
        plan = [{"step_id": 1, "description": "Run tests", "action": "command",
                 "risk": "low", "dependencies": []}]
        llm = make_llm(response=json.dumps(plan))
        node = PlanningNode(llm=llm)
        result = await node(state)
        assert result["plan"] is not None
        assert result["plan"][0].action == "command"


# ── Phase 3.5: Planning Node Goal Summary Tests ──────────────


class TestPlanningNodeGoalSummary:
    """Planning Node 目标摘要测试。"""

    @pytest.mark.asyncio
    async def test_new_format_with_goal_summary(self, state: AgentState) -> None:
        """新格式 JSON 对象包含 plan + original_goal_summary。"""
        response_data = {
            "plan": [
                {"step_id": 1, "description": "Create app", "action": "create",
                 "target_file": "app.py", "risk": "low", "dependencies": []},
            ],
            "original_goal_summary": "Create a Flask application with user login",
        }
        llm = make_llm(response=json.dumps(response_data))
        node = PlanningNode(llm=llm)
        result = await node(state)
        assert result["plan"] is not None
        assert len(result["plan"]) == 1
        assert result["original_goal_summary"] == "Create a Flask application with user login"

    @pytest.mark.asyncio
    async def test_backward_compat_plain_array(self, state: AgentState) -> None:
        """纯数组格式（向后兼容）也应工作，返回空的 original_goal_summary。"""
        llm = make_llm()
        node = PlanningNode(llm=llm)
        result = await node(state)
        assert result["plan"] is not None
        # 纯数组应返回空字符串
        assert result.get("original_goal_summary") == ""

    @pytest.mark.asyncio
    async def test_goal_summary_in_execution_log(self, state: AgentState) -> None:
        """execution_log 应包含 original_goal_summary。"""
        response_data = {
            "plan": [
                {"step_id": 1, "description": "Read config", "action": "read",
                 "target_file": "config.py", "risk": "low", "dependencies": []},
            ],
            "original_goal_summary": "Read and analyze config",
        }
        llm = make_llm(response=json.dumps(response_data))
        node = PlanningNode(llm=llm)
        result = await node(state)
        log = result["execution_log"][0]
        assert log["original_goal_summary"] == "Read and analyze config"


def test_benchmark_prompt_requests_minimal_two_step_plan() -> None:
    benchmark_state = AgentState(
        user_request="Fix regression",
        project_root="/test/project",
        benchmark_instance_id="owner__repo-1",
    )
    prompt = PlanningNode(llm=make_llm())._build_prompt(benchmark_state)

    assert "Prefer exactly two steps" in prompt
    assert "Do not add or modify tests" in prompt
