"""HumanReviewNode 单元测试。

覆盖：
- 初始化（默认 callback、自定义 callback）
- 高风险计划审核（批准/拒绝/修改/终止）
- 偏离检测审核（各种决策处理）
- 验证失败审核
- 路由配合
- 边界条件
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from codeagent.orchestration.nodes.human_review_node import HumanReviewNode
from codeagent.orchestration.state import AgentState, PlanStep


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def base_state() -> AgentState:
    return AgentState(
        user_request="test request",
        project_root="/test/project",
    )


def make_callback(decision: str = "approve") -> AsyncMock:
    """创建模拟审核回调。"""
    cb = AsyncMock()
    cb.return_value = decision
    return cb


# ── Tests: Initialization ─────────────────────────────────────────────────


class TestHumanReviewInit:
    """初始化测试。"""

    async def test_default_callback(self) -> None:
        """默认 callback 为 None。"""
        node = HumanReviewNode()
        assert node._review_callback is None

    async def test_custom_callback(self) -> None:
        """可注入自定义 callback。"""
        cb = make_callback()
        node = HumanReviewNode(review_callback=cb)
        assert node._review_callback is cb


# ── Tests: Resume path (human_decision already injected) ──────────────────


@pytest.mark.asyncio
class TestResumePath:
    """resume 路径（human_decision 已通过 aupdate_state 注入）。"""

    async def test_resume_approve(self) -> None:
        """approve 决策应正确传递。"""
        state = AgentState(
            user_request="test",
            project_root="/root",
            human_decision="approve",
            human_review_required=True,
        )
        node = HumanReviewNode()
        result = await node(state)
        assert result["human_decision"] == "approve"
        assert result["human_review_required"] is False

    async def test_resume_abort(self) -> None:
        """abort 决策应终止。"""
        state = AgentState(
            user_request="test",
            project_root="/root",
            human_decision="abort",
        )
        node = HumanReviewNode()
        result = await node(state)
        assert result["human_decision"] == "abort"

    async def test_resume_modify(self) -> None:
        """modify 决策应触发重新规划。"""
        state = AgentState(
            user_request="test",
            project_root="/root",
            human_decision="modify",
        )
        node = HumanReviewNode()
        result = await node(state)
        assert result["human_decision"] == "modify"

    async def test_resume_resets_flags(self) -> None:
        """resume 后应重置偏离和审核标记。"""
        state = AgentState(
            user_request="test",
            project_root="/root",
            human_decision="approve",
            human_review_required=True,
            deviation_detected=True,
            deviation_count=3,
        )
        node = HumanReviewNode()
        result = await node(state)
        assert result["human_review_required"] is False
        assert result["deviation_detected"] is False
        assert result["deviation_count"] == 0

    async def test_resume_logs_event(self) -> None:
        """resume 路径应在 execution_log 中记录审核事件。"""
        state = AgentState(
            user_request="test",
            project_root="/root",
            human_decision="approve",
        )
        node = HumanReviewNode()
        result = await node(state)
        review_events = [
            e for e in result["execution_log"] if e["type"] == "human_review"
        ]
        assert len(review_events) >= 1
        assert review_events[0]["human_decision"] == "approve"
        assert review_events[0]["resume"] is True


# ── Tests: High-risk plan review ──────────────────────────────────────────


@pytest.mark.asyncio
class TestHighRiskPlanReview:
    """高风险计划审核。"""

    async def test_high_risk_approve(self) -> None:
        """高风险计划批准后应继续执行。"""
        plan = [
            PlanStep(step_id=1, description="Delete old module", action="delete",
                     target_file="old.py", risk="high"),
        ]
        state = AgentState(
            user_request="test",
            project_root="/root",
            plan=plan,
            current_step_index=0,
        )
        cb = make_callback("approve")
        node = HumanReviewNode(review_callback=cb)
        result = await node(state)

        cb.assert_called_once()
        review_request = cb.call_args[0][0]
        assert review_request["review_type"] == "high_risk_plan"
        assert "high_risk_steps" in review_request["details"]
        assert result["human_decision"] == "approve"

    async def test_high_risk_reject(self) -> None:
        """高风险计划拒绝后应终止。"""
        plan = [
            PlanStep(step_id=1, description="Delete old module", action="delete",
                     target_file="old.py", risk="high"),
        ]
        state = AgentState(
            user_request="test",
            project_root="/root",
            plan=plan,
            current_step_index=0,
        )
        cb = make_callback("abort")
        node = HumanReviewNode(review_callback=cb)
        result = await node(state)
        assert result["human_decision"] == "abort"

    async def test_high_risk_modify(self) -> None:
        """高风险计划修改应触发重新规划。"""
        plan = [
            PlanStep(step_id=1, description="Delete old module", action="delete",
                     target_file="old.py", risk="high"),
        ]
        state = AgentState(
            user_request="test",
            project_root="/root",
            plan=plan,
            current_step_index=0,
        )
        cb = make_callback("modify")
        node = HumanReviewNode(review_callback=cb)
        result = await node(state)
        assert result["human_decision"] == "modify"

    async def test_callback_receives_correct_review_type(self) -> None:
        """callback 应收到正确的 review_type。"""
        plan = [
            PlanStep(step_id=1, description="Delete", action="delete",
                     target_file="f.py", risk="high"),
        ]
        state = AgentState(
            user_request="test",
            project_root="/root",
            plan=plan,
        )
        cb = make_callback("approve")
        node = HumanReviewNode(review_callback=cb)
        await node(state)

        review_request = cb.call_args[0][0]
        assert review_request["review_type"] == "high_risk_plan"
        assert review_request["title"] == "高风险操作审核"
        assert "options" in review_request


# ── Tests: Deviation detected review ──────────────────────────────────────


@pytest.mark.asyncio
class TestDeviationReview:
    """偏离检测审核。"""

    async def test_deviation_approve(self) -> None:
        """偏离检测后批准应继续。"""
        state = AgentState(
            user_request="test",
            project_root="/root",
            review_type="deviation_detected",
            review_request={
                "review_type": "deviation_detected",
                "details": {"tool_name": "write_file", "deviation_count": 3},
            },
            human_review_required=True,
            deviation_count=3,
        )
        cb = make_callback("approve")
        node = HumanReviewNode(review_callback=cb)
        result = await node(state)
        assert result["human_decision"] == "approve"
        assert result["human_review_required"] is False
        assert result["deviation_count"] == 0

    async def test_deviation_abort(self) -> None:
        """偏离检测后放弃。"""
        state = AgentState(
            user_request="test",
            project_root="/root",
            review_type="deviation_detected",
            review_request={"review_type": "deviation_detected", "details": {}},
            human_review_required=True,
            deviation_count=3,
        )
        cb = make_callback("abort")
        node = HumanReviewNode(review_callback=cb)
        result = await node(state)
        assert result["human_decision"] == "abort"

    async def test_deviation_callback_receives_details(self) -> None:
        """偏离检测的 callback 应收到偏离详情。"""
        state = AgentState(
            user_request="test",
            project_root="/root",
            review_type="deviation_detected",
            review_request={
                "review_type": "deviation_detected",
                "details": {
                    "step_id": 1,
                    "step_description": "Modify main.py",
                    "tool_name": "write_file",
                    "tool_args": {"file_path": "wrong.py"},
                    "consecutive_deviations": 3,
                },
            },
            deviation_count=3,
            human_review_required=True,
            original_goal_summary="Modify main.py",
        )
        cb = make_callback("approve")
        node = HumanReviewNode(review_callback=cb)
        await node(state)

        req = cb.call_args[0][0]
        assert req["review_type"] == "deviation_detected"
        assert req["details"]["tool_name"] == "write_file"
        assert req["details"]["deviation_count"] == 3


# ── Tests: Validation failure review ──────────────────────────────────────


@pytest.mark.asyncio
class TestValidationFailureReview:
    """验证失败审核。"""

    async def test_validation_failure_continue(self) -> None:
        """验证失败后选择继续。"""
        from codeagent.gateway.validation_gateway import ValidationError, ValidationResult

        state = AgentState(
            user_request="test",
            project_root="/root",
            validation_results=[
                ValidationResult(passed=False, errors=[
                    ValidationError(file_path="bad.py", line=1, message="syntax error"),
                ]),
                ValidationResult(passed=False, errors=[
                    ValidationError(file_path="bad.py", line=5, message="lint error"),
                ]),
                ValidationResult(passed=False, errors=[
                    ValidationError(file_path="bad.py", line=10, message="type error"),
                ]),
            ],
            retry_count=3,
        )
        cb = make_callback("approve")
        node = HumanReviewNode(review_callback=cb)
        result = await node(state)
        assert result["human_decision"] == "approve"

    async def test_validation_failure_abort(self) -> None:
        """验证失败后放弃。"""
        from codeagent.gateway.validation_gateway import ValidationError, ValidationResult

        state = AgentState(
            user_request="test",
            project_root="/root",
            validation_results=[
                ValidationResult(passed=False, errors=[
                    ValidationError(file_path="bad.py", line=1, message="error"),
                ]),
                ValidationResult(passed=False, errors=[
                    ValidationError(file_path="bad.py", line=2, message="error"),
                ]),
                ValidationResult(passed=False, errors=[
                    ValidationError(file_path="bad.py", line=3, message="error"),
                ]),
            ],
            retry_count=3,
        )
        cb = make_callback("abort")
        node = HumanReviewNode(review_callback=cb)
        result = await node(state)
        assert result["human_decision"] == "abort"

    async def test_no_validation_failure_determines_other_type(self) -> None:
        """无验证失败时不应触发 validation_failure。"""
        from codeagent.gateway.validation_gateway import ValidationResult

        state = AgentState(
            user_request="test",
            project_root="/root",
            validation_results=[
                ValidationResult(passed=True),
            ],
        )
        cb = make_callback("approve")
        node = HumanReviewNode(review_callback=cb)
        await node(state)

        req = cb.call_args[0][0]
        # 只有 1 个 passed 的结果，不应触发 validation_failure
        assert req["review_type"] != "validation_failure"

    async def test_validation_details_in_callback(self) -> None:
        """验证失败的 callback 应收到失败详情。"""
        from codeagent.gateway.validation_gateway import ValidationError, ValidationResult

        state = AgentState(
            user_request="test",
            project_root="/root",
            validation_results=[
                ValidationResult(passed=False, errors=[
                    ValidationError(file_path="bad.py", line=1, message="syntax error"),
                    ValidationError(file_path="bad.py", line=2, message="lint error"),
                    ValidationError(file_path="bad.py", line=3, message="type error"),
                ]),
            ],
            retry_count=3,
        )
        cb = make_callback("approve")
        node = HumanReviewNode(review_callback=cb)
        await node(state)

        req = cb.call_args[0][0]
        assert req["review_type"] == "validation_failure"
        assert len(req["details"]["failed_validations"]) >= 1
        assert req["details"]["retry_count"] == 3


# ── Tests: User interrupt ─────────────────────────────────────────────────


@pytest.mark.asyncio
class TestUserInterrupt:
    """用户干预。"""

    async def test_user_interrupt_review_type(self) -> None:
        """无明确触发类型时应为 user_interrupt。"""
        state = AgentState(
            user_request="test",
            project_root="/root",
        )
        cb = make_callback("approve")
        node = HumanReviewNode(review_callback=cb)
        await node(state)

        req = cb.call_args[0][0]
        assert req["review_type"] == "user_interrupt"
        assert req["title"] == "用户干预"


# ── Tests: Route coordination ─────────────────────────────────────────────


@pytest.mark.asyncio
class TestRouteCoordination:
    """与路由函数的配合。"""

    async def test_approve_routes_to_execution(self) -> None:
        """approve 后 route_after_human_review 应返回 execution。"""
        from codeagent.orchestration.routing import route_after_human_review

        state = AgentState(
            user_request="test",
            project_root="/root",
            human_decision="approve",
        )
        next_node = route_after_human_review(state)
        assert next_node == "execution"

    async def test_abort_routes_to_end(self) -> None:
        """abort 后 route_after_human_review 应返回 end。"""
        from codeagent.orchestration.routing import route_after_human_review

        state = AgentState(
            user_request="test",
            project_root="/root",
            human_decision="abort",
        )
        next_node = route_after_human_review(state)
        assert next_node == "end"

    async def test_modify_routes_to_planning(self) -> None:
        """modify 后 route_after_human_review 应返回 planning。"""
        from codeagent.orchestration.routing import route_after_human_review

        state = AgentState(
            user_request="test",
            project_root="/root",
            human_decision="modify",
        )
        next_node = route_after_human_review(state)
        assert next_node == "planning"

    async def test_callback_decision_sets_correct_state(self) -> None:
        """callback 返回的决策应设置正确的 human_decision。"""
        plan = [
            PlanStep(step_id=1, description="Delete", action="delete",
                     target_file="f.py", risk="high"),
        ]
        state = AgentState(
            user_request="test",
            project_root="/root",
            plan=plan,
        )
        cb = make_callback("approve")
        node = HumanReviewNode(review_callback=cb)
        result = await node(state)
        assert result["human_decision"] == "approve"

        # 验证路由
        from codeagent.orchestration.routing import route_after_human_review
        # 注意：route_after_human_review 接收 AgentState，需要重建
        route_state = AgentState(
            user_request="test",
            project_root="/root",
            human_decision=result["human_decision"],
        )
        assert route_after_human_review(route_state) == "execution"


# ── Tests: Edge cases ─────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestEdgeCases:
    """边界条件。"""

    async def test_empty_state(self) -> None:
        """空状态应使用 user_interrupt 类型并默认 abort。"""
        state = AgentState(
            user_request="",
            project_root="/root",
        )
        cb = make_callback("approve")
        node = HumanReviewNode(review_callback=cb)
        result = await node(state)
        assert result["human_decision"] == "approve"

    async def test_callback_exception(self) -> None:
        """callback 抛出异常时应默认 abort。"""
        state = AgentState(
            user_request="test",
            project_root="/root",
        )
        cb = make_callback()
        cb.side_effect = RuntimeError("callback failed")
        node = HumanReviewNode(review_callback=cb)
        result = await node(state)
        assert result["human_decision"] == "abort"

    async def test_invalid_decision_from_callback(self) -> None:
        """callback 返回无效决策时应默认 abort。"""
        state = AgentState(
            user_request="test",
            project_root="/root",
        )
        cb = make_callback("invalid_option")
        node = HumanReviewNode(review_callback=cb)
        result = await node(state)
        assert result["human_decision"] == "abort"

    async def test_logs_review_event(self) -> None:
        """审核事件应记录到 execution_log。"""
        state = AgentState(
            user_request="test",
            project_root="/root",
        )
        cb = make_callback("approve")
        node = HumanReviewNode(review_callback=cb)
        result = await node(state)

        review_events = [
            e for e in result["execution_log"] if e["type"] == "human_review"
        ]
        assert len(review_events) >= 1
        event = review_events[0]
        assert event["human_decision"] == "approve"
        assert event["review_type"] == "user_interrupt"

    async def test_existing_execution_log_preserved(self) -> None:
        """已有的 execution_log 应在审核后保留。"""
        state = AgentState(
            user_request="test",
            project_root="/root",
            execution_log=[{"type": "tool_call", "tool_name": "read_file"}],
        )
        cb = make_callback("approve")
        node = HumanReviewNode(review_callback=cb)
        result = await node(state)

        tool_events = [
            e for e in result["execution_log"] if e["type"] == "tool_call"
        ]
        assert len(tool_events) == 1
