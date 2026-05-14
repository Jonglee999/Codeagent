"""IOrchestrationGateway 接口契约单元测试。

验证接口定义、Mock 实现的兼容性、以及 DTO 数据类的正确性。
"""

from __future__ import annotations

import inspect
import time
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any, get_type_hints

import pytest

from codeagent.gateway.orchestration_gateway import (
    HumanDecision,
    IOrchestrationGateway,
    TaskEvent,
    TaskReport,
    TaskState,
    TaskStatus,
    UserRequest,
)


# =============================================================================
# 测试辅助 — Mock 实现
# =============================================================================


class MockOrchestrationGateway(IOrchestrationGateway):
    """IOrchestrationGateway 的 Mock 实现，用于 isinstance 验证。"""

    async def start_task(self, request: UserRequest) -> str:
        return "mock-task-id"

    async def get_task_status(self, task_id: str) -> TaskStatus:
        return TaskStatus(task_id=task_id, state=TaskState.PENDING)

    async def stream_task(
        self, task_id: str
    ) -> AsyncGenerator[TaskEvent]:
        yield TaskEvent(type="thinking", data={}, timestamp=time.time())
        yield TaskEvent(
            type="done", data={"task_id": task_id}, timestamp=time.time()
        )

    async def submit_decision(
        self, task_id: str, decision: HumanDecision
    ) -> None:
        return None

    async def cancel_task(self, task_id: str) -> bool:
        return True

    async def get_report(self, task_id: str) -> TaskReport:
        return TaskReport(
            task_id=task_id,
            plan=[],
            changes=[],
            validation_results=[],
            duration=1.0,
            token_usage=0,
        )


# =============================================================================
# 测试：抽象接口不能直接实例化
# =============================================================================


class TestInterfaceInstantiation:
    """验证抽象接口不能直接实例化。"""

    def test_cannot_instantiate_abstract(self) -> None:
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IOrchestrationGateway()  # type: ignore[abstract]


# =============================================================================
# 测试：Mock 实现通过 isinstance 检查
# =============================================================================


class TestMockImplementation:
    """验证 Mock 实现可以通过 isinstance 检查。"""

    def test_mock_isinstance(self) -> None:
        mock = MockOrchestrationGateway()
        assert isinstance(mock, IOrchestrationGateway), (
            "MockOrchestrationGateway should satisfy IOrchestrationGateway"
        )


# =============================================================================
# 测试：接口方法签名一致性
# =============================================================================


class TestInterfaceSignatures:
    """验证接口方法签名（参数名、类型注解、返回值）符合预期。"""

    def test_start_task_signature(self) -> None:
        sig = inspect.signature(IOrchestrationGateway.start_task)
        params = sig.parameters

        assert "request" in params
        param_anno = params["request"].annotation
        # Handle string annotations vs actual types
        type_name = getattr(param_anno, "__name__", str(param_anno))
        assert "UserRequest" in type_name or param_anno is UserRequest, (
            f"Expected UserRequest annotation, got {param_anno}"
        )

        hints = get_type_hints(IOrchestrationGateway.start_task)
        assert hints["return"] is str, (
            f"Expected return str, got {hints['return']}"
        )

    def test_get_task_status_signature(self) -> None:
        sig = inspect.signature(IOrchestrationGateway.get_task_status)
        params = sig.parameters

        assert "task_id" in params
        assert params["task_id"].annotation is str

        hints = get_type_hints(IOrchestrationGateway.get_task_status)
        assert hints["return"] is TaskStatus, (
            f"Expected return TaskStatus, got {hints['return']}"
        )

    def test_stream_task_signature(self) -> None:
        sig = inspect.signature(IOrchestrationGateway.stream_task)
        params = sig.parameters

        assert "task_id" in params
        assert params["task_id"].annotation is str

        hints = get_type_hints(IOrchestrationGateway.stream_task)
        return_type = hints["return"]
        assert return_type is not None

    def test_submit_decision_signature(self) -> None:
        sig = inspect.signature(IOrchestrationGateway.submit_decision)
        params = sig.parameters

        assert "task_id" in params
        assert params["task_id"].annotation is str

        assert "decision" in params
        param_anno = params["decision"].annotation
        type_name = getattr(param_anno, "__name__", str(param_anno))
        assert "HumanDecision" in type_name or param_anno is HumanDecision, (
            f"Expected HumanDecision annotation, got {param_anno}"
        )

        hints = get_type_hints(IOrchestrationGateway.submit_decision)
        assert hints["return"] is type(None)

    def test_cancel_task_signature(self) -> None:
        sig = inspect.signature(IOrchestrationGateway.cancel_task)
        params = sig.parameters

        assert "task_id" in params
        assert params["task_id"].annotation is str

        hints = get_type_hints(IOrchestrationGateway.cancel_task)
        assert hints["return"] is bool

    def test_get_report_signature(self) -> None:
        sig = inspect.signature(IOrchestrationGateway.get_report)
        params = sig.parameters

        assert "task_id" in params
        assert params["task_id"].annotation is str

        hints = get_type_hints(IOrchestrationGateway.get_report)
        assert hints["return"] is TaskReport, (
            f"Expected return TaskReport, got {hints['return']}"
        )


# =============================================================================
# 测试：TaskState 枚举
# =============================================================================


class TestTaskState:
    """验证 TaskState 枚举值。"""

    def test_members(self) -> None:
        assert TaskState.PENDING.value == "pending"
        assert TaskState.RUNNING.value == "running"
        assert TaskState.WAITING_REVIEW.value == "waiting_review"
        assert TaskState.COMPLETED.value == "completed"
        assert TaskState.FAILED.value == "failed"
        assert TaskState.CANCELLED.value == "cancelled"

    def test_str_enum(self) -> None:
        assert str(TaskState.PENDING) == "TaskState.PENDING"
        assert TaskState.PENDING == TaskState("pending")


# =============================================================================
# 测试：DTO 数据类
# =============================================================================


class TestUserRequest:
    """验证 UserRequest dataclass 字段类型和默认值。"""

    def test_fields(self) -> None:
        fields = {f.name: f for f in UserRequest.__dataclass_fields__.values()}

        assert fields["query"].type is str
        assert fields["project_root"].type is str
        assert fields["auto_mode"].type is bool
        assert fields["max_retries"].type is int

    def test_defaults(self) -> None:
        req = UserRequest(query="test query", project_root="/project")
        assert req.auto_mode is False
        assert req.max_retries == 3

    def test_full_construction(self) -> None:
        req = UserRequest(
            query="refactor auth",
            project_root="/my-project",
            auto_mode=True,
            max_retries=5,
        )
        assert req.query == "refactor auth"
        assert req.project_root == "/my-project"
        assert req.auto_mode is True
        assert req.max_retries == 5


class TestTaskStatus:
    """验证 TaskStatus dataclass 字段类型和默认值。"""

    def test_fields(self) -> None:
        fields = {f.name: f for f in TaskStatus.__dataclass_fields__.values()}

        assert fields["task_id"].type is str
        assert fields["state"].type is TaskState
        assert fields["progress"].type is float
        # current_step is Optional[str]
        assert fields["errors"].type == list[str]

    def test_defaults(self) -> None:
        status = TaskStatus(task_id="t1", state=TaskState.RUNNING)
        assert status.progress == 0.0
        assert status.current_step is None
        assert status.errors == []

    def test_full_construction(self) -> None:
        status = TaskStatus(
            task_id="t1",
            state=TaskState.WAITING_REVIEW,
            progress=0.5,
            current_step="reviewing changes",
            errors=["error1"],
        )
        assert status.task_id == "t1"
        assert status.state == TaskState.WAITING_REVIEW
        assert status.progress == 0.5
        assert status.current_step == "reviewing changes"
        assert status.errors == ["error1"]


class TestTaskEvent:
    """验证 TaskEvent dataclass 字段。"""

    def test_fields(self) -> None:
        fields = {f.name: f for f in TaskEvent.__dataclass_fields__.values()}

        assert fields["type"].type is str
        assert fields["data"].type is dict
        assert fields["timestamp"].type is float

    def test_construction(self) -> None:
        now = time.time()
        event = TaskEvent(type="tool_call", data={"tool": "read_file"}, timestamp=now)
        assert event.type == "tool_call"
        assert event.data == {"tool": "read_file"}
        assert event.timestamp == now


class TestHumanDecision:
    """验证 HumanDecision dataclass 字段。"""

    def test_fields(self) -> None:
        fields = {
            f.name: f for f in HumanDecision.__dataclass_fields__.values()
        }

        assert fields["task_id"].type is str
        assert fields["decision"].type is str
        assert fields["modifications"].type == dict | None

    def test_defaults(self) -> None:
        d = HumanDecision(task_id="t1", decision="approve")
        assert d.modifications is None

    def test_full_construction(self) -> None:
        d = HumanDecision(
            task_id="t1",
            decision="modify",
            modifications={"step_id": 3, "new_action": "refactor"},
        )
        assert d.task_id == "t1"
        assert d.decision == "modify"
        assert d.modifications == {"step_id": 3, "new_action": "refactor"}


class TestTaskReport:
    """验证 TaskReport dataclass 字段。"""

    def test_fields(self) -> None:
        fields = {f.name: f for f in TaskReport.__dataclass_fields__.values()}

        assert fields["task_id"].type is str
        assert fields["plan"].type is list
        assert fields["changes"].type == list[dict]
        assert fields["validation_results"].type is list
        assert fields["duration"].type is float
        assert fields["token_usage"].type is int

    def test_construction(self) -> None:
        report = TaskReport(
            task_id="t1",
            plan=[{"step_id": 1}],
            changes=[{"file": "main.py"}],
            validation_results=[{"passed": True}],
            duration=12.5,
            token_usage=1500,
        )
        assert report.task_id == "t1"
        assert report.plan == [{"step_id": 1}]
        assert report.changes == [{"file": "main.py"}]
        assert report.duration == 12.5
        assert report.token_usage == 1500


# =============================================================================
# 测试：Mock 行为验证
# =============================================================================


class TestMockBehavior:
    """验证 Mock 实现的基本行为。"""

    @pytest.mark.asyncio
    async def test_mock_start_task_returns_str(self) -> None:
        gw = MockOrchestrationGateway()
        req = UserRequest(query="test", project_root="/p")
        task_id = await gw.start_task(req)
        assert isinstance(task_id, str)
        assert task_id == "mock-task-id"

    @pytest.mark.asyncio
    async def test_mock_get_task_status(self) -> None:
        gw = MockOrchestrationGateway()
        status = await gw.get_task_status("task-1")
        assert isinstance(status, TaskStatus)
        assert status.task_id == "task-1"
        assert status.state == TaskState.PENDING

    @pytest.mark.asyncio
    async def test_mock_stream_task(self) -> None:
        gw = MockOrchestrationGateway()
        events = []
        async for event in gw.stream_task("task-1"):
            events.append(event)
        assert len(events) == 2
        assert events[0].type == "thinking"
        assert events[1].type == "done"

    @pytest.mark.asyncio
    async def test_mock_submit_decision(self) -> None:
        gw = MockOrchestrationGateway()
        decision = HumanDecision(task_id="task-1", decision="approve")
        result = await gw.submit_decision("task-1", decision)
        assert result is None

    @pytest.mark.asyncio
    async def test_mock_cancel_task(self) -> None:
        gw = MockOrchestrationGateway()
        result = await gw.cancel_task("task-1")
        assert result is True

    @pytest.mark.asyncio
    async def test_mock_get_report(self) -> None:
        gw = MockOrchestrationGateway()
        report = await gw.get_report("task-1")
        assert isinstance(report, TaskReport)
        assert report.task_id == "task-1"
        assert report.duration == 1.0
        assert report.token_usage == 0
