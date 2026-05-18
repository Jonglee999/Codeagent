"""OrchestrationGatewayImpl 单元测试。

使用 fakeredis 模拟 Redis，Mock Celery 任务提交。
覆盖：start_task/get_status/stream/submit_decision/cancel/get_report、
Redis 键格式、TTL 设置、异常处理。
"""

from __future__ import annotations

import json
import time
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest

from codeagent.gateway.orchestration_gateway import (
    HumanDecision,
    TaskEvent,
    TaskReport,
    TaskState,
    TaskStatus,
    UserRequest,
)
from codeagent.gateway.orchestration_gateway_impl import OrchestrationGatewayImpl


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_redis_client():
    """Create an AsyncMock Redis client for testing."""
    client = AsyncMock()
    return client


@pytest.fixture
def gateway(mock_redis_client):
    """Create OrchestrationGatewayImpl with mocked Redis and Celery."""
    gw = OrchestrationGatewayImpl(redis_url="redis://localhost:6379")
    # Mock Celery
    gw._celery = MagicMock()
    return gw


@pytest.fixture
def sample_request():
    return UserRequest(
        query="Add type annotations to main.py",
        project_root="/tmp/test_project",
        auto_mode=False,
        max_retries=3,
    )


@pytest.fixture
def sample_request_auto():
    return UserRequest(
        query="Run pytest",
        project_root="/tmp/test_project",
        auto_mode=True,
        max_retries=5,
    )


# =============================================================================
# 辅助函数：在 gateway 方法中 mock aioredis.from_url
# =============================================================================


def _patch_redis(gateway, mock_client):
    """Patch aioredis.from_url to return mock_client."""
    patcher = patch("codeagent.gateway.orchestration_gateway_impl.aioredis.from_url")
    mock_redis_factory = patcher.start()
    mock_redis_factory.return_value = mock_client
    return patcher


# =============================================================================
# 测试：start_task
# =============================================================================


class TestStartTask:
    """验证 start_task 方法的行为。"""

    @pytest.mark.asyncio
    async def test_start_task_returns_string(self, gateway, sample_request, mock_redis_client):
        """start_task 返回字符串类型的 task_id。"""
        patcher = _patch_redis(gateway, mock_redis_client)

        try:
            task_id = await gateway.start_task(sample_request)

            assert isinstance(task_id, str)
            assert len(task_id) > 0

            # 验证 Redis 写入
            status_key = f"task:{task_id}:status"
            mock_redis_client.setex.assert_called_once()
            args, _ = mock_redis_client.setex.call_args
            assert args[0] == status_key
            assert args[1] == 3600  # TTL

            # 验证 Celery 任务提交
            gateway._celery.send_task.assert_called_once()
        finally:
            patcher.stop()

    @pytest.mark.asyncio
    async def test_start_task_initial_status_pending(self, gateway, sample_request, mock_redis_client):
        """初始状态为 PENDING。"""
        patcher = _patch_redis(gateway, mock_redis_client)

        try:
            await gateway.start_task(sample_request)

            # 验证写入的状态数据 — setex 签名为 (key, ttl, value)
            args, _ = mock_redis_client.setex.call_args
            status_data = json.loads(args[2])
            assert status_data["state"] == "pending"
            assert status_data["progress"] == 0.0
            assert status_data["current_step"] is None
            assert status_data["errors"] == []
        finally:
            patcher.stop()

    @pytest.mark.asyncio
    async def test_start_task_idempotent_task_id(self, gateway, sample_request, mock_redis_client):
        """每次 start_task 返回不同的 task_id。"""
        patcher = _patch_redis(gateway, mock_redis_client)

        try:
            task_id_1 = await gateway.start_task(sample_request)
            task_id_2 = await gateway.start_task(sample_request)

            assert task_id_1 != task_id_2
        finally:
            patcher.stop()

    @pytest.mark.asyncio
    async def test_start_task_celery_send_task_called(self, gateway, sample_request, mock_redis_client):
        """验证 Celery send_task 被调用。"""
        patcher = _patch_redis(gateway, mock_redis_client)

        try:
            await gateway.start_task(sample_request)

            # send_task 应被调用一次
            gateway._celery.send_task.assert_called_once()
        finally:
            patcher.stop()

    @pytest.mark.asyncio
    async def test_start_task_celery_parameters(self, gateway, sample_request, mock_redis_client):
        """验证传递给 Celery 的参数正确。"""
        patcher = _patch_redis(gateway, mock_redis_client)

        try:
            task_id = await gateway.start_task(sample_request)

            # 验证 send_task 参数：send_task(name, args=[task_id, request_dict])
            call_args, call_kwargs = gateway._celery.send_task.call_args
            assert call_args[0] == "codeagent.run_agent_task"
            sent_args = call_kwargs.get("args", call_args[1] if len(call_args) > 1 else [])
            assert sent_args[0] == task_id
            assert sent_args[1]["query"] == sample_request.query
            assert sent_args[1]["project_root"] == sample_request.project_root
            assert sent_args[1]["auto_mode"] == sample_request.auto_mode
            assert sent_args[1]["max_retries"] == sample_request.max_retries
        finally:
            patcher.stop()


# =============================================================================
# 测试：get_task_status
# =============================================================================


class TestGetTaskStatus:
    """验证 get_task_status 方法。"""

    @pytest.mark.asyncio
    async def test_get_task_status_pending(self, gateway, mock_redis_client):
        """get_task_status 返回 PENDING 状态。"""
        patcher = _patch_redis(gateway, mock_redis_client)
        mock_redis_client.get.return_value = json.dumps({
            "task_id": "test-id",
            "state": "pending",
            "progress": 0.0,
            "current_step": None,
            "errors": [],
        }, ensure_ascii=False)

        try:
            status = await gateway.get_task_status("test-id")

            assert isinstance(status, TaskStatus)
            assert status.task_id == "test-id"
            assert status.state == TaskState.PENDING
            assert status.progress == 0.0
            assert status.current_step is None
            assert status.errors == []
        finally:
            patcher.stop()

    @pytest.mark.asyncio
    async def test_get_task_status_running(self, gateway, mock_redis_client):
        """get_task_status 返回 RUNNING 状态。"""
        patcher = _patch_redis(gateway, mock_redis_client)
        mock_redis_client.get.return_value = json.dumps({
            "task_id": "test-id",
            "state": "running",
            "progress": 0.5,
            "current_step": "Executing planning",
            "errors": [],
        }, ensure_ascii=False)

        try:
            status = await gateway.get_task_status("test-id")

            assert status.state == TaskState.RUNNING
            assert status.progress == 0.5
            assert status.current_step == "Executing planning"
        finally:
            patcher.stop()

    @pytest.mark.asyncio
    async def test_get_task_status_not_found(self, gateway, mock_redis_client):
        """不存在的 task_id 抛出 KeyError。"""
        patcher = _patch_redis(gateway, mock_redis_client)
        mock_redis_client.get.return_value = None

        try:
            with pytest.raises(KeyError, match="Task not found"):
                await gateway.get_task_status("nonexistent-id")
        finally:
            patcher.stop()

    @pytest.mark.asyncio
    async def test_get_task_status_completed(self, gateway, mock_redis_client):
        """get_task_status 返回 COMPLETED 状态。"""
        patcher = _patch_redis(gateway, mock_redis_client)
        mock_redis_client.get.return_value = json.dumps({
            "task_id": "test-id",
            "state": "completed",
            "progress": 1.0,
            "current_step": "Done",
            "errors": [],
        }, ensure_ascii=False)

        try:
            status = await gateway.get_task_status("test-id")

            assert status.state == TaskState.COMPLETED
            assert status.progress == 1.0
        finally:
            patcher.stop()


# =============================================================================
# 测试：submit_decision
# =============================================================================


class TestSubmitDecision:
    """验证 submit_decision 方法。"""

    @pytest.mark.asyncio
    async def test_submit_decision_approve(self, gateway, mock_redis_client):
        """提交 approve 决策写入 Redis。"""
        patcher = _patch_redis(gateway, mock_redis_client)

        try:
            decision = HumanDecision(task_id="task-1", decision="approve")
            await gateway.submit_decision("task-1", decision)

            decision_key = "task:task-1:decision"
            mock_redis_client.setex.assert_called_once()
            args, _ = mock_redis_client.setex.call_args
            assert args[0] == decision_key
            assert args[1] == 600  # TTL = 600s

            stored = json.loads(args[2])
            assert stored["decision"] == "approve"
            assert stored["task_id"] == "task-1"
        finally:
            patcher.stop()

    @pytest.mark.asyncio
    async def test_submit_decision_with_modifications(self, gateway, mock_redis_client):
        """提交 modify 决策含 modifications。"""
        patcher = _patch_redis(gateway, mock_redis_client)

        try:
            decision = HumanDecision(
                task_id="task-1",
                decision="modify",
                modifications={"step_id": 3, "feedback": "use a different approach"},
            )
            await gateway.submit_decision("task-1", decision)

            args, _ = mock_redis_client.setex.call_args
            stored = json.loads(args[2])
            assert stored["decision"] == "modify"
            assert stored["modifications"] == {"step_id": 3, "feedback": "use a different approach"}
        finally:
            patcher.stop()

    @pytest.mark.asyncio
    async def test_submit_decision_reject(self, gateway, mock_redis_client):
        """提交 reject 决策。"""
        patcher = _patch_redis(gateway, mock_redis_client)

        try:
            decision = HumanDecision(task_id="task-1", decision="reject")
            await gateway.submit_decision("task-1", decision)

            args, _ = mock_redis_client.setex.call_args
            stored = json.loads(args[2])
            assert stored["decision"] == "reject"
        finally:
            patcher.stop()


# =============================================================================
# 测试：cancel_task
# =============================================================================


class TestCancelTask:
    """验证 cancel_task 方法。"""

    @pytest.mark.asyncio
    async def test_cancel_task_updates_status(self, gateway, mock_redis_client):
        """cancel_task 更新状态为 CANCELLED。"""
        patcher = _patch_redis(gateway, mock_redis_client)
        mock_redis_client.get.return_value = json.dumps({
            "task_id": "task-1",
            "state": "running",
            "progress": 0.3,
            "current_step": "executing",
            "errors": [],
        }, ensure_ascii=False)

        try:
            result = await gateway.cancel_task("task-1")

            assert result is True

            # 验证设置 cancel 标志
            cancel_key = "task:task-1:cancel"
            assert any(
                call_args[0] == cancel_key
                for call_args, _ in mock_redis_client.setex.call_args_list
            )

            # 验证最终状态为 CANCELLED
            found_cancelled = False
            for call_args, _ in mock_redis_client.setex.call_args_list:
                if "task:task-1:status" in call_args[0]:
                    updated = json.loads(call_args[2])
                    assert updated["state"] == "cancelled"
                    found_cancelled = True
                    break
            assert found_cancelled
        finally:
            patcher.stop()

    @pytest.mark.asyncio
    async def test_cancel_task_not_found(self, gateway, mock_redis_client):
        """不存在的任务 cancel 返回 False。"""
        patcher = _patch_redis(gateway, mock_redis_client)
        mock_redis_client.get.return_value = None

        try:
            result = await gateway.cancel_task("nonexistent")
            assert result is False
        finally:
            patcher.stop()

    @pytest.mark.asyncio
    async def test_cancel_task_revoke_called(self, gateway, mock_redis_client):
        """cancel_task 调用 Celery revoke。"""
        patcher = _patch_redis(gateway, mock_redis_client)
        mock_redis_client.get.return_value = json.dumps({
            "task_id": "task-1", "state": "running",
        }, ensure_ascii=False)

        try:
            await gateway.cancel_task("task-1")

            gateway._celery.control.revoke.assert_called_once_with("task-1", terminate=True)
        finally:
            patcher.stop()

    @pytest.mark.asyncio
    async def test_cancel_task_cancel_key_ttl(self, gateway, mock_redis_client):
        """cancel 标志 TTL 为 60s。"""
        patcher = _patch_redis(gateway, mock_redis_client)
        mock_redis_client.get.return_value = json.dumps({
            "task_id": "task-1", "state": "running",
        }, ensure_ascii=False)

        try:
            await gateway.cancel_task("task-1")

            # 验证 cancel 键的 TTL 为 60
            found_cancel_ttl = False
            for call_args, _ in mock_redis_client.setex.call_args_list:
                if "task:task-1:cancel" in call_args[0]:
                    assert call_args[1] == 60
                    found_cancel_ttl = True
                    break
            assert found_cancel_ttl
        finally:
            patcher.stop()


# =============================================================================
# 测试：get_report
# =============================================================================


class TestGetReport:
    """验证 get_report 方法。"""

    @pytest.mark.asyncio
    async def test_get_report_success(self, gateway, mock_redis_client):
        """get_report 返回完整报告。"""
        patcher = _patch_redis(gateway, mock_redis_client)
        mock_redis_client.get.return_value = json.dumps({
            "task_id": "task-1",
            "plan": [{"step_id": 1, "action": "create"}],
            "changes": [{"file": "main.py", "action": "modify"}],
            "validation_results": [{"passed": True}],
            "duration": 12.5,
            "token_usage": 1500,
        }, ensure_ascii=False)

        try:
            report = await gateway.get_report("task-1")

            assert isinstance(report, TaskReport)
            assert report.task_id == "task-1"
            assert len(report.plan) == 1
            assert len(report.changes) == 1
            assert report.duration == 12.5
            assert report.token_usage == 1500
        finally:
            patcher.stop()

    @pytest.mark.asyncio
    async def test_get_report_not_found(self, gateway, mock_redis_client):
        """不存在的报告抛出 KeyError。"""
        patcher = _patch_redis(gateway, mock_redis_client)
        mock_redis_client.get.return_value = None

        try:
            with pytest.raises(KeyError, match="Report not found"):
                await gateway.get_report("nonexistent")
        finally:
            patcher.stop()

    @pytest.mark.asyncio
    async def test_get_report_key_format(self, gateway, mock_redis_client):
        """验证 report 键格式。"""
        patcher = _patch_redis(gateway, mock_redis_client)
        mock_redis_client.get.return_value = json.dumps({
            "task_id": "task-1",
            "plan": [], "changes": [],
            "validation_results": [], "duration": 0, "token_usage": 0,
        }, ensure_ascii=False)

        try:
            await gateway.get_report("task-1")

            mock_redis_client.get.assert_called_once_with("task:task-1:report")
        finally:
            patcher.stop()


# =============================================================================
# 测试：stream_task
# =============================================================================


class TestStreamTask:
    """验证 stream_task 方法。"""

    def _setup_pubsub_mock(self, mock_redis_client, messages):
        """Set up pubsub mock on the redis client.

        Uses MagicMock (not AsyncMock) for pubsub() since it's a sync factory method.
        """
        from unittest.mock import MagicMock

        mock_pubsub = MagicMock()
        mock_pubsub.subscribe = AsyncMock()
        mock_pubsub.unsubscribe = AsyncMock()
        mock_pubsub.get_message = AsyncMock(side_effect=messages)
        # pubsub() is a sync method on the Redis client
        mock_redis_client.pubsub = MagicMock(return_value=mock_pubsub)
        return mock_pubsub

    @pytest.mark.asyncio
    async def test_stream_task_receives_events(self, gateway, mock_redis_client):
        """stream_task 接收并 yield 事件。"""
        messages = [
            {"data": json.dumps({
                "type": "node_start", "data": {"node": "planning"},
                "timestamp": time.time(),
            }, ensure_ascii=False)},
            {"data": json.dumps({
                "type": "tool_call", "data": {"tool": "read_file"},
                "timestamp": time.time(),
            }, ensure_ascii=False)},
            {"data": json.dumps({
                "type": "task_complete", "data": {"status": "success"},
                "timestamp": time.time(),
            }, ensure_ascii=False)},
        ]
        # After the last message, keep returning None to end the generator
        async def msg_generator():
            for m in messages:
                yield m
            # Keep returning None after last message
            while True:
                yield None

        gen = msg_generator()

        async def get_message_side_effect(*args, **kwargs):
            return await gen.__anext__()

        mock_pubsub = self._setup_pubsub_mock(mock_redis_client, get_message_side_effect)
        patcher = _patch_redis(gateway, mock_redis_client)

        try:
            events = []
            async for event in gateway.stream_task("task-1"):
                events.append(event)

            assert len(events) == 3
            assert events[0].type == "node_start"
            assert events[1].type == "tool_call"
            assert events[2].type == "task_complete"
        finally:
            patcher.stop()

    @pytest.mark.asyncio
    async def test_stream_task_subscribes_correct_channel(self, gateway, mock_redis_client):
        """stream_task 订阅正确的频道。"""
        async def single_event():
            yield {"data": json.dumps({
                "type": "task_complete", "data": {},
                "timestamp": time.time(),
            }, ensure_ascii=False)}

        gen = single_event()

        async def get_message_side_effect(*args, **kwargs):
            return await gen.__anext__()

        mock_pubsub = self._setup_pubsub_mock(mock_redis_client, get_message_side_effect)
        patcher = _patch_redis(gateway, mock_redis_client)

        try:
            async for _ in gateway.stream_task("task-1"):
                break

            mock_pubsub.subscribe.assert_called_once_with("task:task-1:events")
        finally:
            patcher.stop()

    @pytest.mark.asyncio
    async def test_stream_task_unsubscribes_on_complete(self, gateway, mock_redis_client):
        """stream_task 完成后取消订阅。"""
        async def single_event():
            yield {"data": json.dumps({
                "type": "task_error", "data": {"error": "test"},
                "timestamp": time.time(),
            }, ensure_ascii=False)}

        gen = single_event()

        async def get_message_side_effect(*args, **kwargs):
            return await gen.__anext__()

        mock_pubsub = self._setup_pubsub_mock(mock_redis_client, get_message_side_effect)
        patcher = _patch_redis(gateway, mock_redis_client)

        try:
            # Collect all events from the stream (will stop at task_error automatically)
            events = []
            async for event in gateway.stream_task("task-1"):
                events.append(event)

            # Verify we got the task_error event
            assert len(events) >= 1
            assert events[-1].type == "task_error"

            # Verify unsubscribe was called during generator cleanup
            mock_pubsub.unsubscribe.assert_called_once_with("task:task-1:events")
            mock_redis_client.aclose.assert_called_once()
        finally:
            patcher.stop()


# =============================================================================
# 测试：Redis 键格式和 TTL
# =============================================================================


class TestRedisKeyFormat:
    """验证 Redis 键格式和 TTL。"""

    @pytest.mark.asyncio
    async def test_status_key_format(self, gateway, sample_request, mock_redis_client):
        """状态键格式为 task:{task_id}:status。"""
        patcher = _patch_redis(gateway, mock_redis_client)

        try:
            task_id = await gateway.start_task(sample_request)
            status_key = f"task:{task_id}:status"
            args, _ = mock_redis_client.setex.call_args
            assert args[0] == status_key
        finally:
            patcher.stop()

    @pytest.mark.asyncio
    async def test_decision_key_format(self, gateway, mock_redis_client):
        """决策键格式为 task:{task_id}:decision。"""
        patcher = _patch_redis(gateway, mock_redis_client)

        try:
            decision = HumanDecision(task_id="task-1", decision="approve")
            await gateway.submit_decision("task-1", decision)

            args, _ = mock_redis_client.setex.call_args
            assert args[0] == "task:task-1:decision"
        finally:
            patcher.stop()

    @pytest.mark.asyncio
    async def test_status_ttl_is_3600(self, gateway, sample_request, mock_redis_client):
        """状态 TTL 为 3600s。"""
        patcher = _patch_redis(gateway, mock_redis_client)

        try:
            await gateway.start_task(sample_request)
            args, _ = mock_redis_client.setex.call_args
            assert args[1] == 3600
        finally:
            patcher.stop()


# =============================================================================
# 测试：辅助方法（_publish_event, _set_task_status）
# =============================================================================


class TestInternalMethods:
    """验证内部辅助方法。"""

    @pytest.mark.asyncio
    async def test_publish_event(self, gateway, mock_redis_client):
        """_publish_event 发布到正确的频道。"""
        patcher = _patch_redis(gateway, mock_redis_client)

        try:
            event = TaskEvent(type="test_event", data={"key": "value"}, timestamp=123.0)
            await gateway._publish_event("task-1", event)

            mock_redis_client.publish.assert_called_once()
            args, _ = mock_redis_client.publish.call_args
            assert args[0] == "task:task-1:events"
            published = json.loads(args[1])
            assert published["type"] == "test_event"
            assert published["data"] == {"key": "value"}
        finally:
            patcher.stop()

    @pytest.mark.asyncio
    async def test_set_task_status(self, gateway, mock_redis_client):
        """_set_task_status 写入正确的状态数据。"""
        patcher = _patch_redis(gateway, mock_redis_client)

        try:
            status = TaskStatus(
                task_id="task-1",
                state=TaskState.RUNNING,
                progress=0.5,
                current_step="planning",
                errors=[],
            )
            await gateway._set_task_status("task-1", status)

            mock_redis_client.setex.assert_called_once()
            args, _ = mock_redis_client.setex.call_args
            assert args[0] == "task:task-1:status"
            stored = json.loads(args[2])
            assert stored["state"] == "running"
            assert stored["progress"] == 0.5
            assert stored["current_step"] == "planning"
        finally:
            patcher.stop()


# =============================================================================
# 测试：异常处理
# =============================================================================


class TestErrorHandling:
    """验证异常处理。"""

    @pytest.mark.asyncio
    async def test_get_task_status_redis_connection_error(self, gateway, mock_redis_client):
        """Redis 连接错误时传递异常。"""
        patcher = _patch_redis(gateway, mock_redis_client)
        mock_redis_client.get.side_effect = ConnectionError("Redis unavailable")

        try:
            with pytest.raises(ConnectionError):
                await gateway.get_task_status("task-1")
        finally:
            patcher.stop()

    @pytest.mark.asyncio
    async def test_cancel_task_redis_error_still_sets_cancel_flag(self, gateway, mock_redis_client):
        """即使 Celery revoke 失败，cancel 标志仍写入。"""
        patcher = _patch_redis(gateway, mock_redis_client)
        mock_redis_client.get.return_value = json.dumps({
            "task_id": "task-1", "state": "running",
        }, ensure_ascii=False)
        gateway._celery.control.revoke.side_effect = Exception("Celery error")

        try:
            # 不应抛出异常
            result = await gateway.cancel_task("task-1")
            assert result is True

            # cancel 标志仍然写入
            assert any(
                "task:task-1:cancel" in str(call_args)
                for call_args, _ in mock_redis_client.setex.call_args_list
            )
        finally:
            patcher.stop()


# =============================================================================
# 测试：config 函数
# =============================================================================


class TestConfigFunctions:
    """验证新增的配置函数。"""

    def test_get_redis_url_default(self):
        """默认 REDIS_URL 为 redis://localhost:6379。"""
        from codeagent.config import get_redis_url
        assert get_redis_url() == "redis://127.0.0.1:6379"

    def test_get_redis_url_from_env(self, monkeypatch):
        """环境变量设置 REDIS_URL 后返回正确值。"""
        from codeagent.config import get_redis_url
        monkeypatch.setenv("REDIS_URL", "redis://myredis:6380")
        assert get_redis_url() == "redis://myredis:6380"

    def test_get_celery_task_timeout_default(self):
        """默认 CELERY_TASK_TIMEOUT 为 600。"""
        from codeagent.config import get_celery_task_timeout
        assert get_celery_task_timeout() == 600

    def test_get_celery_task_timeout_from_env(self, monkeypatch):
        """环境变量设置 CELERY_TASK_TIMEOUT 后返回正确值。"""
        from codeagent.config import get_celery_task_timeout
        monkeypatch.setenv("CELERY_TASK_TIMEOUT", "1200")
        assert get_celery_task_timeout() == 1200
