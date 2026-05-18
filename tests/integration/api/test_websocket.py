"""WebSocket 流式输出集成测试。

Mock Redis Pub/Sub 来测试事件接收、心跳、连接关闭、清理等场景。
覆盖：≥10 个测试用例。
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def redis_mocks():
    """Patch aioredis.from_url, return (mock_client, mock_pubsub)."""
    mock_pubsub = AsyncMock()
    mock_pubsub.subscribe = AsyncMock()
    mock_pubsub.unsubscribe = AsyncMock()

    mock_redis_client = AsyncMock()
    mock_redis_client.pubsub = MagicMock(return_value=mock_pubsub)
    mock_redis_client.aclose = AsyncMock()

    with patch(
        "codeagent.interaction.api.websocket.aioredis.from_url"
    ) as mock_from_url:
        mock_from_url.return_value = mock_redis_client
        yield mock_redis_client, mock_pubsub


@pytest.fixture
def app():
    from codeagent.interaction.api.websocket import ws_router

    app = FastAPI()
    app.include_router(ws_router)
    return app


@pytest.fixture
def client(app):
    return TestClient(app)


# =============================================================================
# 辅助：让 ping 测试快速触发（不等待 25s 实际间隔）
# =============================================================================


@pytest.fixture
def fast_ping(mocker):
    """Patch ping/poll intervals to near-zero for fast ping tests."""
    mocker.patch("codeagent.interaction.api.websocket._PING_INTERVAL", 0.0)
    mocker.patch("codeagent.interaction.api.websocket._POLL_INTERVAL", 0.0)


# =============================================================================
# Tests
# =============================================================================


class TestWebSocketStream:
    """WebSocket 事件流核心功能测试。"""

    def test_receive_node_start_event(self, client, redis_mocks):
        """收到 node_start 事件并转发。"""
        _, mock_pubsub = redis_mocks

        calls = [
            {"data": json.dumps({"type": "node_start", "node": "planning"})},
            {"data": json.dumps({"type": "task_complete", "status": "success"})},
        ]

        async def get_message_seq(*args: object, **kwargs: object):
            if calls:
                return calls.pop(0)
            return None

        mock_pubsub.get_message = AsyncMock(side_effect=get_message_seq)

        with client.websocket_connect(
            "/api/v1/tasks/test-123/stream"
        ) as ws:
            event = ws.receive_json()
            assert event["type"] == "node_start"
            assert event["node"] == "planning"

            event2 = ws.receive_json()
            assert event2["type"] == "task_complete"

    def test_receive_multiple_event_types(self, client, redis_mocks):
        """多种事件类型按顺序到达。"""
        _, mock_pubsub = redis_mocks

        messages = [
            {"data": json.dumps({"type": "node_start", "node": "context"})},
            {"data": json.dumps({"type": "node_start", "node": "planning"})},
            {
                "data": json.dumps(
                    {"type": "tool_call", "tool": "read_file", "params": {"path": "main.py"}}
                )
            },
            {
                "data": json.dumps(
                    {"type": "tool_result", "tool": "read_file", "success": True, "summary": "ok"}
                )
            },
            {
                "data": json.dumps(
                    {"type": "validation_result", "layer": "syntax", "passed": True, "errors": []}
                )
            },
            {"data": json.dumps({"type": "task_complete", "status": "success"})},
        ]

        async def get_message_seq(*args: object, **kwargs: object):
            if messages:
                return messages.pop(0)
            return None

        mock_pubsub.get_message = AsyncMock(side_effect=get_message_seq)

        with client.websocket_connect(
            "/api/v1/tasks/test-123/stream"
        ) as ws:
            events = [ws.receive_json() for _ in range(6)]

        assert len(events) == 6
        assert events[0]["type"] == "node_start"
        assert events[0]["node"] == "context"
        assert events[1]["node"] == "planning"
        assert events[2]["type"] == "tool_call"
        assert events[2]["tool"] == "read_file"
        assert events[3]["type"] == "tool_result"
        assert events[3]["success"] is True
        assert events[4]["type"] == "validation_result"
        assert events[4]["passed"] is True
        assert events[5]["type"] == "task_complete"

    def test_timeout_sends_ping(self, client, redis_mocks, fast_ping):
        """接收消息超时后发送 ping 心跳。"""
        _, mock_pubsub = redis_mocks

        # Return None once (triggers ping check), then task_complete
        calls = [None, {"data": json.dumps({"type": "task_complete", "status": "success"})}]

        async def get_message_seq(*args: object, **kwargs: object):
            if calls:
                return calls.pop(0)
            return None

        mock_pubsub.get_message = AsyncMock(side_effect=get_message_seq)

        with client.websocket_connect(
            "/api/v1/tasks/test-123/stream"
        ) as ws:
            event1 = ws.receive_json()
            assert event1["type"] == "ping"

            event2 = ws.receive_json()
            assert event2["type"] == "task_complete"

    def test_task_complete_closes_stream(self, client, redis_mocks):
        """收到 task_complete 后自动关闭流。"""
        _, mock_pubsub = redis_mocks

        async def get_message_seq(*args: object, **kwargs: object):
            return {"data": json.dumps({"type": "task_complete", "status": "success"})}

        mock_pubsub.get_message = AsyncMock(side_effect=get_message_seq)

        with client.websocket_connect(
            "/api/v1/tasks/test-123/stream"
        ) as ws:
            event = ws.receive_json()
            assert event["type"] == "task_complete"

        # Handler 应已退出并清理
        mock_pubsub.unsubscribe.assert_awaited_once()

    def test_task_error_closes_stream(self, client, redis_mocks):
        """收到 task_error 后自动关闭流。"""
        _, mock_pubsub = redis_mocks

        async def get_message_seq(*args: object, **kwargs: object):
            return {
                "data": json.dumps(
                    {"type": "task_error", "error": "LLM API timeout", "data": {}}
                )
            }

        mock_pubsub.get_message = AsyncMock(side_effect=get_message_seq)

        with client.websocket_connect(
            "/api/v1/tasks/test-123/stream"
        ) as ws:
            event = ws.receive_json()
            assert event["type"] == "task_error"
            assert event["error"] == "LLM API timeout"

        mock_pubsub.unsubscribe.assert_awaited_once()

    def test_subscribes_correct_channel(self, client, redis_mocks):
        """订阅正确的 Redis 频道。"""
        _, mock_pubsub = redis_mocks

        async def get_message_seq(*args: object, **kwargs: object):
            return {"data": json.dumps({"type": "task_complete", "status": "success"})}

        mock_pubsub.get_message = AsyncMock(side_effect=get_message_seq)

        with client.websocket_connect(
            "/api/v1/tasks/my-task-42/stream"
        ) as ws:
            ws.receive_json()

        mock_pubsub.subscribe.assert_awaited_once_with("task:my-task-42:events")

    def test_passes_ignore_subscribe_messages(self, client, redis_mocks):
        """get_message 调用时传递 ignore_subscribe_messages=True。"""
        _, mock_pubsub = redis_mocks

        async def get_message_seq(*args: object, **kwargs: object):
            return {"data": json.dumps({"type": "task_complete", "status": "success"})}

        mock_pubsub.get_message = AsyncMock(side_effect=get_message_seq)

        with client.websocket_connect(
            "/api/v1/tasks/test-123/stream"
        ) as ws:
            ws.receive_json()

        # 至少有一次调用携带了 ignore_subscribe_messages=True
        found = any(
            kwargs.get("ignore_subscribe_messages") is True
            for _args, kwargs in mock_pubsub.get_message.call_args_list
        )
        assert found, "Expected ignore_subscribe_messages=True in get_message calls"


# =============================================================================
# 连接生命周期
# =============================================================================


class TestWebSocketLifecycle:
    """WebSocket 连接生命周期测试。"""

    def test_disconnect_cleans_up_redis(self, client, redis_mocks, fast_ping):
        """断开连接后清理 Redis 资源。"""
        mock_redis_client, mock_pubsub = redis_mocks

        # 返回 None 让 handler 进入 None 分支（触发 ping 检查）
        async def get_message_seq(*args: object, **kwargs: object):
            return None

        mock_pubsub.get_message = AsyncMock(side_effect=get_message_seq)

        with client.websocket_connect(
            "/api/v1/tasks/test-123/stream"
        ) as ws:
            # 等待至少一次 ping 以确保 handler 正在运行
            ws.receive_json()

        mock_pubsub.unsubscribe.assert_awaited_once()
        mock_redis_client.aclose.assert_awaited_once()

    def test_disconnect_without_events(self, client, redis_mocks, fast_ping):
        """连接后立即断开也能正常清理。"""
        mock_redis_client, mock_pubsub = redis_mocks

        # 返回 None 让 handler 进入 None 分支
        async def get_message_seq(*args: object, **kwargs: object):
            return None

        mock_pubsub.get_message = AsyncMock(side_effect=get_message_seq)

        with client.websocket_connect(
            "/api/v1/tasks/test-123/stream"
        ) as ws:
            ws.receive_json()  # blocks until first ping

        mock_pubsub.unsubscribe.assert_awaited_once()
        mock_redis_client.aclose.assert_awaited_once()

    def test_event_timestamp_added_if_missing(self, client, redis_mocks):
        """缺少 timestamp 的事件自动添加。"""
        _, mock_pubsub = redis_mocks

        messages = [
            {"data": json.dumps({"type": "tool_call", "tool": "write_file"})},  # no timestamp
            {"data": json.dumps({"type": "task_complete", "status": "success"})},
        ]

        async def get_message_seq(*args: object, **kwargs: object):
            if messages:
                return messages.pop(0)
            return None

        mock_pubsub.get_message = AsyncMock(side_effect=get_message_seq)

        with client.websocket_connect(
            "/api/v1/tasks/test-123/stream"
        ) as ws:
            event = ws.receive_json()
            assert event["type"] == "tool_call"
            assert "timestamp" in event  # auto-added

            event2 = ws.receive_json()
            assert event2["type"] == "task_complete"

    def test_keepalive_then_events(self, client, redis_mocks, fast_ping):
        """ping 心跳后继续正常接收事件。"""
        _, mock_pubsub = redis_mocks

        calls = [None, {"data": json.dumps({"type": "task_complete", "status": "success"})}]

        async def get_message_seq(*args: object, **kwargs: object):
            if calls:
                return calls.pop(0)
            return None

        mock_pubsub.get_message = AsyncMock(side_effect=get_message_seq)

        with client.websocket_connect(
            "/api/v1/tasks/test-123/stream"
        ) as ws:
            event1 = ws.receive_json()
            assert event1["type"] == "ping"

            event2 = ws.receive_json()
            assert event2["type"] == "task_complete"
            assert event2["status"] == "success"
