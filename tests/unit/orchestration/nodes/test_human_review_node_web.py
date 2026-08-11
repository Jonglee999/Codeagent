"""HumanReviewNode Web 模式单元测试。

覆盖：Web 模式正常流程、超时 abort、Redis 异常处理、
CLI 模式不受影响、自动 approve 模式。
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from codeagent.orchestration.nodes.human_review_node import HumanReviewNode
from codeagent.orchestration.state import AgentState


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def base_state() -> AgentState:
    return AgentState(
        user_request="test request",
        project_root="/test/project",
        task_id="test-task-123",
        human_review_required=True,
        review_type="high_risk_plan",
    )


@pytest.fixture
def mock_redis():
    """Create a mock sync Redis client."""
    redis_mock = MagicMock()
    redis_mock.get.return_value = None  # default: no decision
    return redis_mock


# =============================================================================
# Web 模式 — 正常流程
# =============================================================================


@pytest.mark.asyncio
class TestWebModeNormal:
    """Web 模式正常决策流程。"""

    async def test_approve_decision(self, base_state, mock_redis):
        """收到 approve 决策。"""
        mock_redis.get.return_value = json.dumps({"decision": "approve"}).encode()

        with patch("redis.from_url", return_value=mock_redis):
            node = HumanReviewNode(redis_url="redis://localhost:6379")
            result = await node(base_state)

        assert result["human_decision"] == "approve"
        assert result["human_review_required"] is False

    async def test_reject_decision(self, base_state, mock_redis):
        """收到 reject 决策。"""
        mock_redis.get.return_value = json.dumps({"decision": "reject"}).encode()

        with patch("redis.from_url", return_value=mock_redis):
            node = HumanReviewNode(redis_url="redis://localhost:6379")
            result = await node(base_state)

        assert result["human_decision"] == "reject"

    async def test_modify_decision(self, base_state, mock_redis):
        """收到 modify 决策。"""
        mock_redis.get.return_value = json.dumps({"decision": "modify"}).encode()

        with patch("redis.from_url", return_value=mock_redis):
            node = HumanReviewNode(redis_url="redis://localhost:6379")
            result = await node(base_state)

        assert result["human_decision"] == "modify"

    async def test_abort_decision(self, base_state, mock_redis):
        """收到 abort 决策。"""
        mock_redis.get.return_value = json.dumps({"decision": "abort"}).encode()

        with patch("redis.from_url", return_value=mock_redis):
            node = HumanReviewNode(redis_url="redis://localhost:6379")
            result = await node(base_state)

        assert result["human_decision"] == "abort"


# =============================================================================
# Web 模式 — 行为验证
# =============================================================================


@pytest.mark.asyncio
class TestWebModeBehavior:
    """Web 模式行为验证。"""

    async def test_publishes_review_event(self, base_state, mock_redis):
        """发布 human_review_required 事件到 Redis。"""
        mock_redis.get.return_value = json.dumps({"decision": "approve"}).encode()

        with patch("redis.from_url", return_value=mock_redis):
            node = HumanReviewNode(redis_url="redis://localhost:6379")
            await node(base_state)

        mock_redis.publish.assert_called_once()
        channel, payload = mock_redis.publish.call_args[0]
        assert channel == "task:test-task-123:events"
        event_data = json.loads(payload)
        assert event_data["type"] == "human_review_required"
        assert event_data["review_type"] == "high_risk_plan"

    async def test_key_deleted_after_read(self, base_state, mock_redis):
        """读取决策后 key 被删除。"""
        mock_redis.get.return_value = json.dumps({"decision": "approve"}).encode()

        with patch("redis.from_url", return_value=mock_redis):
            node = HumanReviewNode(redis_url="redis://localhost:6379")
            await node(base_state)

        mock_redis.delete.assert_called_once_with("task:test-task-123:decision")

    async def test_timeout_returns_abort(self, base_state, mock_redis):
        """超时后返回 abort。"""
        mock_redis.get.return_value = None  # never returns a decision

        with patch("redis.from_url", return_value=mock_redis), \
             patch("asyncio.sleep", AsyncMock()):  # instant sleep
            node = HumanReviewNode(redis_url="redis://localhost:6379", review_timeout=0.01)
            result = await node(base_state)

        assert result["human_decision"] == "abort"

    async def test_custom_timeout(self, base_state, mock_redis):
        """自定义超时时间生效。"""
        node = HumanReviewNode(redis_url="redis://localhost:6379", review_timeout=600)
        assert node._timeout == 600


# =============================================================================
# Web 模式 — 异常处理
# =============================================================================


@pytest.mark.asyncio
class TestWebModeErrorHandling:
    """Web 模式异常处理。"""

    async def test_publish_error_does_not_block(self, base_state, mock_redis):
        """发布事件时 Redis 异常不阻断流程。"""
        mock_redis.publish.side_effect = ConnectionError("Redis unavailable")
        mock_redis.get.return_value = json.dumps({"decision": "approve"}).encode()

        with patch("redis.from_url", return_value=mock_redis):
            node = HumanReviewNode(redis_url="redis://localhost:6379")
            result = await node(base_state)

        # Should still get decision despite publish error
        assert result["human_decision"] == "approve"

    async def test_poll_error_continues_loop(self, base_state, mock_redis):
        """轮询时 Redis 异常继续循环而非崩溃。"""
        # First call raises, second returns decision
        mock_redis.get.side_effect = [
            ConnectionError("tmp error"),
            json.dumps({"decision": "approve"}).encode(),
        ]

        with patch("redis.from_url", return_value=mock_redis), \
             patch("asyncio.sleep", AsyncMock()):
            node = HumanReviewNode(redis_url="redis://localhost:6379", review_timeout=300)
            result = await node(base_state)

        assert result["human_decision"] == "approve"


# =============================================================================
# Web 模式 — Resume 路径
# =============================================================================


@pytest.mark.asyncio
class TestWebModeResume:
    """Web 模式下 resume 路径仍然优先。"""

    async def test_resume_path_still_works(self, base_state, mock_redis):
        """resume 路径在 Web 模式下仍然优先。"""
        base_state.human_decision = "approve"

        with patch("redis.from_url", return_value=mock_redis):
            node = HumanReviewNode(redis_url="redis://localhost:6379")
            result = await node(base_state)

        # Resume path should NOT touch Redis
        mock_redis.get.assert_not_called()
        mock_redis.publish.assert_not_called()
        assert result["human_decision"] == "approve"

    async def test_resume_path_no_callback(self, base_state):
        """无 callback 无 redis 时 resume 路径仍优先。"""
        base_state.human_decision = "approve"

        node = HumanReviewNode()  # no callback, no redis
        result = await node(base_state)

        assert result["human_decision"] == "approve"


# =============================================================================
# 向后兼容性
# =============================================================================


@pytest.mark.asyncio
class TestBackwardCompatibility:
    """CLI 模式和自动 approve 不受影响。"""

    async def test_cli_mode_with_callback(self, base_state):
        """CLI 模式（callback）不受影响。"""
        callback = AsyncMock(return_value="approve")
        node = HumanReviewNode(review_callback=callback)
        result = await node(base_state)

        assert result["human_decision"] == "approve"
        callback.assert_awaited_once()

    async def test_callback_exception_default_abort(self, base_state):
        """callback 抛异常仍默认 abort。"""
        callback = AsyncMock(side_effect=RuntimeError("fail"))
        node = HumanReviewNode(review_callback=callback)
        result = await node(base_state)

        assert result["human_decision"] == "abort"

    async def test_default_auto_approve(self, base_state):
        """无 callback 和 redis_url 时自动 approve。"""
        node = HumanReviewNode()
        result = await node(base_state)
        assert result["human_decision"] == "approve"

    async def test_redis_url_ignored_when_callback_given(self, base_state, mock_redis):
        """callback 存在时忽略 redis_url。"""
        callback = AsyncMock(return_value="approve")
        node = HumanReviewNode(
            review_callback=callback,
            redis_url="redis://localhost:6379",
        )

        with patch("redis.from_url", return_value=mock_redis):
            result = await node(base_state)

        assert result["human_decision"] == "approve"
        callback.assert_awaited_once()
        mock_redis.publish.assert_not_called()
