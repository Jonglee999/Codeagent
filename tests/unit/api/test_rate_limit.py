"""速率限制单元测试。

覆盖：正常请求放行、超限拒绝、豁免端点、Redis 故障降级。
使用 fakeredis 模拟 Redis，无需外部 Redis 服务。
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _patch_redis():
    """全局替换 RateLimiter 的 Redis 连接为模拟 Redis 客户端。

    使用 fakeredis 作为后端存储，包装为兼容 async redis 的接口。
    """
    import fakeredis
    from unittest.mock import AsyncMock

    fake = fakeredis.FakeRedis()

    class _AsyncFakeRedis:
        """Wrap fakeredis sync methods into async-compatible interface."""

        def __init__(self, backend):
            self._backend = backend
            self.aclose = AsyncMock()

        async def incr(self, key):
            return self._backend.incr(key)

        async def expire(self, key, ttl):
            return self._backend.expire(key, ttl)

    with patch("codeagent.interaction.api.rate_limit.aioredis.from_url") as mock_from_url:
        mock_from_url.return_value = _AsyncFakeRedis(fake)
        yield


@pytest.mark.asyncio
class TestRateLimiter:
    """速率限制器基础功能测试。"""

    async def test_normal_request_passes(self):
        """正常请求不应被限速。"""
        from codeagent.interaction.api.rate_limit import RateLimiter

        limiter = RateLimiter()
        request = _make_request(path="/api/v1/tasks")
        await limiter.check_rate_limit(request)

    async def test_rate_limit_exceeded_returns_429(self):
        """超限请求应返回 429 Too Many Requests。"""
        from fastapi import HTTPException, status

        from codeagent.interaction.api.rate_limit import RateLimiter

        with patch.dict(os.environ, {"RATE_LIMIT_RPM": "5"}, clear=True):
            limiter = RateLimiter()
            request = _make_request(path="/api/v1/tasks")

            # 前 5 次应通过
            for _ in range(5):
                await limiter.check_rate_limit(request)

            # 第 6 次应触发限速
            with pytest.raises(HTTPException) as exc:
                await limiter.check_rate_limit(request)
            assert exc.value.status_code == status.HTTP_429_TOO_MANY_REQUESTS
            assert "Rate limit exceeded" in exc.value.detail
            assert "Retry-After" in exc.value.headers

    async def test_health_endpoint_exempt(self):
        """/health 端点应豁免限速。"""
        from codeagent.interaction.api.rate_limit import RateLimiter

        with patch.dict(os.environ, {"RATE_LIMIT_RPM": "1"}, clear=True):
            limiter = RateLimiter()
            health_request = _make_request(path="/health")

            for _ in range(10):
                await limiter.check_rate_limit(health_request)

    async def test_metrics_endpoint_exempt(self):
        """/metrics 端点应豁免限速。"""
        from codeagent.interaction.api.rate_limit import RateLimiter

        with patch.dict(os.environ, {"RATE_LIMIT_RPM": "1"}, clear=True):
            limiter = RateLimiter()
            metrics_request = _make_request(path="/metrics")

            for _ in range(10):
                await limiter.check_rate_limit(metrics_request)

    async def test_redis_failure_degrades_gracefully(self):
        """Redis 连接失败时应降级放行，不阻塞业务。"""
        from codeagent.interaction.api.rate_limit import RateLimiter

        with patch(
            "codeagent.interaction.api.rate_limit.aioredis.from_url",
            side_effect=ConnectionError("Redis unreachable"),
        ):
            limiter = RateLimiter()
            request = _make_request(path="/api/v1/tasks")
            await limiter.check_rate_limit(request)

    async def test_different_keys_independent_counters(self):
        """不同 API Key 应有独立的计数窗口。"""
        from fastapi import HTTPException

        from codeagent.interaction.api.rate_limit import RateLimiter

        with patch.dict(os.environ, {"RATE_LIMIT_RPM": "3"}, clear=True):
            limiter = RateLimiter()

            req_a = _make_request(path="/api/v1/tasks", auth_header="Bearer key-a")
            req_b = _make_request(path="/api/v1/tasks", auth_header="Bearer key-b")

            # key-a 用完配额
            for _ in range(3):
                await limiter.check_rate_limit(req_a)

            # key-a 超限
            with pytest.raises(HTTPException):
                await limiter.check_rate_limit(req_a)

            # key-b 不受影响
            await limiter.check_rate_limit(req_b)

    async def test_rpm_configurable(self):
        """RATE_LIMIT_RPM 环境变量应能自定义限制值。"""
        from fastapi import HTTPException

        from codeagent.interaction.api.rate_limit import RateLimiter

        with patch.dict(os.environ, {"RATE_LIMIT_RPM": "2"}, clear=True):
            limiter = RateLimiter()
            request = _make_request(path="/api/v1/tasks")

            for _ in range(2):
                await limiter.check_rate_limit(request)

            with pytest.raises(HTTPException):
                await limiter.check_rate_limit(request)

    async def test_retry_after_header_present(self):
        """429 响应应包含 Retry-After 头。"""
        from fastapi import HTTPException

        from codeagent.interaction.api.rate_limit import RateLimiter

        with patch.dict(os.environ, {"RATE_LIMIT_RPM": "1"}, clear=True):
            limiter = RateLimiter()
            request = _make_request(path="/api/v1/tasks")

            await limiter.check_rate_limit(request)  # 第 1 次通过
            with pytest.raises(HTTPException) as exc:
                await limiter.check_rate_limit(request)  # 第 2 次超限

            retry_after = exc.value.headers.get("Retry-After")
            assert retry_after is not None
            assert str(retry_after).isdigit() or isinstance(retry_after, int)


# =============================================================================
# Helpers
# =============================================================================


def _make_request(
    path: str = "/api/v1/tasks",
    auth_header: str = "Bearer test-key",
) -> object:
    """创建一个模拟的 FastAPI Request 对象。"""
    from unittest.mock import MagicMock

    request = MagicMock()
    request.url.path = path
    request.headers = {"Authorization": auth_header}
    return request
