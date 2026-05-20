"""速率限制 — 基于 Redis 的滑动窗口限速。

默认限制：100 req/min（通过 RATE_LIMIT_RPM 环境变量配置）。
/health 和 /metrics 端点豁免限速。
Redis 连接失败时限速降级为放行（不阻塞业务）。
"""

from __future__ import annotations

import os
import time

import redis.asyncio as aioredis
from fastapi import HTTPException, Request, status

logger = __import__("logging").getLogger(__name__)

# 豁免限速的路径前缀
_EXEMPTED_PATHS = {"/health", "/metrics"}


class RateLimiter:
    """基于 Redis 的滑动窗口速率限制器。

    键格式：rate_limit:{key_prefix}:{window}
    窗口大小：60 秒（固定窗口）
    默认限制：100 RPM（通过 RATE_LIMIT_RPM 环境变量配置）

    Usage:
        rate_limiter = RateLimiter()
        # 在 FastAPI 中作为 Depends:
        await rate_limiter.check_rate_limit(request)
    """

    def __init__(self, redis_url: str = "redis://127.0.0.1:6379") -> None:
        self._redis_url = redis_url
        self._rpm = int(os.environ.get("RATE_LIMIT_RPM", "100"))

    def _is_exempted(self, request: Request) -> bool:
        """判断请求路径是否豁免限速。"""
        return request.url.path in _EXEMPTED_PATHS

    async def check_rate_limit(self, request: Request) -> None:
        """FastAPI Depends 调用入口。"""
        if self._is_exempted(request):
            return

        # 从 Authorization 头提取 key（用于区分客户端）
        auth_header = request.headers.get("Authorization", "anonymous")
        if auth_header.startswith("Bearer "):
            key_prefix = auth_header[7:]
        else:
            key_prefix = auth_header

        try:
            redis_client = aioredis.from_url(self._redis_url)
            try:
                window = int(time.time() / 60)  # 60 秒窗口
                redis_key = f"rate_limit:{key_prefix}:{window}"

                count = await redis_client.incr(redis_key)
                if count == 1:
                    await redis_client.expire(redis_key, 120)  # 2 分钟 TTL

                if count > self._rpm:
                    retry_after = 60 - (int(time.time()) % 60)
                    raise HTTPException(
                        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                        detail=f"Rate limit exceeded. Max {self._rpm} req/min",
                        headers={"Retry-After": str(retry_after)},
                    )
            finally:
                await redis_client.aclose()
        except HTTPException:
            raise
        except Exception as exc:
            # Redis 连接失败时降级放行，不阻塞业务
            logger.warning("Rate limiter Redis error, allowing request: %s", exc)
