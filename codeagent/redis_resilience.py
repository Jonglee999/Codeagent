"""Bounded retry helpers for short Redis transport interruptions."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable


@dataclass(frozen=True)
class RedisRetryPolicy:
    max_retries: int = 2
    base_delay_seconds: float = 0.1
    max_delay_seconds: float = 1.0


async def retry_redis_async(
    operation: Callable[[], Awaitable[Any]],
    *,
    policy: RedisRetryPolicy | None = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> tuple[Any, int]:
    active = policy or RedisRetryPolicy()
    for attempt in range(active.max_retries + 1):
        try:
            return await operation(), attempt
        except Exception:
            if attempt >= active.max_retries:
                raise
            delay = min(active.max_delay_seconds, active.base_delay_seconds * (2 ** attempt))
            await sleep(delay)
    raise RuntimeError("unreachable")


def retry_redis_sync(
    operation: Callable[[], Any],
    *,
    policy: RedisRetryPolicy | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[Any, int]:
    active = policy or RedisRetryPolicy()
    for attempt in range(active.max_retries + 1):
        try:
            return operation(), attempt
        except Exception:
            if attempt >= active.max_retries:
                raise
            delay = min(active.max_delay_seconds, active.base_delay_seconds * (2 ** attempt))
            sleep(delay)
    raise RuntimeError("unreachable")
