"""Shared task-control primitives for cancellation and runtime steering."""

from __future__ import annotations

import json
from typing import Any


def steering_key(task_id: str) -> str:
    return f"task:{task_id}:steering"


def cancel_key(task_id: str) -> str:
    return f"task:{task_id}:cancel"


def decode_steering_entries(raw_entries: Any) -> list[str]:
    if raw_entries is None:
        return []
    entries = raw_entries if isinstance(raw_entries, list) else [raw_entries]
    instructions: list[str] = []
    for raw in entries:
        try:
            text = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
            payload = json.loads(text)
            instruction = str(payload.get("instruction", "")).strip()
            if instruction:
                instructions.append(instruction[:4000])
        except (UnicodeDecodeError, ValueError, TypeError, json.JSONDecodeError):
            continue
    return instructions


async def consume_steering(redis_url: str, task_id: str, limit: int = 10) -> list[str]:
    import redis.asyncio as aioredis
    from redis.exceptions import RedisError

    client = aioredis.from_url(redis_url)
    try:
        try:
            raw = await client.lpop(steering_key(task_id), max(1, min(limit, 20)))
            return decode_steering_entries(raw)
        except RedisError:
            return []
    finally:
        await client.aclose()


def cancel_requested_sync(redis_url: str, task_id: str) -> bool:
    import redis
    from redis.exceptions import RedisError

    client = redis.Redis.from_url(redis_url)
    try:
        try:
            return bool(client.exists(cancel_key(task_id)))
        except RedisError:
            return False
    finally:
        client.close()
