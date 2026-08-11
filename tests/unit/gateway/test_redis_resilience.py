from __future__ import annotations

import pytest

from codeagent.interaction.api.main import _publish_event
from codeagent.redis_resilience import RedisRetryPolicy, retry_redis_async, retry_redis_sync


@pytest.mark.asyncio
async def test_async_short_outage_recovers_with_bounded_backoff():
    attempts = 0
    sleeps: list[float] = []

    async def operation():
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise ConnectionError("redis unavailable")
        return "ok"

    async def sleep(delay: float):
        sleeps.append(delay)

    result, retries = await retry_redis_async(
        operation,
        policy=RedisRetryPolicy(max_retries=2, base_delay_seconds=0.1),
        sleep=sleep,
    )

    assert result == "ok"
    assert retries == 2
    assert sleeps == [0.1, 0.2]


@pytest.mark.asyncio
async def test_async_persistent_outage_remains_failure():
    async def operation():
        raise ConnectionError("redis unavailable")

    with pytest.raises(ConnectionError):
        await retry_redis_async(
            operation,
            policy=RedisRetryPolicy(max_retries=1),
            sleep=lambda _delay: _done(),
        )


async def _done():
    return None


def test_sync_short_outage_recovers():
    attempts = 0

    def operation():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ConnectionError("redis unavailable")
        return "ok"

    result, retries = retry_redis_sync(operation, sleep=lambda _delay: None)
    assert result == "ok"
    assert retries == 1


@pytest.mark.asyncio
async def test_event_stream_emits_real_recovery_event(monkeypatch):
    import redis.asyncio as aioredis

    state = {"executes": 0, "payloads": []}

    class Pipeline:
        def publish(self, _channel, payload):
            state["payloads"].append(payload)
            return self

        def rpush(self, *_args):
            return self

        def expire(self, *_args):
            return self

        async def execute(self):
            state["executes"] += 1
            if state["executes"] == 1:
                raise ConnectionError("short outage")

    class Client:
        async def incr(self, _key):
            return 1

        def pipeline(self):
            return Pipeline()

        async def aclose(self):
            return None

    monkeypatch.setattr(aioredis, "from_url", lambda _url: Client())
    runtime = {"retry_count": 0, "recovery_count": 0, "dropped_event_count": 0}

    await _publish_event(
        "redis://test",
        "task-1",
        {"type": "node_start", "node": "worker"},
        runtime,
    )

    assert runtime == {"retry_count": 1, "recovery_count": 1, "dropped_event_count": 0}
    assert any('"type": "infrastructure_recovered"' in item for item in state["payloads"])
