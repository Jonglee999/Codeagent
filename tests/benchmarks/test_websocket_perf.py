"""WebSocket 流式事件延迟基准测试。

测试 WebSocket 事件从发布到接收的端到端延迟。
使用 Docker 中的真实 Redis 服务（redis://127.0.0.1:6379）。

测量指标：
- 事件发布到接收的延迟（latency）
- 事件吞吐量（events/sec）
"""

import asyncio
import json
import time
from datetime import datetime, timezone

import pytest
import redis.asyncio as aioredis

from codeagent.config import get_redis_url

pytestmark = pytest.mark.benchmark

# 测试事件负载
_SAMPLE_EVENT = {
    "type": "node.progress",
    "content": "正在分析代码结构...",
    "timestamp": "",
}

_MANY_EVENTS = 100  # 吞吐量测试事件数


@pytest.mark.asyncio
class TestWebSocketPerformance:
    """WebSocket 事件延迟基准测试（基于真实 Redis）。"""

    async def test_websocket_event_latency(self, benchmark_metrics):
        """WebSocket 事件从发布到消费的延迟，目标 < 200ms。

        使用 Docker 中的真实 Redis Pub/Sub，测量事件从
        publish 到 get_message 的端到端延迟。
        """
        redis_client = aioredis.from_url(get_redis_url())
        pubsub = redis_client.pubsub()

        try:
            channel = "test:events"
            await pubsub.subscribe(channel)

            # 发布事件
            event = dict(_SAMPLE_EVENT)
            event["timestamp"] = datetime.now(timezone.utc).isoformat()
            event_json = json.dumps(event)

            start = time.perf_counter()
            await redis_client.publish(channel, event_json)

            # 轮询直到收到消息或超时
            latencies: list[float] = []
            timeout = 2.0
            deadline = time.monotonic() + timeout

            while time.monotonic() < deadline and len(latencies) < 10:
                message = await pubsub.get_message(
                    ignore_subscribe_messages=True, timeout=0.05,
                )
                if message is not None:
                    elapsed = time.perf_counter() - start
                    latencies.append(elapsed)
                    # 继续发布更多事件
                    for _ in range(9):
                        e = dict(_SAMPLE_EVENT)
                        e["timestamp"] = datetime.now(timezone.utc).isoformat()
                        await redis_client.publish(channel, json.dumps(e))
                        msg = await pubsub.get_message(
                            ignore_subscribe_messages=True, timeout=0.05,
                        )
                        if msg is not None:
                            latencies.append(time.perf_counter() - start)

            if latencies:
                avg_latency = sum(latencies) / len(latencies)
                benchmark_metrics["websocket"]["avg_event_latency_ms"] = avg_latency * 1000
                benchmark_metrics["websocket"]["sample_count"] = len(latencies)
                assert avg_latency < 0.2, (
                    f"WebSocket 事件平均延迟 {avg_latency*1000:.1f}ms，超过阈值 200ms"
                )
            else:
                pytest.fail("未收到任何 WebSocket 事件")

        finally:
            await pubsub.unsubscribe(channel)
            await pubsub.aclose()
            await redis_client.aclose()

    async def test_websocket_throughput(self, benchmark_metrics):
        """事件吞吐量基准测试。

        测量在持续负载下事件的处理速度（事件/秒）。
        """
        redis_client = aioredis.from_url(get_redis_url())
        pubsub = redis_client.pubsub()

        try:
            channel = "test:throughput"
            await pubsub.subscribe(channel)

            start = time.perf_counter()
            received = 0

            async def publisher():
                for i in range(_MANY_EVENTS):
                    event = {
                        "type": "node.progress",
                        "content": f"步骤 {i + 1}/{_MANY_EVENTS}",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                    await redis_client.publish(channel, json.dumps(event))
                    await asyncio.sleep(0.01)  # 模拟真实发布间隔

            # 并发发布和消费
            pub_task = asyncio.create_task(publisher())

            timeout = 10.0
            deadline = time.monotonic() + timeout

            while time.monotonic() < deadline and received < _MANY_EVENTS:
                message = await pubsub.get_message(
                    ignore_subscribe_messages=True, timeout=0.05,
                )
                if message is not None:
                    received += 1

            await pub_task
            elapsed = time.perf_counter() - start
            throughput = received / elapsed if elapsed > 0 else 0

            benchmark_metrics["websocket"]["throughput_events_per_sec"] = throughput
            benchmark_metrics["websocket"]["total_received"] = received

            # 确保收到大部分事件
            assert received >= _MANY_EVENTS * 0.8, (
                f"WebSocket 吞吐量测试仅收到 {received}/{_MANY_EVENTS} 事件"
            )

        finally:
            await pubsub.unsubscribe(channel)
            await pubsub.aclose()
            await redis_client.aclose()
