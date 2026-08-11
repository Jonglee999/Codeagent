"""WebSocket 流式输出 — 实时推送 Agent 执行事件。

通过 Redis Pub/Sub 订阅 Agent 执行事件并转发给 WebSocket 客户端。
连接建立后先回放历史事件（Redis List），再监听新事件（Pub/Sub）。
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone

import redis.asyncio as aioredis
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from codeagent.config import get_redis_url

logger = logging.getLogger(__name__)

ws_router = APIRouter()

_PING_INTERVAL = 25.0   # seconds between keepalive pings
_POLL_INTERVAL = 0.05   # seconds between get_message polls
_INTERNAL_EVENT_TYPES = {
    "memory_recalled",
    "memory_extracted",
    "strategy_recalled",
    "transcript_saved",
}


def _is_user_visible_event(event: dict) -> bool:
    """Enforce the event visibility contract at the server boundary."""
    return (
        event.get("visibility") != "internal"
        and event.get("type") not in _INTERNAL_EVENT_TYPES
    )


@ws_router.websocket("/api/v1/tasks/{task_id}/stream")
async def task_stream(websocket: WebSocket, task_id: str) -> None:
    """实时推送 Agent 执行事件。

    流程：
    1. 接受 WebSocket 连接
    2. 先订阅 Redis Pub/Sub（避免在读历史和订阅之间丢消息）
    3. 回放 Redis List 中的历史事件
    4. 若历史中已有终止事件则直接关闭
    5. 继续监听新事件，每 _PING_INTERVAL 秒发一次 ping 心跳
    6. 收到 task_complete/task_error 后关闭连接
    """
    await websocket.accept()
    redis_client = aioredis.from_url(get_redis_url())
    pubsub = redis_client.pubsub()

    try:
        # 1. 先订阅，再读历史，避免竞态丢消息
        await pubsub.subscribe(f"task:{task_id}:events")

        # 2. 回放历史事件
        log_key = f"task:{task_id}:event_log"
        history_raw = await redis_client.lrange(log_key, 0, -1)
        seen_events: set[str] = set()
        terminal = False

        for raw in history_raw:
            event_data = json.loads(raw)
            event_key = event_data.get("event_id") or (
                f"{event_data.get('type')}-{event_data.get('timestamp', '')}"
            )
            seen_events.add(event_key)
            if not _is_user_visible_event(event_data):
                continue
            await websocket.send_json(event_data)
            if event_data.get("type") in ("task_complete", "task_error", "task_cancelled"):
                terminal = True
                break

        if terminal:
            return

        # 3. 继续监听新事件
        last_ping = asyncio.get_event_loop().time()

        while True:
            message = await pubsub.get_message(ignore_subscribe_messages=True)

            if message is None:
                await asyncio.sleep(_POLL_INTERVAL)
                now = asyncio.get_event_loop().time()
                if now - last_ping >= _PING_INTERVAL:
                    await websocket.send_json(
                        {
                            "type": "ping",
                            "schema_version": 1,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                    )
                    last_ping = now
                continue

            event_data = json.loads(message["data"])
            event_data.setdefault(
                "timestamp", datetime.now(timezone.utc).isoformat()
            )

            # 去重：跳过已在历史回放中发送过的事件
            dedup_key = event_data.get("event_id") or (
                f"{event_data.get('type')}-{event_data.get('timestamp', '')}"
            )
            if dedup_key in seen_events:
                continue
            seen_events.add(dedup_key)

            if not _is_user_visible_event(event_data):
                continue

            await websocket.send_json(event_data)

            if event_data.get("type") in ("task_complete", "task_error", "task_cancelled"):
                break

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected for task %s", task_id)
    except Exception as exc:
        logger.error("WebSocket error for task %s: %s", task_id, exc)
    finally:
        await pubsub.unsubscribe(f"task:{task_id}:events")
        await pubsub.aclose()
        await redis_client.aclose()
