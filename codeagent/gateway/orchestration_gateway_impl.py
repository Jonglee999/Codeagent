"""OrchestrationGatewayImpl — 基于 Celery + Redis 的 IOrchestrationGateway 实现。

任务生命周期：
1. start_task() → 提交到 Celery 队列，返回 task_id
2. Celery Worker 异步执行 Agent 工作流
3. 执行过程中通过 Redis Pub/Sub 发布 TaskEvent
4. get_task_status() → 查询 Redis 中的任务状态
5. stream_task() → 订阅 Redis 频道，转发 TaskEvent
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import AsyncGenerator

import redis.asyncio as aioredis
from celery import Celery

from codeagent.config import get_redis_url
from codeagent.gateway.orchestration_gateway import (
    HumanDecision,
    IOrchestrationGateway,
    TaskEvent,
    TaskReport,
    TaskState,
    TaskStatus,
    UserRequest,
)

logger = logging.getLogger(__name__)

# Redis 键模板
_STATUS_KEY = "task:{task_id}:status"
_EVENTS_CHANNEL = "task:{task_id}:events"
_DECISION_KEY = "task:{task_id}:decision"
_REPORT_KEY = "task:{task_id}:report"
_CANCEL_KEY = "task:{task_id}:cancel"

# TTL 常量（秒）
_STATUS_TTL = 3600
_REPORT_TTL = 3600
_DECISION_TTL = 600
_CANCEL_TTL = 60

# stream_task keepalive 超时
_STREAM_KEEPALIVE_TIMEOUT = 60.0


class OrchestrationGatewayImpl(IOrchestrationGateway):
    """基于 Celery + Redis 的 IOrchestrationGateway 实现。

    将 Agent 长任务（分钟级）从 HTTP 请求中解耦，通过 Celery 异步执行。
    Redis 同时承担 Celery 消息队列和 WebSocket 事件总线两个角色。
    """

    def __init__(self, redis_url: str | None = None) -> None:
        self._redis_url = redis_url or get_redis_url()
        # Only configure broker (not backend) — the API server only publishes tasks,
        # it does not fetch results. Setting backend here causes a blocking connection
        # attempt on first send_task call.
        self._celery = Celery(
            "codeagent",
            broker=self._redis_url,
        )
        self._celery.conf.update(
            broker_connection_timeout=5,
            broker_connection_retry_on_startup=False,
        )

    async def start_task(self, request: UserRequest, skip_celery: bool = False) -> str:
        """将任务提交到 Celery 队列。

        1. 生成 task_id（UUID）
        2. 在 Redis 中初始化任务状态（PENDING）
        3. 提交 Celery 任务（run_agent_task.delay），skip_celery=True 时跳过
        4. 返回 task_id

        Args:
            request: 用户请求
            skip_celery: 设为 True 则跳过 Celery 提交（Inline 模式使用）
        """
        task_id = str(uuid.uuid4())

        status_data = {
            "task_id": task_id,
            "state": TaskState.PENDING.value,
            "progress": 0.0,
            "current_step": None,
            "errors": [],
        }
        redis_client = aioredis.from_url(self._redis_url)
        try:
            await redis_client.setex(
                _STATUS_KEY.format(task_id=task_id),
                _STATUS_TTL,
                json.dumps(status_data, ensure_ascii=False),
            )
        finally:
            await redis_client.aclose()

        if not skip_celery:
            # 使用 send_task 提交 Celery 任务（在线程池中执行，避免阻塞 asyncio 事件循环）
            task_args = [task_id, {
                "query": request.query,
                "project_root": request.project_root,
                "auto_mode": request.auto_mode,
                "max_retries": request.max_retries,
            }]
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self._celery.send_task("codeagent.run_agent_task", args=task_args),
            )
            logger.info("Task %s submitted to Celery (query=%r)", task_id, request.query[:80])
        else:
            logger.info("Task %s initialized (skip_celery=True, query=%r)", task_id, request.query[:80])

        return task_id

    async def get_task_status(self, task_id: str) -> TaskStatus:
        """从 Redis 查询任务状态。

        Raises:
            KeyError: 任务 ID 不存在
        """
        redis_client = aioredis.from_url(self._redis_url)
        try:
            raw = await redis_client.get(_STATUS_KEY.format(task_id=task_id))
            if raw is None:
                raise KeyError(f"Task not found: {task_id}")
            data = json.loads(raw)
            return TaskStatus(
                task_id=data["task_id"],
                state=TaskState(data["state"]),
                progress=data.get("progress", 0.0),
                current_step=data.get("current_step"),
                errors=data.get("errors", []),
            )
        finally:
            await redis_client.aclose()

    async def stream_task(self, task_id: str) -> AsyncGenerator[TaskEvent, None]:
        """订阅 Redis Pub/Sub 频道 task:{task_id}:events。

        将收到的 JSON 消息反序列化为 TaskEvent 并 yield。
        任务完成（task_complete / task_error 事件）后自动结束订阅。
        超时 _STREAM_KEEPALIVE_TIMEOUT 无消息时发送 keepalive 并继续。
        """
        redis_client = aioredis.from_url(self._redis_url)
        pubsub = redis_client.pubsub()
        try:
            await pubsub.subscribe(_EVENTS_CHANNEL.format(task_id=task_id))

            while True:
                try:
                    message = await asyncio.wait_for(
                        pubsub.get_message(ignore_subscribe_messages=True),
                        timeout=_STREAM_KEEPALIVE_TIMEOUT,
                    )
                except asyncio.TimeoutError:
                    # 超时无消息，发送 keepalive 事件
                    yield TaskEvent(
                        type="keepalive",
                        data={},
                        timestamp=time.time(),
                    )
                    continue

                if message is None:
                    continue

                event_data = json.loads(message["data"])
                event = TaskEvent(
                    type=event_data.get("type", "unknown"),
                    data=event_data.get("data", {}),
                    timestamp=event_data.get("timestamp", time.time()),
                )
                yield event

                if event.type in ("task_complete", "task_error"):
                    break
        finally:
            await pubsub.unsubscribe(_EVENTS_CHANNEL.format(task_id=task_id))
            await redis_client.aclose()

    async def submit_decision(self, task_id: str, decision: HumanDecision) -> None:
        """将用户决策写入 Redis，Human Review 节点轮询此键恢复执行。

        Args:
            task_id: 任务唯一标识
            decision: 用户决策对象
        """
        redis_client = aioredis.from_url(self._redis_url)
        try:
            decision_data = {
                "task_id": decision.task_id,
                "decision": decision.decision,
                "modifications": decision.modifications,
            }
            await redis_client.setex(
                _DECISION_KEY.format(task_id=task_id),
                _DECISION_TTL,
                json.dumps(decision_data, ensure_ascii=False),
            )
            logger.info("Decision %s submitted for task %s", decision.decision, task_id)
        finally:
            await redis_client.aclose()

    async def cancel_task(self, task_id: str) -> bool:
        """取消任务。

        撤销 Celery 任务，设置 Redis 取消标志，更新状态为 CANCELLED。
        """
        loop = asyncio.get_event_loop()
        redis_client = aioredis.from_url(self._redis_url)
        try:
            # 设置取消标志
            await redis_client.setex(
                _CANCEL_KEY.format(task_id=task_id),
                _CANCEL_TTL,
                b"1",
            )

            # 尝试撤销 Celery 任务（在线程池中执行，避免阻塞事件循环）
            try:
                await loop.run_in_executor(
                    None,
                    lambda: self._celery.control.revoke(task_id, terminate=True),
                )
            except Exception as exc:
                logger.warning("Celery revoke failed for %s: %s", task_id, exc)

            # 更新状态为 CANCELLED
            raw = await redis_client.get(_STATUS_KEY.format(task_id=task_id))
            if raw:
                status_data = json.loads(raw)
                status_data["state"] = TaskState.CANCELLED.value
                await redis_client.setex(
                    _STATUS_KEY.format(task_id=task_id),
                    _STATUS_TTL,
                    json.dumps(status_data, ensure_ascii=False),
                )
                return True
            return False
        finally:
            await redis_client.aclose()

    async def get_report(self, task_id: str) -> TaskReport:
        """从 Redis 读取任务完整报告。

        Raises:
            KeyError: 报告不存在
        """
        redis_client = aioredis.from_url(self._redis_url)
        try:
            raw = await redis_client.get(_REPORT_KEY.format(task_id=task_id))
            if raw is None:
                raise KeyError(f"Report not found for task: {task_id}")
            data = json.loads(raw)
            return TaskReport(
                task_id=data["task_id"],
                plan=data.get("plan", []),
                changes=data.get("changes", []),
                validation_results=data.get("validation_results", []),
                duration=data.get("duration", 0.0),
                token_usage=data.get("token_usage", 0),
            )
        finally:
            await redis_client.aclose()

    # ── 内部辅助方法（Worker 端调用） ────────────────────────────

    async def _publish_event(self, task_id: str, event: TaskEvent) -> None:
        """向 Redis Pub/Sub 频道发布事件（Worker 端调用）。"""
        redis_client = aioredis.from_url(self._redis_url)
        try:
            event_data = {
                "type": event.type,
                "data": event.data,
                "timestamp": event.timestamp,
            }
            await redis_client.publish(
                _EVENTS_CHANNEL.format(task_id=task_id),
                json.dumps(event_data, ensure_ascii=False),
            )
        finally:
            await redis_client.aclose()

    async def _set_task_status(
        self, task_id: str, status: TaskStatus
    ) -> None:
        """更新 Redis 中的任务状态，TTL = _STATUS_TTL。"""
        redis_client = aioredis.from_url(self._redis_url)
        try:
            status_data = {
                "task_id": status.task_id,
                "state": status.state.value,
                "progress": status.progress,
                "current_step": status.current_step,
                "errors": status.errors,
            }
            await redis_client.setex(
                _STATUS_KEY.format(task_id=task_id),
                _STATUS_TTL,
                json.dumps(status_data, ensure_ascii=False),
            )
        finally:
            await redis_client.aclose()

    def _get_redis_url(self) -> str:
        """获取 Redis URL（供 Worker 端同步代码调用）。"""
        return self._redis_url
