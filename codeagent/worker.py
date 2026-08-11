"""Celery Worker 入口 — 异步执行 Agent 工作流。

在 Worker 进程中执行 Agent 任务，并通过 Redis Pub/Sub 发布执行事件。
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import traceback
import uuid
from datetime import datetime, timezone
from typing import Any

import redis as sync_redis

from celery import Celery

from codeagent.chat_response import request_visible_chat
from codeagent.config import (
    get_env,
    get_redis_url,
    load_env_file,
)
from codeagent.orchestration.intent import classify_response_mode
from codeagent.product_state import product_state_store

# 在模块加载时加载 .env 文件
from pathlib import Path
load_env_file(Path.cwd() / ".env")

logger = logging.getLogger(__name__)

# Redis 键模板（与 orchestration_gateway_impl.py 保持一致）
_STATUS_KEY = "task:{task_id}:status"
_EVENTS_CHANNEL = "task:{task_id}:events"
_REPORT_KEY = "task:{task_id}:report"

# ── Celery 应用 ────────────────────────────────────────────────

app = Celery("codeagent", broker=get_redis_url(), backend=get_redis_url())

app.conf.update(
    task_track_started=True,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
)


# ── 事件发布辅助函数（同步，供 Celery Worker 调用） ──────────


def _publish_event_sync(
    redis_url: str,
    task_id: str,
    event: dict,
    redis_runtime: dict[str, int] | None = None,
) -> None:
    """同步发布事件到 Redis，并持久化到 List 供历史回放。"""
    # 统一 timestamp 为 ISO 8601 字符串
    if "timestamp" not in event or not isinstance(event["timestamp"], str):
        event["timestamp"] = datetime.now(timezone.utc).isoformat()
    event.setdefault("event_id", uuid.uuid4().hex)
    event.setdefault("schema_version", 1)
    from codeagent.redis_resilience import retry_redis_sync

    def operation() -> None:
        r = sync_redis.from_url(redis_url)
        try:
            if "seq" not in event:
                event["seq"] = int(r.incr(f"task:{task_id}:event_seq"))
            payload = json.dumps(event, ensure_ascii=False)
            channel = _EVENTS_CHANNEL.format(task_id=task_id)
            log_key = f"task:{task_id}:event_log"
            pipe = r.pipeline()
            pipe.publish(channel, payload)
            pipe.rpush(log_key, payload)
            pipe.expire(log_key, 3600)
            pipe.execute()
        finally:
            r.close()

    try:
        _, retries = retry_redis_sync(operation)
        if retries and redis_runtime is not None:
            redis_runtime["retry_count"] = redis_runtime.get("retry_count", 0) + retries
            redis_runtime["recovery_count"] = redis_runtime.get("recovery_count", 0) + 1
    except Exception as exc:
        if redis_runtime is not None:
            redis_runtime["dropped_event_count"] = redis_runtime.get("dropped_event_count", 0) + 1
        logger.warning("Failed to publish event for task %s: %s", task_id, exc)


def _update_status_sync(
    redis_url: str, task_id: str, state: str, *,
    progress: float = 0.0,
    current_step: str | None = None,
    errors: list[str] | None = None,
    redis_runtime: dict[str, int] | None = None,
) -> None:
    """同步更新 Redis 中的任务状态。"""
    from codeagent.redis_resilience import retry_redis_sync

    def operation() -> None:
        r = sync_redis.from_url(redis_url)
        try:
            raw = r.get(_STATUS_KEY.format(task_id=task_id))
            if raw:
                status_data = json.loads(raw)
            else:
                status_data = {
                    "task_id": task_id,
                    "state": "pending",
                    "progress": 0.0,
                    "current_step": None,
                    "errors": [],
                }
            status_data["state"] = state
            status_data["progress"] = progress
            if current_step is not None:
                status_data["current_step"] = current_step
            if errors is not None:
                status_data["errors"] = errors
            r.setex(_STATUS_KEY.format(task_id=task_id), 3600, json.dumps(status_data, ensure_ascii=False))
        finally:
            r.close()

    _, retries = retry_redis_sync(operation)
    product_state_store.update_run_state(task_id, state)
    if retries and redis_runtime is not None:
        redis_runtime["retry_count"] = redis_runtime.get("retry_count", 0) + retries
        redis_runtime["recovery_count"] = redis_runtime.get("recovery_count", 0) + 1


def _save_report_sync(
    redis_url: str,
    task_id: str,
    report: dict,
    redis_runtime: dict[str, int] | None = None,
) -> None:
    """Persist a task report; unlike telemetry, report loss is not degradable."""
    from codeagent.redis_resilience import retry_redis_sync

    def operation() -> None:
        r = sync_redis.from_url(redis_url)
        try:
            r.setex(
                _REPORT_KEY.format(task_id=task_id),
                3600,
                json.dumps(report, ensure_ascii=False),
            )
        finally:
            r.close()

    _, retries = retry_redis_sync(operation)
    product_state_store.save_report(task_id, report)
    if retries and redis_runtime is not None:
        redis_runtime["retry_count"] = redis_runtime.get("retry_count", 0) + retries
        redis_runtime["recovery_count"] = redis_runtime.get("recovery_count", 0) + 1


# ── 组件构建（参考 CLI main.py） ──────────────────────────────


def _build_llm(model_name: str, event_callback: Any | None = None) -> Any:
    """Build the shared classified/retrying/fallback model gateway."""
    from dataclasses import replace
    from codeagent.model_gateway import ModelGateway
    from codeagent.model_routing import ModelRouter

    gateway = ModelRouter.from_env(event_callback=event_callback)
    if model_name and model_name != gateway.primary.model:
        gateway = ModelGateway(
            replace(gateway.primary, model=model_name),
            None,
            event_callback=event_callback,
        )
    return gateway


def _build_tool_gateway(
    project_root: str,
    *,
    allow_high_risk_extensions: bool = False,
    context_engine: Any | None = None,
) -> Any:
    """构建工具 Gateway。"""
    from codeagent.tools.gateway import ToolGateway
    return ToolGateway(
        project_root=project_root,
        allow_high_risk_extensions=allow_high_risk_extensions,
        context_engine=context_engine,
    )


def _build_validation_gateway(project_root: str = "") -> Any:
    """构建验证 Gateway。"""
    from codeagent.gateway.validation_gateway_impl import ValidationGateway
    return ValidationGateway(project_root=project_root)


def _build_context_gateway(project_root: str) -> Any:
    """构建上下文 Gateway。"""
    from codeagent.context_engine.engine import ContextEngine, ContextConfig

    return ContextEngine(config=ContextConfig.from_env())


# ── Celery 任务 ────────────────────────────────────────────────


@app.task(bind=True, name="codeagent.run_agent_task")
def run_agent_task(self, task_id: str, request_dict: dict) -> dict:
    """Celery 任务：在 Worker 进程中执行 Agent 工作流。

    1. 从 request_dict 重建 UserRequest
    2. 更新 Redis 任务状态为 RUNNING
    3. 构建 Orchestrator（从 config 读取 LLM 配置）
    4. 在事件回调中向 Redis 发布 TaskEvent
    5. 执行完成后将 TaskReport 写入 Redis
    6. 更新任务状态为 COMPLETED 或 FAILED
    """
    redis_url = get_redis_url()
    query = request_dict.get("query", "")
    project_root = request_dict.get("project_root", "")
    auto_mode = request_dict.get("auto_mode", False)
    max_retries = request_dict.get("max_retries", 3)
    conversation_history = request_dict.get("conversation_history", [])
    direct_execution = bool(request_dict.get("direct_execution", False))
    benchmark_instance_id = request_dict.get("benchmark_instance_id")
    benchmark_fail_to_pass = request_dict.get("benchmark_fail_to_pass", [])
    benchmark_pass_to_pass = request_dict.get("benchmark_pass_to_pass", [])
    recovered_from_task_id = request_dict.get("recovered_from_task_id")
    response_mode = classify_response_mode(
        query,
        request_dict.get("response_mode", "auto"),
    )
    redis_runtime = {"retry_count": 0, "recovery_count": 0, "dropped_event_count": 0}
    tool_gateway = None
    llm = None

    logger.info("Worker starting task %s: %r", task_id, query[:80])

    # 1. 更新状态为 RUNNING
    _update_status_sync(
        redis_url, task_id, "running", progress=0.0, redis_runtime=redis_runtime
    )

    # 2. 发布 task_started 事件
    _publish_event_sync(redis_url, task_id, {
        "type": "node_start",
        "node": "worker",
        "summary": query[:200],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }, redis_runtime)

    try:
        # 创建事件发布回调
        def make_event_callback(tid: str) -> Any:
            """创建进度回调，将进度 dict 发布为 Redis 事件。"""
            def _publish_progress(progress: dict) -> None:
                from codeagent.task_control import cancel_requested_sync

                if cancel_requested_sync(redis_url, tid):
                    raise asyncio.CancelledError(f"Task {tid} was cancelled")
                """将进度 dict 发布为 Redis 事件。"""
                if "timestamp" not in progress:
                    progress["timestamp"] = time.time()
                _publish_event_sync(redis_url, tid, progress, redis_runtime)

            return _publish_progress

        progress_callback = make_event_callback(task_id)
        llm = _build_llm(
            get_env("LLM_MODEL", "deepseek/deepseek-v4-flash"),
            event_callback=progress_callback,
        )

        # 对话模式不构建上下文、工具、验证器或 MCP 连接。这样简单追问
        # 与 inline runner 保持一致，也不会在前端制造无关的工具活动。
        if response_mode == "chat":
            messages = [{
                "role": "system",
                "content": (
                    "你是 AGENT4CODE。简洁、准确地回答用户关于当前开发任务或项目的问题。"
                    "这是对话模式：不要声称已经运行工具、修改文件或完成验证。"
                ),
            }]
            for item in conversation_history[-20:]:
                role = item.get("role")
                content = item.get("content")
                if role in {"user", "assistant"} and isinstance(content, str):
                    messages.append({"role": role, "content": content[:12000]})
            messages.append({"role": "user", "content": query})
            chat_response = asyncio.run(request_visible_chat(
                llm,
                model=get_env("LLM_MODEL", "deepseek/deepseek-v4-flash"),
                messages=messages,
            ))
            assistant_response = chat_response.text
            duration = (
                time.time() - self.request.received_timestamp
                if hasattr(self.request, "received_timestamp")
                else 0.0
            )
            from codeagent.memory.transcript import persist_transcript

            chat_entries = [
                {
                    "kind": "message",
                    "role": item.get("role"),
                    "content": item.get("content", ""),
                }
                for item in conversation_history[-20:]
                if item.get("role") in {"user", "assistant"}
            ]
            chat_entries.extend([
                {"kind": "message", "role": "user", "content": query},
                {"kind": "message", "role": "assistant", "content": assistant_response},
                {"kind": "outcome", "success": True, "mode": "chat"},
            ])
            transcript_path = persist_transcript(project_root, task_id, chat_entries)
            report = {
                "task_id": task_id,
                "status": "completed",
                "plan": [],
                "changes": [],
                "validation_results": [],
                "duration": duration,
                "token_usage": chat_response.token_usage,
                "assistant_response": assistant_response,
                "response_mode": "chat",
                "memory_hits": [],
                "resolved_skills": [],
                "warnings": [],
                "reflection": None,
                "transcript_path": transcript_path,
                "model_runtime": llm.snapshot(),
                "infrastructure_runtime": {"redis": dict(redis_runtime)},
            }
            _save_report_sync(redis_url, task_id, report, redis_runtime)
            _publish_event_sync(redis_url, task_id, {
                "type": "assistant_message",
                "content": assistant_response,
            }, redis_runtime)
            _update_status_sync(
                redis_url, task_id, "completed", progress=1.0,
                redis_runtime=redis_runtime,
            )
            _publish_event_sync(redis_url, task_id, {
                "type": "task_complete",
                "status": "success",
                "duration": duration,
                "token_usage": report["token_usage"],
            }, redis_runtime)
            from codeagent.orchestration.policy import requests_durable_memory

            if requests_durable_memory(query):
                from codeagent.orchestration.runtime import build_memory_service

                memory = build_memory_service(project_root, llm, auto_mode=True)
                if memory is not None:
                    try:
                        asyncio.run(memory.auto_extract(
                            [
                                *[
                                    {"role": item.get("role"), "content": item.get("content", "")}
                                    for item in conversation_history[-20:]
                                    if item.get("role") in {"user", "assistant"}
                                ],
                                {"role": "user", "content": query},
                                {"role": "assistant", "content": assistant_response},
                            ],
                            trigger="explicit_user_memory",
                        ))
                    except Exception:
                        logger.warning(
                            "Chat memory writeback failed for %s", task_id, exc_info=True,
                        )
            return report

        # 3. 执行模式才构建上下文、工具、验证器并按需发现 MCP。
        context_gateway = _build_context_gateway(project_root)
        tool_gateway = _build_tool_gateway(
            project_root,
            allow_high_risk_extensions=bool(auto_mode),
            context_engine=context_gateway,
        )
        validation_gateway = _build_validation_gateway(project_root)

        # 4. 构建 Orchestrator
        from codeagent.orchestration.orchestrator import Orchestrator
        from codeagent.orchestration.runtime import build_learning_services

        memory_gateway, evolution_manager, strategy_applier = build_learning_services(
            project_root=project_root,
            llm_client=llm,
            auto_mode=auto_mode,
        )

        from codeagent.task_control import consume_steering

        async def steering_provider() -> list[str]:
            return await consume_steering(redis_url, task_id)

        orchestrator = None

        # 5. 异步执行
        async def _run_async() -> Any:
            nonlocal orchestrator
            checkpointer = None
            try:
                from codeagent.config import get_checkpoint_enabled

                if get_checkpoint_enabled():
                    from codeagent.orchestration.checkpoint import get_async_checkpointer

                    checkpointer = await get_async_checkpointer()
                orchestrator = Orchestrator(
                    context_gateway=context_gateway,
                    tool_gateway=tool_gateway,
                    validation_gateway=validation_gateway,
                    llm=llm,
                    model_name=get_env("LLM_MODEL", "deepseek/deepseek-v4-flash"),
                    progress_callback=progress_callback,
                    memory_gateway=memory_gateway,
                    evolution_manager=evolution_manager,
                    strategy_applier=strategy_applier,
                    max_retries=int(max_retries),
                    steering_provider=steering_provider,
                    checkpointer=checkpointer,
                )
                extension_status = await tool_gateway.initialize_extensions(
                    progress_callback,
                    query=query,
                    external_enabled=not bool(benchmark_instance_id),
                )
                progress_callback({
                    "type": "capabilities_resolved",
                    "summary": extension_status["mcp"]["detail"],
                    "data": extension_status,
                })
                return await orchestrator.run(
                    query,
                    project_root,
                    auto_mode=auto_mode,
                    task_id=task_id,
                    conversation_history=conversation_history,
                    direct_execution=direct_execution,
                    benchmark_instance_id=benchmark_instance_id,
                    benchmark_fail_to_pass=benchmark_fail_to_pass,
                    benchmark_pass_to_pass=benchmark_pass_to_pass,
                    recovered_from_task_id=recovered_from_task_id,
                )
            finally:
                await tool_gateway.aclose()
                if checkpointer is not None and hasattr(checkpointer, "conn"):
                    await checkpointer.conn.close()

        try:
            final_state = asyncio.run(_run_async())
        except asyncio.CancelledError:
            logger.info("Worker task %s stopped after cancellation request", task_id)
            _update_status_sync(
                redis_url,
                task_id,
                "cancelled",
                progress=1.0,
                redis_runtime=redis_runtime,
            )
            return {"task_id": task_id, "status": "cancelled"}

        # 6. 构建报告
        duration = time.time() - self.request.received_timestamp if hasattr(self.request, 'received_timestamp') else 0.0
        token_usage = final_state.estimated_tokens if hasattr(final_state, "estimated_tokens") else 0
        from codeagent.orchestration.runtime import (
            assess_task_completion,
            build_execution_report,
            persist_run_artifacts,
        )

        success, errors = assess_task_completion(
            final_state,
            require_patch=bool(benchmark_instance_id),
            require_test_evidence=bool(benchmark_instance_id),
        )

        def _to_dict(obj: Any) -> Any:
            """递归将 dataclass 转换为 dict（确保 JSON 可序列化）。"""
            if hasattr(obj, "__dataclass_fields__"):
                return {f: _to_dict(getattr(obj, f)) for f in obj.__dataclass_fields__}
            if isinstance(obj, (list, tuple)):
                return [_to_dict(i) for i in obj]
            if isinstance(obj, dict):
                return {k: _to_dict(v) for k, v in obj.items()}
            return obj

        extension_runtime = tool_gateway.extension_status()["mcp"]
        report = build_execution_report(
            task_id,
            final_state,
            duration=duration,
            success=success,
            errors=errors,
            extension_warnings=extension_runtime.get("warnings", []),
            mcp_servers=extension_runtime.get("servers", []),
            model_runtime=llm.snapshot(),
            infrastructure_runtime={
                "redis": dict(redis_runtime),
                "tools": tool_gateway.resilience_status(),
            },
        )
        persist_run_artifacts(project_root, task_id, report)

        # 7. 写入报告
        _save_report_sync(redis_url, task_id, report, redis_runtime)

        # 8. 更新状态为 COMPLETED/FAILED
        state = "completed" if success else "failed"
        _update_status_sync(
            redis_url, task_id, state,
            progress=1.0,
            errors=errors if errors else None,
            redis_runtime=redis_runtime,
        )

        # 9. 发布完成事件
        _publish_event_sync(redis_url, task_id, {
            "type": "task_complete" if success else "task_error",
            "status": "success" if success else "failed",
            "duration": duration,
            "token_usage": token_usage,
            "error": errors[0] if errors else None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }, redis_runtime)

        # The terminal event is already visible to the user. Memory writeback is
        # internal post-processing and never publishes progress events.
        try:
            if orchestrator is not None:
                asyncio.run(orchestrator.persist_memory(
                final_state,
                user_request=query,
                task_id=task_id,
                success=success,
                completion_errors=errors,
                ))
        except Exception:
            logger.warning("Background memory writeback failed for %s", task_id, exc_info=True)

        logger.info("Task %s completed (success=%s, duration=%.1fs)", task_id, success, duration)
        return report

    except Exception as exc:
        error_msg = f"{type(exc).__name__}: {exc}"
        logger.error("Task %s failed: %s", task_id, error_msg)
        traceback.print_exc()

        # 更新状态为 FAILED
        _update_status_sync(
            redis_url, task_id, "failed",
            progress=1.0,
            errors=[error_msg],
            redis_runtime=redis_runtime,
        )

        # 发布失败事件
        _publish_event_sync(redis_url, task_id, {
            "type": "task_error",
            "error": error_msg,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }, redis_runtime)

        # 写入失败报告
        try:
            _save_report_sync(
                redis_url,
                task_id,
                {
                    "task_id": task_id,
                    "status": "failed",
                    "recovered_from_task_id": recovered_from_task_id,
                    "plan": [],
                    "changes": [],
                    "validation_results": [],
                    "duration": 0.0,
                    "token_usage": 0,
                    "error": error_msg,
                    "model_runtime": llm.snapshot() if llm is not None else {},
                    "infrastructure_runtime": {
                        "redis": dict(redis_runtime),
                        "tools": (
                            tool_gateway.resilience_status()
                            if tool_gateway is not None else {}
                        ),
                    },
                },
                redis_runtime,
            )
        except Exception:
            pass

        raise
