"""FastAPI 应用入口。

支持两种运行模式：
- Celery 模式（默认）：任务由 Celery Worker 异步执行
- Inline 模式（USE_INLINE_RUNNER=true）：任务直接在 API 进程的 asyncio 后台任务中执行
  适用于本地开发，不需要单独启动 Celery Worker
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from codeagent.config import (
    get_a2a_api_keys,
    get_a2a_max_message_bytes,
    get_a2a_server_enabled,
    get_checkpoint_enabled,
    get_env,
    get_redis_url,
    load_env_file,
)
from codeagent.chat_response import request_visible_chat
from codeagent.interaction.api.metrics import get_metrics_endpoint
from codeagent.product_state import product_state_store
from codeagent.orchestration.intent import classify_response_mode

load_env_file(Path.cwd() / ".env")
logger = logging.getLogger(__name__)

# ── Inline Runner（仅当 USE_INLINE_RUNNER=true 时使用） ──────────

_INLINE_MODE = get_env("USE_INLINE_RUNNER", "false").lower() in ("true", "1", "yes")

_STATUS_KEY = "task:{task_id}:status"
_EVENTS_CHANNEL = "task:{task_id}:events"
_REPORT_KEY = "task:{task_id}:report"
_BACKGROUND_TASKS: set[asyncio.Task] = set()
_CHECKPOINTER = None


def _schedule_background(coroutine: object, *, label: str) -> asyncio.Task:
    """Track an internal coroutine without exposing it in the task event stream."""
    task = asyncio.create_task(coroutine)  # type: ignore[arg-type]
    _BACKGROUND_TASKS.add(task)

    def finished(completed: asyncio.Task) -> None:
        _BACKGROUND_TASKS.discard(completed)
        if completed.cancelled():
            return
        try:
            completed.result()
        except Exception:
            logger.warning("Background operation failed: %s", label, exc_info=True)

    task.add_done_callback(finished)
    return task


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _classify_response_mode(query: str, requested_mode: str) -> str:
    """Backward-compatible wrapper around the shared runner classifier."""
    return classify_response_mode(query, requested_mode)


def _execution_response(final_state: object, success: bool, errors: list[str]) -> str:
    execution_log = getattr(final_state, "execution_log", []) or []
    for entry in reversed(execution_log):
        if entry.get("type") == "llm_response" and entry.get("content"):
            content = str(entry["content"]).strip()
            if content:
                return content[:6000]

    changes = getattr(final_state, "accumulated_changes", []) or []
    validations = getattr(final_state, "validation_results", []) or []
    if success:
        parts = ["任务已完成。"]
        if changes:
            parts.append(f"已记录 {len(changes)} 项文件变更。")
        if validations:
            passed = sum(bool(getattr(item, "passed", False)) for item in validations)
            parts.append(f"验证 {passed}/{len(validations)} 项通过。")
        return " ".join(parts)
    detail = "；".join(errors[:3]) if errors else "执行或验证未通过"
    return f"任务未能完成：{detail}"


async def _publish_event(
    redis_url: str,
    task_id: str,
    event: dict,
    redis_runtime: dict[str, int] | None = None,
    *,
    _emit_recovery: bool = True,
) -> None:
    """向 Redis 发布事件并持久化到 List。"""
    import redis.asyncio as aioredis

    if "timestamp" not in event or not isinstance(event.get("timestamp"), str):
        event["timestamp"] = _now_iso()
    event.setdefault("event_id", uuid.uuid4().hex)
    event.setdefault("schema_version", 1)
    channel = _EVENTS_CHANNEL.format(task_id=task_id)
    log_key = f"task:{task_id}:event_log"
    sequence_key = f"task:{task_id}:event_seq"

    from codeagent.redis_resilience import retry_redis_async

    async def operation() -> None:
        r = aioredis.from_url(redis_url)
        try:
            if "seq" not in event:
                event["seq"] = int(await r.incr(sequence_key))
            payload = json.dumps(event, ensure_ascii=False)
            pipe = r.pipeline()
            pipe.publish(channel, payload)
            pipe.rpush(log_key, payload)
            pipe.expire(log_key, 3600)
            await pipe.execute()
        finally:
            await r.aclose()

    try:
        _, retries = await retry_redis_async(operation)
        if retries and redis_runtime is not None:
            redis_runtime["retry_count"] = redis_runtime.get("retry_count", 0) + retries
            redis_runtime["recovery_count"] = redis_runtime.get("recovery_count", 0) + 1
        if retries and _emit_recovery:
            await _publish_event(
                redis_url,
                task_id,
                {
                    "type": "infrastructure_recovered",
                    "summary": f"Redis event stream recovered after {retries} retries",
                    "data": {"capability": "redis", "retry_count": retries},
                },
                redis_runtime,
                _emit_recovery=False,
            )
    except Exception:
        if redis_runtime is not None:
            redis_runtime["dropped_event_count"] = redis_runtime.get("dropped_event_count", 0) + 1
        logger.warning("Could not publish task event after Redis retries", exc_info=True)


async def _update_status(
    redis_url: str, task_id: str, state: str,
    progress: float = 0.0, current_step: str | None = None, errors: list[str] | None = None,
    redis_runtime: dict[str, int] | None = None,
) -> None:
    import redis.asyncio as aioredis

    from codeagent.redis_resilience import retry_redis_async

    async def operation() -> None:
        r = aioredis.from_url(redis_url)
        try:
            raw = await r.get(_STATUS_KEY.format(task_id=task_id))
            status_data = json.loads(raw) if raw else {
                "task_id": task_id, "state": "pending", "progress": 0.0,
                "current_step": None, "errors": [],
            }
            status_data["state"] = state
            status_data["progress"] = progress
            if current_step is not None:
                status_data["current_step"] = current_step
            if errors is not None:
                status_data["errors"] = errors
            await r.setex(
                _STATUS_KEY.format(task_id=task_id),
                3600,
                json.dumps(status_data, ensure_ascii=False),
            )
        finally:
            await r.aclose()

    _, retries = await retry_redis_async(operation)
    product_state_store.update_run_state(task_id, state)
    if retries and redis_runtime is not None:
        redis_runtime["retry_count"] = redis_runtime.get("retry_count", 0) + retries
        redis_runtime["recovery_count"] = redis_runtime.get("recovery_count", 0) + 1


async def _save_report(
    redis_url: str,
    task_id: str,
    report: dict,
    redis_runtime: dict[str, int] | None = None,
) -> None:
    import redis.asyncio as aioredis

    from codeagent.redis_resilience import retry_redis_async

    async def operation() -> None:
        r = aioredis.from_url(redis_url)
        try:
            await r.setex(
                _REPORT_KEY.format(task_id=task_id),
                3600,
                json.dumps(report, ensure_ascii=False),
            )
        finally:
            await r.aclose()

    _, retries = await retry_redis_async(operation)
    product_state_store.save_report(task_id, report)
    if retries and redis_runtime is not None:
        redis_runtime["retry_count"] = redis_runtime.get("retry_count", 0) + retries
        redis_runtime["recovery_count"] = redis_runtime.get("recovery_count", 0) + 1


async def _run_inline(task_id: str, request_dict: dict, redis_url: str) -> None:
    """在 API 进程的 asyncio 中直接运行 Agent，无需 Celery Worker。"""
    query = request_dict.get("query", "")
    project_root = request_dict.get("project_root", "")
    auto_mode = request_dict.get("auto_mode", False)
    max_retries = int(request_dict.get("max_retries", 3))
    conversation_history = request_dict.get("conversation_history", [])
    requested_mode = request_dict.get("response_mode", "auto")
    response_mode = _classify_response_mode(query, requested_mode)
    direct_execution = bool(request_dict.get("direct_execution", False))
    benchmark_instance_id = request_dict.get("benchmark_instance_id")
    benchmark_fail_to_pass = request_dict.get("benchmark_fail_to_pass", [])
    benchmark_pass_to_pass = request_dict.get("benchmark_pass_to_pass", [])
    recovered_from_task_id = request_dict.get("recovered_from_task_id")
    redis_runtime = {"retry_count": 0, "recovery_count": 0, "dropped_event_count": 0}

    logger.info("Inline runner starting task %s: %r", task_id, query[:80])
    await _update_status(redis_url, task_id, "running", progress=0.0, redis_runtime=redis_runtime)
    await _publish_event(redis_url, task_id, {
        "type": "node_start",
        "node": "worker",
        "summary": query[:200],
        "timestamp": _now_iso(),
    }, redis_runtime)

    t0 = time.monotonic()
    event_tasks: set[asyncio.Task] = set()
    tool_gateway = None
    model_gateway = None

    try:
        from codeagent.model_routing import ModelRouter

        async def model_event(event: dict) -> None:
            event.setdefault("timestamp", _now_iso())
            await _publish_event(redis_url, task_id, event, redis_runtime)

        model_gateway = ModelRouter.from_env(event_callback=model_event)
        model_name = model_gateway.primary.model

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
            await _publish_event(redis_url, task_id, {
                "type": "node_start", "node": "conversation",
                "summary": "正在生成回复", "timestamp": _now_iso(),
            }, redis_runtime)
            chat_response = await request_visible_chat(
                model_gateway,
                model=model_name,
                messages=messages,
                model_role="chat",
                routing_context={"request": query},
            )
            assistant_response = chat_response.text
            duration = time.monotonic() - t0
            from codeagent.memory.transcript import persist_transcript

            chat_entries = [
                {"kind": "message", "role": item.get("role"), "content": item.get("content", "")}
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
                "model_runtime": model_gateway.snapshot(),
                "infrastructure_runtime": {"redis": dict(redis_runtime)},
            }
            await _save_report(redis_url, task_id, report, redis_runtime)
            await _publish_event(redis_url, task_id, {
                "type": "assistant_message",
                "content": assistant_response,
                "timestamp": _now_iso(),
            }, redis_runtime)
            await _update_status(redis_url, task_id, "completed", progress=1.0, redis_runtime=redis_runtime)
            await _publish_event(redis_url, task_id, {
                "type": "task_complete", "status": "success",
                "duration": duration, "token_usage": report["token_usage"],
                "timestamp": _now_iso(),
            }, redis_runtime)
            from codeagent.orchestration.policy import requests_durable_memory

            if requests_durable_memory(query):
                from codeagent.orchestration.runtime import build_memory_service

                memory = build_memory_service(project_root, model_gateway, auto_mode=True)
                if memory is not None:
                    _schedule_background(
                        memory.auto_extract(
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
                        ),
                        label=f"chat memory writeback for {task_id}",
                    )
            return

        from codeagent.context_engine.engine import ContextEngine, ContextConfig
        context_gateway = ContextEngine(config=ContextConfig.from_env())

        from codeagent.tools.gateway import ToolGateway

        tool_gateway = ToolGateway(
            project_root=project_root,
            allow_high_risk_extensions=auto_mode,
            context_engine=context_gateway,
        )

        from codeagent.gateway.validation_gateway_impl import ValidationGateway
        validation_gateway = ValidationGateway(project_root=project_root)

        def progress_callback(progress: dict) -> asyncio.Task:
            progress.setdefault("timestamp", _now_iso())
            task = asyncio.create_task(_publish_event(redis_url, task_id, progress, redis_runtime))
            event_tasks.add(task)
            task.add_done_callback(event_tasks.discard)
            return task

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

        from codeagent.orchestration.orchestrator import Orchestrator
        from codeagent.orchestration.runtime import (
            assess_task_completion,
            build_execution_report,
            build_learning_services,
            persist_run_artifacts,
        )
        memory_gateway, evolution_manager, strategy_applier = build_learning_services(
            project_root=project_root,
            llm_client=model_gateway,
            auto_mode=auto_mode,
        )
        from codeagent.task_control import consume_steering

        async def steering_provider() -> list[str]:
            return await consume_steering(redis_url, task_id)

        orchestrator = Orchestrator(
            context_gateway=context_gateway,
            tool_gateway=tool_gateway,
            validation_gateway=validation_gateway,
            llm=model_gateway,
            model_name=model_name,
            progress_callback=progress_callback,
            memory_gateway=memory_gateway,
            evolution_manager=evolution_manager,
            strategy_applier=strategy_applier,
            max_retries=max_retries,
            steering_provider=steering_provider,
            checkpointer=_CHECKPOINTER,
        )

        final_state = await orchestrator.run(
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
        if event_tasks:
            await asyncio.gather(*tuple(event_tasks), return_exceptions=True)

        duration = time.monotonic() - t0
        success, errors = assess_task_completion(
            final_state,
            require_patch=bool(benchmark_instance_id),
            require_test_evidence=bool(benchmark_instance_id),
        )

        extension_runtime = tool_gateway.extension_status()["mcp"]
        report = build_execution_report(
            task_id,
            final_state,
            duration=duration,
            success=success,
            errors=errors,
            extension_warnings=extension_runtime.get("warnings", []),
            mcp_servers=extension_runtime.get("servers", []),
            model_runtime=model_gateway.snapshot(),
            infrastructure_runtime={
                "redis": dict(redis_runtime),
                "tools": tool_gateway.resilience_status(),
            },
        )
        persist_run_artifacts(project_root, task_id, report)
        await _save_report(redis_url, task_id, report, redis_runtime)

        await _publish_event(redis_url, task_id, {
            "type": "assistant_message",
            "content": report["assistant_response"],
            "timestamp": _now_iso(),
        }, redis_runtime)

        await _update_status(
            redis_url,
            task_id,
            "completed" if success else "failed",
            progress=1.0,
            errors=errors or None,
            redis_runtime=redis_runtime,
        )
        await _publish_event(redis_url, task_id, {
            "type": "task_complete" if success else "task_error",
            "status": "success" if success else "failed",
            "duration": duration,
            "token_usage": report["token_usage"],
            "error": errors[0] if errors else None,
            "timestamp": _now_iso(),
        }, redis_runtime)
        _schedule_background(
            orchestrator.persist_memory(
                final_state,
                user_request=query,
                task_id=task_id,
                success=success,
                completion_errors=errors,
            ),
            label=f"memory writeback for {task_id}",
        )

    except asyncio.CancelledError:
        logger.info("Inline task %s cancelled", task_id)
        for event_task in tuple(event_tasks):
            event_task.cancel()
        if event_tasks:
            await asyncio.gather(*tuple(event_tasks), return_exceptions=True)
        raise
    except Exception as exc:
        error_msg = f"{type(exc).__name__}: {exc}"
        logger.error("Inline task %s failed: %s", task_id, error_msg)
        traceback.print_exc()
        failure_report = {
            "task_id": task_id,
            "status": "failed",
            "recovered_from_task_id": recovered_from_task_id,
            "plan": [],
            "changes": [],
            "validation_results": [],
            "duration": time.monotonic() - t0,
            "token_usage": 0,
            "assistant_response": "",
            "response_mode": response_mode,
            "memory_hits": [],
            "resolved_skills": [],
            "warnings": [error_msg],
            "reflection": None,
            "transcript_path": None,
            "mcp_servers": (
                tool_gateway.extension_status()["mcp"].get("servers", [])
                if tool_gateway is not None else []
            ),
            "model_runtime": model_gateway.snapshot() if model_gateway is not None else {},
            "infrastructure_runtime": {
                "redis": dict(redis_runtime),
                "tools": tool_gateway.resilience_status() if tool_gateway is not None else {},
            },
        }
        try:
            await _save_report(redis_url, task_id, failure_report, redis_runtime)
        except Exception:
            logger.warning("Could not persist failure report for task %s", task_id, exc_info=True)
        await _update_status(
            redis_url,
            task_id,
            "failed",
            progress=1.0,
            errors=[error_msg],
            redis_runtime=redis_runtime,
        )
        await _publish_event(redis_url, task_id, {
            "type": "task_error",
            "error": error_msg,
            "timestamp": _now_iso(),
        }, redis_runtime)
    finally:
        if tool_gateway is not None:
            await tool_gateway.aclose()


# ── App setup ─────────────────────────────────────────────────


async def resume_inline_checkpoint(task_id: str, decision: str, redis_url: str) -> None:
    """Rebuild runtime dependencies and continue a persisted interrupted graph."""
    run = product_state_store.get_run(task_id)
    if run is None:
        raise ValueError(f"Task not found: {task_id}")
    project_root = str(run["workspace_root"])
    query = str(run["query"])
    started = time.monotonic()
    redis_runtime = {"retry_count": 0, "recovery_count": 1, "dropped_event_count": 0}
    tool_gateway = None
    try:
        from codeagent.context_engine.engine import ContextConfig, ContextEngine
        from codeagent.gateway.validation_gateway_impl import ValidationGateway
        from codeagent.model_routing import ModelRouter
        from codeagent.orchestration.orchestrator import Orchestrator
        from codeagent.orchestration.runtime import (
            assess_task_completion,
            build_execution_report,
            build_learning_services,
            persist_run_artifacts,
        )
        from codeagent.tools.gateway import ToolGateway

        async def model_event(event: dict) -> None:
            event.setdefault("timestamp", _now_iso())
            await _publish_event(redis_url, task_id, event, redis_runtime)

        model_gateway = ModelRouter.from_env(event_callback=model_event)
        context_gateway = ContextEngine(config=ContextConfig.from_env())
        tool_gateway = ToolGateway(project_root=project_root, context_engine=context_gateway)
        validation_gateway = ValidationGateway(project_root=project_root)

        def progress_callback(progress: dict) -> asyncio.Task:
            progress.setdefault("timestamp", _now_iso())
            return asyncio.create_task(
                _publish_event(redis_url, task_id, progress, redis_runtime)
            )

        await tool_gateway.initialize_extensions(progress_callback, query=query)
        memory_gateway, evolution_manager, strategy_applier = build_learning_services(
            project_root=project_root, llm_client=model_gateway, auto_mode=False
        )
        orchestrator = Orchestrator(
            context_gateway=context_gateway,
            tool_gateway=tool_gateway,
            validation_gateway=validation_gateway,
            llm=model_gateway,
            model_name=model_gateway.primary.model,
            progress_callback=progress_callback,
            memory_gateway=memory_gateway,
            evolution_manager=evolution_manager,
            strategy_applier=strategy_applier,
            checkpointer=_CHECKPOINTER,
        )
        await _update_status(
            redis_url, task_id, "running", progress=0.5, redis_runtime=redis_runtime
        )
        final_state = await orchestrator.resume(task_id, decision)
        success, errors = assess_task_completion(final_state)
        report = build_execution_report(
            task_id,
            final_state,
            duration=time.monotonic() - started,
            success=success,
            errors=errors,
            model_runtime=model_gateway.snapshot(),
            infrastructure_runtime={
                "redis": redis_runtime,
                "checkpoint": {"resumed": True, "thread_id": task_id},
            },
        )
        persist_run_artifacts(project_root, task_id, report)
        await _save_report(redis_url, task_id, report, redis_runtime)
        state = "completed" if success else "failed"
        await _update_status(
            redis_url,
            task_id,
            state,
            progress=1.0,
            errors=errors or None,
            redis_runtime=redis_runtime,
        )
        await _publish_event(redis_url, task_id, {
            "type": "task_complete" if success else "task_error",
            "status": "success" if success else "failed",
            "timestamp": _now_iso(),
            "checkpoint_resumed": True,
        }, redis_runtime)
    except Exception as exc:
        logger.exception("Failed to resume checkpoint for task %s", task_id)
        await _update_status(
            redis_url,
            task_id,
            "failed",
            progress=1.0,
            errors=[str(exc)],
            redis_runtime=redis_runtime,
        )
        await _publish_event(redis_url, task_id, {
            "type": "task_error",
            "error": str(exc),
            "timestamp": _now_iso(),
            "checkpoint_resumed": True,
        }, redis_runtime)
    finally:
        if tool_gateway is not None:
            await tool_gateway.aclose()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _CHECKPOINTER
    from codeagent.gateway.orchestration_gateway_impl import OrchestrationGatewayImpl

    gateway = OrchestrationGatewayImpl(redis_url=get_redis_url())
    app.state.gateway = gateway
    app.state.inline_mode = _INLINE_MODE
    app.state.redis_url = get_redis_url()
    app.state.inline_tasks = {}
    app.state.resume_inline_checkpoint = None

    a2a_executor = getattr(app.state, "a2a_executor", None)
    if a2a_executor is not None:
        async def start_a2a_inline(task_id: str, request_dict: dict) -> None:
            task = asyncio.create_task(_run_inline(task_id, request_dict, get_redis_url()))
            app.state.inline_tasks[task_id] = task
            task.add_done_callback(lambda _done: app.state.inline_tasks.pop(task_id, None))

        a2a_executor.configure(gateway, start_a2a_inline if _INLINE_MODE else None)

    if get_checkpoint_enabled():
        from codeagent.orchestration.checkpoint import get_async_checkpointer

        _CHECKPOINTER = await get_async_checkpointer()
        app.state.checkpointer = _CHECKPOINTER
        if _INLINE_MODE:
            app.state.resume_inline_checkpoint = resume_inline_checkpoint

    if _INLINE_MODE:
        logger.info("CodeAgent API starting in INLINE mode (no Celery Worker needed)")
    else:
        logger.info("CodeAgent API starting in Celery mode")

    try:
        yield
    finally:
        inline_tasks = list(app.state.inline_tasks.values())
        for task in inline_tasks:
            task.cancel()
        if inline_tasks:
            await asyncio.gather(*inline_tasks, return_exceptions=True)
        background_tasks = list(_BACKGROUND_TASKS)
        if background_tasks:
            _done, pending = await asyncio.wait(background_tasks, timeout=5)
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
        if _CHECKPOINTER is not None and hasattr(_CHECKPOINTER, "conn"):
            await _CHECKPOINTER.conn.close()
            _CHECKPOINTER = None


app = FastAPI(
    title="CodeAgent API",
    description="AI-powered coding assistant REST API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "inline_mode": _INLINE_MODE,
        "llm_configured": bool(get_env("LLM_API_KEY", "")),
        "model": get_env("LLM_MODEL", "deepseek/deepseek-v4-flash"),
        "redis": get_redis_url().split("@")[-1] if "@" in get_redis_url() else get_redis_url(),
    }


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint (text/plain; charset=utf-8)."""
    from starlette.responses import Response
    return Response(content=get_metrics_endpoint(), media_type="text/plain; charset=utf-8")


from .routes import router as api_router  # noqa: E402
from .websocket import ws_router  # noqa: E402

# 注入 inline runner 到 routes
app.state.inline_runner = _run_inline if _INLINE_MODE else None

app.include_router(api_router)
app.include_router(ws_router)

if get_a2a_server_enabled():
    import hmac

    from fastapi import Request
    from starlette.responses import JSONResponse

    from codeagent.a2a.server import CodeAgentExecutor, install_a2a_routes

    _a2a_executor = CodeAgentExecutor()
    app.state.a2a_executor = _a2a_executor
    install_a2a_routes(app, _a2a_executor)

    @app.middleware("http")
    async def enforce_a2a_boundary(request: Request, call_next):
        if request.url.path == "/a2a":
            length = int(request.headers.get("content-length", "0") or 0)
            if length > get_a2a_max_message_bytes():
                return JSONResponse({"detail": "A2A request too large"}, status_code=413)
            body = await request.body()
            if len(body) > get_a2a_max_message_bytes():
                return JSONResponse({"detail": "A2A request too large"}, status_code=413)
            keys = get_a2a_api_keys()
            supplied = request.headers.get("authorization", "").removeprefix("Bearer ")
            if keys and not any(hmac.compare_digest(supplied, key) for key in keys):
                return JSONResponse({"detail": "Invalid A2A credentials"}, status_code=401)
        return await call_next(request)
