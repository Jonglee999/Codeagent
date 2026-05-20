"""Celery Worker 入口 — 异步执行 Agent 工作流。

在 Worker 进程中执行 Agent 任务，并通过 Redis Pub/Sub 发布执行事件。
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import traceback
from datetime import datetime, timezone
from typing import Any

import redis as sync_redis

from celery import Celery

from codeagent.config import (
    get_env,
    get_redis_url,
    load_env_file,
)
from codeagent.interaction.api.metrics import wrap_llm_call

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


def _publish_event_sync(redis_url: str, task_id: str, event: dict) -> None:
    """同步发布事件到 Redis，并持久化到 List 供历史回放。"""
    # 统一 timestamp 为 ISO 8601 字符串
    if "timestamp" not in event or not isinstance(event["timestamp"], str):
        event["timestamp"] = datetime.now(timezone.utc).isoformat()
    try:
        r = sync_redis.from_url(redis_url)
        payload = json.dumps(event, ensure_ascii=False)
        channel = _EVENTS_CHANNEL.format(task_id=task_id)
        log_key = f"task:{task_id}:event_log"
        pipe = r.pipeline()
        pipe.publish(channel, payload)
        pipe.rpush(log_key, payload)
        pipe.expire(log_key, 3600)
        pipe.execute()
    except Exception as exc:
        logger.warning("Failed to publish event for task %s: %s", task_id, exc)


def _update_status_sync(
    redis_url: str, task_id: str, state: str, *,
    progress: float = 0.0,
    current_step: str | None = None,
    errors: list[str] | None = None,
) -> None:
    """同步更新 Redis 中的任务状态。"""
    try:
        r = sync_redis.from_url(redis_url)
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
    except Exception as exc:
        logger.warning("Failed to update status for task %s: %s", task_id, exc)


# ── 组件构建（参考 CLI main.py） ──────────────────────────────


def _build_llm(model_name: str) -> Any:
    """构建 LLM 调用函数。"""
    api_key = get_env("LLM_API_KEY", "")
    api_base = get_env("LLM_API_BASE", "")
    timeout = int(get_env("LLM_TIMEOUT", "60"))

    import litellm
    litellm.set_verbose = False

    async def llm_call(**kwargs: Any) -> Any:
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                call_kwargs: dict[str, Any] = {
                    **{k: v for k, v in kwargs.items() if v is not None},
                    "timeout": timeout,
                }
                if api_key:
                    call_kwargs["api_key"] = api_key
                if api_base:
                    call_kwargs["api_base"] = api_base
                return await litellm.acompletion(**call_kwargs)
            except Exception as e:
                last_exc = e
                if attempt < 2:
                    await asyncio.sleep(1.5 * (attempt + 1))
        raise last_exc  # type: ignore[misc]

    return wrap_llm_call(model_name, llm_call)


def _build_tool_gateway(project_root: str) -> Any:
    """构建工具 Gateway。"""
    from codeagent.tools.gateway import ToolGateway
    from codeagent.tools.terminal.run_terminal import RunTerminalTool

    gateway = ToolGateway(project_root=project_root)
    try:
        gateway._registry.register(RunTerminalTool(project_root=project_root))
    except Exception:
        logger.warning("RunTerminalTool not available (Docker may not be installed)")
    return gateway


def _build_validation_gateway(project_root: str = "") -> Any:
    """构建验证 Gateway。"""
    from codeagent.gateway.validation_gateway_impl import ValidationGateway
    return ValidationGateway()


def _build_context_gateway(project_root: str) -> Any:
    """构建上下文 Gateway。"""
    from codeagent.context_engine.engine import ContextEngine, ContextConfig

    budget = int(get_env("CONTEXT_BUDGET_TOKENS", "8000"))
    config_ctx = ContextConfig(total_budget=budget)
    return ContextEngine(config=config_ctx)


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

    logger.info("Worker starting task %s: %r", task_id, query[:80])

    # 1. 更新状态为 RUNNING
    _update_status_sync(redis_url, task_id, "running", progress=0.0)

    # 2. 发布 task_started 事件
    _publish_event_sync(redis_url, task_id, {
        "type": "node_start",
        "node": "worker",
        "summary": query[:200],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

    try:
        # 3. 构建组件
        llm = _build_llm(get_env("LLM_MODEL", "deepseek/deepseek-v4-flash"))
        tool_gateway = _build_tool_gateway(project_root)
        context_gateway = _build_context_gateway(project_root)
        validation_gateway = _build_validation_gateway(project_root)

        # 创建事件发布回调
        def make_event_callback(tid: str) -> Any:
            """创建进度回调，将进度 dict 发布为 Redis 事件。"""
            def _publish_progress(progress: dict) -> None:
                """将进度 dict 发布为 Redis 事件。"""
                if "timestamp" not in progress:
                    progress["timestamp"] = time.time()
                _publish_event_sync(redis_url, tid, progress)

            return _publish_progress

        progress_callback = make_event_callback(task_id)

        # 4. 构建 Orchestrator
        from codeagent.orchestration.orchestrator import Orchestrator

        orchestrator = Orchestrator(
            context_gateway=context_gateway,
            tool_gateway=tool_gateway,
            validation_gateway=validation_gateway,
            llm=llm,
            model_name=get_env("LLM_MODEL", "deepseek/deepseek-v4-flash"),
            progress_callback=progress_callback,
        )

        # 5. 异步执行
        async def _run_async() -> Any:
            result = await orchestrator.run(
                query,
                project_root,
                auto_mode=auto_mode,
            )
            return result

        final_state = asyncio.run(_run_async())

        # 6. 构建报告
        duration = time.time() - self.request.received_timestamp if hasattr(self.request, 'received_timestamp') else 0.0
        token_usage = final_state.estimated_tokens if hasattr(final_state, "estimated_tokens") else 0
        errors = final_state.errors
        success = len(errors) == 0

        def _to_dict(obj: Any) -> Any:
            """递归将 dataclass 转换为 dict（确保 JSON 可序列化）。"""
            if hasattr(obj, "__dataclass_fields__"):
                return {f: _to_dict(getattr(obj, f)) for f in obj.__dataclass_fields__}
            if isinstance(obj, (list, tuple)):
                return [_to_dict(i) for i in obj]
            if isinstance(obj, dict):
                return {k: _to_dict(v) for k, v in obj.items()}
            return obj

        report = {
            "task_id": task_id,
            "plan": [_to_dict(s) for s in (final_state.plan or [])],
            "changes": _to_dict(final_state.accumulated_changes) if hasattr(final_state, "accumulated_changes") else [],
            "validation_results": _to_dict(final_state.validation_results or []),
            "duration": duration,
            "token_usage": token_usage,
        }

        # 7. 写入报告
        r = sync_redis.from_url(redis_url)
        r.setex(
            _REPORT_KEY.format(task_id=task_id),
            3600,
            json.dumps(report, ensure_ascii=False),
        )

        # 8. 更新状态为 COMPLETED/FAILED
        state = "completed" if success else "failed"
        _update_status_sync(
            redis_url, task_id, state,
            progress=1.0,
            errors=errors if errors else None,
        )

        # 9. 发布完成事件
        _publish_event_sync(redis_url, task_id, {
            "type": "task_complete" if success else "task_error",
            "status": "success" if success else "failed",
            "duration": duration,
            "token_usage": token_usage,
            "error": errors[0] if errors else None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

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
        )

        # 发布失败事件
        _publish_event_sync(redis_url, task_id, {
            "type": "task_error",
            "error": error_msg,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        # 写入失败报告
        try:
            r = sync_redis.from_url(redis_url)
            r.setex(
                _REPORT_KEY.format(task_id=task_id),
                3600,
                json.dumps({
                    "task_id": task_id,
                    "plan": [],
                    "changes": [],
                    "validation_results": [],
                    "duration": 0.0,
                    "token_usage": 0,
                    "error": error_msg,
                }, ensure_ascii=False),
            )
        except Exception:
            pass

        raise
