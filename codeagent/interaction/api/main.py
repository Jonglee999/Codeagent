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
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from codeagent.config import get_env, get_redis_url, load_env_file

load_env_file(Path.cwd() / ".env")
logger = logging.getLogger(__name__)

# ── Inline Runner（仅当 USE_INLINE_RUNNER=true 时使用） ──────────

_INLINE_MODE = get_env("USE_INLINE_RUNNER", "false").lower() in ("true", "1", "yes")

_STATUS_KEY = "task:{task_id}:status"
_EVENTS_CHANNEL = "task:{task_id}:events"
_REPORT_KEY = "task:{task_id}:report"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


async def _publish_event(redis_url: str, task_id: str, event: dict) -> None:
    """向 Redis 发布事件并持久化到 List。"""
    import redis.asyncio as aioredis

    if "timestamp" not in event or not isinstance(event.get("timestamp"), str):
        event["timestamp"] = _now_iso()

    payload = json.dumps(event, ensure_ascii=False)
    channel = _EVENTS_CHANNEL.format(task_id=task_id)
    log_key = f"task:{task_id}:event_log"

    r = aioredis.from_url(redis_url)
    try:
        pipe = r.pipeline()
        pipe.publish(channel, payload)
        pipe.rpush(log_key, payload)
        pipe.expire(log_key, 3600)
        await pipe.execute()
    finally:
        await r.aclose()


async def _update_status(
    redis_url: str, task_id: str, state: str,
    progress: float = 0.0, current_step: str | None = None, errors: list[str] | None = None,
) -> None:
    import redis.asyncio as aioredis

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


async def _run_inline(task_id: str, request_dict: dict, redis_url: str) -> None:
    """在 API 进程的 asyncio 中直接运行 Agent，无需 Celery Worker。"""
    import litellm

    query = request_dict.get("query", "")
    project_root = request_dict.get("project_root", "")
    auto_mode = request_dict.get("auto_mode", False)

    logger.info("Inline runner starting task %s: %r", task_id, query[:80])
    await _update_status(redis_url, task_id, "running", progress=0.0)
    await _publish_event(redis_url, task_id, {
        "type": "node_start",
        "node": "worker",
        "summary": query[:200],
        "timestamp": _now_iso(),
    })

    t0 = time.monotonic()

    try:
        api_key = get_env("LLM_API_KEY", "")
        api_base = get_env("LLM_API_BASE", "")
        timeout = int(get_env("LLM_TIMEOUT", "60"))
        model_name = get_env("LLM_MODEL", "deepseek/deepseek-v4-flash")

        litellm.set_verbose = False

        async def llm_call(**kwargs):
            call_kwargs = {**{k: v for k, v in kwargs.items() if v is not None}, "timeout": timeout}
            if api_key:
                call_kwargs["api_key"] = api_key
            if api_base:
                call_kwargs["api_base"] = api_base
            return await litellm.acompletion(**call_kwargs)

        from codeagent.tools.gateway import ToolGateway
        from codeagent.tools.terminal.run_terminal import RunTerminalTool

        tool_gateway = ToolGateway(project_root=project_root)
        try:
            tool_gateway._registry.register(RunTerminalTool(project_root=project_root))
        except Exception:
            pass

        from codeagent.gateway.validation_gateway_impl import ValidationGateway
        validation_gateway = ValidationGateway()

        from codeagent.context_engine.engine import ContextEngine, ContextConfig
        budget = int(get_env("CONTEXT_BUDGET_TOKENS", "8000"))
        context_gateway = ContextEngine(config=ContextConfig(total_budget=budget))

        def progress_callback(progress: dict) -> None:
            progress.setdefault("timestamp", _now_iso())
            asyncio.create_task(_publish_event(redis_url, task_id, progress))

        from codeagent.orchestration.orchestrator import Orchestrator
        orchestrator = Orchestrator(
            context_gateway=context_gateway,
            tool_gateway=tool_gateway,
            validation_gateway=validation_gateway,
            llm=llm_call,
            model_name=model_name,
            progress_callback=progress_callback,
        )

        final_state = await orchestrator.run(query, project_root, auto_mode=auto_mode)

        duration = time.monotonic() - t0
        errors = final_state.errors
        success = len(errors) == 0

        def _to_dict(obj):
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
            "token_usage": final_state.estimated_tokens if hasattr(final_state, "estimated_tokens") else 0,
        }

        import redis.asyncio as aioredis
        r = aioredis.from_url(redis_url)
        try:
            await r.setex(_REPORT_KEY.format(task_id=task_id), 3600, json.dumps(report, ensure_ascii=False))
        finally:
            await r.aclose()

        await _update_status(redis_url, task_id, "completed" if success else "failed", progress=1.0, errors=errors or None)
        await _publish_event(redis_url, task_id, {
            "type": "task_complete" if success else "task_error",
            "status": "success" if success else "failed",
            "duration": duration,
            "token_usage": report["token_usage"],
            "error": errors[0] if errors else None,
            "timestamp": _now_iso(),
        })

    except Exception as exc:
        error_msg = f"{type(exc).__name__}: {exc}"
        logger.error("Inline task %s failed: %s", task_id, error_msg)
        traceback.print_exc()
        await _update_status(redis_url, task_id, "failed", progress=1.0, errors=[error_msg])
        await _publish_event(redis_url, task_id, {
            "type": "task_error",
            "error": error_msg,
            "timestamp": _now_iso(),
        })


# ── App setup ─────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    from codeagent.gateway.orchestration_gateway_impl import OrchestrationGatewayImpl

    gateway = OrchestrationGatewayImpl(redis_url=get_redis_url())
    app.state.gateway = gateway
    app.state.inline_mode = _INLINE_MODE
    app.state.redis_url = get_redis_url()

    if _INLINE_MODE:
        logger.info("CodeAgent API starting in INLINE mode (no Celery Worker needed)")
    else:
        logger.info("CodeAgent API starting in Celery mode")

    yield


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
        "redis": get_redis_url().split("@")[-1] if "@" in get_redis_url() else get_redis_url(),
    }


from .routes import router as api_router
from .websocket import ws_router

# 注入 inline runner 到 routes
app.state.inline_runner = _run_inline if _INLINE_MODE else None

app.include_router(api_router)
app.include_router(ws_router)
