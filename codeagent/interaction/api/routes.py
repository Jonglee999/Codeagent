"""FastAPI REST routes — 任务生命周期管理端点。"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse

from codeagent.gateway.orchestration_gateway import (
    HumanDecision,
    UserRequest,
)
from codeagent.gateway.orchestration_gateway_impl import OrchestrationGatewayImpl

from .auth import verify_api_key
from .models import (
    ApiResponse,
    DecisionRequest,
    TaskCreateRequest,
    TaskReportResponse,
    TaskStatusResponse,
)
from .rate_limit import RateLimiter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1")


def _get_gateway(request: Request) -> OrchestrationGatewayImpl:
    """从 app.state 获取 Gateway 单例。"""
    gateway: OrchestrationGatewayImpl | None = getattr(
        request.app.state, "gateway", None
    )
    if gateway is None:
        raise HTTPException(status_code=503, detail="Gateway not initialized")
    return gateway


# 速率限制器（单例，使用 Redis URL）
_rate_limiter = RateLimiter()


@router.post(
    "/tasks", response_model=ApiResponse, status_code=status.HTTP_202_ACCEPTED
)
async def create_task(
    req: TaskCreateRequest,
    request: Request,
    gateway: OrchestrationGatewayImpl = Depends(_get_gateway),
    _auth: str = Depends(verify_api_key),
    _rl: None = Depends(_rate_limiter.check_rate_limit),
) -> ApiResponse:
    """提交新 Agent 任务。

    支持两种模式：
    - Celery 模式：通过 gateway.start_task 提交到 Celery 队列
    - Inline 模式：直接在当前 asyncio 事件循环中启动后台任务
    """
    try:
        user_req = UserRequest(
            query=req.query,
            project_root=req.project_root,
            auto_mode=req.auto_mode,
            max_retries=req.max_retries,
        )
        inline_runner = getattr(request.app.state, "inline_runner", None)
        task_id = await gateway.start_task(user_req, skip_celery=inline_runner is not None)

        # Inline 模式：直接在后台协程中运行 Agent
        if inline_runner is not None:
            redis_url = getattr(request.app.state, "redis_url", None)
            asyncio.create_task(
                inline_runner(task_id, {
                    "query": req.query,
                    "project_root": req.project_root,
                    "auto_mode": req.auto_mode,
                    "max_retries": req.max_retries,
                }, redis_url)
            )

        return ApiResponse(
            success=True, data={"task_id": task_id, "status": "pending"}
        )
    except Exception as exc:
        logger.exception("Failed to create task")
        return JSONResponse(
            status_code=500,
            content=ApiResponse(success=False, error=str(exc)).model_dump(),
        )


@router.get("/tasks/{task_id}", response_model=ApiResponse)
async def get_task_status(
    task_id: str,
    gateway: OrchestrationGatewayImpl = Depends(_get_gateway),
    _auth: str = Depends(verify_api_key),
    _rl: None = Depends(_rate_limiter.check_rate_limit),
) -> ApiResponse:
    """查询任务状态。"""
    try:
        status = await gateway.get_task_status(task_id)
        resp = TaskStatusResponse(
            task_id=status.task_id,
            state=status.state.value,
            progress=status.progress,
            current_step=status.current_step,
            errors=status.errors,
        )
        return ApiResponse(success=True, data=resp.model_dump())
    except KeyError:
        return JSONResponse(
            status_code=404,
            content=ApiResponse(
                success=False, error=f"Task not found: {task_id}"
            ).model_dump(),
        )
    except Exception as exc:
        logger.exception("Failed to get task status")
        return JSONResponse(
            status_code=500,
            content=ApiResponse(success=False, error=str(exc)).model_dump(),
        )


@router.post("/tasks/{task_id}/decision", response_model=ApiResponse)
async def submit_decision(
    task_id: str,
    body: DecisionRequest,
    gateway: OrchestrationGatewayImpl = Depends(_get_gateway),
    _auth: str = Depends(verify_api_key),
    _rl: None = Depends(_rate_limiter.check_rate_limit),
) -> ApiResponse:
    """提交 Human Review 决策。"""
    try:
        decision = HumanDecision(
            task_id=task_id,
            decision=body.decision,
            modifications=body.modifications,
        )
        await gateway.submit_decision(task_id, decision)
        return ApiResponse(
            success=True,
            data={"task_id": task_id, "decision": body.decision},
        )
    except Exception as exc:
        logger.exception("Failed to submit decision")
        return JSONResponse(
            status_code=500,
            content=ApiResponse(success=False, error=str(exc)).model_dump(),
        )


@router.delete("/tasks/{task_id}", response_model=ApiResponse)
async def cancel_task(
    task_id: str,
    gateway: OrchestrationGatewayImpl = Depends(_get_gateway),
    _auth: str = Depends(verify_api_key),
    _rl: None = Depends(_rate_limiter.check_rate_limit),
) -> ApiResponse:
    """取消任务。"""
    try:
        result = await gateway.cancel_task(task_id)
        return ApiResponse(success=True, data={"cancelled": result})
    except Exception as exc:
        logger.exception("Failed to cancel task")
        return JSONResponse(
            status_code=500,
            content=ApiResponse(success=False, error=str(exc)).model_dump(),
        )


@router.get("/tasks/{task_id}/report", response_model=ApiResponse)
async def get_report(
    task_id: str,
    gateway: OrchestrationGatewayImpl = Depends(_get_gateway),
    _auth: str = Depends(verify_api_key),
    _rl: None = Depends(_rate_limiter.check_rate_limit),
) -> ApiResponse:
    """获取任务完整报告。"""
    try:
        report = await gateway.get_report(task_id)
        resp = TaskReportResponse(
            task_id=report.task_id,
            plan=report.plan,
            changes=report.changes,
            validation_results=report.validation_results,
            duration=report.duration,
            token_usage=report.token_usage,
        )
        return ApiResponse(success=True, data=resp.model_dump())
    except KeyError:
        return JSONResponse(
            status_code=404,
            content=ApiResponse(
                success=False, error=f"Report not found for task: {task_id}"
            ).model_dump(),
        )
    except Exception as exc:
        logger.exception("Failed to get report")
        return JSONResponse(
            status_code=500,
            content=ApiResponse(success=False, error=str(exc)).model_dump(),
        )
