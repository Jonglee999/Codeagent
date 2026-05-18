"""Pydantic models for REST API request/response."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class TaskCreateRequest(BaseModel):
    query: str = Field(..., description="用户任务描述", min_length=1, max_length=10000)
    project_root: str = Field(..., description="项目根目录绝对路径")
    auto_mode: bool = Field(False, description="是否跳过 Human Review")
    max_retries: int = Field(3, ge=0, le=10)


class TaskStatusResponse(BaseModel):
    task_id: str
    state: str
    progress: float = 0.0
    current_step: str | None = None
    errors: list[str] = []


class DecisionRequest(BaseModel):
    decision: str = Field(..., pattern="^(approve|reject|modify)$")
    modifications: dict[str, Any] | None = None


class TaskReportResponse(BaseModel):
    task_id: str
    plan: list
    changes: list[dict]
    validation_results: list
    duration: float
    token_usage: int


class ApiResponse(BaseModel):
    """统一响应包装器。"""

    success: bool
    data: Any | None = None
    error: str | None = None
