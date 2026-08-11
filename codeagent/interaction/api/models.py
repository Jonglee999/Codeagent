"""Pydantic models for REST API request and response payloads."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class TaskCreateRequest(BaseModel):
    query: str = Field(
        ...,
        description="Software engineering request or benchmark issue statement",
        min_length=1,
        max_length=100_000,
    )
    project_root: str | None = Field(
        None,
        description="Optional workspace; omitted requests get a project-local chat workspace",
    )
    auto_mode: bool = Field(False, description="Proceed without low-risk review pauses")
    max_retries: int = Field(3, ge=0, le=10)
    conversation_history: list[dict[str, str]] = Field(default_factory=list)
    response_mode: Literal["auto", "execute", "chat"] = "auto"
    direct_execution: bool = False
    conversation_id: str | None = Field(None, max_length=100)
    benchmark_instance_id: str | None = Field(None, max_length=200)
    recovered_from_task_id: str | None = Field(None, max_length=200)


class ProjectCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=80)


class ConversationDeleteRequest(BaseModel):
    workspace_root: str | None = None


class ConversationImportRequest(BaseModel):
    conversations: list[dict[str, Any]] = Field(default_factory=list, max_length=500)


class TaskStatusResponse(BaseModel):
    task_id: str
    state: str
    progress: float = 0.0
    current_step: str | None = None
    errors: list[str] = Field(default_factory=list)


class DecisionRequest(BaseModel):
    decision: str = Field(..., pattern="^(approve|reject|modify)$")
    modifications: dict[str, Any] | None = None


class SteeringRequest(BaseModel):
    instruction: str = Field(..., min_length=1, max_length=4000)


class RecoveryRequest(BaseModel):
    instruction: str = Field(
        "Inspect the existing workspace changes, continue the unfinished work, and validate the result.",
        min_length=1,
        max_length=4000,
    )
    auto_mode: bool = True
    max_retries: int = Field(3, ge=0, le=10)


class TaskReportResponse(BaseModel):
    task_id: str
    status: str = "completed"
    plan: list
    changes: list[dict]
    validation_results: list
    duration: float
    token_usage: int
    assistant_response: str = ""
    response_mode: str = "execute"
    run_profile: dict[str, Any] = Field(default_factory=dict)
    tool_manifest: dict[str, Any] = Field(default_factory=dict)
    context_manifest: dict[str, Any] = Field(default_factory=dict)
    steering_instructions: list[str] = Field(default_factory=list)
    memory_hits: list[dict[str, Any]] = Field(default_factory=list)
    resolved_skills: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    reflection: dict[str, Any] | None = None
    transcript_path: str | None = None
    mcp_servers: list[dict[str, Any]] = Field(default_factory=list)
    model_runtime: dict[str, Any] = Field(default_factory=dict)
    infrastructure_runtime: dict[str, Any] = Field(default_factory=dict)
    benchmark_metrics: dict[str, Any] = Field(default_factory=dict)
    benchmark_instance_id: str | None = None
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    error: str | None = None
    recovered_from_task_id: str | None = None


class ApiResponse(BaseModel):
    success: bool
    data: Any | None = None
    error: str | None = None
