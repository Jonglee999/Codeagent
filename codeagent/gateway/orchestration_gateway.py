"""Orchestration Gateway — 交互层与编排核心的通信契约。

将交互层（CLI/API/WebSocket）与编排核心（Orchestrator）解耦。
交互层只依赖此接口，不直接依赖 Orchestrator。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncGenerator


class TaskState(str, Enum):
    """任务状态枚举。"""

    PENDING = "pending"
    RUNNING = "running"
    WAITING_REVIEW = "waiting_review"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class UserRequest:
    """用户请求的数据传输对象。

    Attributes:
        query: 用户请求文本
        project_root: 项目根目录路径
        auto_mode: 是否自动模式（跳过人工审核）
        max_retries: 最大重试次数
    """

    query: str
    project_root: str
    auto_mode: bool = False
    max_retries: int = 3
    conversation_history: list[dict[str, str]] = field(default_factory=list)
    response_mode: str = "auto"
    direct_execution: bool = False
    conversation_id: str | None = None
    benchmark_instance_id: str | None = None
    benchmark_fail_to_pass: list[str] = field(default_factory=list)
    benchmark_pass_to_pass: list[str] = field(default_factory=list)
    recovered_from_task_id: str | None = None


@dataclass
class TaskStatus:
    """任务状态查询结果。

    Attributes:
        task_id: 任务唯一标识
        state: 当前任务状态
        progress: 执行进度（0.0 ~ 1.0）
        current_step: 当前执行步骤描述
        errors: 错误信息列表
    """

    task_id: str
    state: TaskState
    progress: float = 0.0
    current_step: str | None = None
    errors: list[str] = field(default_factory=list)


@dataclass
class TaskEvent:
    """任务执行过程中产生的流式事件。

    Attributes:
        type: 事件类型
            "thinking" | "tool_call" | "file_change" | "validation"
            | "review_request" | "progress" | "done" | "error"
        data: 事件相关数据
        timestamp: 事件发生时间戳
    """

    type: str
    data: dict
    timestamp: float
    event_id: str | None = None
    seq: int | None = None
    schema_version: int = 1


@dataclass
class HumanDecision:
    """用户在 Human Review 节点提交的决策。

    Attributes:
        task_id: 任务唯一标识
        decision: 决策（"approve" | "reject" | "modify"）
        modifications: 修改建议（modify 时使用）
    """

    task_id: str
    decision: str
    modifications: dict | None = None


@dataclass
class TaskReport:
    """任务执行完成后的完整报告。

    Attributes:
        task_id: 任务唯一标识
        plan: 执行计划
        changes: 文件变更列表
        validation_results: 验证结果列表
        duration: 执行耗时（秒）
        token_usage: 消耗的 token 数
    """

    task_id: str
    plan: list
    changes: list[dict]
    validation_results: list
    duration: float
    token_usage: int
    status: str = "completed"
    assistant_response: str = ""
    response_mode: str = "execute"
    run_profile: dict = field(default_factory=dict)
    tool_manifest: dict = field(default_factory=dict)
    context_manifest: dict = field(default_factory=dict)
    steering_instructions: list[str] = field(default_factory=list)
    memory_hits: list[dict] = field(default_factory=list)
    resolved_skills: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    reflection: dict | None = None
    transcript_path: str | None = None
    mcp_servers: list[dict] = field(default_factory=list)
    model_runtime: dict = field(default_factory=dict)
    infrastructure_runtime: dict = field(default_factory=dict)
    benchmark_metrics: dict[str, Any] = field(default_factory=dict)
    benchmark_instance_id: str | None = None
    artifacts: list[dict] = field(default_factory=list)
    error: str | None = None
    recovered_from_task_id: str | None = None


class IOrchestrationGateway(ABC):
    """交互层与编排核心的通信契约。

    将交互层（CLI/API/WebSocket）与编排核心（Orchestrator）解耦。
    交互层只依赖此接口，不直接依赖 Orchestrator。
    """

    @abstractmethod
    async def start_task(self, request: UserRequest) -> str:
        """启动一个 Agent 任务，返回 task_id。"""
        ...

    @abstractmethod
    async def get_task_status(self, task_id: str) -> TaskStatus:
        """查询任务状态。"""
        ...

    @abstractmethod
    async def stream_task(self, task_id: str) -> AsyncGenerator[TaskEvent, None]:
        """流式获取任务执行事件。"""
        ...

    @abstractmethod
    async def submit_decision(
        self, task_id: str, decision: HumanDecision
    ) -> None:
        """提交用户在 Human Review 节点的决策。"""
        ...

    @abstractmethod
    async def cancel_task(self, task_id: str) -> bool:
        """取消任务。"""
        ...

    @abstractmethod
    async def steer_task(self, task_id: str, instruction: str) -> bool:
        """Queue a user instruction for the next safe execution boundary."""
        ...

    @abstractmethod
    async def get_report(self, task_id: str) -> TaskReport:
        """获取完整的任务执行报告。"""
        ...
