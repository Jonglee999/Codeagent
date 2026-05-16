"""AgentState — Agent 执行状态定义（Phase 1b 升级版）。

包含 PlanStep 定义和完整的 AgentState，在 LangGraph 工作流各节点间流转。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from codeagent.gateway.validation_gateway import ValidationResult


# ── 计划相关类型 ────────────────────────────────────────────

PlanAction = Literal["create", "modify", "delete", "read", "command"]
RiskLevel = Literal["low", "high"]


@dataclass
class PlanStep:
    """单个执行步骤的描述。

    Attributes:
        step_id: 步骤编号
        description: 步骤描述
        action: 操作类型
        target_file: 目标文件路径（可选）
        risk: 风险等级
        dependencies: 依赖的步骤 ID 列表
    """

    step_id: int
    description: str
    action: PlanAction
    target_file: str | None = None
    risk: RiskLevel = "low"
    dependencies: list[int] = field(default_factory=list)


# ── Phase 5.4: 修复上下文 ──────────────────────────────────


@dataclass
class StructuredError:
    """单个结构化错误/警告项。

    Attributes:
        file_path: 文件路径
        line_number: 行号（可选）
        error_type: 错误类型（syntax/lint/test/runtime）
        message: 错误消息
        is_pre_existing: 是否为预存在的错误（无需修复）
    """

    file_path: str = ""
    line_number: int | None = None
    error_type: Literal["syntax", "lint", "test", "runtime"] = "syntax"
    message: str = ""
    is_pre_existing: bool = False


@dataclass
class RepairContext:
    """修复上下文，跟踪修复尝试的状态。

    Attributes:
        attempt_number: 当前修复尝试次数（从 1 开始）
        errors: 需要修复的结构化错误列表（非 pre_existing）
        pre_existing_errors: 预存在的错误描述列表（无需修复）
        last_fix_summary: 上次修复的内容摘要（防止 LLM 重复）
    """

    attempt_number: int = 0
    errors: list[StructuredError] = field(default_factory=list)
    pre_existing_errors: list[str] = field(default_factory=list)
    last_fix_summary: str | None = None


# ── Agent 状态 ──────────────────────────────────────────────


@dataclass
class AgentState:
    """Agent 执行状态，在工作流各节点间流转。

    Phase 1a 基础字段:
        user_request: 用户原始需求
        project_root: 项目根目录路径
        file_tree: 文件树结构（可选）
        context: 通用上下文信息
        auto_mode: 是否自动模式
        execution_log: 执行日志列表
        errors: 错误列表

    Phase 1b 新增字段:
        semantic_context: Context Node 输出的语义上下文
        current_file_context: 当前文件上下文
        dependency_graph: 依赖关系图
        plan: 多步骤执行计划
        current_step_index: 当前执行步骤索引
        accumulated_changes: 累计修改记录
        validation_results: 验证结果列表
        human_decision: 人工决策输入
        retry_count: 重试计数
        degraded_mode: 是否降级模式
    """

    # ── Phase 1a 已有字段 ────────────────────────────────────
    user_request: str
    project_root: str
    file_tree: Any = None
    context: str = ""
    auto_mode: bool = False
    execution_log: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    # ── Phase 1b 新增字段 ────────────────────────────────────
    semantic_context: str | None = None
    current_file_context: str | None = None
    dependency_graph: dict | None = None
    plan: list[PlanStep] | None = None
    current_step_index: int = 0
    accumulated_changes: list[dict] = field(default_factory=list)
    validation_results: list[ValidationResult] = field(default_factory=list)
    human_decision: str | None = None
    retry_count: int = 0
    degraded_mode: bool = False

    # ── Phase 3.5 TaskFocus + Human Review 字段 ──────────────
    original_goal_summary: str = ""               # 原始目标摘要（Planning Node 写入）
    completed_steps_summary: str = ""              # 已完成步骤摘要
    tasks_remaining: list[str] = field(default_factory=list)  # 剩余步骤清单
    deviation_detected: bool = False              # 偏离标记
    deviation_count: int = 0                      # 连续偏离计数
    conversation_history: list[dict] = field(default_factory=list)  # 多轮对话历史
    human_review_required: bool = False           # 是否需要人工审核
    review_request: dict | None = None            # 当前审核请求详情
    review_type: str | None = None                # 审核触发类型

    # ── Phase 5.4 新增字段 ────────────────────────────────────
    repair_context: RepairContext | None = None    # 修复上下文（结构化）

    # ── Phase 5.5 新增字段 ────────────────────────────────────
    llm_call_count: int = 0              # 当前任务已调用 LLM 次数
    estimated_tokens: int = 0            # 当前任务估算 token 消耗

    # ── Phase 6.6 新增字段 ────────────────────────────────────
    memory_extracted: bool = False       # 防止重复提取记忆
