"""Conditional Routing — LangGraph 工作流条件路由函数。

每个路由函数接收 AgentState，返回下一节点的名称。
路由逻辑基于 AgentState 中的字段值做条件判断。
"""

from __future__ import annotations

from typing import Literal

from codeagent.orchestration.state import AgentState
from codeagent.orchestration.nodes.validation_node import (
    benchmark_runtime_environment_unavailable,
)

# 路由目标类型
AfterPlanning = Literal["context", "execution", "human_review", "end"]
AfterContext = Literal["planning", "execution"]
AfterExecution = Literal["validation", "execution", "end", "human_review"]
AfterValidation = Literal["execution", "planning", "human_review", "end"]
AfterReflection = Literal["execution", "planning", "human_review", "end"]
AfterValidationEvidence = Literal["reflection", "end"]
AfterHumanReview = Literal["execution", "planning", "end"]

# 验证修复循环常量
_MAX_VALIDATION_RETRIES = 3          # 单轮最大重试次数
_MAX_CONSECUTIVE_FAILURES = 9        # 连续验证失败上限（3 轮 × 3 次）


def _benchmark_runtime_environment_unavailable(state: AgentState) -> bool:
    """Detect a benchmark test collection failure caused by the generic sandbox.

    Benchmark repositories are validated authoritatively by SWE-bench's project-specific
    Docker image. The lightweight Agent sandbox intentionally doesn't install every
    benchmark project's dependencies, so a collection-time missing-module error must not
    spend more model calls trying to repair otherwise valid source code.
    """
    return benchmark_runtime_environment_unavailable(
        state, state.validation_results
    )


def route_after_context(state: AgentState) -> AfterContext:
    """Follow-up code requests bypass the planning node entirely."""
    return "execution" if state.direct_execution else "planning"


def route_after_planning(state: AgentState) -> AfterPlanning:
    """规划完成后的路由。

    路由逻辑：
    - plan 中有 high_risk 步骤且非 auto_mode → "human_review"
    - context 不足（semantic_context 为空且文件树过大）→ "context"
    - 有严重错误 → "end"
    - 否则 → "execution"

    Args:
        state: 当前 Agent 状态

    Returns:
        下一节点名称
    """
    # 有严重错误时终止
    if state.direct_execution:
        return "execution"

    if state.errors and any("Plan generation failed" in e for e in state.errors):
        return "end"

    # 计划为空时终止
    if state.plan is None or len(state.plan) == 0:
        return "end"

    # high_risk 步骤需人工审核（非 auto_mode 时）
    has_high_risk = any(
        step.risk == "high" for step in state.plan
    )
    if has_high_risk and not state.auto_mode:
        return "human_review"

    # context 不足时重新收集
    if not state.semantic_context and state.file_tree:
        return "context"

    return "execution"


def route_after_execution(state: AgentState) -> AfterExecution:
    """执行完成后的路由。

    路由逻辑：
    - 所有步骤执行成功 → "validation"
    - 还有未执行步骤 → "execution"
    - 执行出错且用户放弃 → "end"
    - 否则（无 plan，直连模式）→ "validation"

    Args:
        state: 当前 Agent 状态

    Returns:
        下一节点名称
    """
    # 有 fatal 错误时终止
    if state.errors and any(
        "Exceeded max tool calls" in e
        or "LLM call failed" in e
        or "Step evidence missing" in e
        for e in state.errors
    ):
        return "end"

    # 用户明确放弃
    if state.human_decision == "abort":
        return "end"

    # Phase 3.5: 偏离检测触发人工审核
    if state.human_review_required:
        return "human_review"

    # 有 plan 时检查步骤执行状态
    if state.plan is not None and len(state.plan) > 0:
        if state.current_step_index < len(state.plan):
            return "execution"
        return "validation"

    # Read/run-only direct turns have no mutation to validate. This is common
    # for conversational follow-ups such as "run it again and show the result";
    # sending them through every repository validator adds noise and latency.
    if state.direct_execution and not state.accumulated_changes:
        return "end"

    # 无 plan（直连模式）且产生了文件修改 → 验证
    return "validation"


def route_after_validation(state: AgentState) -> AfterValidation:
    """验证完成后的路由。

    路由逻辑：
    - 全部验证通过 → "end"
    - 验证失败且 retry_count < MAX_RETRIES → "execution"（修复模式）
    - 验证失败且 retry_count >= MAX_CONSECUTIVE_FAILURES → "human_review"（连续失败过多）
    - 验证失败且 retry_count >= MAX_RETRIES → "planning"（需要重新规划）
    - 非自动模式且重试耗尽 → "human_review"

    Args:
        state: 当前 Agent 状态

    Returns:
        下一节点名称
    """
    if not state.validation_results:
        return "end"

    all_passed = all(r.passed for r in state.validation_results)
    if all_passed:
        return "end"

    if _benchmark_runtime_environment_unavailable(state):
        return "end"

    # ── Phase 4.A.5 增强：修复循环路由 ─────────────────────────
    if state.retry_count < _MAX_VALIDATION_RETRIES:
        return "execution"

    # 超过连续失败上限（3 轮共 9 次）→ 人工审核
    if state.retry_count >= _MAX_CONSECUTIVE_FAILURES:
        return "human_review"

    # 重试耗尽，需要重新规划
    if not state.auto_mode:
        return "human_review"

    return "planning"


def route_after_validation_evidence(state: AgentState) -> AfterValidationEvidence:
    """Reflect only when validation produced evidence that needs revision."""
    if state.validation_results and all(result.passed for result in state.validation_results):
        return "end"
    if not state.validation_results:
        return "end"
    if _benchmark_runtime_environment_unavailable(state):
        return "end"
    return "reflection"


def route_after_reflection(state: AgentState) -> AfterReflection:
    action = (state.reflection or {}).get("next_action")
    return {
        "finish": "end",
        "repair": "execution",
        "replan": "planning",
        "review": "human_review",
    }.get(action, "end")  # type: ignore[return-value]


def route_after_human_review(state: AgentState) -> AfterHumanReview:
    """人工审核节点完成后的路由。

    路由逻辑：
    - human_decision == "approve" → "execution"（继续执行）
    - human_decision == "abort" → "end"（终止）
    - human_decision == "modify" → "planning"（重新规划）
    - 其他 → "end"（安全默认）

    Args:
        state: 当前 Agent 状态

    Returns:
        下一节点名称
    """
    if state.human_decision == "approve":
        return "execution"
    elif state.human_decision == "abort":
        return "end"
    elif state.human_decision == "modify":
        return "planning"
    return "end"
