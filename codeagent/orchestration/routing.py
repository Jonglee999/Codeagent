"""Conditional Routing — LangGraph 工作流条件路由函数。

每个路由函数接收 AgentState，返回下一节点的名称。
路由逻辑基于 AgentState 中的字段值做条件判断。
"""

from __future__ import annotations

from typing import Literal

from codeagent.orchestration.state import AgentState

# 路由目标类型
AfterPlanning = Literal["context", "execution", "human_review", "end"]
AfterExecution = Literal["validation", "execution", "end"]
AfterValidation = Literal["execution", "planning", "human_review", "end"]
AfterHumanReview = Literal["execution", "planning", "end"]


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
        "Exceeded max tool calls" in e or "LLM call failed" in e
        for e in state.errors
    ):
        return "end"

    # 用户明确放弃
    if state.human_decision == "abort":
        return "end"

    # 有 plan 时检查步骤执行状态
    if state.plan is not None and len(state.plan) > 0:
        if state.current_step_index < len(state.plan):
            return "execution"
        return "validation"

    # 无 plan（直连模式）→ 验证
    return "validation"


def route_after_validation(state: AgentState) -> AfterValidation:
    """验证完成后的路由。

    路由逻辑：
    - 全部验证通过 → "end"
    - 验证失败且 retry_count < 3 → "execution"（重试）
    - 验证失败且 retry_count >= 3 → "planning"（重新规划）
    - 连续失败且非 auto_mode → "human_review"

    Args:
        state: 当前 Agent 状态

    Returns:
        下一节点名称
    """
    if not state.validation_results:
        # 无验证结果 → 默认通过
        return "end"

    all_passed = all(r.passed for r in state.validation_results)

    if all_passed:
        return "end"

    # 验证失败，检查重试次数
    if state.retry_count < 3:
        return "execution"

    # 重试已耗尽，需要重新规划
    if not state.auto_mode:
        return "human_review"

    return "planning"


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
