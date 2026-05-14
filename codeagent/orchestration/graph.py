"""LangGraph 工作流图组装。

将 Context、Planning、Execution、Validation、Human Review 节点
通过 StateGraph 组装为完整的工作流图。
"""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from codeagent.orchestration.routing import (
    route_after_execution,
    route_after_human_review,
    route_after_planning,
    route_after_validation,
)
from codeagent.orchestration.state import AgentState


def _default_human_review(state: AgentState) -> dict[str, Any]:
    """默认 Human Review 节点：记录审核事件到日志。"""
    import time

    return {
        "execution_log": [
            *state.execution_log,
            {
                "type": "human_review",
                "timestamp": time.time(),
                "human_decision": state.human_decision,
            },
        ],
    }


def build_workflow(
    context_node: Any,
    planning_node: Any,
    execution_node: Any,
    validation_node: Any,
    human_review_node: Any | None = None,
) -> StateGraph:
    """构建完整的 LangGraph StateGraph 工作流。

    Args:
        context_node: Context Node 实例（callable, state → dict）
        planning_node: Planning Node 实例
        execution_node: Execution Node 实例
        validation_node: Validation Node 实例
        human_review_node: Human Review 节点（默认使用 _default_human_review）

    Returns:
        编译后的 StateGraph（可 ainvoke）
    """
    workflow = StateGraph(AgentState)

    # 添加所有节点
    workflow.add_node("context", context_node)
    workflow.add_node("planning", planning_node)
    workflow.add_node("execution", execution_node)
    workflow.add_node("validation", validation_node)
    workflow.add_node(
        "human_review", human_review_node or _default_human_review
    )

    # 入口点
    workflow.set_entry_point("context")

    # 普通边
    workflow.add_edge("context", "planning")

    # 条件边
    workflow.add_conditional_edges(
        "planning",
        route_after_planning,
        {
            "context": "context",
            "execution": "execution",
            "human_review": "human_review",
            "end": END,
        },
    )
    workflow.add_conditional_edges(
        "execution",
        route_after_execution,
        {
            "execution": "execution",
            "validation": "validation",
            "human_review": "human_review",
            "end": END,
        },
    )
    workflow.add_conditional_edges(
        "validation",
        route_after_validation,
        {
            "execution": "execution",
            "planning": "planning",
            "human_review": "human_review",
            "end": END,
        },
    )
    workflow.add_conditional_edges(
        "human_review",
        route_after_human_review,
        {
            "execution": "execution",
            "planning": "planning",
            "end": END,
        },
    )

    # 编译图，在 human_review 前设置中断点
    memory = MemorySaver()
    return workflow.compile(
        checkpointer=memory,
        interrupt_before=["human_review"],
    )
