"""LangGraph 工作流图组装。

将 Context、Planning、Execution、Validation、Human Review 节点
通过 StateGraph 组装为完整的工作流图。
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from langgraph.graph import END, StateGraph

from codeagent import config

logger = logging.getLogger(__name__)


def _build_checkpointer() -> Any:
    """构建 Checkpointer。

    默认返回 MemorySaver（内存级，CLI/开发模式适用）。
    生产环境通过 build_workflow 的 checkpointer 参数注入 AsyncSqliteSaver。
    """
    from langgraph.checkpoint.memory import MemorySaver

    return MemorySaver()


from codeagent.context_engine.evolution import (
    SelfEvolutionManager,
    TrajectoryRecorder,
    TrajectoryStep,
)
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


def _summarize_state(state: AgentState, max_chars: int = 200) -> str:
    """提取 AgentState 的摘要（不包含完整消息体）。

    Args:
        state: Agent 状态
        max_chars: 最大字符数

    Returns:
        JSON 摘要字符串
    """
    summary = {
        "current_step": state.current_step_index,
        "plan_steps": len(state.plan) if state.plan else 0,
        "changes_count": len(state.accumulated_changes),
        "retry_count": state.retry_count,
        "repair_rounds": state.repair_rounds,
    }
    return json.dumps(summary, ensure_ascii=False)[:max_chars]


def _summarize_output(output: dict, max_chars: int = 200) -> str:
    """提取节点输出的摘要。

    Args:
        output: 节点输出字典
        max_chars: 最大字符数

    Returns:
        JSON 摘要字符串
    """
    summary = {
        k: v for k, v in output.items()
        if k in ("plan", "errors", "current_step_index")
    }
    # 对于 plan 只记录长度
    if "plan" in summary and isinstance(summary["plan"], list):
        summary["plan"] = f"[{len(summary['plan'])} steps]"
    return json.dumps(summary, ensure_ascii=False)[:max_chars]


def _wrap_node_with_recording(
    node_name: str,
    node_fn: Callable,
    recorder: TrajectoryRecorder,
) -> Callable:
    """包装节点函数：在调用前后插入轨迹记录点。

    Args:
        node_name: 节点名称
        node_fn: 原始节点函数
        recorder: TrajectoryRecorder 实例

    Returns:
        包装后的异步节点函数
    """
    async def wrapped(state: AgentState) -> dict:
        step_id = str(uuid.uuid4())
        input_summary = _summarize_state(state)

        # 确定步骤类型
        if node_name in ("planning", "execution"):
            step_type = "llm_call"
        elif node_name == "validation":
            step_type = "validation"
        elif node_name == "human_review":
            step_type = "human_review"
        else:
            step_type = "llm_call"

        start_time = datetime.now()

        try:
            result = await node_fn(state) if asyncio.iscoroutinefunction(node_fn) else node_fn(state)
            output_summary = _summarize_output(result)

            duration = int((datetime.now() - start_time).total_seconds() * 1000)

            recorder.record_step(state.task_id, TrajectoryStep(
                step_id=step_id,
                node_name=node_name,
                step_type=step_type,  # type: ignore[arg-type]
                timestamp=start_time,
                input_summary=input_summary,
                output_summary=output_summary,
                duration_ms=duration,
                success=True,
            ))
            return result
        except Exception as e:
            duration = int((datetime.now() - start_time).total_seconds() * 1000)
            recorder.record_step(state.task_id, TrajectoryStep(
                step_id=step_id,
                node_name=node_name,
                step_type=step_type,  # type: ignore[arg-type]
                timestamp=start_time,
                input_summary=input_summary,
                output_summary=f"error: {e}",
                duration_ms=duration,
                success=False,
                error=str(e),
            ))
            raise

    return wrapped


def _ensure_agent_state(state: dict | AgentState) -> AgentState:
    """将 LangGraph 返回的 dict 转换回 AgentState。

    LangGraph 1.2.0 的 graph.ainvoke() 即使传入 dataclass schema，
    返回值也是 plain dict 而非 AgentState 实例。
    """
    if isinstance(state, AgentState):
        return state
    valid_fields = AgentState.__dataclass_fields__
    filtered = {k: v for k, v in state.items() if k in valid_fields}
    return AgentState(**filtered)


def _all_validations_passed(state: AgentState) -> bool:
    """检查所有验证是否通过。

    Args:
        state: Agent 状态

    Returns:
        所有验证通过返回 True
    """
    if not state.validation_results:
        return False
    return all(r.passed for r in state.validation_results)


async def run_workflow(
    graph: StateGraph,
    initial_state: AgentState,
    config: dict | None = None,
    evolution_manager: Optional[SelfEvolutionManager] = None,
) -> AgentState:
    """运行工作流，带自进化生命周期钩子。

    任务开始时调用 evolution_manager.on_task_start()，
    任务完成时调用 evolution_manager.on_task_complete()。

    Args:
        graph: 编译后的 StateGraph
        initial_state: 初始状态
        config: LangGraph 配置字典，None 时自动生成默认配置
        evolution_manager: SelfEvolutionManager 实例，None 时跳过

    Returns:
        最终 AgentState
    """
    # 自动生成默认配置（checkpointer 需要 thread_id）
    if config is None:
        config = {"configurable": {"thread_id": f"run-{uuid.uuid4().hex[:8]}"}}

    # 任务开始时
    if evolution_manager and initial_state.evolution_enabled:
        await evolution_manager.on_task_start(
            task_id=initial_state.task_id,
            user_request=initial_state.user_request,
        )

    # 执行工作流（LangGraph 1.2.0 返回 dict，需转回 AgentState）
    final_state_dict = await graph.ainvoke(initial_state, config)
    final_state = _ensure_agent_state(final_state_dict)

    # 任务完成时
    if evolution_manager and initial_state.evolution_enabled:
        success = len(final_state.errors) == 0
        await evolution_manager.on_task_complete(
            task_id=final_state.task_id or initial_state.task_id,
            success=success,
            repair_rounds=final_state.repair_rounds,
            validation_passed=_all_validations_passed(final_state),
            applied_strategy_ids=final_state.applied_strategy_ids,
        )

    return final_state


def build_workflow(
    context_node: Any,
    planning_node: Any,
    execution_node: Any,
    validation_node: Any,
    human_review_node: Any | None = None,
    trajectory_recorder: Optional[TrajectoryRecorder] = None,
    evolution_manager: Optional[SelfEvolutionManager] = None,
    checkpointer: Any | None = None,
) -> StateGraph:
    """构建完整的 LangGraph StateGraph 工作流。

    Args:
        context_node: Context Node 实例（callable, state → dict）
        planning_node: Planning Node 实例
        execution_node: Execution Node 实例
        validation_node: Validation Node 实例
        human_review_node: Human Review 节点（默认使用 _default_human_review）
        trajectory_recorder: TrajectoryRecorder 实例，None 时不记录轨迹
        checkpointer: 持久化 Checkpointer（默认使用 SqliteSaver，可通过 get_checkpointer 创建）

    Returns:
        编译后的 StateGraph（可 ainvoke）
    """
    # Phase 7.1: 包装节点以记录轨迹
    if trajectory_recorder is not None:
        node_configs = [
            ("context", context_node),
            ("planning", planning_node),
            ("execution", execution_node),
            ("validation", validation_node),
        ]
        for name, node in node_configs:
            wrapped = _wrap_node_with_recording(name, node, trajectory_recorder)
            if name == "context":
                context_node = wrapped
            elif name == "planning":
                planning_node = wrapped
            elif name == "execution":
                execution_node = wrapped
            elif name == "validation":
                validation_node = wrapped

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
    if checkpointer is None:
        checkpointer = _build_checkpointer()
    return workflow.compile(
        checkpointer=checkpointer,
        interrupt_before=["human_review"],
    )
