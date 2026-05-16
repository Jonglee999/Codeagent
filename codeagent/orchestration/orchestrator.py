"""Orchestrator — LangGraph 工作流编排器。

封装完整的 Agent 工作流：创建节点、组装图、执行 run/resume 接口。
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Optional

from codeagent.context_engine.evolution import SelfEvolutionManager
from codeagent.gateway.tool_gateway import IToolGateway
from codeagent.gateway.validation_gateway import IValidationGateway
from codeagent.orchestration.graph import build_workflow, run_workflow
from codeagent.orchestration.nodes.context_node import ContextNode
from codeagent.orchestration.nodes.execution_node import ExecutionNode
from codeagent.orchestration.nodes.human_review_node import HumanReviewNode
from codeagent.orchestration.nodes.planning_node import PlanningNode
from codeagent.orchestration.nodes.validation_node import ValidationNode
from codeagent.orchestration.state import AgentState

logger = logging.getLogger(__name__)

_CONFIG_TTL_SECONDS = 3600 * 24  # 24 小时后过期


class Orchestrator:
    """工作流编排器。

    组装所有节点和 LangGraph 图，提供 run/resume 接口。
    """

    def __init__(
        self,
        context_gateway: Any,
        tool_gateway: IToolGateway,
        validation_gateway: IValidationGateway,
        llm: Any,
        model_name: str = "deepseek/deepseek-v4-flash",
        progress_callback: Any | None = None,
        evolution_manager: Optional[SelfEvolutionManager] = None,
    ) -> None:
        """初始化 Orchestrator。

        Args:
            context_gateway: 上下文 Gateway
            tool_gateway: 工具 Gateway
            validation_gateway: 验证 Gateway
            llm: LLM 调用函数
            model_name: 模型名称
            progress_callback: 执行进度回调
        """
        self._llm = llm
        self._model_name = model_name

        # 创建节点
        self._context_node = ContextNode(context_gateway)
        self._planning_node = PlanningNode(llm, model_name)
        self._execution_node = ExecutionNode(
            llm=llm,
            tool_gateway=tool_gateway,
            validation_gateway=validation_gateway,
            model_name=model_name,
            progress_callback=progress_callback,
        )
        self._validation_node = ValidationNode(validation_gateway)
        self._human_review_node = HumanReviewNode()
        self._evolution_manager = evolution_manager

        # 组装并编译图
        self._graph = build_workflow(
            context_node=self._context_node,
            planning_node=self._planning_node,
            execution_node=self._execution_node,
            validation_node=self._validation_node,
            human_review_node=self._human_review_node,
            evolution_manager=evolution_manager,
        )

        # 运行配置追踪
        self._configs: dict[str, dict[str, Any]] = {}
        self._config_timestamps: dict[str, float] = {}

    def _cleanup_stale_configs(self) -> None:
        """清理超过 TTL 的 thread_id 记录。"""
        now = time.monotonic()
        stale = [
            tid
            for tid, ts in self._config_timestamps.items()
            if now - ts > _CONFIG_TTL_SECONDS
        ]
        for tid in stale:
            self._configs.pop(tid, None)
            self._config_timestamps.pop(tid, None)
        if stale:
            logger.debug("Cleaned %d stale config(s)", len(stale))

    async def run(
        self,
        user_request: str,
        project_root: str,
        auto_mode: bool = False,
    ) -> AgentState:
        """运行完整工作流。

        Args:
            user_request: 用户请求
            project_root: 项目根目录
            auto_mode: 是否自动模式（跳过人工审核）

        Returns:
            最终 AgentState
        """
        thread_id = str(uuid.uuid4())
        initial_state = AgentState(
            user_request=user_request,
            project_root=project_root,
            auto_mode=auto_mode,
        )
        config = {"configurable": {"thread_id": thread_id}}
        self._cleanup_stale_configs()
        self._configs[thread_id] = config
        self._config_timestamps[thread_id] = time.monotonic()

        logger.info(
            "Orchestrator.run starting (thread=%s, auto_mode=%s)",
            thread_id, auto_mode,
        )

        try:
            final_state = await run_workflow(
                graph=self._graph,
                initial_state=initial_state,
                config=config,
                evolution_manager=self._evolution_manager,
            )
            logger.info("Orchestrator.run completed (thread=%s)", thread_id)
            return final_state
        except Exception as e:
            logger.error("Orchestrator.run failed (thread=%s): %s", thread_id, e)
            initial_state.errors.append(f"Orchestrator run failed: {e}")
            return initial_state

    async def resume(
        self,
        thread_id: str,
        human_decision: str,
    ) -> AgentState:
        """从中断点恢复工作流（用于 Human Review 后继续）。

        Args:
            thread_id: 线程 ID（来自 run 的返回或 get_checkpoints）
            human_decision: 人工决定（"approve", "abort", "modify"）

        Returns:
            最终 AgentState
        """
        config = self._configs.get(thread_id)
        if not config:
            raise ValueError(f"Unknown thread_id: {thread_id}")

        logger.info(
            "Orchestrator.resume (thread=%s, decision=%s)",
            thread_id, human_decision,
        )

        # 更新当前状态注入 human_decision
        await self._graph.aupdate_state(
            config,
            {"human_decision": human_decision},
        )

        # 继续执行
        try:
            final_state = await self._graph.ainvoke(None, config)
            logger.info("Orchestrator.resume completed (thread=%s)", thread_id)
            return final_state
        except Exception as e:
            logger.error(
                "Orchestrator.resume failed (thread=%s): %s", thread_id, e
            )
            current = await self._graph.aget_state(config)
            state: AgentState = current.values if hasattr(current, "values") else AgentState(user_request="", project_root="")  # type: ignore[assignment]
            if hasattr(state, "errors"):
                state.errors.append(f"Orchestrator resume failed: {e}")
            return state

    def get_checkpoints(self) -> list[dict[str, Any]]:
        """获取所有已知的运行会话信息。

        Returns:
            会话信息列表
        """
        return [
            {
                "thread_id": tid,
                "config": cfg,
            }
            for tid, cfg in self._configs.items()
        ]
