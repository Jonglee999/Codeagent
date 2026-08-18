"""Orchestrator — LangGraph 工作流编排器。

封装完整的 Agent 工作流：创建节点、组装图、执行 run/resume 接口。
"""

from __future__ import annotations

import logging
import inspect
import time
import uuid
from datetime import datetime, timedelta
from typing import Any, Optional

from codeagent import config as codeagent_config
from codeagent.context_engine.evolution import SelfEvolutionManager
from codeagent.context_engine.evolution import StrategyApplier
from codeagent.gateway.memory_gateway import IMemoryGateway
from codeagent.gateway.tool_gateway import IToolGateway
from codeagent.gateway.validation_gateway import IValidationGateway
from codeagent.interaction.api.metrics import active_tasks, observe_task_duration
from codeagent.orchestration.graph import build_workflow, run_workflow
from codeagent.tracing import trace_llm_call
from codeagent.orchestration.nodes.context_node import ContextNode
from codeagent.orchestration.nodes.execution_node import ExecutionNode
from codeagent.orchestration.nodes.human_review_node import HumanReviewNode
from codeagent.orchestration.nodes.planning_node import PlanningNode
from codeagent.orchestration.nodes.validation_node import ValidationNode
from codeagent.orchestration.capabilities import select_tool_capabilities
from codeagent.orchestration.policy import select_run_profile
from codeagent.orchestration.state import AgentState
from codeagent.memory.store import MemoryEntry, MemoryType
from codeagent.memory.transcript import build_transcript, memory_messages, persist_transcript

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
        memory_gateway: Optional[IMemoryGateway] = None,
        strategy_applier: Optional[StrategyApplier] = None,
        max_retries: int = 3,
        steering_provider: Any | None = None,
        checkpointer: Any | None = None,
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
        self._tool_gateway = tool_gateway
        self._checkpointer = checkpointer

        # 为 LLM 调用添加 OpenTelemetry 追踪包装
        traced_llm = trace_llm_call(model_name)(llm)

        # 创建节点
        self._context_node = ContextNode(
            context_gateway,
            progress_callback=progress_callback,
            memory_gateway=memory_gateway,
        )
        self._memory_gateway = memory_gateway
        self._progress_callback = progress_callback
        self._planning_node = PlanningNode(
            traced_llm,
            model_name,
            max_retries=max_retries,
            memory_gateway=memory_gateway,
            strategy_applier=strategy_applier,
            progress_callback=progress_callback,
        )
        self._execution_node = ExecutionNode(
            llm=traced_llm,
            tool_gateway=tool_gateway,
            validation_gateway=validation_gateway,
            model_name=model_name,
            progress_callback=progress_callback,
            memory_gateway=memory_gateway,
            max_retries=max_retries,
            steering_provider=steering_provider,
        )
        self._validation_node = ValidationNode(
            validation_gateway,
            static_analyzer=getattr(validation_gateway, "static_analyzer", None),
            runtime_validator=getattr(validation_gateway, "runtime_validator", None),
        )
        self._human_review_node = HumanReviewNode()
        self._evolution_manager = evolution_manager

        delegation_node = None
        if codeagent_config.get_a2a_client_enabled() and codeagent_config.get_a2a_delegate_url():
            from codeagent.a2a.client import A2AClientGateway
            from codeagent.orchestration.nodes.delegation_node import DelegationNode

            delegation_node = DelegationNode(
                A2AClientGateway(), codeagent_config.get_a2a_delegate_url()
            )

        # 组装并编译图
        self._graph = build_workflow(
            context_node=self._context_node,
            planning_node=self._planning_node,
            execution_node=self._execution_node,
            validation_node=self._validation_node,
            human_review_node=self._human_review_node,
            trajectory_recorder=(evolution_manager.recorder if evolution_manager else None),
            evolution_manager=evolution_manager,
            progress_callback=progress_callback,
            checkpointer=checkpointer,
            delegation_node=delegation_node,
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

    async def _emit_progress(self, event: dict[str, Any]) -> None:
        if self._progress_callback is None:
            return
        try:
            result = self._progress_callback(event)
            if inspect.isawaitable(result):
                await result
        except Exception as exc:
            logger.warning("Orchestrator progress event failed: %s", exc)

    async def run(
        self,
        user_request: str,
        project_root: str,
        auto_mode: bool = False,
        task_id: str | None = None,
        conversation_history: list[dict[str, Any]] | None = None,
        direct_execution: bool = False,
        benchmark_instance_id: str | None = None,
        benchmark_fail_to_pass: list[str] | None = None,
        benchmark_pass_to_pass: list[str] | None = None,
        recovered_from_task_id: str | None = None,
    ) -> AgentState:
        """运行完整工作流。

        Args:
            user_request: 用户请求
            project_root: 项目根目录
            auto_mode: 是否自动模式（跳过人工审核）

        Returns:
            最终 AgentState
        """
        # External task ids are stable across process restarts and therefore
        # also serve as persistent LangGraph thread ids.
        thread_id = task_id or str(uuid.uuid4())
        external_task_id = task_id or thread_id
        history = list(conversation_history or [])
        run_profile = select_run_profile(
            user_request,
            conversation_history=history,
            direct_execution=direct_execution,
            benchmark_instance_id=benchmark_instance_id,
        )
        tool_selection = select_tool_capabilities(
            user_request,
            self._tool_gateway.list_tools(),
            run_profile,
        )
        if not history or history[-1].get("role") != "user" or history[-1].get("content") != user_request:
            history.append({"role": "user", "content": user_request})
        initial_state = AgentState(
            user_request=user_request,
            project_root=project_root,
            auto_mode=auto_mode,
            task_id=external_task_id,
            conversation_history=history,
            direct_execution=run_profile.workflow == "direct",
            benchmark_instance_id=benchmark_instance_id,
            benchmark_fail_to_pass=list(benchmark_fail_to_pass or []),
            benchmark_pass_to_pass=list(benchmark_pass_to_pass or []),
            recovered_from_task_id=recovered_from_task_id,
            run_profile=run_profile.public_metadata(),
            tool_manifest=tool_selection.public_metadata(),
            context_mode=run_profile.context,
            memory_mode=run_profile.memory,
            learning_mode=run_profile.learning,
            evolution_enabled=run_profile.learning == "capture",
        )
        config = {"configurable": {"thread_id": thread_id}}
        self._cleanup_stale_configs()
        self._configs[thread_id] = config
        self._config_timestamps[thread_id] = time.monotonic()
        if self._checkpointer is not None:
            from codeagent.orchestration.checkpoint import register_checkpoint_thread

            await register_checkpoint_thread(
                self._checkpointer,
                thread_id=thread_id,
                task_id=external_task_id,
                status="running",
            )

        logger.info(
            "Orchestrator.run starting (thread=%s, auto_mode=%s, workflow=%s)",
            thread_id, auto_mode, run_profile.workflow,
        )
        await self._emit_progress({
            "type": "run_profile_selected",
            "summary": run_profile.reason,
            "data": run_profile.public_metadata(),
        })
        await self._emit_progress({
            "type": "tools_selected",
            "summary": (
                f"Exposed {len(tool_selection.selected)} relevant tool(s); "
                f"deferred {len(tool_selection.deferred)}"
            ),
            "data": tool_selection.public_metadata(),
        })
        if recovered_from_task_id:
            await self._emit_progress({
                "type": "recovery_started",
                "summary": f"Continuing from task {recovered_from_task_id}",
                "data": {"recovered_from_task_id": recovered_from_task_id},
            })
        if self._memory_gateway is None and run_profile.memory != "off":
            from codeagent.config import get_memory_enabled

            if not get_memory_enabled():
                initial_state.warnings.append("Memory disabled by MEMORY_ENABLED=false")
                await self._emit_progress({
                    "type": "capability_degraded",
                    "visibility": "internal",
                    "summary": "Memory is disabled; task continues without recall",
                    "data": {"capability": "memory", "reason": "MEMORY_ENABLED=false"},
                })

        run_start = time.monotonic()
        active_tasks.inc()

        try:
            final_state = await run_workflow(
                graph=self._graph,
                initial_state=initial_state,
                config=config,
                evolution_manager=self._evolution_manager,
            )
            from codeagent.orchestration.runtime import assess_task_completion

            success, completion_errors = assess_task_completion(
                final_state,
                require_patch=bool(final_state.benchmark_instance_id),
                require_test_evidence=bool(final_state.benchmark_instance_id),
            )
            transcript = build_transcript(final_state, success=success)
            try:
                final_state.transcript_path = persist_transcript(
                    project_root=project_root,
                    task_id=external_task_id,
                    entries=transcript,
                )
                await self._emit_progress({
                    "type": "transcript_saved",
                    "visibility": "internal",
                    "summary": f"Saved {len(transcript)} structured transcript entries",
                    "data": {"entry_count": len(transcript)},
                })
            except Exception as exc:
                final_state.warnings.append(f"Transcript persistence degraded: {exc}")

            duration = time.monotonic() - run_start
            has_errors = bool(final_state.errors)
            status = "failed" if has_errors else "completed"
            observe_task_duration(duration, status)
            active_tasks.dec()
            logger.info("Orchestrator.run completed (thread=%s)", thread_id)
            if self._checkpointer is not None:
                from codeagent.orchestration.checkpoint import register_checkpoint_thread

                await register_checkpoint_thread(
                    self._checkpointer,
                    thread_id=thread_id,
                    task_id=external_task_id,
                    status=(
                        "waiting_review"
                        if final_state.human_review_required
                        else "failed" if final_state.errors else "completed"
                    ),
                )
            return final_state
        except Exception as e:
            duration = time.monotonic() - run_start
            observe_task_duration(duration, "failed")
            active_tasks.dec()
            logger.error("Orchestrator.run failed (thread=%s): %s", thread_id, e)
            initial_state.errors.append(f"Orchestrator run failed: {e}")
            return initial_state

    async def persist_memory(
        self,
        final_state: AgentState,
        *,
        user_request: str,
        task_id: str,
        success: bool,
        completion_errors: list[str],
    ) -> int:
        """Persist evidence-gated memory after the user-visible run completes.

        This method intentionally emits no progress events. Runners invoke it
        only after saving the report and publishing the terminal event, so model-
        backed extraction cannot delay or appear in the user conversation.
        """
        if self._memory_gateway is None or final_state.learning_mode != "capture":
            return 0
        try:
            if success:
                changed_files = [
                    str(change.get("file_path") or change.get("path") or "")
                    for change in final_state.accumulated_changes
                ]
                self._memory_gateway.save(MemoryEntry(
                    name=f"verified-run-{task_id[:12]}",
                    memory_type=MemoryType.SESSION,
                    description=user_request[:180],
                    body=(
                        f"Verified request: {user_request}\n"
                        f"Changed files: {', '.join(path for path in changed_files if path)}\n"
                        "Validation passed: true\n"
                        f"Reflection: {final_state.reflection}"
                    ),
                    confidence=0.75,
                    tags=["verified", "episodic", "evidence-gated"],
                    expires_at=datetime.now() + timedelta(days=30),
                ), scope="project")
                transcript = build_transcript(final_state, success=True)
                saved = await self._memory_gateway.auto_extract(
                    conversation_history=memory_messages(transcript),
                    trigger="task_complete_evidence_passed",
                )
                saved_count = 1 + len(saved)
            else:
                self._memory_gateway.save(MemoryEntry(
                    name=f"counterexample-{task_id[:12]}",
                    memory_type=MemoryType.SESSION,
                    description="Failed task counterexample; do not repeat unchanged approach",
                    body=(
                        f"Request: {user_request}\n"
                        f"Evidence: {'; '.join(completion_errors[:5])}\n"
                        f"Reflection: {final_state.reflection}"
                    ),
                    confidence=0.5,
                    tags=["counterexample", "validation-failed"],
                    expires_at=datetime.now() + timedelta(days=14),
                ), scope="project")
                saved_count = 1
            final_state.memory_extracted = True
            logger.info("Background memory writeback saved %d item(s) for %s", saved_count, task_id)
            return saved_count
        except Exception as exc:
            logger.warning("Background memory writeback failed for %s: %s", task_id, exc)
            return 0

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
        config = self._configs.get(thread_id) or {
            "configurable": {"thread_id": thread_id}
        }
        if thread_id not in self._configs:
            if self._checkpointer is None:
                raise ValueError(f"Unknown thread_id: {thread_id}")
            try:
                snapshot = await self._graph.aget_state(config)
                if not getattr(snapshot, "values", None):
                    raise ValueError(f"Unknown thread_id: {thread_id}")
            except ValueError:
                raise
            except Exception as exc:
                raise ValueError(f"Unknown thread_id: {thread_id}") from exc
            self._configs[thread_id] = config
            self._config_timestamps[thread_id] = time.monotonic()

        logger.info(
            "Orchestrator.resume (thread=%s, decision=%s)",
            thread_id, human_decision,
        )
        if self._checkpointer is not None:
            from codeagent.orchestration.checkpoint import register_checkpoint_thread

            await register_checkpoint_thread(
                self._checkpointer,
                thread_id=thread_id,
                task_id=thread_id,
                status="running",
            )

        # 更新当前状态注入 human_decision
        await self._graph.aupdate_state(
            config,
            {"human_decision": human_decision},
        )

        # 继续执行
        try:
            result = await self._graph.ainvoke(None, config)
            # LangGraph 1.2.0 返回 dict 而非 AgentState
            if isinstance(result, dict):
                valid_fields = AgentState.__dataclass_fields__
                filtered = {k: v for k, v in result.items() if k in valid_fields}
                final_state = AgentState(**filtered)
            else:
                final_state = result
            logger.info("Orchestrator.resume completed (thread=%s)", thread_id)
            return final_state
        except Exception as e:
            logger.error(
                "Orchestrator.resume failed (thread=%s): %s", thread_id, e
            )
            current = await self._graph.aget_state(config)
            raw = current.values if hasattr(current, "values") else {}
            if isinstance(raw, dict):
                valid_fields = AgentState.__dataclass_fields__
                filtered = {k: v for k, v in raw.items() if k in valid_fields}
                state = AgentState(**filtered)
            else:
                state = raw
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
