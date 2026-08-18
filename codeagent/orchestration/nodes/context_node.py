"""Context Node — 上下文收集节点。

接收 IContextGateway 作为依赖，调用 build_context 获取 ContextPackage，
格式化后更新 AgentState 的语义上下文、当前文件上下文、依赖关系图。
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from typing import Any

from codeagent import config as codeagent_config
from codeagent.context_engine.context_assembler import ContextAssembler
from codeagent.extensions import resolve_project_instructions
from codeagent.gateway.context_gateway import IContextGateway
from codeagent.gateway.memory_gateway import IMemoryGateway
from codeagent.orchestration.state import AgentState
from codeagent.tracing import trace_node

logger = logging.getLogger(__name__)


class ContextNode:
    """上下文收集节点。

    调用 IContextGateway 收集项目上下文，通过 ContextAssembler 格式化为
    LLM-ready 字符串，更新 AgentState 对应字段。
    发生异常时进入降级模式（degraded_mode=True）。
    """

    def __init__(
        self,
        context_gateway: IContextGateway,
        context_assembler: ContextAssembler | None = None,
        progress_callback: Any | None = None,
        memory_gateway: IMemoryGateway | None = None,
    ) -> None:
        """初始化 ContextNode。

        Args:
            context_gateway: 上下文引擎 Gateway
            context_assembler: 上下文组装器（可选，默认创建）
        """
        self._gateway = context_gateway
        self._assembler = (
            context_assembler
            or getattr(context_gateway, "context_assembler", None)
            or ContextAssembler(
                total_budget=codeagent_config.get_context_budget(),
                model_name=codeagent_config.get_model(),
            )
        )
        self._progress_callback = progress_callback
        self._memory_gateway = memory_gateway

    async def _emit(self, event: dict[str, Any]) -> None:
        if self._progress_callback is None:
            return
        try:
            result = self._progress_callback(event)
            if inspect.isawaitable(result):
                await result
        except Exception as exc:
            logger.warning("Skill resolution event failed: %s", exc)

    @trace_node("context")
    async def __call__(self, state: AgentState) -> dict[str, Any]:
        """执行上下文收集。

        调用上下文引擎获取 ContextPackage，组装为 LLM-ready 格式，
        返回状态更新字典。

        Args:
            state: 当前 AgentState

        Returns:
            dict: 更新的状态字段
        """
        memory_task = asyncio.create_task(self._prefetch_memory(state))
        if state.context_mode == "minimal":
            skills_enabled = codeagent_config.get_skills_enabled()
            resolution = resolve_project_instructions(
                state.project_root,
                state.user_request,
                skills_enabled=skills_enabled,
            )
            if not skills_enabled:
                resolution.warnings.append("Skills disabled by SKILLS_ENABLED=false")
            skill_metadata = [skill.public_metadata() for skill in resolution.skills]
            await self._emit({
                "type": "skill_resolved",
                "summary": (
                    f"Matched {len(skill_metadata)} Skill(s)"
                    if skill_metadata else "No Skill matched this request"
                ),
                "data": {
                    "skills": skill_metadata,
                    "repository_instructions": resolution.repository_instructions,
                    "allowed_tools": resolution.allowed_tools,
                    "warnings": resolution.warnings,
                    "context_mode": "minimal",
                },
            })
            context_manifest = {
                "mode": "minimal",
                "strategy": "just_in_time_tools",
                "total_budget": 0,
                "total_used": max(0, len(resolution.instructions) // 4),
                "sources": [
                    {
                        "kind": "repository_instructions",
                        "included": bool(resolution.repository_instructions),
                        "tokens": max(0, len(resolution.instructions) // 4),
                        "reason": "persistent project instructions",
                    },
                    {
                        "kind": "skills",
                        "included": bool(skill_metadata),
                        "count": len(skill_metadata),
                        "reason": "request-matched workflow instructions",
                    },
                    {
                        "kind": "repository_index",
                        "included": False,
                        "reason": "deferred until search/read tools request code",
                    },
                ],
            }
            await self._emit({
                "type": "context_selected",
                "summary": "Using project instructions and just-in-time code discovery",
                "data": context_manifest,
            })
            memory_update = await memory_task
            return {
                "context": resolution.instructions,
                "semantic_context": resolution.instructions or None,
                "resolved_skills": skill_metadata,
                "allowed_tools": resolution.allowed_tools,
                "warnings": [*state.warnings, *resolution.warnings],
                "context_manifest": context_manifest,
                **memory_update,
            }

        try:
            package, memory_update = await asyncio.gather(
                self._gateway.build_context(
                    project_root=state.project_root,
                    query=state.user_request,
                ),
                memory_task,
            )

            # 格式化为 LLM-ready 字符串（Phase 1a 兼容）
            assembled = self._assembler.assemble(package)
            skills_enabled = codeagent_config.get_skills_enabled()
            resolution = resolve_project_instructions(
                state.project_root,
                state.user_request,
                skills_enabled=skills_enabled,
            )
            if not skills_enabled:
                resolution.warnings.append("Skills disabled by SKILLS_ENABLED=false")
                await self._emit({
                    "type": "capability_degraded",
                    "summary": "Skill injection is disabled; core tools remain available",
                    "data": {"capability": "skills", "reason": "SKILLS_ENABLED=false"},
                })
            if resolution.instructions:
                assembled = f"{resolution.instructions}\n\n## Retrieved project context\n{assembled}"
            skill_metadata = [skill.public_metadata() for skill in resolution.skills]
            await self._emit({
                "type": "skill_resolved",
                "summary": (
                    f"Matched {len(skill_metadata)} Skill(s)"
                    if skill_metadata else "No Skill matched this request"
                ),
                "data": {
                    "skills": skill_metadata,
                    "repository_instructions": resolution.repository_instructions,
                    "allowed_tools": resolution.allowed_tools,
                    "warnings": resolution.warnings,
                },
            })

            # 语义上下文：相关代码片段
            semantic_ctx = self._format_related_text(package.related_code)

            # 当前文件上下文：文件树文本
            file_tree_text = self._assembler._format_file_tree(
                package.file_tree
            ) if package.file_tree else ""

            budget_report = self._assembler.get_budget_report()
            allocations = [
                {
                    "kind": allocation.section,
                    "included": allocation.used > 0,
                    "budget": allocation.budget,
                    "tokens": allocation.used,
                    "trimmed": allocation.trimmed,
                    "reason": "selected by full repository context policy",
                }
                for allocation in (budget_report.allocations if budget_report else [])
            ]
            context_manifest = {
                "mode": "full",
                "strategy": "budgeted_repository_context",
                "preinject": {
                    "strategy": "parallel_semantic_and_memory",
                    "max_files": codeagent_config.get_context_preinject_max_files(),
                    "code_snippets": len(package.related_code),
                    "memory_items": len(memory_update.get("memory_hits", [])),
                },
                "total_budget": budget_report.total_budget if budget_report else 0,
                "total_used": budget_report.total_used if budget_report else 0,
                "symbol_count": budget_report.symbol_count if budget_report else 0,
                "dependency_count": budget_report.dependency_count if budget_report else 0,
                "tokenizer": budget_report.tokenizer if budget_report else "unknown",
                "trim_reasons": budget_report.trim_reasons if budget_report else [],
                "sources": [
                    {
                        "kind": "repository_instructions",
                        "included": bool(resolution.repository_instructions),
                        "reason": "persistent project instructions",
                    },
                    {
                        "kind": "skills",
                        "included": bool(skill_metadata),
                        "count": len(skill_metadata),
                        "reason": "request-matched workflow instructions",
                    },
                    *allocations,
                ],
            }
            await self._emit({
                "type": "context_selected",
                "summary": (
                    f"Selected {context_manifest['total_used']} of "
                    f"{context_manifest['total_budget']} context tokens"
                ),
                "data": context_manifest,
            })

            result: dict[str, Any] = {
                "file_tree": package.file_tree,
                "context": assembled,
                "semantic_context": semantic_ctx or assembled,
                "current_file_context": file_tree_text,
                "dependency_graph": package.dependency_info,
                "resolved_skills": skill_metadata,
                "allowed_tools": resolution.allowed_tools,
                "warnings": [*state.warnings, *resolution.warnings],
                "context_manifest": context_manifest,
                **memory_update,
            }
            if not skills_enabled:
                result["degraded_mode"] = True
            return result
        except Exception as exc:
            logger.warning(
                "Context build failed, entering degraded mode: %s", exc
            )
            await self._emit({
                "type": "capability_degraded",
                "summary": "Context index unavailable; continuing with direct file and search tools",
                "data": {
                    "capability": "context",
                    "reason": type(exc).__name__,
                    "fallback": "direct_tools",
                },
            })
            return {
                "warnings": [*state.warnings, f"Context build degraded: {exc}"],
                "degraded_mode": True,
                "context_manifest": {
                    "mode": state.context_mode,
                    "strategy": "direct_tool_fallback",
                    "total_budget": 0,
                    "total_used": 0,
                    "sources": [],
                    "degraded": True,
                    "reason": type(exc).__name__,
                },
                **(await memory_task if not memory_task.done() else memory_task.result()),
            }

    async def _prefetch_memory(self, state: AgentState) -> dict[str, Any]:
        """Recall bounded persistent memory before the first reasoning call."""
        if state.memory_mode == "off" or self._memory_gateway is None:
            return {}
        try:
            token_budget = codeagent_config.get_memory_token_budget()
            memory_xml = await self._memory_gateway.recall(
                query=state.user_request,
                token_budget=token_budget,
            )
            if not memory_xml:
                return {"memory_recalled": True, "memory_context": ""}
            from codeagent.memory.audit import parse_memory_hits

            hits = parse_memory_hits(memory_xml, token_budget)[:5]
            await self._emit({
                "type": "memory_recalled",
                "summary": f"Pre-injected {len(hits)} relevant memory item(s)",
                "data": {"hits": hits, "stage": "pre_reasoning"},
            })
            return {
                "memory_recalled": True,
                "memory_context": f"## Relevant Memories\n\n{memory_xml}",
                "memory_hits": hits,
            }
        except Exception as exc:
            logger.warning("Memory pre-injection failed (non-blocking): %s", exc)
            return {}

    def _format_related_text(
        self, related_code: list[Any]
    ) -> str:
        """将相关代码片段列表格式化为文本。"""
        if not related_code:
            return ""

        parts: list[str] = []
        for snippet in related_code:
            header = (
                f"=== {snippet.file_path}:{snippet.start_line}-"
                f"{snippet.end_line} ==="
            )
            parts.append(header)
            parts.append(snippet.code.rstrip("\n"))

        return "\n\n".join(parts)
