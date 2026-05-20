"""Context Node — 上下文收集节点。

接收 IContextGateway 作为依赖，调用 build_context 获取 ContextPackage，
格式化后更新 AgentState 的语义上下文、当前文件上下文、依赖关系图。
"""

from __future__ import annotations

import logging
from typing import Any

from codeagent.context_engine.context_assembler import ContextAssembler
from codeagent.gateway.context_gateway import IContextGateway
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
    ) -> None:
        """初始化 ContextNode。

        Args:
            context_gateway: 上下文引擎 Gateway
            context_assembler: 上下文组装器（可选，默认创建）
        """
        self._gateway = context_gateway
        self._assembler = context_assembler or ContextAssembler()

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
        try:
            package = await self._gateway.build_context(
                project_root=state.project_root,
                query=state.user_request,
            )

            # 格式化为 LLM-ready 字符串（Phase 1a 兼容）
            assembled = self._assembler.assemble(package)

            # 语义上下文：相关代码片段
            semantic_ctx = self._format_related_text(package.related_code)

            # 当前文件上下文：文件树文本
            file_tree_text = self._assembler._format_file_tree(
                package.file_tree
            ) if package.file_tree else ""

            return {
                "file_tree": package.file_tree,
                "context": assembled,
                "semantic_context": semantic_ctx or assembled,
                "current_file_context": file_tree_text,
                "dependency_graph": package.dependency_info,
            }
        except Exception as exc:
            logger.warning(
                "Context build failed, entering degraded mode: %s", exc
            )
            return {
                "errors": [f"Context build failed: {exc}"],
                "degraded_mode": True,
            }

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
