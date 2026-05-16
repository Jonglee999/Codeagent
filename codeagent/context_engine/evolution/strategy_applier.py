"""StrategyApplier — 策略应用器。

在任务启动时从 StrategyStore 检索相关策略并注入 System Prompt，
在任务完成后根据实际效果更新策略置信度。

检索时机：Planning Node 构建 System Prompt 时
注入位置：System Prompt 的 Memory 层之后（新增 Strategy 层）
效果追踪：任务完成后记录应用中哪些策略被使用及其效果
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from codeagent.context_engine.evolution.strategy_extractor import Strategy
from codeagent.context_engine.evolution.strategy_store import StrategyStore

logger = logging.getLogger(__name__)


class StrategyApplier:
    """策略应用器——在任务启动时检索相关策略并注入 System Prompt。

    任务完成后根据实际效果更新策略置信度。

    Args:
        store: StrategyStore 实例
        confidence_threshold: 最低置信度阈值（默认 0.5）
        max_strategies: 每次注入的最大策略数（默认 5）
    """

    def __init__(
        self,
        store: StrategyStore,
        confidence_threshold: float = 0.5,
        max_strategies: int = 5,
    ) -> None:
        self._store = store
        self._confidence_threshold = confidence_threshold
        self._max_strategies = max_strategies

    # ── 公开接口 ──────────────────────────────────────────────

    async def get_relevant_strategies(
        self,
        task_description: str,
        top_k: int = 5,
    ) -> list[Strategy]:
        """检索与当前任务相关的策略。

        1. 调用 store.search(query=task_description, top_k=top_k)
        2. 过滤 confidence >= confidence_threshold
        3. 按 confidence 降序排列
        4. 取 top_k（不超过 max_strategies）

        Args:
            task_description: 任务描述（用户请求）
            top_k: 检索条数

        Returns:
            相关策略列表（按 confidence 降序）
        """
        try:
            strategies = self._store.search(query=task_description, top_k=top_k)
        except Exception as exc:
            logger.warning("Strategy search failed (non-blocking): %s", exc)
            return []

        # 过滤低置信度
        filtered = [
            s for s in strategies
            if s.confidence >= self._confidence_threshold
        ]

        # 按 confidence 降序
        filtered.sort(key=lambda s: s.confidence, reverse=True)

        # 取 top_k（不超过 max_strategies）
        return filtered[:min(top_k, self._max_strategies)]

    def format_for_prompt(self, strategies: list[Strategy]) -> str:
        """将策略列表格式化为 XML 块，供 System Prompt 注入。

        输出格式：
        <strategies>
        <strategy confidence="0.85" category="workflow">
          <condition>当需要在 FastAPI 项目中添加新路由时</condition>
          <action>应同时在 tests/ 下添加对应的测试文件</action>
          <rationale>在 3 次任务中，未同步写测试导致 CI 失败 2 次</rationale>
        </strategy>
        ...
        </strategies>

        Args:
            strategies: 策略列表

        Returns:
            XML 格式的字符串，无策略时返回空字符串
        """
        if not strategies:
            return ""

        parts = ["<strategies>"]
        for s in strategies:
            parts.append(
                f'<strategy confidence="{s.confidence}" category="{s.category}">'
            )
            parts.append(f"  <condition>{self._xml_escape(s.condition)}</condition>")
            parts.append(f"  <action>{self._xml_escape(s.action)}</action>")
            parts.append(f"  <rationale>{self._xml_escape(s.rationale)}</rationale>")
            parts.append("</strategy>")
        parts.append("</strategies>")

        return "\n".join(parts)

    async def record_outcome(
        self,
        strategy_ids: list[str],
        success: bool,
    ) -> None:
        """任务完成后更新策略置信度。

        对每条策略调用 store.increment_applied() 和 store.update_confidence()。
        异常时只记录日志，不抛异常。

        Args:
            strategy_ids: 应用的策略 ID 列表
            success: 任务是否成功
        """
        if not strategy_ids:
            return

        for sid in strategy_ids:
            try:
                self._store.increment_applied(sid, success=success)
                self._store.update_confidence(sid, success=success)
            except Exception as exc:
                logger.warning(
                    "Failed to record outcome for strategy %s (non-blocking): %s",
                    sid, exc,
                )

    # ── 辅助 ──────────────────────────────────────────────────

    @staticmethod
    def _xml_escape(text: str) -> str:
        """对 XML 特殊字符进行转义。

        Args:
            text: 原始文本

        Returns:
            转义后的文本
        """
        text = text.replace("&", "&amp;")
        text = text.replace("<", "&lt;")
        text = text.replace(">", "&gt;")
        text = text.replace('"', "&quot;")
        text = text.replace("'", "&apos;")
        return text
