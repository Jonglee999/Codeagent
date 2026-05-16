"""MemoryManager — 记忆系统统一 Facade。

聚合 MemoryStore / MemoryRetriever / MemoryExtractor，
提供简洁接口供上层模块（Context Engine、Orchestration Node）调用。

支持定期维护（session 归档 + confidence 衰减）和工厂方法创建。
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Optional

from codeagent.gateway.memory_gateway import IMemoryGateway
from codeagent.memory.extractor import MemoryExtractor
from codeagent.memory.retriever import MemoryRetriever, RetrievalResult
from codeagent.memory.store import MemoryEntry, MemoryStore, MemoryType

logger = logging.getLogger(__name__)

_DEFAULT_DECAY_DAYS = 30
_DECAY_AMOUNT = 0.1
_DECAY_DELETE_THRESHOLD = 0.2


class MemoryManager(IMemoryGateway):
    """记忆系统统一 Facade。

    Args:
        store: MemoryStore 实例
        retriever: MemoryRetriever 实例
        extractor: MemoryExtractor 实例
    """

    def __init__(
        self,
        store: MemoryStore,
        retriever: MemoryRetriever,
        extractor: MemoryExtractor,
    ) -> None:
        self._store = store
        self._retriever = retriever
        self._extractor = extractor

    # ── IMemoryGateway 接口实现 ────────────────────────────────

    async def recall(self, query: str, token_budget: int = 800) -> str:
        """检索相关记忆并返回 XML 字符串。

        内部更新被检索记忆的 last_accessed_at。

        Args:
            query: 查询字符串
            token_budget: token 预算上限

        Returns:
            XML 格式的记忆字符串，无结果时返回空字符串
        """
        try:
            results = self._retriever.retrieve(query, token_budget=token_budget)
            if not results:
                return ""

            # 更新被检索记忆的 last_accessed_at
            for r in results:
                try:
                    entry = r.entry
                    entry.last_accessed_at = datetime.now()
                    self._store.save(entry, scope=r.scope)
                except Exception as exc:
                    logger.debug(
                        "Failed to update last_accessed_at for %s: %s",
                        r.entry.name, exc,
                    )

            return self._retriever.to_xml(results)
        except Exception as exc:
            logger.warning("Memory recall failed (non-blocking): %s", exc)
            return ""

    def save(self, entry: MemoryEntry, scope: str = "auto") -> None:
        """保存一条记忆。

        Args:
            entry: 记忆条目
            scope: "auto"、"global" 或 "project"
        """
        try:
            self._store.save(entry, scope=scope)
        except Exception as exc:
            logger.warning("Memory save failed (non-blocking): %s", exc)

    async def auto_extract(
        self,
        conversation_history: list[dict],
        trigger: str = "task_complete",
    ) -> list[MemoryEntry]:
        """从对话历史自动提取并保存记忆。

        过滤 is_duplicate=True 和低置信度的候选，非重复的自动保存。

        Args:
            conversation_history: 对话历史
            trigger: 触发原因

        Returns:
            实际保存的 MemoryEntry 列表
        """
        try:
            candidates = await self._extractor.extract_from_conversation(
                conversation_history, trigger=trigger,
            )
        except Exception as exc:
            logger.warning("Memory extraction failed (non-blocking): %s", exc)
            return []

        saved: list[MemoryEntry] = []
        for cand in candidates:
            if cand.is_duplicate:
                logger.debug("Skipping duplicate memory: %s", cand.entry.name)
                continue
            if cand.confidence <= 0.0:
                continue
            try:
                self._store.save(cand.entry)
                saved.append(cand.entry)
                logger.debug("Auto-saved memory: %s", cand.entry.name)
            except Exception as exc:
                logger.warning(
                    "Failed to save extracted memory %s: %s",
                    cand.entry.name, exc,
                )

        return saved

    def forget(self, name: str) -> bool:
        """删除指定记忆。

        Args:
            name: 记忆条目的 name

        Returns:
            bool: 是否成功删除
        """
        try:
            return self._store.delete(name)
        except Exception as exc:
            logger.warning("Memory forget failed (non-blocking): %s", exc)
            return False

    def list_memories(
        self,
        memory_type: Optional[MemoryType] = None,
        scope: str = "both",
    ) -> list[MemoryEntry]:
        """列出记忆条目。

        Args:
            memory_type: 筛选特定类型，None 表示全部
            scope: "global"、"project" 或 "both"

        Returns:
            list[MemoryEntry]: 记忆条目列表
        """
        try:
            return self._store.list_all(memory_type=memory_type, scope=scope)
        except Exception as exc:
            logger.warning("Memory list failed (non-blocking): %s", exc)
            return []

    # ── 维护方法 ───────────────────────────────────────────────

    def run_maintenance(self) -> dict:
        """执行定期维护。

        1. 归档过期 session 记忆
        2. 对超过 30 天未更新的记忆降低 confidence

        Returns:
            dict: 维护报告 {"archived_sessions": int, "decayed": int}
        """
        result: dict = {"archived_sessions": 0, "decayed": 0}

        try:
            result["archived_sessions"] = self._store.archive_expired_sessions()
        except Exception as exc:
            logger.warning("Session archiving failed (non-blocking): %s", exc)

        try:
            result["decayed"] = self._decay_confidence()
        except Exception as exc:
            logger.warning("Confidence decay failed (non-blocking): %s", exc)

        return result

    def _decay_confidence(self) -> int:
        """对超过 30 天未更新的记忆降低 confidence。

        每超过 30 天（以 updated_at 为准）confidence -= 0.1。
        confidence < 0.2 时删除该记忆。

        Returns:
            int: 受影响的条目数量（含被删除的）
        """
        count = 0
        cutoff = datetime.now() - timedelta(days=_DEFAULT_DECAY_DAYS)

        all_entries = self._store.list_all()
        for entry in all_entries:
            if entry.updated_at >= cutoff:
                continue

            # 超过 30 天未更新，降低 confidence
            entry.confidence = max(0.0, entry.confidence - _DECAY_AMOUNT)
            count += 1

            if entry.confidence < _DECAY_DELETE_THRESHOLD:
                # 置信度过低，删除
                try:
                    self._store.delete(entry.name)
                    logger.debug(
                        "Memory deleted due to low confidence: %s (%.2f)",
                        entry.name, entry.confidence,
                    )
                except Exception as exc:
                    logger.warning(
                        "Failed to delete low-confidence memory %s: %s",
                        entry.name, exc,
                    )
            else:
                # 写回更新后的 confidence
                try:
                    self._store.save(entry)
                except Exception as exc:
                    logger.warning(
                        "Failed to update confidence for %s: %s",
                        entry.name, exc,
                    )

        return count

    # ── 工厂方法 ───────────────────────────────────────────────

    @classmethod
    def create(
        cls,
        project_path: Optional[str] = None,
        llm_client: Optional[Callable[..., Any]] = None,
        auto_mode: bool = False,
    ) -> MemoryManager:
        """工厂方法：从 config.py 读取路径配置，自动构建所有组件。

        Args:
            project_path: 项目根目录路径，None 时只使用全局存储
            llm_client: LLM 调用函数，MemoryExtractor 依赖
            auto_mode: MemoryExtractor 是否自动保存模式

        Returns:
            MemoryManager 实例
        """
        from codeagent import config

        global_root = config.get_memory_global_root()
        project_root: Optional[Path] = None
        if project_path:
            project_root = config.get_memory_project_root(project_path)

        store = MemoryStore(global_root=global_root, project_root=project_root)
        retriever = MemoryRetriever(store=store)
        extractor = MemoryExtractor(
            store=store,
            retriever=retriever,
            llm_client=llm_client or cls._noop_llm,
            auto_mode=auto_mode,
        )

        return cls(store=store, retriever=retriever, extractor=extractor)

    @staticmethod
    async def _noop_llm(**kwargs: Any) -> Any:
        """默认 LLM 客户端（什么都不做，返回空响应）。"""
        from types import SimpleNamespace

        return SimpleNamespace(choices=[])
