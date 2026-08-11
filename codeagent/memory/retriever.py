"""MemoryRetriever — 轻量级混合检索。

支持关键词匹配 + 可选向量相似度，合并两级存储结果，
输出 XML 格式供 System Prompt 注入。
"""

from __future__ import annotations

import importlib.util
import logging
import re
import time
from dataclasses import dataclass

from codeagent.interaction.api.metrics import observe_memory_retrieval
from codeagent.memory.store import MemoryEntry, MemoryStore, MemoryType

logger = logging.getLogger(__name__)

# 类型在 token 截断时的优先级（越小越优先保留）
_TYPE_PRIORITY: dict[MemoryType, int] = {
    MemoryType.FEEDBACK: 0,
    MemoryType.CODE_PATTERN: 1,
    MemoryType.USER: 2,
    MemoryType.PROJECT: 3,
    MemoryType.SESSION: 4,
}


@dataclass
class RetrievalResult:
    """一条检索结果。"""

    entry: MemoryEntry
    score: float  # 0.0~1.0 相关性分数
    scope: str  # "global" 或 "project"
    match_reason: str  # 匹配原因，用于调试


class MemoryRetriever:
    """记忆检索器 — 混合检索 + XML 输出。

    Args:
        store: MemoryStore 实例
        use_vector: 是否启用向量检索（需要 sentence-transformers）
        top_k: 最多返回条数
        high_threshold: 高相关性阈值（≥此值直接注入）
        medium_threshold: 中等相关性阈值（≥此值按预算注入）
    """

    def __init__(
        self,
        store: MemoryStore,
        use_vector: bool = False,
        top_k: int = 10,
        high_threshold: float = 0.7,
        medium_threshold: float = 0.4,
    ) -> None:
        self._store = store
        self._use_vector = use_vector
        self._top_k = top_k
        self._high_threshold = high_threshold
        self._medium_threshold = medium_threshold

    # ── 公共接口 ──────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        token_budget: int = 800,
    ) -> list[RetrievalResult]:
        """检索与 query 相关的记忆。

        步骤：
        1. 从 store.list_all() 获取所有记忆
        2. 对每条记忆计算关键词分数
        3. 若 use_vector=True，计算向量分数并加权合并
        4. 过滤 score < medium_threshold 的结果
        5. 项目层记忆 score += 0.05 bonus
        6. 按 score 降序排列 + 类型优先级，取 top_k
        7. 按 token_budget 截断

        Args:
            query: 查询字符串
            token_budget: token 预算上限，默认 800

        Returns:
            按分数降序排列的 RetrievalResult 列表
        """
        t0 = time.monotonic()
        try:
            return self._retrieve_impl(query, token_budget)
        finally:
            observe_memory_retrieval(time.monotonic() - t0)

    def _retrieve_impl(
        self,
        query: str,
        token_budget: int = 800,
    ) -> list[RetrievalResult]:
        if not query or not query.strip():
            return []

        all_entries = self._store.list_all()
        results: list[RetrievalResult] = []

        for entry in all_entries:
            keyword_score = self._keyword_score(query, entry)
            score = keyword_score

            if self._use_vector:
                vector_score = self._vector_score(query, entry)
                score = keyword_score * 0.6 + vector_score * 0.4

            if score < self._medium_threshold:
                continue

            scope = self._determine_scope(entry)
            if scope == "project":
                score = min(1.0, score + 0.05)

            results.append(RetrievalResult(
                entry=entry,
                score=min(1.0, score),
                scope=scope,
                match_reason=f"keyword={keyword_score:.2f}",
            ))

        # 按分数降序 + 类型优先级排序
        results.sort(key=lambda r: (
            -r.score,
            _TYPE_PRIORITY.get(r.entry.memory_type, 5),
        ))

        # 按 token_budget 截断
        results = self._truncate_by_budget(results, token_budget)

        return results[:self._top_k]

    def to_xml(self, results: list[RetrievalResult]) -> str:
        """将检索结果序列化为 XML 格式。

        Args:
            results: 检索结果列表

        Returns:
            XML 字符串，无结果时返回空字符串
        """
        if not results:
            return ""

        from xml.sax.saxutils import escape

        parts = ["<relevant_memories>"]
        for r in results:
            body_escaped = escape(r.entry.body)
            parts.append(
                f'  <memory type="{escape(r.entry.memory_type.value)}" '
                f'name="{escape(r.entry.name)}" '
                f'score="{r.score:.2f}" '
                f'scope="{escape(r.scope)}">'
            )
            parts.append(f"    {body_escaped}")
            parts.append("  </memory>")
        parts.append("</relevant_memories>")

        return "\n".join(parts)

    # ── 关键词匹配 ────────────────────────────────────────────

    def _keyword_score(self, query: str, entry: MemoryEntry) -> float:
        """计算关键词匹配分数（0.0~1.0）。

        将 query 分词后计算在 entry 中的覆盖率，
        description 中命中的词加权 1.5x。
        """
        query_words = self._tokenize(query)
        if not query_words:
            return 0.0

        description_words = self._tokenize(entry.description)
        body_words = self._tokenize(entry.body)

        # entry 总词集
        entry_words = description_words | body_words

        if not entry_words:
            return 0.0

        weighted_hits = 0
        for qw in query_words:
            if qw in entry_words:
                if qw in description_words:
                    weighted_hits += 1.5
                else:
                    weighted_hits += 1.0

        score = weighted_hits / len(query_words)
        return min(1.0, score)

    # ── 向量检索（可选） ──────────────────────────────────────

    def _vector_score(self, query: str, entry: MemoryEntry) -> float:
        """计算向量相似度分数（0.0~1.0）。

        sentence-transformers 不可用时返回 0.0，优雅降级。
        """
        if importlib.util.find_spec("sentence_transformers") is None:
            logger.debug("sentence-transformers not available, vector score = 0.0")
            return 0.0

        # 延迟加载模型（只在首次调用时加载）
        model = self._get_vector_model()
        if model is None:
            return 0.0

        try:
            query_emb = model.encode([query], normalize_embeddings=True)
            entry_text = f"{entry.description} {entry.body}"
            entry_emb = model.encode([entry_text], normalize_embeddings=True)
            similarity = float(query_emb @ entry_emb.T)
            return max(0.0, min(1.0, similarity))
        except Exception as exc:
            logger.warning("Vector score computation failed: %s", exc)
            return 0.0

    def _get_vector_model(self):
        """获取或创建向量模型实例。"""
        if not hasattr(self, "_vector_model"):
            try:
                from sentence_transformers import SentenceTransformer
                self._vector_model = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception as exc:
                logger.warning("Failed to load vector model: %s", exc)
                self._vector_model = None
        return self._vector_model

    # ── 内部辅助 ──────────────────────────────────────────────

    def _determine_scope(self, entry: MemoryEntry) -> str:
        """判断记忆属于全局层还是项目层。"""
        if self._store._project_root:
            file_path = (
                self._store._project_root
                / entry.memory_type.value
                / f"{entry.name}.md"
            )
            if file_path.exists():
                return "project"
        return "global"

    def _truncate_by_budget(
        self,
        results: list[RetrievalResult],
        budget: int,
    ) -> list[RetrievalResult]:
        """按 token 预算截断。

        高相关结果（≥high_threshold）始终保留，
        中等相关结果按类型优先级保留。
        """
        kept_high: list[RetrievalResult] = []
        kept_medium: list[RetrievalResult] = []
        budget_remaining = budget

        for r in results:
            tokens = max(1, len(r.entry.body) // 4)
            if r.score >= self._high_threshold:
                kept_high.append(r)
                budget_remaining -= tokens
            elif r.score >= self._medium_threshold:
                if budget_remaining >= tokens:
                    kept_medium.append(r)
                    budget_remaining -= tokens

        return kept_high + kept_medium

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """分词：小写 + 按非字母字符分割。"""
        return set(re.findall(r"[a-zA-Z0-9一-鿿]+", text.lower()))
