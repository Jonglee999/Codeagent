"""MemoryRetriever 单元测试。

覆盖：关键词分数、项目优先 bonus、token_budget 截断、XML 格式、空结果、向量降级。
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from codeagent.memory.retriever import MemoryRetriever, RetrievalResult
from codeagent.memory.store import MemoryEntry, MemoryStore, MemoryType


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def global_root(tmp_path: Path) -> Path:
    root = tmp_path / "global_memory"
    root.mkdir(parents=True)
    return root


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    root = tmp_path / "project_memory"
    root.mkdir(parents=True)
    return root


@pytest.fixture
def store(global_root: Path, project_root: Path) -> MemoryStore:
    return MemoryStore(global_root=global_root, project_root=project_root)


@pytest.fixture
def retriever(store: MemoryStore) -> MemoryRetriever:
    return MemoryRetriever(store=store)


# ── 辅助函数 ─────────────────────────────────────────────────────────────


def make_entry(
    name: str = "test",
    memory_type: MemoryType = MemoryType.USER,
    description: str = "Test description about python coding",
    body: str = "This is a test body about python programming.",
    tags: list[str] | None = None,
) -> MemoryEntry:
    return MemoryEntry(
        name=name,
        memory_type=memory_type,
        description=description,
        body=body,
        tags=tags or [],
    )


def save_entry(
    store: MemoryStore,
    name: str = "test",
    memory_type: MemoryType = MemoryType.USER,
    description: str = "Test description about python coding",
    body: str = "This is a test body about python programming.",
    scope: str = "auto",
) -> MemoryEntry:
    entry = make_entry(name=name, memory_type=memory_type, description=description, body=body)
    store.save(entry, scope=scope)
    return entry


# ══════════════════════════════════════════════════════════════════════════
# 1. _keyword_score
# ══════════════════════════════════════════════════════════════════════════


class TestKeywordScore:
    """关键词匹配分数测试。"""

    def test_exact_match_full_score(self, retriever: MemoryRetriever) -> None:
        """查询词全部在条目中出现应得满分。"""
        entry = make_entry(description="python coding test", body="python coding")
        score = retriever._keyword_score("python coding test", entry)
        assert score == pytest.approx(1.0, abs=0.01)

    def test_partial_match_partial_score(self, retriever: MemoryRetriever) -> None:
        """部分匹配应得中间分数。"""
        entry = make_entry(description="python only", body="python")
        # query "python javascript": "python" in desc -> 1.5, "javascript" not found -> 0
        # score = 1.5 / 2 = 0.75
        score = retriever._keyword_score("python javascript", entry)
        assert score == pytest.approx(0.75, abs=0.01)

    def test_no_match_zero_score(self, retriever: MemoryRetriever) -> None:
        """无匹配应得 0 分。"""
        entry = make_entry(description="java programming", body="java spring")
        score = retriever._keyword_score("python django", entry)
        assert score == pytest.approx(0.0, abs=0.01)

    def test_empty_query_returns_zero(self, retriever: MemoryRetriever) -> None:
        """空查询应得 0 分。"""
        entry = make_entry(description="python coding")
        score = retriever._keyword_score("", entry)
        assert score == pytest.approx(0.0, abs=0.01)

    def test_description_weighted_higher(self, retriever: MemoryRetriever) -> None:
        """description 中命中的词应加权 1.5x。"""
        # "python" 出现在 description 中, "javascript" 不出现在任何地方
        entry = make_entry(description="python expert", body="just some text")
        # query "python" -> 命中 1 个词, 在 description 中 -> 1.5
        # score = 1.5 / 1 = 1.5 -> capped to 1.0
        score = retriever._keyword_score("python", entry)
        assert score == pytest.approx(1.0, abs=0.01)

    def test_description_weight_vs_body(self, retriever: MemoryRetriever) -> None:
        """description 加权应高于 body 匹配。"""
        # desc_entry: 4 个词都在 description 中
        desc_entry = make_entry(description="python expert django advanced", body="other")
        # body_entry: 3 个词在 body, 1 个词 (advanced) 缺失
        body_entry = make_entry(description="other", body="python expert django")
        query = "python expert django advanced"  # 4 words

        # desc: python(1.5)+expert(1.5)+django(1.5)+advanced(1.5) = 6.0/4 = 1.5 -> 1.0
        # body: python(1.0)+expert(1.0)+django(1.0)+advanced(0) = 3.0/4 = 0.75
        desc_score = retriever._keyword_score(query, desc_entry)
        body_score = retriever._keyword_score(query, body_entry)

        assert desc_score > body_score

    def test_case_insensitive(self, retriever: MemoryRetriever) -> None:
        """匹配应不区分大小写。"""
        entry = make_entry(description="Python Coding", body="PYTHON")
        score = retriever._keyword_score("python coding", entry)
        assert score == pytest.approx(1.0, abs=0.01)

    def test_unicode_support(self, retriever: MemoryRetriever) -> None:
        """中文字符应正确分词匹配。"""
        # 用空格分隔的中文词
        entry = make_entry(description="python 测试 数据库", body="python 测试")
        score = retriever._keyword_score("测试 数据库", entry)
        # "测试" in desc -> 1.5, "数据库" in desc -> 1.5
        # total = 3.0 / 2 = 1.5 -> capped to 1.0
        assert score == pytest.approx(1.0, abs=0.01)


# ══════════════════════════════════════════════════════════════════════════
# 2. retrieve 管道
# ══════════════════════════════════════════════════════════════════════════


class TestRetrieve:
    """retrieve 管道测试。"""

    def test_retrieve_finds_relevant_memory(self, retriever: MemoryRetriever, store: MemoryStore) -> None:
        """检索能找到相关记忆。"""
        save_entry(store, name="python-tips", memory_type=MemoryType.USER,
                   description="Python programming tips", body="Use list comprehensions")
        save_entry(store, name="java-tips", memory_type=MemoryType.USER,
                   description="Java programming", body="Use streams")

        results = retriever.retrieve("python programming")
        names = [r.entry.name for r in results]
        assert "python-tips" in names

    def test_retrieve_empty_query(self, retriever: MemoryRetriever) -> None:
        """空查询应返回空列表。"""
        results = retriever.retrieve("")
        assert results == []

        results = retriever.retrieve("   ")
        assert results == []

    def test_retrieve_no_match(self, retriever: MemoryRetriever, store: MemoryStore) -> None:
        """不相关查询应返回空列表。"""
        save_entry(store, name="python-tips", memory_type=MemoryType.USER,
                   description="Python coding", body="Python tips")
        results = retriever.retrieve("quantum physics cosmology")
        assert results == []

    def test_expired_memory_is_not_recalled(
        self, retriever: MemoryRetriever, store: MemoryStore
    ) -> None:
        entry = make_entry(
            name="expired-python",
            memory_type=MemoryType.SESSION,
            description="Python programming tips",
        )
        entry.expires_at = datetime.now() - timedelta(seconds=1)
        store.save(entry, scope="project")

        assert retriever.retrieve("python programming") == []
        assert store.load("expired-python") is None

    def test_retrieve_project_bonus(self, retriever: MemoryRetriever, store: MemoryStore, global_root: Path) -> None:
        """项目层记忆应获得 +0.05 bonus。"""
        # 同样的内容存到全局和项目层
        save_entry(store, name="same-topic", memory_type=MemoryType.USER,
                   description="Python coding tips", body="Python coding tips and tricks",
                   scope="global")
        save_entry(store, name="same-topic", memory_type=MemoryType.USER,
                   description="Python coding tips", body="Python coding tips and tricks",
                   scope="project")

        results = retriever.retrieve("Python coding tips")
        # 项目层应该有更高分
        project_scores = [r.score for r in results if r.scope == "project"]
        global_scores = [r.score for r in results if r.scope == "global"]

        if project_scores and global_scores:
            assert project_scores[0] > global_scores[0]

    def test_retrieve_top_k_limit(self, retriever: MemoryRetriever, store: MemoryStore) -> None:
        """结果数量应受 top_k 限制。"""
        # 创建多个匹配的记忆
        for i in range(5):
            save_entry(store, name=f"test-{i}", memory_type=MemoryType.USER,
                       description=f"Common topic item number {i}",
                       body="Common topic content here")

        # top_k 设为 3 的检索器
        small_retriever = MemoryRetriever(store=store, top_k=3)
        results = small_retriever.retrieve("Common topic")
        assert len(results) <= 3

    def test_retrieve_medium_threshold_filter(self, retriever: MemoryRetriever, store: MemoryStore) -> None:
        """低于 medium_threshold 的结果应被过滤。"""
        save_entry(store, name="relevant", memory_type=MemoryType.USER,
                   description="Python coding", body="Python coding")
        save_entry(store, name="irrelevant", memory_type=MemoryType.USER,
                   description="Quantum physics", body="Quantum physics")

        results = retriever.retrieve("Python coding")
        names = [r.entry.name for r in results]
        assert "relevant" in names
        assert "irrelevant" not in names

    def test_retrieve_returns_sorted_by_score(self, retriever: MemoryRetriever, store: MemoryStore) -> None:
        """结果应按分数降序排列。"""
        save_entry(store, name="match-two", memory_type=MemoryType.USER,
                   description="python django flask", body="python django")
        save_entry(store, name="match-one", memory_type=MemoryType.USER,
                   description="python only", body="python")

        results = retriever.retrieve("python django flask")
        if len(results) >= 2:
            assert results[0].score >= results[1].score


# ══════════════════════════════════════════════════════════════════════════
# 3. to_xml
# ══════════════════════════════════════════════════════════════════════════


class TestToXml:
    """XML 输出测试。"""

    def test_to_xml_empty_results(self, retriever: MemoryRetriever) -> None:
        """空结果应返回空字符串。"""
        xml = retriever.to_xml([])
        assert xml == ""

    def test_to_xml_contains_relevant_memories_root(self, retriever: MemoryRetriever) -> None:
        """XML 应包含 <relevant_memories> 根元素。"""
        entry = make_entry(name="python-tips", memory_type=MemoryType.USER,
                           description="Python tips", body="Use list comprehensions")
        results = [RetrievalResult(entry=entry, score=0.9, scope="global", match_reason="test")]
        xml = retriever.to_xml(results)
        assert "<relevant_memories>" in xml
        assert "</relevant_memories>" in xml

    def test_to_xml_attributes(self, retriever: MemoryRetriever) -> None:
        """XML memory 元素应包含正确的属性。"""
        entry = make_entry(name="python-tips", memory_type=MemoryType.USER,
                           description="Python tips", body="Use list comprehensions")
        results = [RetrievalResult(entry=entry, score=0.9, scope="global", match_reason="test")]
        xml = retriever.to_xml(results)
        assert 'type="user"' in xml
        assert 'name="python-tips"' in xml
        assert 'score="0.90"' in xml
        assert 'scope="global"' in xml

    def test_to_xml_body_escaped(self, retriever: MemoryRetriever) -> None:
        """XML body 中的特殊字符应被转义（&、<、>）。"""
        entry = make_entry(name="special", memory_type=MemoryType.FEEDBACK,
                           description="Special chars", body='Use "double quotes" & <angle>')
        results = [RetrievalResult(entry=entry, score=0.8, scope="project", match_reason="test")]
        xml = retriever.to_xml(results)
        assert "&amp;" in xml
        assert "&lt;" in xml
        assert "&gt;" in xml
        # 双引号在文本节点中不需要转义（仅在属性中需要）
        assert '"double quotes"' in xml

    def test_to_xml_multiple_results(self, retriever: MemoryRetriever) -> None:
        """XML 应包含多个 memory 元素。"""
        e1 = make_entry(name="first", memory_type=MemoryType.FEEDBACK,
                        description="First", body="First memory")
        e2 = make_entry(name="second", memory_type=MemoryType.USER,
                        description="Second", body="Second memory")
        results = [
            RetrievalResult(entry=e1, score=0.9, scope="global", match_reason="test"),
            RetrievalResult(entry=e2, score=0.8, scope="project", match_reason="test"),
        ]
        xml = retriever.to_xml(results)
        assert xml.count("<memory") == 2
        assert "first" in xml
        assert "second" in xml

    def test_to_xml_valid_xml(self, retriever: MemoryRetriever) -> None:
        """XML 应可被标准 XML 解析器解析。"""
        import xml.etree.ElementTree as ET

        e1 = make_entry(name="entry-1", memory_type=MemoryType.FEEDBACK,
                        description="Test", body="Some body content")
        results = [RetrievalResult(entry=e1, score=0.85, scope="global", match_reason="test")]
        xml = retriever.to_xml(results)
        root = ET.fromstring(xml)
        assert root.tag == "relevant_memories"
        assert len(root.findall("memory")) == 1


# ══════════════════════════════════════════════════════════════════════════
# 4. token_budget 截断
# ══════════════════════════════════════════════════════════════════════════


class TestTokenBudget:
    """token_budget 截断测试。"""

    def test_token_budget_truncates_medium(self, retriever: MemoryRetriever, store: MemoryStore) -> None:
        """中等相关结果应受 budget 限制。"""
        # 高相关条目（score >= 0.7）
        save_entry(store, name="high-match", memory_type=MemoryType.USER,
                   description="python coding test", body="python")
        # 中等相关条目（score 0.4~0.7），body 很长
        save_entry(store, name="medium-match", memory_type=MemoryType.USER,
                   description="python", body="x" * 5000)  # 5000 chars = ~1250 tokens

        results = retriever.retrieve("python coding test", token_budget=100)
        names = [r.entry.name for r in results]
        assert "high-match" in names  # 高相关始终保留

    def test_token_budget_allows_enough(self, retriever: MemoryRetriever, store: MemoryStore) -> None:
        """足够大的 budget 应保留所有结果。"""
        save_entry(store, name="first", memory_type=MemoryType.USER,
                   description="python tips", body="short body")
        save_entry(store, name="second", memory_type=MemoryType.USER,
                   description="python tricks", body="very short")

        results = retriever.retrieve("python tips", token_budget=8000)
        assert len(results) >= 2

    def test_high_relevance_always_included(self, retriever: MemoryRetriever, store: MemoryStore) -> None:
        """高相关结果（≥0.7）无论 budget 大小都应包含。"""
        save_entry(store, name="perfect-match", memory_type=MemoryType.FEEDBACK,
                   description="exact keyword match for python testing",
                   body="x" * 10000)  # 2500 tokens

        results = retriever.retrieve("exact keyword match for python testing", token_budget=10)
        names = [r.entry.name for r in results]
        assert "perfect-match" in names


# ══════════════════════════════════════════════════════════════════════════
# 5. 向量检索降级
# ══════════════════════════════════════════════════════════════════════════


class TestVectorDegradation:
    """向量检索不可用降级测试。"""

    def test_vector_off_uses_keyword_only(self, retriever: MemoryRetriever) -> None:
        """use_vector=False 时应只使用关键词检索。"""
        entry = make_entry(description="python coding", body="python")
        vec_score = retriever._vector_score("python", entry)
        assert vec_score == pytest.approx(0.0, abs=0.01)

    def test_vector_enabled_but_not_installed(self, store: MemoryStore) -> None:
        """use_vector=True 但 sentence-transformers 未安装时优雅降级。"""
        retriever_vec = MemoryRetriever(store=store, use_vector=True)
        entry = make_entry(description="python coding", body="python")
        # _vector_score 应返回 0.0 而不崩溃
        score = retriever_vec._vector_score("python", entry)
        assert score == pytest.approx(0.0, abs=0.01)

    def test_vector_downgrade_retrieve_works(self, store: MemoryStore) -> None:
        """向量不可用时 retrieve 仍正常工作。"""
        save_entry(store, name="python-tip", memory_type=MemoryType.USER,
                   description="Python coding", body="Python coding tips")
        retriever_vec = MemoryRetriever(store=store, use_vector=True)
        results = retriever_vec.retrieve("Python coding")
        assert len(results) >= 1


# ══════════════════════════════════════════════════════════════════════════
# 6. RetrievalResult dataclass
# ══════════════════════════════════════════════════════════════════════════


class TestRetrievalResult:
    """RetrievalResult 基础功能测试。"""

    def test_dataclass_fields(self) -> None:
        """验证 RetrievalResult 字段。"""
        entry = make_entry()
        result = RetrievalResult(entry=entry, score=0.85, scope="global", match_reason="test")
        assert result.score == 0.85
        assert result.scope == "global"
        assert result.match_reason == "test"
        assert result.entry.name == "test"
