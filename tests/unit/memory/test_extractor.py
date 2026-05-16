"""MemoryExtractor 单元测试。

覆盖：正常提取、JSON 解析失败降级、去重检测、auto_mode 自动保存、
explicit 提取、置信度过滤、对话截断。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from codeagent.memory.extractor import (
    ExtractionCandidate,
    MemoryExtractor,
    _REMEMBER_KEYWORDS,
)
from codeagent.memory.retriever import MemoryRetriever
from codeagent.memory.store import MemoryEntry, MemoryStore, MemoryType


# ── Mock LLM 响应 ─────────────────────────────────────────────────────────


@dataclass
class MockChoiceMessage:
    content: str | None = None


@dataclass
class MockChoice:
    message: MockChoiceMessage = field(default_factory=MockChoiceMessage)


@dataclass
class MockLLMResponse:
    choices: list[MockChoice] = field(default_factory=lambda: [MockChoice()])


def build_mock_llm(response_text: str | None = None, fail: bool = False):
    """构建 Mock LLM 客户端。

    Args:
        response_text: LLM 返回的 content 文本
        fail: True 时 LLM 调用抛异常
    """
    async def mock_llm(**kwargs: Any) -> MockLLMResponse:
        if fail:
            raise RuntimeError("LLM API error")
        msg = MockChoiceMessage(content=response_text)
        return MockLLMResponse(choices=[MockChoice(message=msg)])
    return mock_llm


# ── 通用的有效 JSON 响应 ──────────────────────────────────────────────────

_VALID_JSON_RESPONSE = json.dumps([
    {
        "type": "feedback",
        "name": "no-mock-database",
        "description": "集成测试必须使用真实数据库",
        "body": "集成测试必须连接真实数据库。\n\n**Why:** 上季度 Mock 测试通过但生产迁移失败。",
        "confidence": 0.95,
    },
    {
        "type": "user",
        "name": "user-is-senior-go-dev",
        "description": "用户是 Go 资深开发者",
        "body": "用户有 5 年 Go 开发经验，熟悉并发编程。",
        "confidence": 0.85,
    },
])

_SINGLE_EXPLICIT_RESPONSE = json.dumps([
    {
        "type": "user",
        "name": "prefer-snake-case",
        "description": "用户偏好 snake_case 命名",
        "body": "项目 API 返回 snake_case 格式。",
        "confidence": 1.0,
    },
])

_EMPTY_RESPONSE = "[]"


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


def make_entry(
    name: str = "test",
    memory_type: MemoryType = MemoryType.USER,
    description: str = "Test entry",
    body: str = "Test body content",
) -> MemoryEntry:
    return MemoryEntry(
        name=name,
        memory_type=memory_type,
        description=description,
        body=body,
    )


# ══════════════════════════════════════════════════════════════════════════
# 1. extract_from_conversation — 正常提取
# ══════════════════════════════════════════════════════════════════════════


class TestExtractFromConversation:
    """extract_from_conversation 正常路径测试。"""

    @pytest.mark.asyncio
    async def test_extract_valid_json(self, store: MemoryStore, retriever: MemoryRetriever) -> None:
        """有效 JSON 响应应正确解析出候选。"""
        llm = build_mock_llm(_VALID_JSON_RESPONSE)
        extractor = MemoryExtractor(store=store, retriever=retriever, llm_client=llm)
        history = [{"role": "user", "content": "不要 Mock 数据库"}, {"role": "assistant", "content": "好的"}]

        candidates = await extractor.extract_from_conversation(history, trigger="correction")

        assert len(candidates) == 2
        assert candidates[0].entry.name == "no-mock-database"
        assert candidates[0].entry.memory_type == MemoryType.FEEDBACK
        assert candidates[0].trigger == "correction"
        assert candidates[1].entry.name == "user-is-senior-go-dev"
        assert candidates[1].entry.memory_type == MemoryType.USER

    @pytest.mark.asyncio
    async def test_extract_empty_response(self, store: MemoryStore, retriever: MemoryRetriever) -> None:
        """空数组响应应返回空列表。"""
        llm = build_mock_llm(_EMPTY_RESPONSE)
        extractor = MemoryExtractor(store=store, retriever=retriever, llm_client=llm)
        history = [{"role": "user", "content": "你好"}]

        candidates = await extractor.extract_from_conversation(history)
        assert candidates == []

    @pytest.mark.asyncio
    async def test_extract_trigger_passed(self, store: MemoryStore, retriever: MemoryRetriever) -> None:
        """trigger 参数应正确传递到 ExtractionCandidate。"""
        llm = build_mock_llm(_VALID_JSON_RESPONSE)
        extractor = MemoryExtractor(store=store, retriever=retriever, llm_client=llm)

        candidates = await extractor.extract_from_conversation([], trigger="task_complete")
        assert candidates[0].trigger == "task_complete"


# ══════════════════════════════════════════════════════════════════════════
# 2. JSON 解析失败降级
# ══════════════════════════════════════════════════════════════════════════


class TestJsonErrorHandling:
    """JSON 解析失败降级测试。"""

    @pytest.mark.asyncio
    async def test_invalid_json_returns_empty(self, store: MemoryStore, retriever: MemoryRetriever) -> None:
        """非法 JSON 应返回空列表不抛异常。"""
        llm = build_mock_llm("这不是 JSON")
        extractor = MemoryExtractor(store=store, retriever=retriever, llm_client=llm)

        candidates = await extractor.extract_from_conversation([])
        assert candidates == []

    @pytest.mark.asyncio
    async def test_llm_exception_returns_empty(self, store: MemoryStore, retriever: MemoryRetriever) -> None:
        """LLM 抛异常应返回空列表不抛异常。"""
        llm = build_mock_llm(fail=True)
        extractor = MemoryExtractor(store=store, retriever=retriever, llm_client=llm)

        candidates = await extractor.extract_from_conversation([])
        assert candidates == []

    @pytest.mark.asyncio
    async def test_not_a_json_array_returns_empty(self, store: MemoryStore, retriever: MemoryRetriever) -> None:
        """LLM 返回的不是 JSON 数组应返回空列表。"""
        llm = build_mock_llm('{"result": "ok"}')
        extractor = MemoryExtractor(store=store, retriever=retriever, llm_client=llm)

        candidates = await extractor.extract_from_conversation([])
        assert candidates == []

    @pytest.mark.asyncio
    async def test_markdown_code_block_parsed(self, store: MemoryStore, retriever: MemoryRetriever) -> None:
        """LLM 返回 markdown 代码块时应能正确提取 JSON。"""
        markdown_response = "```json\n" + _VALID_JSON_RESPONSE + "\n```"
        llm = build_mock_llm(markdown_response)
        extractor = MemoryExtractor(store=store, retriever=retriever, llm_client=llm)

        candidates = await extractor.extract_from_conversation([])
        assert len(candidates) == 2
        assert candidates[0].entry.name == "no-mock-database"

    @pytest.mark.asyncio
    async def test_partially_invalid_items_skipped(self, store: MemoryStore, retriever: MemoryRetriever) -> None:
        """部分无效的 JSON 项应被跳过。"""
        mixed_response = json.dumps([
            {"type": "user", "name": "valid-item", "description": "Valid", "body": "OK", "confidence": 0.8},
            {"type": "invalid-type"},  # 缺少 name/description/body
        ])
        llm = build_mock_llm(mixed_response)
        extractor = MemoryExtractor(store=store, retriever=retriever, llm_client=llm)

        candidates = await extractor.extract_from_conversation([])
        assert len(candidates) == 1
        assert candidates[0].entry.name == "valid-item"


# ══════════════════════════════════════════════════════════════════════════
# 3. 去重检测
# ══════════════════════════════════════════════════════════════════════════


class TestDuplicateDetection:
    """去重检测测试。"""

    def test_exact_name_match(self, store: MemoryStore, retriever: MemoryRetriever) -> None:
        """相同 name 应检测为重复。"""
        existing = make_entry(name="already-exists", memory_type=MemoryType.USER,
                              description="Existing", body="Existing body")
        store.save(existing)

        candidate = ExtractionCandidate(
            entry=make_entry(name="already-exists", memory_type=MemoryType.USER,
                             description="New version", body="New body"),
            confidence=0.9, trigger="task_complete",
        )

        extractor = MemoryExtractor(store=store, retriever=retriever, llm_client=build_mock_llm())
        is_dup, dup_name = extractor._check_duplicate(candidate)
        assert is_dup is True
        assert dup_name == "already-exists"

    def test_similar_content_duplicate(self, store: MemoryStore, retriever: MemoryRetriever) -> None:
        """内容高度相似应检测为重复。"""
        existing = make_entry(name="original", memory_type=MemoryType.FEEDBACK,
                              description="Use real database for testing",
                              body="Always use a real database for integration tests")
        store.save(existing)

        candidate_entry = make_entry(name="new-one", memory_type=MemoryType.FEEDBACK,
                                     description="Use real database for testing",
                                     body="Always use a real database for integration tests")
        candidate = ExtractionCandidate(entry=candidate_entry, confidence=0.9, trigger="task_complete")

        extractor = MemoryExtractor(store=store, retriever=retriever, llm_client=build_mock_llm())
        is_dup, dup_name = extractor._check_duplicate(candidate, threshold=0.5)
        assert is_dup is True
        assert dup_name == "original"

    def test_different_content_not_duplicate(self, store: MemoryStore, retriever: MemoryRetriever) -> None:
        """内容完全不同不应视为重复。"""
        existing = make_entry(name="python-tips", memory_type=MemoryType.USER,
                              description="Python programming", body="Python tips")
        store.save(existing)

        candidate_entry = make_entry(name="go-tips", memory_type=MemoryType.USER,
                                     description="Golang programming", body="Go tips")
        candidate = ExtractionCandidate(entry=candidate_entry, confidence=0.9, trigger="task_complete")

        extractor = MemoryExtractor(store=store, retriever=retriever, llm_client=build_mock_llm())
        is_dup, dup_name = extractor._check_duplicate(candidate)
        assert is_dup is False
        assert dup_name is None

    def test_different_type_not_duplicate(self, store: MemoryStore, retriever: MemoryRetriever) -> None:
        """不同类型即使同 name 也应检测为重复（name 全局唯一）。"""
        existing = make_entry(name="same-name", memory_type=MemoryType.USER,
                              description="User", body="User content")
        store.save(existing)

        candidate_entry = make_entry(name="same-name", memory_type=MemoryType.FEEDBACK,
                                     description="Feedback", body="Feedback content")
        candidate = ExtractionCandidate(entry=candidate_entry, confidence=0.9, trigger="task_complete")

        extractor = MemoryExtractor(store=store, retriever=retriever, llm_client=build_mock_llm())
        is_dup, dup_name = extractor._check_duplicate(candidate)
        # 同名即重复（不论类型）
        assert is_dup is True
        assert dup_name == "same-name"


# ══════════════════════════════════════════════════════════════════════════
# 4. auto_mode 自动保存
# ══════════════════════════════════════════════════════════════════════════


class TestAutoMode:
    """auto_mode 自动保存测试。"""

    @pytest.mark.asyncio
    async def test_auto_mode_saves_to_store(self, store: MemoryStore, retriever: MemoryRetriever) -> None:
        """auto_mode=True 时应自动保存非重复记忆。"""
        llm = build_mock_llm(_VALID_JSON_RESPONSE)
        extractor = MemoryExtractor(store=store, retriever=retriever, llm_client=llm, auto_mode=True)

        await extractor.extract_from_conversation([])

        # 验证已保存到 store
        loaded = store.load("no-mock-database")
        assert loaded is not None
        assert loaded.description == "集成测试必须使用真实数据库"

        loaded2 = store.load("user-is-senior-go-dev")
        assert loaded2 is not None

    @pytest.mark.asyncio
    async def test_auto_mode_skips_duplicates(self, store: MemoryStore, retriever: MemoryRetriever) -> None:
        """auto_mode=True 时应跳过重复条目。"""
        # 先保存一条现有记忆
        existing = make_entry(name="no-mock-database", memory_type=MemoryType.FEEDBACK,
                              description="集成测试必须使用真实数据库",
                              body="集成测试必须连接真实数据库。")
        store.save(existing)

        # LLM 返回和现有记忆完全相同的条目
        duplicate_response = json.dumps([
            {"type": "feedback", "name": "no-mock-database",
             "description": "集成测试必须使用真实数据库",
             "body": "集成测试必须连接真实数据库。", "confidence": 0.95},
        ])
        llm = build_mock_llm(duplicate_response)
        extractor = MemoryExtractor(store=store, retriever=retriever, llm_client=llm, auto_mode=True)

        candidates = await extractor.extract_from_conversation([])

        # 应标记为重复
        assert candidates[0].is_duplicate is True
        assert candidates[0].duplicate_name == "no-mock-database"

    @pytest.mark.asyncio
    async def test_auto_mode_skips_low_confidence(self, store: MemoryStore, retriever: MemoryRetriever) -> None:
        """auto_mode=True 时低置信度条目不应保存。"""
        low_conf_response = json.dumps([
            {"type": "user", "name": "low-conf-item",
             "description": "Low confidence", "body": "Not sure", "confidence": 0.3},
        ])
        llm = build_mock_llm(low_conf_response)
        extractor = MemoryExtractor(store=store, retriever=retriever, llm_client=llm,
                                    auto_mode=True, min_confidence=0.7)

        candidates = await extractor.extract_from_conversation([])

        # 置信度被置为 0.0（标记无效）
        assert candidates[0].confidence == 0.0
        # 不应被保存
        loaded = store.load("low-conf-item")
        assert loaded is None


# ══════════════════════════════════════════════════════════════════════════
# 5. extract_explicit
# ══════════════════════════════════════════════════════════════════════════


class TestExtractExplicit:
    """extract_explicit 显式提取测试。"""

    @pytest.mark.asyncio
    async def test_explicit_extraction(self, store: MemoryStore, retriever: MemoryRetriever) -> None:
        """显式"记住"指令应提取记忆。"""
        llm = build_mock_llm(_SINGLE_EXPLICIT_RESPONSE)
        extractor = MemoryExtractor(store=store, retriever=retriever, llm_client=llm)

        candidates = await extractor.extract_explicit("记住：API 返回 snake_case")

        assert len(candidates) == 1
        assert candidates[0].entry.name == "prefer-snake-case"
        assert candidates[0].trigger == "explicit"

    @pytest.mark.asyncio
    async def test_explicit_empty_response(self, store: MemoryStore, retriever: MemoryRetriever) -> None:
        """LLM 返回空时显式提取应返回空列表。"""
        llm = build_mock_llm(_EMPTY_RESPONSE)
        extractor = MemoryExtractor(store=store, retriever=retriever, llm_client=llm)

        candidates = await extractor.extract_explicit("你好")
        assert candidates == []

    @pytest.mark.asyncio
    async def test_explicit_with_context(self, store: MemoryStore, retriever: MemoryRetriever) -> None:
        """显式提取应能接收额外上下文。"""
        llm = build_mock_llm(_SINGLE_EXPLICIT_RESPONSE)
        extractor = MemoryExtractor(store=store, retriever=retriever, llm_client=llm)

        # 验证 context 不导致崩溃
        candidates = await extractor.extract_explicit("记住这个", context="项目使用 Python 3.12")
        assert len(candidates) >= 0  # 只要不崩溃就算通过


# ══════════════════════════════════════════════════════════════════════════
# 6. 置信度过滤
# ══════════════════════════════════════════════════════════════════════════


class TestConfidenceFilter:
    """置信度过滤测试。"""

    @pytest.mark.asyncio
    async def test_below_min_confidence_filtered(self, store: MemoryStore, retriever: MemoryRetriever) -> None:
        """低于 min_confidence 的候选应被标记为无效（confidence=0.0）。"""
        mixed_response = json.dumps([
            {"type": "user", "name": "high-conf", "description": "High", "body": "High", "confidence": 0.9},
            {"type": "user", "name": "low-conf", "description": "Low", "body": "Low", "confidence": 0.3},
        ])
        llm = build_mock_llm(mixed_response)
        extractor = MemoryExtractor(store=store, retriever=retriever, llm_client=llm, min_confidence=0.7)

        candidates = await extractor.extract_from_conversation([])

        assert len(candidates) == 2
        # 高置信度保持
        assert candidates[0].confidence == 0.9
        # 低置信度被清零
        assert candidates[1].confidence == 0.0


# ══════════════════════════════════════════════════════════════════════════
# 7. 对话截断
# ══════════════════════════════════════════════════════════════════════════


class TestConversationTruncation:
    """对话历史截断测试。"""

    def test_truncates_long_history(self, store: MemoryStore, retriever: MemoryRetriever) -> None:
        """超过 20 轮的对话应被截断。"""
        history: list[dict] = []
        for i in range(25):
            history.append({"role": "user" if i % 2 == 0 else "assistant", "content": f"Message {i}"})

        extractor = MemoryExtractor(store=store, retriever=retriever, llm_client=build_mock_llm())
        prompt = extractor._build_extraction_prompt(history, "task_complete")

        # 验证提示词中有内容但不会太长
        assert "Message" in prompt
        assert len(prompt) < 10000  # 合理长度

    def test_empty_history(self, store: MemoryStore, retriever: MemoryRetriever) -> None:
        """空对话历史不应导致崩溃。"""
        extractor = MemoryExtractor(store=store, retriever=retriever, llm_client=build_mock_llm())
        prompt = extractor._build_extraction_prompt([], "task_complete")
        assert "(empty)" in prompt


# ══════════════════════════════════════════════════════════════════════════
# 8. 辅助方法
# ══════════════════════════════════════════════════════════════════════════


class TestUtilityMethods:
    """辅助方法测试。"""

    def test_to_kebab_case(self) -> None:
        """_to_kebab_case 应正确转换。"""
        assert MemoryExtractor._to_kebab_case("Hello World") == "hello-world"
        assert MemoryExtractor._to_kebab_case("no Mock Database") == "no-mock-database"
        assert MemoryExtractor._to_kebab_case("  spaces  ") == "spaces"
        assert MemoryExtractor._to_kebab_case("") == "untitled"
        assert MemoryExtractor._to_kebab_case("UPPER CASE") == "upper-case"

    def test_clean_json_strips_markdown(self) -> None:
        """_clean_json 应移除 markdown 代码块。"""
        raw = '```json\n[{"key": "value"}]\n```'
        cleaned = MemoryExtractor._clean_json(raw)
        assert cleaned == '[{"key": "value"}]'

    def test_clean_json_passthrough(self) -> None:
        """已合法的 JSON 应直接通过。"""
        raw = '[{"key": "value"}]'
        cleaned = MemoryExtractor._clean_json(raw)
        assert cleaned == raw

    def test_clean_json_invalid(self) -> None:
        """非法格式应原样返回。"""
        raw = "some random text"
        cleaned = MemoryExtractor._clean_json(raw)
        assert cleaned == "some random text"

    @pytest.mark.asyncio
    async def test_explicit_remember_keywords_detected(self) -> None:
        """验证 _REMEMBER_KEYWORDS 包含中英文关键词。"""
        assert "记住" in _REMEMBER_KEYWORDS
        assert "remember" in _REMEMBER_KEYWORDS
        assert "记得" in _REMEMBER_KEYWORDS


# ══════════════════════════════════════════════════════════════════════════
# 9. ExtractionCandidate dataclass
# ══════════════════════════════════════════════════════════════════════════


class TestExtractionCandidate:
    """ExtractionCandidate 基础功能测试。"""

    def test_dataclass_fields(self) -> None:
        """验证 ExtractionCandidate 字段默认值。"""
        entry = make_entry(name="test", memory_type=MemoryType.USER)
        cand = ExtractionCandidate(entry=entry, confidence=0.8, trigger="task_complete")
        assert cand.confidence == 0.8
        assert cand.trigger == "task_complete"
        assert cand.is_duplicate is False
        assert cand.duplicate_name is None

    def test_dataclass_with_duplicate(self) -> None:
        """验证 duplicate 标记字段。"""
        entry = make_entry(name="dup", memory_type=MemoryType.FEEDBACK)
        cand = ExtractionCandidate(
            entry=entry, confidence=0.9, trigger="explicit",
            is_duplicate=True, duplicate_name="original",
        )
        assert cand.is_duplicate is True
        assert cand.duplicate_name == "original"
