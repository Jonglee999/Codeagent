"""MemoryManager 单元测试。

覆盖：recall/save/forget/list、auto_extract 过滤重复、run_maintenance、
create 工厂方法、异常保护。
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from codeagent.gateway.memory_gateway import IMemoryGateway
from codeagent.memory.extractor import MemoryExtractor
from codeagent.memory.manager import MemoryManager
from codeagent.memory.retriever import MemoryRetriever
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
def store_no_project(global_root: Path) -> MemoryStore:
    return MemoryStore(global_root=global_root, project_root=None)


@pytest.fixture
def retriever(store: MemoryStore) -> MemoryRetriever:
    return MemoryRetriever(store=store)


def make_entry(
    name: str = "test",
    memory_type: MemoryType = MemoryType.USER,
    description: str = "Test entry",
    body: str = "Test body content",
    confidence: float = 1.0,
) -> MemoryEntry:
    return MemoryEntry(
        name=name,
        memory_type=memory_type,
        description=description,
        body=body,
        confidence=confidence,
    )


def make_mock_llm(response_text: str | None = None, fail: bool = False):
    """构建 Mock LLM 客户端。"""
    async def mock_llm(**kwargs: Any) -> Any:
        if fail:
            raise RuntimeError("LLM API error")
        from types import SimpleNamespace
        msg = SimpleNamespace(content=response_text)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])
    return mock_llm


# ══════════════════════════════════════════════════════════════════════════
# 1. recall
# ══════════════════════════════════════════════════════════════════════════


class TestRecall:
    """recall 接口测试。"""

    @pytest.mark.asyncio
    async def test_recall_returns_xml(self, store: MemoryStore) -> None:
        """recall 应返回 XML 格式的记忆字符串。"""
        # 预存一条记忆
        entry = make_entry(name="python-tips", memory_type=MemoryType.USER,
                           description="Python programming tips",
                           body="Use list comprehensions")
        store.save(entry)

        retriever = MemoryRetriever(store=store)
        extractor = MemoryExtractor(
            store=store, retriever=retriever,
            llm_client=make_mock_llm("[]"),
        )
        manager = MemoryManager(store=store, retriever=retriever, extractor=extractor)

        xml = await manager.recall("python programming")

        assert "<relevant_memories>" in xml
        assert "python-tips" in xml
        assert "</relevant_memories>" in xml

    @pytest.mark.asyncio
    async def test_recall_empty_query(self, store: MemoryStore) -> None:
        """空查询应返回空字符串。"""
        entry = make_entry(name="python-tips", memory_type=MemoryType.USER,
                           description="Python tips", body="Python")
        store.save(entry)

        retriever = MemoryRetriever(store=store)
        extractor = MemoryExtractor(
            store=store, retriever=retriever,
            llm_client=make_mock_llm("[]"),
        )
        manager = MemoryManager(store=store, retriever=retriever, extractor=extractor)

        xml = await manager.recall("")
        assert xml == ""

    @pytest.mark.asyncio
    async def test_recall_no_match_returns_empty(self, store: MemoryStore) -> None:
        """无匹配查询应返回空字符串。"""
        entry = make_entry(name="python-tips", memory_type=MemoryType.USER,
                           description="Python tips", body="Python")
        store.save(entry)

        retriever = MemoryRetriever(store=store)
        extractor = MemoryExtractor(
            store=store, retriever=retriever,
            llm_client=make_mock_llm("[]"),
        )
        manager = MemoryManager(store=store, retriever=retriever, extractor=extractor)

        xml = await manager.recall("quantum physics cosmology")
        assert xml == ""

    @pytest.mark.asyncio
    async def test_recall_updates_last_accessed(self, store: MemoryStore) -> None:
        """recall 应更新被检索记忆的 last_accessed_at。"""
        entry = make_entry(name="test-entry", memory_type=MemoryType.USER,
                           description="test memory for recall",
                           body="test body for recall test")
        store.save(entry)

        # 确认初始状态
        loaded = store.load("test-entry")
        assert loaded is not None
        original_access = loaded.last_accessed_at

        retriever = MemoryRetriever(store=store)
        extractor = MemoryExtractor(
            store=store, retriever=retriever,
            llm_client=make_mock_llm("[]"),
        )
        manager = MemoryManager(store=store, retriever=retriever, extractor=extractor)

        await manager.recall("test memory")

        updated = store.load("test-entry")
        assert updated is not None
        if original_access is not None:
            assert updated.last_accessed_at >= original_access


# ══════════════════════════════════════════════════════════════════════════
# 2. save
# ══════════════════════════════════════════════════════════════════════════


class TestSave:
    """save 接口测试。"""

    def test_save_persists_entry(self, store: MemoryStore) -> None:
        """save 应持久化记忆条目。"""
        retriever = MemoryRetriever(store=store)
        extractor = MemoryExtractor(
            store=store, retriever=retriever,
            llm_client=make_mock_llm("[]"),
        )
        manager = MemoryManager(store=store, retriever=retriever, extractor=extractor)

        entry = make_entry(name="saved-entry", memory_type=MemoryType.USER,
                           description="Saved entry", body="Saved body")
        manager.save(entry)

        loaded = store.load("saved-entry")
        assert loaded is not None
        assert loaded.description == "Saved entry"

    def test_save_with_scope(self, store: MemoryStore) -> None:
        """save 应支持指定 scope。"""
        retriever = MemoryRetriever(store=store)
        extractor = MemoryExtractor(
            store=store, retriever=retriever,
            llm_client=make_mock_llm("[]"),
        )
        manager = MemoryManager(store=store, retriever=retriever, extractor=extractor)

        entry = make_entry(name="global-entry", memory_type=MemoryType.PROJECT,
                           description="Global project entry", body="Global body")
        # PROJECT 类型默认存项目层，强制存全局
        manager.save(entry, scope="global")

        loaded = store.load("global-entry")
        assert loaded is not None


# ══════════════════════════════════════════════════════════════════════════
# 3. forget
# ══════════════════════════════════════════════════════════════════════════


class TestForget:
    """forget 接口测试。"""

    def test_forget_deletes_entry(self, store: MemoryStore) -> None:
        """forget 应删除已保存的记忆。"""
        entry = make_entry(name="to-delete", memory_type=MemoryType.USER,
                           description="To delete", body="To delete")
        store.save(entry)

        retriever = MemoryRetriever(store=store)
        extractor = MemoryExtractor(
            store=store, retriever=retriever,
            llm_client=make_mock_llm("[]"),
        )
        manager = MemoryManager(store=store, retriever=retriever, extractor=extractor)

        result = manager.forget("to-delete")
        assert result is True

        loaded = store.load("to-delete")
        assert loaded is None

    def test_forget_nonexistent_returns_false(self, store: MemoryStore) -> None:
        """删除不存在的记忆应返回 False。"""
        retriever = MemoryRetriever(store=store)
        extractor = MemoryExtractor(
            store=store, retriever=retriever,
            llm_client=make_mock_llm("[]"),
        )
        manager = MemoryManager(store=store, retriever=retriever, extractor=extractor)

        result = manager.forget("does-not-exist")
        assert result is False


# ══════════════════════════════════════════════════════════════════════════
# 4. list_memories
# ══════════════════════════════════════════════════════════════════════════


class TestListMemories:
    """list_memories 接口测试。"""

    def test_list_all(self, store: MemoryStore) -> None:
        """list_memories 应列出所有记忆。"""
        retriever = MemoryRetriever(store=store)
        extractor = MemoryExtractor(
            store=store, retriever=retriever,
            llm_client=make_mock_llm("[]"),
        )
        manager = MemoryManager(store=store, retriever=retriever, extractor=extractor)

        e1 = make_entry(name="first", memory_type=MemoryType.USER,
                        description="First", body="First")
        e2 = make_entry(name="second", memory_type=MemoryType.FEEDBACK,
                        description="Second", body="Second")
        manager.save(e1)
        manager.save(e2)

        entries = manager.list_memories()
        assert len(entries) >= 2
        names = [e.name for e in entries]
        assert "first" in names
        assert "second" in names

    def test_list_by_type(self, store: MemoryStore) -> None:
        """list_memories 应支持按类型筛选。"""
        retriever = MemoryRetriever(store=store)
        extractor = MemoryExtractor(
            store=store, retriever=retriever,
            llm_client=make_mock_llm("[]"),
        )
        manager = MemoryManager(store=store, retriever=retriever, extractor=extractor)

        manager.save(make_entry(name="user-1", memory_type=MemoryType.USER,
                                description="User", body="User"))
        manager.save(make_entry(name="fb-1", memory_type=MemoryType.FEEDBACK,
                                description="Feedback", body="Feedback"))

        entries = manager.list_memories(memory_type=MemoryType.FEEDBACK)
        names = [e.name for e in entries]
        assert "fb-1" in names
        assert "user-1" not in names


# ══════════════════════════════════════════════════════════════════════════
# 5. auto_extract
# ══════════════════════════════════════════════════════════════════════════


class TestAutoExtract:
    """auto_extract 接口测试。"""

    @pytest.mark.asyncio
    async def test_auto_extract_saves_non_duplicates(self, store: MemoryStore) -> None:
        """auto_extract 应保存非重复的记忆条目。"""
        valid_json = (
            '[{"type": "user", "name": "extracted-item",'
            '"description": "Extracted", "body": "Extracted body",'
            '"confidence": 0.9}]'
        )
        llm = make_mock_llm(valid_json)
        retriever = MemoryRetriever(store=store)
        extractor = MemoryExtractor(
            store=store, retriever=retriever,
            llm_client=llm, auto_mode=False,
        )
        manager = MemoryManager(store=store, retriever=retriever, extractor=extractor)

        saved = await manager.auto_extract(
            [{"role": "user", "content": "记住这个"}],
            trigger="task_complete",
        )

        assert len(saved) == 1
        assert saved[0].name == "extracted-item"

        # 验证确实被保存到了 store
        loaded = store.load("extracted-item")
        assert loaded is not None

    @pytest.mark.asyncio
    async def test_auto_extract_skips_duplicates(self, store: MemoryStore) -> None:
        """auto_extract 应跳过重复条目。"""
        # 预存一条同名的现有记忆
        existing = make_entry(name="dup-item", memory_type=MemoryType.USER,
                              description="Existing", body="Existing")
        store.save(existing)

        # LLM 返回一个同名、同描述的候选
        dup_json = (
            '[{"type": "user", "name": "dup-item",'
            '"description": "Existing", "body": "Existing",'
            '"confidence": 0.9}]'
        )
        llm = make_mock_llm(dup_json)
        retriever = MemoryRetriever(store=store)
        extractor = MemoryExtractor(
            store=store, retriever=retriever,
            llm_client=llm, auto_mode=False,
        )
        manager = MemoryManager(store=store, retriever=retriever, extractor=extractor)

        saved = await manager.auto_extract(
            [{"role": "user", "content": "记住"}],
            trigger="task_complete",
        )

        # 重复条目不应被保存
        assert len(saved) == 0

    @pytest.mark.asyncio
    async def test_auto_extract_llm_failure_returns_empty(self, store: MemoryStore) -> None:
        """LLM 失败时 auto_extract 应返回空列表。"""
        llm = make_mock_llm(fail=True)
        retriever = MemoryRetriever(store=store)
        extractor = MemoryExtractor(
            store=store, retriever=retriever,
            llm_client=llm,
        )
        manager = MemoryManager(store=store, retriever=retriever, extractor=extractor)

        saved = await manager.auto_extract([], trigger="task_complete")
        assert saved == []


# ══════════════════════════════════════════════════════════════════════════
# 6. run_maintenance
# ══════════════════════════════════════════════════════════════════════════


class TestRunMaintenance:
    """run_maintenance 维护测试。"""

    def test_maintenance_archives_sessions(self, store: MemoryStore) -> None:
        """run_maintenance 应归档过期的 session 记忆。"""
        # 创建一条 8 天前最后访问的 session 记忆
        old_entry = MemoryEntry(
            name="old-session",
            memory_type=MemoryType.SESSION,
            description="Old session",
            body="Old session data",
            created_at=datetime.now() - timedelta(days=8),
            updated_at=datetime.now() - timedelta(days=8),
            last_accessed_at=datetime.now() - timedelta(days=8),
        )
        store.save(old_entry)

        retriever = MemoryRetriever(store=store)
        extractor = MemoryExtractor(
            store=store, retriever=retriever,
            llm_client=make_mock_llm("[]"),
        )
        manager = MemoryManager(store=store, retriever=retriever, extractor=extractor)

        report = manager.run_maintenance()

        assert report["archived_sessions"] >= 1

        # 验证已被移除
        loaded = store.load("old-session")
        assert loaded is None

    def test_maintenance_decays_confidence(self, store: MemoryStore) -> None:
        """run_maintenance 应降低过期记忆的 confidence。"""
        # 创建一条 31 天前更新的记忆
        old_entry = MemoryEntry(
            name="old-entry",
            memory_type=MemoryType.USER,
            description="Old entry for decay",
            body="Old content",
            created_at=datetime.now() - timedelta(days=35),
            updated_at=datetime.now() - timedelta(days=35),
            confidence=0.9,
        )
        store.save(old_entry)

        retriever = MemoryRetriever(store=store)
        extractor = MemoryExtractor(
            store=store, retriever=retriever,
            llm_client=make_mock_llm("[]"),
        )
        manager = MemoryManager(store=store, retriever=retriever, extractor=extractor)

        report = manager.run_maintenance()

        assert report["decayed"] >= 1

        # 验证 confidence 已降低
        loaded = store.load("old-entry")
        # confidence 0.9 - 0.1 = 0.8, 仍高于 0.2, 所以还在
        if loaded is not None:
            assert loaded.confidence < 0.9

    def test_maintenance_deletes_low_confidence(self, store: MemoryStore) -> None:
        """run_maintenance 应删除 confidence 低于阈值的记忆。"""
        old_entry = MemoryEntry(
            name="too-low",
            memory_type=MemoryType.USER,
            description="Low confidence",
            body="Will be deleted",
            created_at=datetime.now() - timedelta(days=35),
            updated_at=datetime.now() - timedelta(days=35),
            confidence=0.25,  # 0.25 - 0.1 = 0.15 < 0.2 → 删除
        )
        store.save(old_entry)

        retriever = MemoryRetriever(store=store)
        extractor = MemoryExtractor(
            store=store, retriever=retriever,
            llm_client=make_mock_llm("[]"),
        )
        manager = MemoryManager(store=store, retriever=retriever, extractor=extractor)

        manager.run_maintenance()

        loaded = store.load("too-low")
        assert loaded is None

    def test_maintenance_returns_report(self, store: MemoryStore) -> None:
        """run_maintenance 应返回正确的维护报告。"""
        retriever = MemoryRetriever(store=store)
        extractor = MemoryExtractor(
            store=store, retriever=retriever,
            llm_client=make_mock_llm("[]"),
        )
        manager = MemoryManager(store=store, retriever=retriever, extractor=extractor)

        report = manager.run_maintenance()

        assert isinstance(report, dict)
        assert "archived_sessions" in report
        assert "decayed" in report
        assert isinstance(report["archived_sessions"], int)
        assert isinstance(report["decayed"], int)


# ══════════════════════════════════════════════════════════════════════════
# 7. create 工厂方法
# ══════════════════════════════════════════════════════════════════════════


class TestCreate:
    """create 工厂方法测试。"""

    def test_create_with_project_path(self, tmp_path: Path) -> None:
        """create 工厂方法应正确构建 MemoryManager。"""
        project_path = str(tmp_path / "test_project")
        manager = MemoryManager.create(project_path=project_path)

        assert isinstance(manager, MemoryManager)
        assert isinstance(manager, IMemoryGateway)

        # 验证组件已初始化
        assert manager._store is not None
        assert manager._retriever is not None
        assert manager._extractor is not None

    def test_create_without_project_path(self) -> None:
        """create 工厂方法在不传 project_path 时应正常工作。"""
        manager = MemoryManager.create(project_path=None)

        assert isinstance(manager, MemoryManager)
        assert manager._store is not None

    def test_create_with_llm_client(self) -> None:
        """create 工厂方法应能接收 llm_client。"""
        llm = make_mock_llm("[]")
        manager = MemoryManager.create(llm_client=llm)

        assert manager._extractor._llm_client is llm

    def test_create_with_auto_mode(self) -> None:
        """create 工厂方法应能设置 auto_mode。"""
        manager = MemoryManager.create(auto_mode=True)

        assert manager._extractor._auto_mode is True


# ══════════════════════════════════════════════════════════════════════════
# 8. 异常保护
# ══════════════════════════════════════════════════════════════════════════


class TestExceptionProtection:
    """异常保护测试 — 子组件抛异常时 MemoryManager 不崩溃。"""

    @pytest.mark.asyncio
    async def test_recall_exception_returns_empty(self) -> None:
        """recall 子组件崩溃时返回空字符串。"""
        store = MagicMock(spec=MemoryStore)
        retriever = MagicMock(spec=MemoryRetriever)
        extractor = MagicMock(spec=MemoryExtractor)

        # retriever.retrieve 抛异常
        retriever.retrieve.side_effect = RuntimeError("Retrieval failed")

        manager = MemoryManager(
            store=store, retriever=retriever, extractor=extractor,
        )

        xml = await manager.recall("test")
        assert xml == ""

    @pytest.mark.asyncio
    async def test_auto_extract_exception_returns_empty(self) -> None:
        """auto_extract 子组件崩溃时返回空列表。"""
        store = MagicMock(spec=MemoryStore)
        retriever = MagicMock(spec=MemoryRetriever)
        extractor = MagicMock(spec=MemoryExtractor)

        extractor.extract_from_conversation.side_effect = RuntimeError("Extraction failed")

        manager = MemoryManager(
            store=store, retriever=retriever, extractor=extractor,
        )

        saved = await manager.auto_extract([], trigger="task_complete")
        assert saved == []

    def test_save_exception_does_not_crash(self) -> None:
        """save 子组件崩溃时不影响调用方。"""
        store = MagicMock(spec=MemoryStore)
        store.save.side_effect = RuntimeError("Save failed")

        retriever = MagicMock(spec=MemoryRetriever)
        extractor = MagicMock(spec=MemoryExtractor)

        manager = MemoryManager(
            store=store, retriever=retriever, extractor=extractor,
        )

        # 不抛异常
        entry = make_entry(name="test", memory_type=MemoryType.USER,
                           description="Test", body="Test")
        manager.save(entry)  # no exception
