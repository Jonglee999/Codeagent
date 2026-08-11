"""记忆系统集成测试 — 真实文件系统环境。

场景 A：跨会话持久化（两次 MemoryManager 实例，同一 tmp_path）
场景 B：项目层优先（全局+项目同名记忆，recall 返回项目层）
场景 C：提取去重（两次 auto_extract 相同对话，store 只有一条）
场景 D：session 归档（last_accessed_at 设为 8 天前，run_maintenance 后归档）

所有测试使用 pytest tmp_path 创建真实文件系统，测试结束后自动清理。
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from codeagent.memory.extractor import MemoryExtractor
from codeagent.memory.manager import MemoryManager
from codeagent.memory.retriever import MemoryRetriever
from codeagent.memory.store import MemoryEntry, MemoryStore, MemoryType


# ── Mock LLM 工具函数 ────────────────────────────────────────────────────


def make_mock_llm(response_text: str | None = None, fail: bool = False):
    """构建模拟 LLM 客户端，返回固定 JSON 响应。

    MemoryExtractor 的 _call_llm 调用方式：
      response = await llm_client(model=..., messages=[...])
      content = response.choices[0].message.content
    """
    async def mock_llm(**kwargs: Any) -> Any:
        if fail:
            raise RuntimeError("LLM API error")
        from types import SimpleNamespace
        msg = SimpleNamespace(content=response_text)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])
    return mock_llm


def make_retriever_llm(response_text: str | None = None):
    """构建模拟 LLM 客户端并返回 SimpleNamespace 格式。"""
    return make_mock_llm(response_text)


# ── 辅助函数 ────────────────────────────────────────────────────────────


def make_manager(
    global_root: Path,
    project_root: Path | None = None,
    llm_client: Any = None,
) -> MemoryManager:
    """使用指定路径构建完整 MemoryManager 实例。"""
    store = MemoryStore(global_root=global_root, project_root=project_root)
    retriever = MemoryRetriever(store=store)
    if llm_client is None:
        llm_client = make_mock_llm("[]")
    extractor = MemoryExtractor(
        store=store,
        retriever=retriever,
        llm_client=llm_client,
        auto_mode=True,
        min_confidence=0.0,
    )
    return MemoryManager(store=store, retriever=retriever, extractor=extractor)


def make_entry(
    name: str = "test-entry",
    memory_type: MemoryType = MemoryType.FEEDBACK,
    description: str = "Test entry",
    body: str = "Test body",
) -> MemoryEntry:
    return MemoryEntry(
        name=name,
        memory_type=memory_type,
        description=description,
        body=body,
    )


# ══════════════════════════════════════════════════════════════════════════
# 场景 A：跨会话持久化
# ══════════════════════════════════════════════════════════════════════════


class TestCrossSessionPersistence:
    """验证记忆在 MemoryManager 实例间持久化。"""

    @pytest.mark.asyncio
    async def test_save_then_recall_in_new_session(self, tmp_path: Path) -> None:
        """会话 1 写入 feedback 记忆，会话 2 recall 应能检索到。"""
        global_root = tmp_path / "global"
        global_root.mkdir(parents=True)

        # 会话 1：写入
        mgr1 = make_manager(global_root)
        mgr1.save(MemoryEntry(
            name="no-mock-database",
            memory_type=MemoryType.FEEDBACK,
            description="Don't mock database",
            body="Don't mock DB - always use real database",
        ))

        # 会话 2：全新实例，相同路径
        mgr2 = make_manager(global_root)
        xml = await mgr2.recall("mock database")
        assert "no-mock-database" in xml
        assert "Don't mock DB" in xml

    @pytest.mark.asyncio
    async def test_multiple_memories_survive_new_session(self, tmp_path: Path) -> None:
        """多个记忆条目在一新会话中全部可检索。"""
        global_root = tmp_path / "global"
        global_root.mkdir(parents=True)

        mgr1 = make_manager(global_root)
        mgr1.save(make_entry(name="use-pytest", memory_type=MemoryType.FEEDBACK, body="Use pytest"))
        mgr1.save(make_entry(name="async-pattern", memory_type=MemoryType.CODE_PATTERN, body="Async functions"))

        mgr2 = make_manager(global_root)
        xml_pytest = await mgr2.recall("pytest")
        xml_async = await mgr2.recall("async")

        assert "use-pytest" in xml_pytest
        assert "async-pattern" in xml_async

    @pytest.mark.asyncio
    async def test_project_memory_persists(self, tmp_path: Path) -> None:
        """项目层记忆同样跨会话持久化。"""
        global_root = tmp_path / "global"
        project_root = tmp_path / "project"
        global_root.mkdir(parents=True)
        project_root.mkdir(parents=True)

        mgr1 = make_manager(global_root, project_root)
        entry = MemoryEntry(
            name="project-api-style", memory_type=MemoryType.PROJECT,
            description="Project API style convention", body="Use snake_case for API",
        )
        mgr1.save(entry)

        mgr2 = make_manager(global_root, project_root)
        xml = await mgr2.recall("API style")
        assert "project-api-style" in xml

    @pytest.mark.asyncio
    async def test_recall_empty_when_no_memory(self, tmp_path: Path) -> None:
        """无记忆时 recall 返回空字符串。"""
        global_root = tmp_path / "empty_global"
        global_root.mkdir(parents=True)
        mgr = make_manager(global_root)
        xml = await mgr.recall("anything")
        assert xml == ""


# ══════════════════════════════════════════════════════════════════════════
# 场景 B：项目层优先于全局层
# ══════════════════════════════════════════════════════════════════════════


class TestProjectLayerPriority:
    """验证项目层记忆优先于全局层同名记忆。"""

    @pytest.mark.asyncio
    async def test_project_over_global_same_name(self, tmp_path: Path) -> None:
        """全局和项目各有同名记忆 use-pytest，recall 返回项目层版本。"""
        global_root = tmp_path / "global"
        project_root = tmp_path / "project"
        global_root.mkdir(parents=True)
        project_root.mkdir(parents=True)

        # 先写全局
        mgr_global = make_manager(global_root, project_root)
        global_entry = MemoryEntry(
            name="use-pytest", memory_type=MemoryType.FEEDBACK,
            description="Use pytest", body="使用 pytest",
        )
        mgr_global.save(global_entry, scope="global")

        # 再写项目（同名）
        project_entry = MemoryEntry(
            name="use-pytest", memory_type=MemoryType.FEEDBACK,
            description="Use pytest with coverage", body="使用 pytest + coverage",
        )
        mgr_global.save(project_entry, scope="project")

        # 新会话中检索
        mgr = make_manager(global_root, project_root)
        xml = await mgr.recall("use pytest")

        # 应返回项目层版本（scope="project"，body 含 "coverage"）
        assert 'scope="project"' in xml, f"XML should have scope='project': {xml}"
        assert "coverage" in xml, f"XML should contain 'coverage': {xml}"

    @pytest.mark.asyncio
    async def test_global_only_when_no_project_match(self, tmp_path: Path) -> None:
        """只有全局层有时返回全局层结果。"""
        global_root = tmp_path / "global"
        project_root = tmp_path / "project"
        global_root.mkdir(parents=True)
        project_root.mkdir(parents=True)

        mgr = make_manager(global_root, project_root)
        mgr.save(MemoryEntry(
            name="serializer-style", memory_type=MemoryType.FEEDBACK,
            description="Use JSON serializer", body="Use json.dumps",
        ), scope="global")

        xml = await mgr.recall("serializer")
        assert 'scope="global"' in xml, f"XML should have scope='global': {xml}"

    @pytest.mark.asyncio
    async def test_project_priority_independent_retrieval(self, tmp_path: Path) -> None:
        """项目层高 confidence 记忆在全局层同名低 confidence 时优先。"""
        global_root = tmp_path / "global"
        project_root = tmp_path / "project"
        global_root.mkdir(parents=True)
        project_root.mkdir(parents=True)

        mgr = make_manager(global_root, project_root)
        mgr.save(MemoryEntry(
            name="api-version", memory_type=MemoryType.FEEDBACK,
            description="API version header", body="v1", confidence=0.5,
        ), scope="global")
        mgr.save(MemoryEntry(
            name="api-version", memory_type=MemoryType.FEEDBACK,
            description="API version header v2", body="v2 with details", confidence=0.9,
        ), scope="project")

        xml = await mgr.recall("API version")
        assert 'scope="project"' in xml
        assert "v2" in xml


# ══════════════════════════════════════════════════════════════════════════
# 场景 C：提取→去重→合并
# ══════════════════════════════════════════════════════════════════════════


class TestExtractDedup:
    """验证 auto_extract 去重逻辑。"""

    EXTRACTION_JSON = json.dumps([
        {
            "type": "feedback",
            "name": "no-mock-database",
            "description": "不要 Mock 数据库",
            "body": "集成测试必须连接真实数据库\n\n**Why:** 线上事故\n**How to apply:** 禁止使用 Mock",
            "confidence": 0.95,
        },
    ])

    @pytest.mark.asyncio
    async def test_first_extraction_saves(self, tmp_path: Path) -> None:
        """首次 auto_extract 应保存新记忆。"""
        global_root = tmp_path / "global"
        global_root.mkdir(parents=True)

        mock_llm = make_mock_llm(self.EXTRACTION_JSON)
        mgr = make_manager(global_root, llm_client=mock_llm)

        conversation = [
            {"role": "user", "content": "不要用 Mock，用真实数据库"},
            {"role": "assistant", "content": "好的，我会使用真实数据库"},
        ]

        saved = await mgr.auto_extract(conversation)
        assert len(saved) == 1
        assert saved[0].name == "no-mock-database"

        # store 中应有一条记忆
        memories = mgr.list_memories()
        assert len(memories) == 1

    @pytest.mark.asyncio
    async def test_second_extraction_dedup(self, tmp_path: Path) -> None:
        """相同对话第二次 auto_extract 应检测重复。"""
        global_root = tmp_path / "global"
        global_root.mkdir(parents=True)

        mock_llm = make_mock_llm(self.EXTRACTION_JSON)
        mgr = make_manager(global_root, llm_client=mock_llm)

        conversation = [
            {"role": "user", "content": "不要用 Mock，用真实数据库"},
        ]

        # 第一次提取
        saved1 = await mgr.auto_extract(conversation)
        assert len(saved1) == 1

        # 第二次提取（用同一个 mock_llm 返回相同 JSON）
        saved2 = await mgr.auto_extract(conversation)
        # 去重后不应返回新的 saved（但 auto_extract 在 manager 层过滤了 is_duplicate=True 的候选）
        # 注意：manager.auto_extract 已经在过滤 duplicate candidates，所以 saved2 应该为空
        assert len(saved2) == 0, f"Expected empty saved list, got {len(saved2)}"

        # store 中仍只有一条
        memories = mgr.list_memories()
        assert len(memories) == 1

    @pytest.mark.asyncio
    async def test_different_content_not_duplicate(self, tmp_path: Path) -> None:
        """不同内容的提取不会视为重复。"""
        global_root = tmp_path / "global"
        global_root.mkdir(parents=True)

        extraction_1 = json.dumps([
            {"type": "feedback", "name": "no-mock-database",
             "description": "Don't mock DB", "body": "Use real DB", "confidence": 0.95},
        ])
        extraction_2 = json.dumps([
            {"type": "feedback", "name": "use-pytest",
             "description": "Use pytest", "body": "Use pytest framework", "confidence": 0.90},
        ])

        mgr = make_manager(global_root, llm_client=make_mock_llm(extraction_1))
        await mgr.auto_extract([{"role": "user", "content": "不要 Mock"}])
        assert len(mgr.list_memories()) == 1

        # 第二次不同内容
        mgr2 = make_manager(global_root, llm_client=make_mock_llm(extraction_2))
        await mgr2.auto_extract([{"role": "user", "content": "用 pytest"}])
        assert len(mgr.list_memories()) == 2

    @pytest.mark.asyncio
    async def test_extract_multiple_candidates(self, tmp_path: Path) -> None:
        """一次提取多个候选时全部保存。"""
        global_root = tmp_path / "global"
        global_root.mkdir(parents=True)

        extraction_json = json.dumps([
            {"type": "feedback", "name": "use-pytest",
             "description": "Use pytest", "body": "Use pytest", "confidence": 0.9},
            {"type": "code_pattern", "name": "async-handlers",
             "description": "Async handlers", "body": "All handlers are async", "confidence": 0.85},
        ])

        mgr = make_manager(global_root, llm_client=make_mock_llm(extraction_json))
        saved = await mgr.auto_extract([{"role": "user", "content": "用 pytest 和 async"}])
        assert len(saved) == 2


# ══════════════════════════════════════════════════════════════════════════
# 场景 D：session 类型 7 天归档
# ══════════════════════════════════════════════════════════════════════════


class TestSessionArchive:
    """验证 session 类型记忆过期归档。"""

    def _save_session_with_old_timestamp(
        self, store: MemoryStore, entry: MemoryEntry, days_ago: int,
    ) -> None:
        """保存一条 last_accessed_at 为 days_ago 天前的 session 记忆。"""
        old_time = datetime.now() - timedelta(days=days_ago)
        entry.last_accessed_at = old_time
        entry.created_at = old_time
        store.save(entry)

    @pytest.mark.asyncio
    async def test_expired_session_archived(self, tmp_path: Path) -> None:
        """超过 7 天未访问的 session 应被归档。"""
        project_root = tmp_path / "project"
        project_root.mkdir(parents=True)

        mgr = make_manager(tmp_path / "global", project_root)

        # 写入一条 8 天前的 session
        entry = MemoryEntry(
            name="temp-decision", memory_type=MemoryType.SESSION,
            description="Temporary decision", body="Use X library",
        )
        self._save_session_with_old_timestamp(mgr._store, entry, days_ago=8)

        # 写入一条最近的 session（不应归档）
        recent_entry = MemoryEntry(
            name="recent-decision", memory_type=MemoryType.SESSION,
            description="Recent decision", body="Use Y library",
        )
        mgr.save(recent_entry)

        # 执行维护
        report = mgr.run_maintenance()

        # 验证过期 session 已归档
        archived_dir = project_root / "archived" / "session"
        assert (archived_dir / "temp-decision.md").exists(), \
            "Expired session should be moved to archived/"
        assert report["archived_sessions"] >= 1

        # 验证最近 session 未被归档
        session_dir = project_root / "session"
        assert (session_dir / "recent-decision.md").exists(), \
            "Recent session should remain in session/"

    @pytest.mark.asyncio
    async def test_recent_session_not_archived(self, tmp_path: Path) -> None:
        """未过期的 session 不应被归档。"""
        project_root = tmp_path / "project"
        project_root.mkdir(parents=True)

        mgr = make_manager(tmp_path / "global", project_root)

        entry = MemoryEntry(
            name="fresh-decision", memory_type=MemoryType.SESSION,
            description="Fresh", body="Fresh decision",
        )
        mgr.save(entry)

        report = mgr.run_maintenance()
        assert report["archived_sessions"] == 0

        archived_dir = project_root / "archived" / "session"
        assert not (archived_dir / "fresh-decision.md").exists()

    @pytest.mark.asyncio
    async def test_archive_updates_index(self, tmp_path: Path) -> None:
        """归档后 MEMORY.md 中不应包含被归档的 session。"""
        project_root = tmp_path / "project"
        project_root.mkdir(parents=True)

        mgr = make_manager(tmp_path / "global", project_root)

        entry = MemoryEntry(
            name="old-session", memory_type=MemoryType.SESSION,
            description="Old session", body="Old",
        )
        self._save_session_with_old_timestamp(mgr._store, entry, days_ago=8)

        # 归档前索引中有该条目
        index_path = project_root / "MEMORY.md"
        before = index_path.read_text(encoding="utf-8") if index_path.exists() else ""
        assert "old-session" in before

        mgr.run_maintenance()

        # 归档后重建索引
        after = index_path.read_text(encoding="utf-8") if index_path.exists() else ""
        assert "old-session" not in after

    @pytest.mark.asyncio
    async def test_non_session_not_archived(self, tmp_path: Path) -> None:
        """非 session 类型即使时间久也不应归档。"""
        project_root = tmp_path / "project"
        project_root.mkdir(parents=True)

        mgr = make_manager(tmp_path / "global", project_root)

        old_entry = MemoryEntry(
            name="old-feedback", memory_type=MemoryType.FEEDBACK,
            description="Old feedback", body="Old feedback",
        )
        self._save_session_with_old_timestamp(mgr._store, old_entry, days_ago=30)

        report = mgr.run_maintenance()
        # feedback 不会被归档（只有 session 会被归档）
        # 但可能会被 confidence 衰减删除
        archived = report["archived_sessions"]
        assert archived == 0, "Non-session memories should not be archived"
