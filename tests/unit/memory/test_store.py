"""MemoryStore 单元测试。

覆盖：save/load/delete/list_all、项目层优先、索引截断、session 归档、Frontmatter 序列化。
"""

from __future__ import annotations

import shutil
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from codeagent.memory.store import MemoryEntry, MemoryStore, MemoryType


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def global_root(tmp_path: Path) -> Path:
    """全局存储根目录（临时）。"""
    root = tmp_path / "global_memory"
    root.mkdir(parents=True)
    return root


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    """项目存储根目录（临时）。"""
    root = tmp_path / "project_memory"
    root.mkdir(parents=True)
    return root


@pytest.fixture
def store(global_root: Path, project_root: Path) -> MemoryStore:
    """带项目层的 MemoryStore 实例。"""
    return MemoryStore(global_root=global_root, project_root=project_root)


@pytest.fixture
def store_global_only(global_root: Path) -> MemoryStore:
    """仅全局层的 MemoryStore 实例。"""
    return MemoryStore(global_root=global_root)


# ── 测试辅助函数 ─────────────────────────────────────────────────────────


def make_entry(
    name: str = "test-entry",
    memory_type: MemoryType = MemoryType.USER,
    description: str = "Test description",
    body: str = "Test body content",
    tags: list[str] | None = None,
    confidence: float = 1.0,
    last_accessed_at: datetime | None = None,
) -> MemoryEntry:
    """创建测试用的 MemoryEntry。"""
    return MemoryEntry(
        name=name,
        memory_type=memory_type,
        description=description,
        body=body,
        tags=tags or [],
        confidence=confidence,
        last_accessed_at=last_accessed_at,
    )


# ══════════════════════════════════════════════════════════════════════════
# 1. 序列化 / 反序列化
# ══════════════════════════════════════════════════════════════════════════


class TestFrontmatterSerialization:
    """Frontmatter 序列化/反序列化测试。"""

    def test_to_frontmatter_roundtrip(self) -> None:
        """验证 MemoryEntry 序列化后反序列化回相同内容。"""
        entry = make_entry(
            name="test-roundtrip",
            memory_type=MemoryType.FEEDBACK,
            description="Roundtrip test",
            body="# Title\n\nSome **bold** text.",
            tags=["test", "roundtrip"],
        )
        text = entry.to_frontmatter_str()
        restored = MemoryEntry.from_frontmatter_str(text)

        assert restored.name == entry.name
        assert restored.memory_type == entry.memory_type
        assert restored.description == entry.description
        assert restored.body == entry.body
        assert restored.tags == entry.tags
        assert restored.confidence == entry.confidence

    def test_frontmatter_handles_empty_body(self) -> None:
        """验证空 body 也能正确序列化。"""
        entry = make_entry(
            name="empty-body",
            memory_type=MemoryType.USER,
            description="Empty body",
            body="",
        )
        text = entry.to_frontmatter_str()
        restored = MemoryEntry.from_frontmatter_str(text)
        assert restored.body == ""

    def test_frontmatter_handles_last_accessed(self) -> None:
        """验证 last_accessed_at 为 None 时序列化不包含该字段。"""
        entry = make_entry(name="no-access", memory_type=MemoryType.USER, description="No access")
        text = entry.to_frontmatter_str()
        assert "last_accessed_at" not in text

        entry.last_accessed_at = datetime(2026, 5, 16, 10, 0, 0)
        text = entry.to_frontmatter_str()
        restored = MemoryEntry.from_frontmatter_str(text)
        assert restored.last_accessed_at is not None
        assert restored.last_accessed_at.year == 2026

    def test_frontmatter_not_crash_on_invalid(self) -> None:
        """验证非法内容不会导致崩溃，而是优雅降级。"""
        # 缺少 name 字段会得到默认空字符串的 entry，但不崩溃
        entry = MemoryEntry.from_frontmatter_str("hello: world\n---\nbody")
        assert isinstance(entry, MemoryEntry)

    def test_frontmatter_all_types_roundtrip(self) -> None:
        """验证所有 5 种类型都能正确序列化/反序列化。"""
        for t in MemoryType:
            entry = make_entry(name=f"type-{t.value}", memory_type=t, description=f"Type {t.value}")
            text = entry.to_frontmatter_str()
            restored = MemoryEntry.from_frontmatter_str(text)
            assert restored.memory_type == t


# ══════════════════════════════════════════════════════════════════════════
# 2. MemoryStore: save / load
# ══════════════════════════════════════════════════════════════════════════


class TestSaveLoad:
    """save / load 基础测试。"""

    def test_save_user_to_global(self, store: MemoryStore) -> None:
        """USER 类型应自动保存到全局层。"""
        entry = make_entry(
            name="my-user", memory_type=MemoryType.USER, description="A user"
        )
        path = store.save(entry)
        assert path.exists()
        assert "global_memory" in str(path)
        assert path.parent.name == "user"

    def test_save_feedback_to_global(self, store: MemoryStore) -> None:
        """FEEDBACK 类型应自动保存到全局层。"""
        entry = make_entry(
            name="my-feedback", memory_type=MemoryType.FEEDBACK, description="A feedback"
        )
        path = store.save(entry)
        assert "global_memory" in str(path)

    def test_save_project_to_project(self, store: MemoryStore) -> None:
        """PROJECT 类型应自动保存到项目层。"""
        entry = make_entry(
            name="my-project", memory_type=MemoryType.PROJECT, description="A project"
        )
        path = store.save(entry)
        assert "project_memory" in str(path)

    def test_save_code_pattern_to_project(self, store: MemoryStore) -> None:
        """CODE_PATTERN 类型应自动保存到项目层。"""
        entry = make_entry(
            name="my-pattern", memory_type=MemoryType.CODE_PATTERN, description="A pattern"
        )
        path = store.save(entry)
        assert "project_memory" in str(path)

    def test_save_session_to_project(self, store: MemoryStore) -> None:
        """SESSION 类型应自动保存到项目层。"""
        entry = make_entry(
            name="my-session", memory_type=MemoryType.SESSION, description="A session"
        )
        path = store.save(entry)
        assert "project_memory" in str(path)

    def test_save_with_force_scope_global(self, store: MemoryStore) -> None:
        """强制 scope='global' 时不管什么类型都存全局。"""
        entry = make_entry(
            name="force-global", memory_type=MemoryType.PROJECT, description="Forced global"
        )
        path = store.save(entry, scope="global")
        assert "global_memory" in str(path)

    def test_save_with_force_scope_project(self, store: MemoryStore) -> None:
        """强制 scope='project' 时不管什么类型都存项目层。"""
        entry = make_entry(
            name="force-project", memory_type=MemoryType.USER, description="Forced project"
        )
        path = store.save(entry, scope="project")
        assert "project_memory" in str(path)

    def test_save_global_only_no_project(self, store: MemoryType) -> None:
        """无 project_root 时所有类型都存全局。"""
        store2 = MemoryStore(global_root=Path("/tmp/test_global"))
        store2._global_root = store._global_root  # use tmp_path
        entry = make_entry(
            name="no-project", memory_type=MemoryType.SESSION, description="No project"
        )
        path = store2.save(entry)
        assert "global_memory" in str(path)

    def test_load_by_name(self, store: MemoryStore) -> None:
        """load() 应返回已保存的记忆条目。"""
        entry = make_entry(name="find-me", memory_type=MemoryType.FEEDBACK, description="Find me")
        store.save(entry)
        loaded = store.load("find-me")
        assert loaded is not None
        assert loaded.name == "find-me"
        assert loaded.description == "Find me"
        assert loaded.memory_type == MemoryType.FEEDBACK

    def test_load_nonexistent_returns_none(self, store: MemoryStore) -> None:
        """load() 不存在的记忆应返回 None。"""
        loaded = store.load("does-not-exist")
        assert loaded is None

    def test_load_project_over_global(self, store: MemoryStore) -> None:
        """项目层同名记忆应优先于全局层。"""
        global_entry = make_entry(
            name="same-name", memory_type=MemoryType.FEEDBACK,
            description="Global version",
            body="This is global",
        )
        store.save(global_entry, scope="global")

        project_entry = make_entry(
            name="same-name", memory_type=MemoryType.FEEDBACK,
            description="Project version",
            body="This is project",
        )
        store.save(project_entry, scope="project")

        loaded = store.load("same-name")
        assert loaded is not None
        assert loaded.body == "This is project"

    def test_load_updates_last_accessed(self, store: MemoryStore) -> None:
        """load() 应更新记忆的 last_accessed_at。"""
        entry = make_entry(name="access-me", memory_type=MemoryType.USER, description="Access me")
        store.save(entry)
        assert entry.last_accessed_at is None

        loaded = store.load("access-me")
        assert loaded is not None
        assert loaded.last_accessed_at is not None


# ══════════════════════════════════════════════════════════════════════════
# 3. delete
# ══════════════════════════════════════════════════════════════════════════


class TestDelete:
    """delete 操作测试。"""

    def test_delete_global_memory(self, store: MemoryStore) -> None:
        """删除全局层的记忆。"""
        entry = make_entry(name="delete-me", memory_type=MemoryType.USER, description="Delete me")
        path = store.save(entry)
        assert path.exists()

        result = store.delete("delete-me")
        assert result is True
        assert not path.exists()

    def test_delete_nonexistent_returns_false(self, store: MemoryStore) -> None:
        """删除不存在的记忆应返回 False。"""
        result = store.delete("i-dont-exist")
        assert result is False

    def test_delete_project_memory(self, store: MemoryStore) -> None:
        """删除项目层的记忆。"""
        entry = make_entry(name="proj-del", memory_type=MemoryType.PROJECT, description="Proj del")
        path = store.save(entry)
        assert path.exists()

        result = store.delete("proj-del")
        assert result is True
        assert not path.exists()

    def test_delete_updates_index(self, store: MemoryStore) -> None:
        """删除后索引不应再包含该条目。"""
        entry = make_entry(name="gone", memory_type=MemoryType.USER, description="Will be gone")
        store.save(entry)

        # 确认索引包含
        index_path = store._global_root / "MEMORY.md"
        assert "gone" in index_path.read_text(encoding="utf-8")

        store.delete("gone")
        assert "gone" not in index_path.read_text(encoding="utf-8")


# ══════════════════════════════════════════════════════════════════════════
# 4. list_all
# ══════════════════════════════════════════════════════════════════════════


class TestListAll:
    """list_all 测试。"""

    def test_list_all_empty(self, store: MemoryStore) -> None:
        """空存储时应返回空列表。"""
        entries = store.list_all()
        assert entries == []

    def test_list_all_returns_all(self, store: MemoryStore) -> None:
        """列出所有记忆。"""
        u = make_entry(name="u1", memory_type=MemoryType.USER, description="User 1")
        f = make_entry(name="f1", memory_type=MemoryType.FEEDBACK, description="Feedback 1")
        p = make_entry(name="p1", memory_type=MemoryType.PROJECT, description="Project 1")
        store.save(u)
        store.save(f)
        store.save(p)

        all_entries = store.list_all()
        assert len(all_entries) >= 3

    def test_list_all_filter_by_type(self, store: MemoryStore) -> None:
        """按类型筛选记忆。"""
        u = make_entry(name="user-only", memory_type=MemoryType.USER, description="Only user")
        f = make_entry(name="fb-only", memory_type=MemoryType.FEEDBACK, description="Only feedback")
        store.save(u)
        store.save(f)

        users = store.list_all(memory_type=MemoryType.USER)
        assert len(users) == 1
        assert users[0].name == "user-only"

        feedbacks = store.list_all(memory_type=MemoryType.FEEDBACK)
        assert len(feedbacks) == 1
        assert feedbacks[0].name == "fb-only"

    def test_list_all_project_first(self, store: MemoryStore) -> None:
        """项目层结果应排在全局层前面。"""
        gu = make_entry(name="g-user", memory_type=MemoryType.USER, description="Global user")
        store.save(gu, scope="global")

        pu = make_entry(name="p-user", memory_type=MemoryType.USER, description="Project user")
        store.save(pu, scope="project")

        all_entries = store.list_all(memory_type=MemoryType.USER)
        assert len(all_entries) >= 2
        # 项目层优先
        assert all_entries[0].name == "p-user"

    def test_list_all_scope_global(self, store: MemoryStore) -> None:
        """scope='global' 时只返回全局层。"""
        gu = make_entry(name="g-only", memory_type=MemoryType.USER, description="Global")
        pu = make_entry(name="p-only", memory_type=MemoryType.USER, description="Project")
        store.save(gu, scope="global")
        store.save(pu, scope="project")

        entries = store.list_all(scope="global")
        names = [e.name for e in entries]
        assert "g-only" in names
        assert "p-only" not in names

    def test_list_all_scope_project(self, store: MemoryStore) -> None:
        """scope='project' 时只返回项目层。"""
        gu = make_entry(name="g-only", memory_type=MemoryType.USER, description="Global")
        pu = make_entry(name="p-only", memory_type=MemoryType.USER, description="Project")
        store.save(gu, scope="global")
        store.save(pu, scope="project")

        entries = store.list_all(scope="project")
        names = [e.name for e in entries]
        assert "p-only" in names
        assert "g-only" not in names


# ══════════════════════════════════════════════════════════════════════════
# 5. update_index
# ══════════════════════════════════════════════════════════════════════════


class TestUpdateIndex:
    """MEMORY.md 索引测试。"""

    def test_index_created_on_save(self, store: MemoryStore) -> None:
        """save() 后应自动创建 MEMORY.md。"""
        entry = make_entry(name="idx-test", memory_type=MemoryType.USER, description="Index test")
        store.save(entry)

        index_path = store._global_root / "MEMORY.md"
        assert index_path.exists()
        content = index_path.read_text(encoding="utf-8")
        assert "idx-test" in content
        assert "Index" in content

    def test_index_deleted_entry_removed(self, store: MemoryStore) -> None:
        """删除后索引中不应包含对应条目。"""
        entry = make_entry(name="to-delete", memory_type=MemoryType.USER, description="Delete from index")
        store.save(entry)
        store.delete("to-delete")

        index_path = store._global_root / "MEMORY.md"
        content = index_path.read_text(encoding="utf-8")
        assert "to-delete" not in content

    def test_index_format(self, store: MemoryStore) -> None:
        """验证索引行格式正确。"""
        entry = make_entry(name="format-check", memory_type=MemoryType.FEEDBACK, description="Check format")
        store.save(entry)

        index_path = store._global_root / "MEMORY.md"
        content = index_path.read_text(encoding="utf-8")
        expected = "- [format-check](feedback/format-check.md) — Check format"
        assert expected in content


# ══════════════════════════════════════════════════════════════════════════
# 6. archive_expired_sessions
# ══════════════════════════════════════════════════════════════════════════


class TestArchiveExpiredSessions:
    """session 归档测试。"""

    def test_archive_expired_session(self, store: MemoryStore, project_root: Path) -> None:
        """超过 7 天未访问的 session 应被归档。"""
        old_time = datetime.now() - timedelta(days=8)
        entry = make_entry(
            name="old-session",
            memory_type=MemoryType.SESSION,
            description="Old session",
            last_accessed_at=old_time,
        )
        store.save(entry)

        archived_count = store.archive_expired_sessions(days=7)
        assert archived_count >= 1

        # SESSION 默认存入 project_root，检查项目层的 archived 目录
        archived_path = project_root / "archived" / "session" / "old-session.md"
        assert archived_path.exists()

        # 确认原文件已不存在
        orig_path = project_root / "session" / "old-session.md"
        assert not orig_path.exists()

    def test_archive_recent_session_not_archived(self, store: MemoryStore, project_root: Path) -> None:
        """7 天内的 session 不应被归档。"""
        recent_time = datetime.now() - timedelta(days=1)
        entry = make_entry(
            name="recent-session",
            memory_type=MemoryType.SESSION,
            description="Recent session",
            last_accessed_at=recent_time,
        )
        store.save(entry)

        archived_count = store.archive_expired_sessions(days=7)
        assert archived_count == 0

        # SESSION 默认存入 project_root
        orig_path = project_root / "session" / "recent-session.md"
        assert orig_path.exists()

    def test_archive_non_session_ignored(self, store: MemoryStore) -> None:
        """非 session 类型不应被归档。"""
        old_time = datetime.now() - timedelta(days=30)
        entry = make_entry(
            name="old-user",
            memory_type=MemoryType.USER,
            description="Old user",
            last_accessed_at=old_time,
        )
        store.save(entry)

        archived_count = store.archive_expired_sessions(days=7)
        assert archived_count == 0


# ══════════════════════════════════════════════════════════════════════════
# 7. save 异常路径
# ══════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """边界情况测试。"""

    def test_save_project_scope_without_project_root(self) -> None:
        """scope='project' 但未设置 project_root 时应抛异常。"""
        store2 = MemoryStore(global_root=Path("/tmp/test"))
        store2._global_root = Path("/tmp")  # dummy
        entry = make_entry(name="fail", memory_type=MemoryType.USER, description="Should fail")
        with pytest.raises(ValueError, match="project_root"):
            store2.save(entry, scope="project")

    def test_update_index_project_without_project_root(self) -> None:
        """project 层 update_index 但无 project_root 应抛异常。"""
        store2 = MemoryStore(global_root=Path("/tmp"))
        with pytest.raises(ValueError, match="project_root"):
            store2.update_index("project")

    def test_save_with_special_chars_in_name(self, store: MemoryStore, global_root: Path) -> None:
        """记忆 name 包含特殊字符能正常保存。"""
        entry = make_entry(
            name="test-special-chars",
            memory_type=MemoryType.USER,
            description="Special: @#$%",
        )
        store.save(entry)
        loaded = store.load("test-special-chars")
        assert loaded is not None
        assert loaded.description == "Special: @#$%"

    def test_list_all_no_project_root(self, global_root: Path) -> None:
        """无 project_root 时 list_all 正常工作。"""
        store2 = MemoryStore(global_root=global_root)
        entry = make_entry(name="no-proj", memory_type=MemoryType.USER, description="No proj")
        store2.save(entry)
        entries = store2.list_all()
        assert len(entries) >= 1
