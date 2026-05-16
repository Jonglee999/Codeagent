"""StrategyStore 单元测试。

覆盖：
- save/get 完整 CRUD
- list_active / list_archived 分类正确
- update_confidence 成功+0.1 / 失败-0.15 / 低于 0.2 归档
- increment_applied 计数更新
- apply_decay 30 天衰减 / 90 天归档
- search 关键词匹配 / 加权排序
- get_stats 统计正确
- 文件不存在的降级
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from codeagent.context_engine.evolution import Strategy, StrategyStore


# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture
def store(tmp_path: Path) -> StrategyStore:
    """使用临时目录创建 StrategyStore。"""
    base_path = tmp_path / ".codeagent" / "strategies"
    return StrategyStore(base_path=str(base_path))


@pytest.fixture
def sample_strategy() -> Strategy:
    """创建一个示例策略。"""
    return Strategy(
        strategy_id=str(uuid.uuid4()),
        title="添加 FastAPI 路由时同步更新测试",
        condition="当需要在 FastAPI 项目中添加新路由时",
        action="应同时在 tests/ 下添加对应的测试文件",
        rationale="未同步写测试导致 CI 失败",
        category="workflow",
        source_task_ids=["task-1", "task-2"],
        confidence=0.85,
    )


def make_strategy(
    strategy_id: str = "",
    title: str = "Test strategy",
    condition: str = "when X happens",
    action: str = "do Y",
    rationale: str = "because Z",
    category: str = "workflow",
    confidence: float = 0.8,
    source_task_ids: list[str] | None = None,
    applied_count: int = 0,
    success_count: int = 0,
    last_applied: datetime | None = None,
) -> Strategy:
    """创建测试用 Strategy 对象。"""
    return Strategy(
        strategy_id=strategy_id or str(uuid.uuid4()),
        title=title,
        condition=condition,
        action=action,
        rationale=rationale,
        category=category,  # type: ignore[arg-type]
        source_task_ids=source_task_ids or [],
        confidence=confidence,
        applied_count=applied_count,
        success_count=success_count,
        last_applied=last_applied,
    )


# ── save / get 测试 ───────────────────────────────────────────


class TestSaveGet:
    """save 和 get 方法测试。"""

    def test_save_and_get(self, store: StrategyStore, sample_strategy: Strategy) -> None:
        sid = store.save(sample_strategy)
        assert sid == sample_strategy.strategy_id

        loaded = store.get(sample_strategy.strategy_id)
        assert loaded is not None
        assert loaded.strategy_id == sample_strategy.strategy_id
        assert loaded.title == sample_strategy.title
        assert loaded.condition == sample_strategy.condition
        assert loaded.action == sample_strategy.action
        assert loaded.rationale == sample_strategy.rationale
        assert loaded.category == sample_strategy.category
        assert loaded.source_task_ids == sample_strategy.source_task_ids
        assert loaded.confidence == sample_strategy.confidence

    def test_save_creates_file(self, store: StrategyStore, sample_strategy: Strategy) -> None:
        store.save(sample_strategy)
        file_path = store._base_path / sample_strategy.category / f"{sample_strategy.strategy_id}.md"
        assert file_path.exists()
        content = file_path.read_text(encoding="utf-8")
        assert sample_strategy.title in content
        assert sample_strategy.condition in content

    def test_get_nonexistent(self, store: StrategyStore) -> None:
        result = store.get("nonexistent-id")
        assert result is None

    def test_save_multiple_categories(self, store: StrategyStore) -> None:
        categories = ["workflow", "coding_style", "error_avoidance", "tool_usage"]
        saved_ids = []
        for i, cat in enumerate(categories):
            s = make_strategy(
                strategy_id=f"multi-{i}",
                title=f"Strategy {i}",
                category=cat,
                source_task_ids=["t1"],
            )
            store.save(s)
            saved_ids.append(s.strategy_id)

        for sid in saved_ids:
            loaded = store.get(sid)
            assert loaded is not None

    def test_save_invalid_category_fallback(self, store: StrategyStore) -> None:
        s = make_strategy(
            strategy_id="invalid-cat",
            title="Invalid cat",
            category="invalid_category",
            source_task_ids=["t1"],
        )
        store.save(s)
        # Should have been saved under 'workflow'
        file_path = store._base_path / "workflow" / f"{s.strategy_id}.md"
        assert file_path.exists()

    def test_save_preserves_optional_fields(self, store: StrategyStore) -> None:
        now = datetime.now()
        s = Strategy(
            strategy_id="full-strategy",
            title="Full fields",
            condition="condition",
            action="action",
            rationale="rationale",
            category="error_avoidance",
            source_task_ids=["t1", "t2"],
            confidence=0.95,
            created_at=now - timedelta(days=10),
            applied_count=5,
            success_count=4,
            last_applied=now - timedelta(days=1),
        )
        store.save(s)
        loaded = store.get(s.strategy_id)
        assert loaded is not None
        assert loaded.applied_count == 5
        assert loaded.success_count == 4
        assert loaded.confidence == 0.95
        assert loaded.last_applied is not None
        assert abs((loaded.last_applied - (now - timedelta(days=1))).total_seconds()) < 5


# ── list_active / list_archived 测试 ──────────────────────────


class TestListActive:
    """list_active 方法测试。"""

    def test_list_active_empty(self, store: StrategyStore) -> None:
        assert store.list_active() == []

    def test_list_active_returns_all_active(self, store: StrategyStore) -> None:
        for i in range(3):
            s = make_strategy(strategy_id=f"active-{i}", title=f"Active {i}", source_task_ids=[])
            store.save(s)
        active = store.list_active()
        assert len(active) == 3

    def test_list_active_filters_low_confidence(self, store: StrategyStore) -> None:
        good = make_strategy(strategy_id="good-1", title="Good", confidence=0.8, source_task_ids=[])
        low = make_strategy(strategy_id="low-1", title="Low", confidence=0.1, source_task_ids=[])
        store.save(good)
        store.save(low)
        active = store.list_active()
        ids = [s.strategy_id for s in active]
        assert "good-1" in ids
        assert "low-1" not in ids

    def test_list_active_archived_not_included(self, store: StrategyStore) -> None:
        s = make_strategy(strategy_id="to-archive", title="Archived", confidence=0.8, source_task_ids=[])
        store.save(s)
        store._archive(s.strategy_id)
        active = store.list_active()
        assert s.strategy_id not in [a.strategy_id for a in active]


class TestListArchived:
    """list_archived 方法测试。"""

    def test_list_archived_empty(self, store: StrategyStore) -> None:
        assert store.list_archived() == []

    def test_list_archived_after_archive(self, store: StrategyStore) -> None:
        s = make_strategy(strategy_id="arch-1", title="To archive", confidence=0.8, source_task_ids=[])
        store.save(s)
        store._archive(s.strategy_id)
        archived = store.list_archived()
        assert len(archived) == 1
        assert archived[0].strategy_id == "arch-1"

    def test_list_archived_only_archived(self, store: StrategyStore) -> None:
        s1 = make_strategy(strategy_id="active-keep", title="Keep", source_task_ids=[])
        s2 = make_strategy(strategy_id="archived-move", title="Move", source_task_ids=[])
        store.save(s1)
        store.save(s2)
        store._archive(s2.strategy_id)
        archived = store.list_archived()
        assert len(archived) == 1
        assert archived[0].strategy_id == "archived-move"


# ── update_confidence 测试 ────────────────────────────────────


class TestUpdateConfidence:
    """update_confidence 方法测试。"""

    def test_update_confidence_success(self, store: StrategyStore) -> None:
        s = make_strategy(strategy_id="conf-succ", title="Test", confidence=0.5, source_task_ids=[])
        store.save(s)
        store.update_confidence(s.strategy_id, success=True)
        loaded = store.get(s.strategy_id)
        assert loaded is not None
        assert loaded.confidence == 0.6  # 0.5 + 0.1

    def test_update_confidence_failure(self, store: StrategyStore) -> None:
        s = make_strategy(strategy_id="conf-fail", title="Test", confidence=0.8, source_task_ids=[])
        store.save(s)
        store.update_confidence(s.strategy_id, success=False)
        loaded = store.get(s.strategy_id)
        assert loaded is not None
        assert loaded.confidence == 0.65  # 0.8 - 0.15

    def test_update_confidence_capped_at_1_0(self, store: StrategyStore) -> None:
        s = make_strategy(strategy_id="conf-cap", title="Test", confidence=0.95, source_task_ids=[])
        store.save(s)
        store.update_confidence(s.strategy_id, success=True)
        loaded = store.get(s.strategy_id)
        assert loaded is not None
        assert loaded.confidence == 1.0  # capped

    def test_update_confidence_floor_at_0_0(self, store: StrategyStore) -> None:
        s = make_strategy(strategy_id="conf-floor", title="Test", confidence=0.1, source_task_ids=[])
        store.save(s)
        store.update_confidence(s.strategy_id, success=False)
        loaded = store.get(s.strategy_id)
        assert loaded is not None
        assert loaded.confidence == 0.0  # floored

    def test_update_confidence_archives_below_0_2(self, store: StrategyStore) -> None:
        """置信度低于 0.2 时自动归档。"""
        s = make_strategy(strategy_id="conf-arch", title="Test", confidence=0.3, source_task_ids=[])
        store.save(s)
        store.update_confidence(s.strategy_id, success=False)  # 0.3 - 0.15 = 0.15 < 0.2

        # Should be in archived
        active = store.list_active()
        assert s.strategy_id not in [a.strategy_id for a in active]
        archived = store.list_archived()
        assert s.strategy_id in [a.strategy_id for a in archived]

    def test_update_confidence_nonexistent(self, store: StrategyStore) -> None:
        """不存在的策略不应抛异常。"""
        store.update_confidence("nonexistent", success=True)
        # Should not raise


# ── increment_applied 测试 ────────────────────────────────────


class TestIncrementApplied:
    """increment_applied 方法测试。"""

    def test_increment_applied_success(self, store: StrategyStore) -> None:
        s = make_strategy(strategy_id="inc-succ", title="Test", source_task_ids=[])
        store.save(s)
        store.increment_applied(s.strategy_id, success=True)
        loaded = store.get(s.strategy_id)
        assert loaded is not None
        assert loaded.applied_count == 1
        assert loaded.success_count == 1
        assert loaded.last_applied is not None

    def test_increment_applied_failure(self, store: StrategyStore) -> None:
        s = make_strategy(strategy_id="inc-fail", title="Test", source_task_ids=[])
        store.save(s)
        store.increment_applied(s.strategy_id, success=False)
        loaded = store.get(s.strategy_id)
        assert loaded is not None
        assert loaded.applied_count == 1
        assert loaded.success_count == 0  # not incremented

    def test_increment_applied_multiple_times(self, store: StrategyStore) -> None:
        s = make_strategy(strategy_id="inc-multi", title="Test", source_task_ids=[])
        store.save(s)
        for i in range(5):
            store.increment_applied(s.strategy_id, success=(i % 2 == 0))
        loaded = store.get(s.strategy_id)
        assert loaded is not None
        assert loaded.applied_count == 5
        assert loaded.success_count == 3  # 3 successes, 2 failures

    def test_increment_applied_nonexistent(self, store: StrategyStore) -> None:
        """不存在的策略不应抛异常。"""
        store.increment_applied("nonexistent", success=True)
        # Should not raise


# ── apply_decay 测试 ──────────────────────────────────────────


class TestApplyDecay:
    """apply_decay 方法测试。"""

    def test_apply_decay_no_decay_for_recent(self, store: StrategyStore) -> None:
        """最近应用的策略不应衰减。"""
        now = datetime.now()
        s = make_strategy(
            strategy_id="recent", title="Recent", confidence=0.9,
            last_applied=now - timedelta(days=5), source_task_ids=[],
        )
        store.save(s)
        count = store.apply_decay()
        loaded = store.get(s.strategy_id)
        assert loaded is not None
        assert loaded.confidence == 0.9
        assert count >= 0

    def test_apply_decay_30_days(self, store: StrategyStore) -> None:
        """超过 30 天未应用应衰减 0.1。"""
        now = datetime.now()
        s = make_strategy(
            strategy_id="old-30d", title="30 days old", confidence=0.8,
            last_applied=now - timedelta(days=35), source_task_ids=[],
        )
        store.save(s)
        count = store.apply_decay()
        loaded = store.get(s.strategy_id)
        assert loaded is not None
        assert loaded.confidence == pytest.approx(0.7, rel=1e-9)
        assert count >= 1

    def test_apply_decay_90_days_archives(self, store: StrategyStore) -> None:
        """超过 90 天未应用应归档。"""
        now = datetime.now()
        s = make_strategy(
            strategy_id="old-90d", title="90 days old", confidence=0.9,
            last_applied=now - timedelta(days=95), source_task_ids=[],
        )
        store.save(s)
        count = store.apply_decay()
        # Should be archived
        active = store.list_active()
        assert s.strategy_id not in [a.strategy_id for a in active]
        archived = store.list_archived()
        assert s.strategy_id in [a.strategy_id for a in archived]
        assert count >= 1

    def test_apply_decay_uses_created_at_if_no_last_applied(self, store: StrategyStore) -> None:
        """没有 last_applied 时使用 created_at 判断。"""
        s = make_strategy(
            strategy_id="no-last", title="No last_applied",
            confidence=0.8, last_applied=None, source_task_ids=[],
        )
        # Override created_at to be 40 days ago
        s.created_at = datetime.now() - timedelta(days=40)
        store.save(s)
        count = store.apply_decay()
        loaded = store.get(s.strategy_id)
        assert loaded is not None
        assert loaded.confidence == pytest.approx(0.7, rel=1e-9)  # decayed
        assert count >= 1

    def test_apply_decay_none_active(self, store: StrategyStore) -> None:
        """无活跃策略时返回 0。"""
        count = store.apply_decay()
        assert count == 0


# ── search 测试 ───────────────────────────────────────────────


class TestSearch:
    """search 方法测试。"""

    def test_search_empty_query(self, store: StrategyStore) -> None:
        s = make_strategy(strategy_id="s1", title="Test", source_task_ids=[])
        store.save(s)
        results = store.search("")
        assert results == []

    def test_search_by_condition(self, store: StrategyStore) -> None:
        s = make_strategy(
            strategy_id="s-fastapi", title="FastAPI route",
            condition="在 FastAPI 项目中添加新路由",
            action="do something", rationale="reason",
            source_task_ids=[],
        )
        store.save(s)
        results = store.search("FastAPI")
        assert len(results) >= 1

    def test_search_by_action(self, store: StrategyStore) -> None:
        s = make_strategy(
            strategy_id="s-migration", title="DB migration",
            condition="something", action="执行数据库迁移脚本",
            rationale="reason", source_task_ids=[],
        )
        store.save(s)
        results = store.search("数据库迁移")
        assert len(results) >= 1

    def test_search_ranking_condition_has_higher_weight(self, store: StrategyStore) -> None:
        """condition 匹配的策略应排在前。"""
        s1 = make_strategy(
            strategy_id="s-condition", title="Match condition",
            condition="FastAPI route 当需要添加路由时",
            action="other stuff", rationale="other",
            source_task_ids=[],
        )
        s2 = make_strategy(
            strategy_id="s-action", title="Match action",
            condition="other stuff", action="FastAPI route 添加路由",
            rationale="other", source_task_ids=[],
        )
        store.save(s1)
        store.save(s2)
        results = store.search("FastAPI route")
        assert len(results) >= 2
        # s1 should rank higher (condition match has 2x weight)
        assert results[0].strategy_id == "s-condition"

    def test_search_no_match(self, store: StrategyStore) -> None:
        s = make_strategy(
            strategy_id="s-unrelated", title="Unrelated",
            condition="something unrelated", action="do other",
            rationale="xyz", source_task_ids=[],
        )
        store.save(s)
        results = store.search("nonexistentkeyword")
        assert results == []

    def test_search_top_k(self, store: StrategyStore) -> None:
        for i in range(10):
            s = make_strategy(
                strategy_id=f"s-{i}", title=f"Strategy {i}",
                condition=f"keyword condition {i}", action="action",
                rationale="reason", source_task_ids=[],
            )
            store.save(s)
        results = store.search("keyword", top_k=3)
        assert len(results) <= 3


# ── get_stats 测试 ────────────────────────────────────────────


class TestGetStats:
    """get_stats 方法测试。"""

    def test_get_stats_empty(self, store: StrategyStore) -> None:
        stats = store.get_stats()
        assert stats["total"] == 0
        assert stats["active"] == 0
        assert stats["archived"] == 0
        assert stats["avg_confidence"] == 0.0

    def test_get_stats_with_data(self, store: StrategyStore) -> None:
        s1 = make_strategy(strategy_id="stat-1", title="S1", confidence=0.9, source_task_ids=[])
        s2 = make_strategy(strategy_id="stat-2", title="S2", confidence=0.7, source_task_ids=[])
        s3 = make_strategy(strategy_id="stat-3", title="S3", confidence=0.1, source_task_ids=[])
        store.save(s1)
        store.save(s2)
        store.save(s3)
        # s3 will be filtered from active but still on disk
        stats = store.get_stats()
        # s3 confidence < 0.2, so it won't show in active
        assert stats["active"] == 2
        assert stats["avg_confidence"] == 0.8  # (0.9 + 0.7) / 2

    def test_get_stats_archived_counted(self, store: StrategyStore) -> None:
        s1 = make_strategy(strategy_id="arch-stat", title="Arch", confidence=0.8, source_task_ids=[])
        store.save(s1)
        store._archive(s1.strategy_id)
        stats = store.get_stats()
        assert stats["active"] == 0
        assert stats["archived"] == 1
        assert stats["total"] == 1


# ── 边界情况和错误处理测试 ────────────────────────────────────


class TestEdgeCases:
    """边界情况和错误处理测试。"""

    def test_base_path_auto_creation(self, tmp_path: Path) -> None:
        """不存在的目录应自动创建。"""
        base = tmp_path / "new" / "deep" / "path" / "strategies"
        s = StrategyStore(base_path=str(base))
        strategy = make_strategy(strategy_id="auto-dir", title="Auto dir")
        s.save(strategy)
        loaded = s.get(strategy.strategy_id)
        assert loaded is not None

    def test_get_after_delete_file(self, store: StrategyStore) -> None:
        s = make_strategy(strategy_id="del-test", title="Delete test", source_task_ids=[])
        store.save(s)
        store._delete_file(s.strategy_id)
        loaded = store.get(s.strategy_id)
        assert loaded is None

    def test_get_from_archived(self, store: StrategyStore) -> None:
        s = make_strategy(strategy_id="arch-get", title="Get from archived", source_task_ids=[])
        store.save(s)
        store._archive(s.strategy_id)
        loaded = store.get(s.strategy_id)
        assert loaded is not None

    def test_multiple_saves_update(self, store: StrategyStore) -> None:
        """多次保存同一策略应更新文件。"""
        s = make_strategy(strategy_id="update-test", title="Original", confidence=0.5, source_task_ids=[])
        store.save(s)
        s.confidence = 0.9
        store.save(s)  # save again with updated confidence
        loaded = store.get(s.strategy_id)
        assert loaded is not None
        assert loaded.confidence == 0.9

    def test_search_excludes_archived(self, store: StrategyStore) -> None:
        """search 不应返回已归档的策略。"""
        s = make_strategy(strategy_id="arch-search", title="Archived",
                          condition="test keyword match", action="x", rationale="y",
                          source_task_ids=[])
        store.save(s)
        store._archive(s.strategy_id)
        results = store.search("test keyword")
        assert s.strategy_id not in [r.strategy_id for r in results]
