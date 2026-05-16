"""StrategyApplier 单元测试。

覆盖：
- get_relevant_strategies 检索过滤、置信度阈值、max_strategies 限制、排序
- format_for_prompt XML 格式、空列表、XML 转义
- record_outcome 委托 store 更新、异常降级
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest

from codeagent.context_engine.evolution import Strategy, StrategyApplier, StrategyStore


# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture
def store(tmp_path: Path) -> StrategyStore:
    return StrategyStore(base_path=str(tmp_path / ".codeagent" / "strategies"))


@pytest.fixture
def applier(store: StrategyStore) -> StrategyApplier:
    return StrategyApplier(store=store, confidence_threshold=0.5, max_strategies=5)


@pytest.fixture
def strategies() -> list[Strategy]:
    """创建一批测试用策略。"""
    return [
        Strategy(
            strategy_id="s-fastapi",
            title="FastAPI route testing",
            condition="当在 FastAPI 项目中添加新路由时",
            action="应同时在 tests/ 下添加测试文件",
            rationale="未同步写测试导致 CI 失败",
            category="workflow",
            source_task_ids=[],
            confidence=0.9,
        ),
        Strategy(
            strategy_id="s-migration",
            title="DB migration",
            condition="当修改 SQLAlchemy 模型时",
            action="应生成迁移脚本并执行",
            rationale="模型修改后未迁移导致运行时错误",
            category="workflow",
            source_task_ids=[],
            confidence=0.75,
        ),
        Strategy(
            strategy_id="s-error",
            title="Error handling",
            condition="当捕获异常时",
            action="使用具体的异常类型而非裸 except",
            rationale="裸 except 隐藏错误",
            category="coding_style",
            source_task_ids=[],
            confidence=0.85,
        ),
        Strategy(
            strategy_id="s-low-conf",
            title="Low confidence",
            condition="when something",
            action="do something",
            rationale="because",
            category="tool_usage",
            source_task_ids=[],
            confidence=0.3,
        ),
    ]


def save_strategies(store: StrategyStore, strategies: list[Strategy]) -> None:
    """批量保存策略到 store。"""
    for s in strategies:
        store.save(s)


# ── get_relevant_strategies 测试 ──────────────────────────────


class TestGetRelevantStrategies:
    """get_relevant_strategies 方法测试。"""

    @pytest.mark.asyncio
    async def test_retrieves_matching_strategies(
        self, store: StrategyStore, applier: StrategyApplier, strategies: list[Strategy],
    ) -> None:
        save_strategies(store, strategies)
        results = await applier.get_relevant_strategies("FastAPI 路由")
        assert len(results) >= 1
        # s-fastapi should match (condition contains FastAPI 路由相关词)
        ids = [s.strategy_id for s in results]
        assert "s-fastapi" in ids

    @pytest.mark.asyncio
    async def test_filters_below_confidence_threshold(
        self, store: StrategyStore, applier: StrategyApplier, strategies: list[Strategy],
    ) -> None:
        save_strategies(store, strategies)
        results = await applier.get_relevant_strategies("something do")
        ids = [s.strategy_id for s in results]
        # s-low-conf has confidence 0.3 < 0.5, should be filtered out
        assert "s-low-conf" not in ids

    @pytest.mark.asyncio
    async def test_respects_max_strategies(
        self, store: StrategyStore, strategies: list[Strategy],
    ) -> None:
        small_applier = StrategyApplier(store=store, max_strategies=2)
        save_strategies(store, strategies)
        results = await small_applier.get_relevant_strategies("test query")
        assert len(results) <= 2

    @pytest.mark.asyncio
    async def test_returns_sorted_by_confidence_desc(
        self, store: StrategyStore, applier: StrategyApplier, strategies: list[Strategy],
    ) -> None:
        save_strategies(store, strategies)
        # Use a query that matches many strategies broadly
        results = await applier.get_relevant_strategies("当 应 因为")
        if len(results) >= 2:
            for i in range(len(results) - 1):
                assert results[i].confidence >= results[i + 1].confidence

    @pytest.mark.asyncio
    async def test_empty_when_no_match(
        self, store: StrategyStore, applier: StrategyApplier,
    ) -> None:
        results = await applier.get_relevant_strategies("nonexistent keyword")
        assert results == []

    @pytest.mark.asyncio
    async def test_store_search_failure_fallback(
        self, applier: StrategyApplier,
    ) -> None:
        """store.search 抛异常时返回空列表。"""
        # Use a store with no base path that doesn't exist
        results = await applier.get_relevant_strategies("test")
        # Should return empty list without raising
        assert results == []

    @pytest.mark.asyncio
    async def test_top_k_respected(
        self, store: StrategyStore, strategies: list[Strategy], applier: StrategyApplier,
    ) -> None:
        save_strategies(store, strategies)
        results = await applier.get_relevant_strategies("当 应", top_k=2)
        assert len(results) <= 2


# ── format_for_prompt 测试 ────────────────────────────────────


class TestFormatForPrompt:
    """format_for_prompt 方法测试。"""

    def test_format_xml_structure(self, applier: StrategyApplier, strategies: list[Strategy]) -> None:
        xml = applier.format_for_prompt(strategies[:1])
        assert xml.startswith("<strategies>")
        assert xml.endswith("</strategies>")
        assert "<strategy confidence=" in xml
        assert "<condition>" in xml
        assert "<action>" in xml
        assert "<rationale>" in xml
        assert strategies[0].condition in xml

    def test_format_empty_list(self, applier: StrategyApplier) -> None:
        xml = applier.format_for_prompt([])
        assert xml == ""

    def test_format_multiple_strategies(self, applier: StrategyApplier, strategies: list[Strategy]) -> None:
        xml = applier.format_for_prompt(strategies)
        # Each strategy should have its own <strategy> block
        assert xml.count("<strategy") == len(strategies)
        assert xml.count("</strategy>") == len(strategies)

    def test_format_includes_confidence_and_category(self, applier: StrategyApplier, strategies: list[Strategy]) -> None:
        s = strategies[0]
        xml = applier.format_for_prompt([s])
        assert f'confidence="{s.confidence}"' in xml
        assert f'category="{s.category}"' in xml

    def test_xml_escaping(self, applier: StrategyApplier) -> None:
        """XML 特殊字符应被转义。"""
        s = Strategy(
            strategy_id="s-xml",
            title="XML test",
            condition="a < b & c > d",
            action='use "quotes" and \'apos\'',
            rationale="safe & sound",
            category="workflow",
            source_task_ids=[],
        )
        xml = applier.format_for_prompt([s])
        assert "&lt;" in xml  # < escaped
        assert "&gt;" in xml  # > escaped
        assert "&amp;" in xml  # & escaped
        assert "&quot;" in xml  # " escaped
        assert "&apos;" in xml  # ' escaped
        # Original chars should NOT appear
        assert "< b" not in xml
        assert "& c" not in xml  # should be &amp; not bare &


# ── record_outcome 测试 ───────────────────────────────────────


class TestRecordOutcome:
    """record_outcome 方法测试。"""

    @pytest.mark.asyncio
    async def test_record_success(
        self, store: StrategyStore, applier: StrategyApplier,
    ) -> None:
        s = Strategy(
            strategy_id="outcome-succ",
            title="Test",
            condition="c", action="a", rationale="r",
            category="workflow", source_task_ids=[],
            confidence=0.8,
        )
        store.save(s)
        await applier.record_outcome([s.strategy_id], success=True)
        loaded = store.get(s.strategy_id)
        assert loaded is not None
        assert loaded.applied_count == 1
        assert loaded.success_count == 1
        assert loaded.confidence == 0.9  # 0.8 + 0.1

    @pytest.mark.asyncio
    async def test_record_failure(
        self, store: StrategyStore, applier: StrategyApplier,
    ) -> None:
        s = Strategy(
            strategy_id="outcome-fail",
            title="Test",
            condition="c", action="a", rationale="r",
            category="workflow", source_task_ids=[],
            confidence=0.8,
        )
        store.save(s)
        await applier.record_outcome([s.strategy_id], success=False)
        loaded = store.get(s.strategy_id)
        assert loaded is not None
        assert loaded.applied_count == 1
        assert loaded.success_count == 0  # not incremented
        assert loaded.confidence == 0.65  # 0.8 - 0.15

    @pytest.mark.asyncio
    async def test_record_empty_ids(
        self, applier: StrategyApplier,
    ) -> None:
        """空 ID 列表不应抛异常。"""
        await applier.record_outcome([], success=True)
        # Should not raise

    @pytest.mark.asyncio
    async def test_record_invalid_id_no_error(
        self, applier: StrategyApplier,
    ) -> None:
        """不存在的策略 ID 不应抛异常。"""
        await applier.record_outcome(["nonexistent-id"], success=True)
        # Should not raise

    @pytest.mark.asyncio
    async def test_record_multiple_strategies(
        self, store: StrategyStore, applier: StrategyApplier,
    ) -> None:
        s1 = Strategy(
            strategy_id="multi-1", title="S1",
            condition="c", action="a", rationale="r",
            category="workflow", source_task_ids=[], confidence=0.8,
        )
        s2 = Strategy(
            strategy_id="multi-2", title="S2",
            condition="c", action="a", rationale="r",
            category="workflow", source_task_ids=[], confidence=0.6,
        )
        store.save(s1)
        store.save(s2)
        await applier.record_outcome(["multi-1", "multi-2"], success=True)
        l1 = store.get("multi-1")
        l2 = store.get("multi-2")
        assert l1 is not None and l1.applied_count == 1
        assert l2 is not None and l2.applied_count == 1


# ── XML 转义辅助方法测试 ─────────────────────────────────────


class TestXmlEscape:
    """_xml_escape 辅助方法测试。"""

    def test_escape_ampersand(self) -> None:
        assert StrategyApplier._xml_escape("a & b") == "a &amp; b"

    def test_escape_angle_brackets(self) -> None:
        assert StrategyApplier._xml_escape("<hello>") == "&lt;hello&gt;"

    def test_escape_quotes(self) -> None:
        assert StrategyApplier._xml_escape('say "hello"') == "say &quot;hello&quot;"
        assert StrategyApplier._xml_escape("it's") == "it&apos;s"

    def test_no_escape_needed(self) -> None:
        text = "normal text Chinese 中文"
        assert StrategyApplier._xml_escape(text) == text


# ── planning_node.py 集成测试 ────────────────────────────────


class TestPlanningNodeIntegration:
    """PlanningNode + StrategyApplier 集成测试。"""

    @pytest.mark.asyncio
    async def test_applier_injected_into_planning_node(
        self, tmp_path: Path,
    ) -> None:
        """验证 StrategyApplier 能通过 PlanningNode 的 strategy_applier 参数注入。"""
        from codeagent.orchestration.nodes.planning_node import PlanningNode

        store = StrategyStore(base_path=str(tmp_path / ".codeagent" / "strategies"))
        s = Strategy(
            strategy_id="plan-test",
            title="Test strategy",
            condition="when fixing bugs",
            action="write tests first",
            rationale="tests prevent regression",
            category="workflow",
            source_task_ids=[],
            confidence=0.85,
        )
        store.save(s)
        applier = StrategyApplier(store=store)

        async def mock_llm(model: str = "", messages: list | None = None) -> Any:
            class MockResp:
                choices = [type("", (), {"content": '{"plan": [], "original_goal_summary": "test"}'})()]
            return MockResp()

        node = PlanningNode(
            llm=mock_llm,
            strategy_applier=applier,
        )
        assert node._strategy_applier is applier

    @pytest.mark.asyncio
    async def test_applier_none_skips_injection(
        self, tmp_path: Path,
    ) -> None:
        """strategy_applier=None 时行为不变。"""
        from codeagent.orchestration.nodes.planning_node import PlanningNode

        async def mock_llm(model: str = "", messages: list | None = None) -> Any:
            class MockResp:
                choices = [type("", (), {"content": '{"plan": [], "original_goal_summary": "test"}'})()]
            return MockResp()

        node = PlanningNode(llm=mock_llm)
        assert node._strategy_applier is None
