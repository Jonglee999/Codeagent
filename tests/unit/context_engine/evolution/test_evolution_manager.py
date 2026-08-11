"""SelfEvolutionManager 单元测试。

覆盖：
- on_task_start 正常流程
- on_task_complete 触发/不触发提炼
- 子模块抛异常时降级
- run_maintenance 委托 store
- get_stats 聚合统计
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from codeagent.context_engine.evolution.manager import SelfEvolutionManager
from codeagent.context_engine.evolution.trajectory_recorder import TrajectoryStep


@pytest.fixture
def mock_recorder() -> MagicMock:
    recorder = MagicMock()
    recorder.start_task = MagicMock()
    recorder.record_step = MagicMock()
    recorder.complete_task = MagicMock()
    return recorder


@pytest.fixture
def mock_extractor() -> MagicMock:
    extractor = MagicMock()
    extractor.should_extract = AsyncMock(return_value=False)
    extractor.extract_from_recent = AsyncMock(return_value=[])
    return extractor


@pytest.fixture
def mock_store() -> MagicMock:
    store = MagicMock()
    store.save = MagicMock(return_value="strategy-id")
    store.list_active = MagicMock(return_value=[])
    store.apply_decay = MagicMock(return_value=0)
    store.get_stats = MagicMock(return_value={
        "total": 5, "active": 4, "archived": 1, "avg_confidence": 0.75,
    })
    return store


@pytest.fixture
def mock_applier() -> MagicMock:
    applier = MagicMock()
    applier.get_relevant_strategies = AsyncMock(return_value=[])
    applier.format_for_prompt = MagicMock(return_value="")
    applier.record_outcome = AsyncMock()
    return applier


@pytest.fixture
def manager(mock_recorder, mock_extractor, mock_store, mock_applier):
    return SelfEvolutionManager(
        recorder=mock_recorder,
        extractor=mock_extractor,
        store=mock_store,
        applier=mock_applier,
        task_count=0,
    )


class TestOnTaskStart:
    """on_task_start 方法测试。"""

    async def test_start_records_trajectory(self, manager, mock_recorder):
        """应调用 recorder.start_task 开始轨迹记录。"""
        await manager.on_task_start(task_id="task-1", user_request="fix bug")
        mock_recorder.start_task.assert_called_once_with("task-1", "fix bug")

    async def test_returns_strategy_xml(self, manager, mock_applier):
        """当有相关策略时，应返回策略 XML。"""
        mock_strategies = [
            MagicMock(
                strategy_id="s1",
                condition="when adding routes",
                action="add tests",
                rationale="prevents CI failures",
                category="workflow",
                confidence=0.85,
            ),
        ]
        mock_applier.get_relevant_strategies = AsyncMock(return_value=mock_strategies)
        mock_applier.format_for_prompt = MagicMock(return_value="<strategies>...</strategies>")

        result = await manager.on_task_start(task_id="task-1", user_request="add route")
        assert result == "<strategies>...</strategies>"

    async def test_returns_empty_when_no_strategies(self, manager):
        """当无相关策略时，应返回空字符串。"""
        result = await manager.on_task_start(task_id="task-1", user_request="simple task")
        assert result == ""

    async def test_recorder_exception_does_not_propagate(self, manager, mock_recorder):
        """recorder.start_task 抛异常时不应传播。"""
        mock_recorder.start_task.side_effect = RuntimeError("recorder error")
        result = await manager.on_task_start(task_id="task-1", user_request="task")
        # Should return empty string gracefully
        assert isinstance(result, str)

    async def test_applier_exception_returns_empty(self, manager, mock_applier):
        """applier.get_relevant_strategies 抛异常时返回空字符串。"""
        mock_applier.get_relevant_strategies.side_effect = RuntimeError("applier error")
        result = await manager.on_task_start(task_id="task-1", user_request="task")
        assert result == ""


class TestRecordStep:
    """record_step 方法测试。"""

    def test_records_step(self, manager, mock_recorder):
        """应委托 recorder.record_step。"""
        step = TrajectoryStep(
            step_id="s1", node_name="execution",
            step_type="tool_call", timestamp=MagicMock(),
            input_summary="in", output_summary="out",
        )
        manager.record_step(task_id="task-1", step=step)
        mock_recorder.record_step.assert_called_once_with("task-1", step)

    def test_recorder_exception_does_not_propagate(self, manager, mock_recorder):
        """recorder.record_step 抛异常时不应传播。"""
        mock_recorder.record_step.side_effect = RuntimeError("record error")
        step = TrajectoryStep(
            step_id="s1", node_name="execution",
            step_type="tool_call", timestamp=MagicMock(),
            input_summary="in", output_summary="out",
        )
        # Should not raise
        manager.record_step(task_id="task-1", step=step)


class TestOnTaskComplete:
    """on_task_complete 方法测试。"""

    async def test_completes_trajectory(self, manager, mock_recorder):
        """应调用 recorder.complete_task。"""
        await manager.on_task_complete(
            task_id="task-1", success=True,
            repair_rounds=0, validation_passed=True,
        )
        mock_recorder.complete_task.assert_called_once_with(
            task_id="task-1", status="success",
            repair_rounds=0, validation_passed=True,
        )

    async def test_records_outcome_for_applied_strategies(self, manager, mock_applier):
        """当有应用的策略时，应记录结果。"""
        strategy_ids = ["s1", "s2"]
        await manager.on_task_complete(
            task_id="task-1", success=True,
            repair_rounds=0, validation_passed=True,
            applied_strategy_ids=strategy_ids,
        )
        mock_applier.record_outcome.assert_called_once_with(
            strategy_ids=strategy_ids, success=True,
        )

    async def test_triggers_extraction_when_should(self, manager, mock_extractor, mock_store):
        """当 should_extract 返回 True 时，应触发异步提取。"""
        mock_extractor.should_extract = AsyncMock(return_value=True)
        mock_extractor.extract_from_recent = AsyncMock(return_value=[])

        await manager.on_task_complete(
            task_id="task-1", success=True,
            repair_rounds=1, validation_passed=False,
        )

        mock_extractor.should_extract.assert_called_once()
        # extraction_from_recent should be called (async)
        import asyncio
        await asyncio.sleep(0.01)  # Give async task time to start
        mock_extractor.extract_from_recent.assert_called_once()

    async def test_does_not_trigger_extraction_when_not_should(self, manager, mock_extractor):
        """当 should_extract 返回 False 时，不应触发提取。"""
        mock_extractor.should_extract = AsyncMock(return_value=False)

        await manager.on_task_complete(
            task_id="task-1", success=True,
            repair_rounds=0, validation_passed=True,
        )

        mock_extractor.extract_from_recent.assert_not_called()

    async def test_complete_recorder_exception_does_not_propagate(self, manager, mock_recorder):
        """recorder.complete_task 抛异常时不应传播。"""
        mock_recorder.complete_task.side_effect = RuntimeError("complete error")
        # Should not raise
        await manager.on_task_complete(
            task_id="task-1", success=True,
            repair_rounds=0, validation_passed=True,
        )

    async def test_increments_task_count(self, manager):
        """每次 on_task_complete 后 task_count 应递增。"""
        assert manager._task_count == 0
        await manager.on_task_complete(
            task_id="task-1", success=True,
            repair_rounds=0, validation_passed=True,
        )
        assert manager._task_count == 1
        await manager.on_task_complete(
            task_id="task-2", success=False,
            repair_rounds=1, validation_passed=False,
        )
        assert manager._task_count == 2

    async def test_repair_triggers_extraction(self, manager, mock_extractor):
        """修复过的任务应触发策略提炼。"""
        mock_extractor.should_extract = AsyncMock(return_value=True)
        mock_extractor.extract_from_recent = AsyncMock(return_value=[])

        await manager.on_task_complete(
            task_id="task-1", success=True,
            repair_rounds=2, validation_passed=True,
        )

        import asyncio
        await asyncio.sleep(0.01)
        mock_extractor.extract_from_recent.assert_called_once()

    async def test_saves_extracted_strategies(self, manager, mock_extractor, mock_store):
        """提炼出的策略应保存到 store。"""
        mock_strategy = MagicMock(strategy_id="new-strategy")
        mock_extractor.should_extract = AsyncMock(return_value=True)
        mock_extractor.extract_from_recent = AsyncMock(return_value=[mock_strategy])

        await manager.on_task_complete(
            task_id="task-1", success=True,
            repair_rounds=1, validation_passed=True,
        )

        import asyncio
        await asyncio.sleep(0.01)
        mock_store.save.assert_called_once_with(mock_strategy)


class TestRunMaintenance:
    """run_maintenance 方法测试。"""

    def test_delegates_to_store(self, manager, mock_store):
        """应委托 store.apply_decay。"""
        result = manager.run_maintenance()
        mock_store.apply_decay.assert_called_once()
        assert result == {"decayed": 0, "archived": 0}

    def test_store_exception_does_not_propagate(self, manager, mock_store):
        """store.apply_decay 抛异常时不应传播。"""
        mock_store.apply_decay.side_effect = RuntimeError("decay error")
        result = manager.run_maintenance()
        assert "error" in result


class TestGetStats:
    """get_stats 方法测试。"""

    def test_returns_aggregated_stats(self, manager, mock_store):
        """应返回聚合后的统计信息。"""
        stats = manager.get_stats()
        assert stats["task_count"] == 0
        assert stats["total"] == 5
        assert stats["active"] == 4
        assert stats["archived"] == 1
        assert stats["avg_confidence"] == 0.75

    def test_includes_task_count(self, manager):
        """应包含 task_count。"""
        manager._task_count = 3
        stats = manager.get_stats()
        assert stats["task_count"] == 3

    def test_store_exception_does_not_propagate(self, manager, mock_store):
        """store.get_stats 抛异常时不应传播。"""
        mock_store.get_stats.side_effect = RuntimeError("stats error")
        stats = manager.get_stats()
        assert stats["task_count"] == 0
        # Other fields should be absent or handled
        assert "error" not in stats
