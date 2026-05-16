"""自进化系统集成测试 — 真实文件系统环境。

Scene A：完整管道（轨迹记录 → 策略提炼 → 策略存储 → 策略检索）
Scene B：置信度衰减和归档
Scene C：异常降级

所有测试使用 pytest tmp_path 创建真实文件系统，测试结束后自动清理。
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from codeagent.context_engine.evolution.manager import SelfEvolutionManager
from codeagent.context_engine.evolution.strategy_applier import StrategyApplier
from codeagent.context_engine.evolution.strategy_extractor import (
    Strategy,
    StrategyExtractor,
)
from codeagent.context_engine.evolution.strategy_store import StrategyStore
from codeagent.context_engine.evolution.trajectory_recorder import (
    Trajectory,
    TrajectoryRecorder,
    TrajectoryStep,
)


# ── Mock LLM 工具函数 ────────────────────────────────────────────────────


def make_mock_llm(response_text: str | None = None):
    """构建模拟 LLM 客户端，返回固定 JSON 响应。"""
    async def mock_llm(**kwargs: Any) -> Any:
        from types import SimpleNamespace
        msg = SimpleNamespace(content=response_text)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])
    return mock_llm


# ── 辅助函数 ─────────────────────────────────────────────────────────────


def make_strategy(
    strategy_id: str = "s1",
    title: str = "Add tests when modifying routes",
    condition: str = "when modifying API routes",
    action: str = "add corresponding test files",
    rationale: str = "prevents CI failures",
    category: str = "workflow",
    confidence: float = 0.85,
) -> Strategy:
    return Strategy(
        strategy_id=strategy_id,
        title=title,
        condition=condition,
        action=action,
        rationale=rationale,
        category=category,  # type: ignore[arg-type]
        source_task_ids=["t1", "t2"],
        confidence=confidence,
        created_at=datetime.now(),
    )


# ══════════════════════════════════════════════════════════════════════════
# Scene A：完整管道
# ══════════════════════════════════════════════════════════════════════════


class TestSceneA_FullPipeline:
    """Scene A：轨迹记录 → 策略提炼 → 策略存储 → 策略检索 完整管道。"""

    def test_trajectory_recorder_to_strategy_extractor(self, tmp_path: Path) -> None:
        """验证轨迹可被成功记录、加载并用于策略提炼。"""
        base_path = str(tmp_path / "trajectories")
        recorder = TrajectoryRecorder(base_path=base_path)

        # 记录 2 条轨迹
        recorder.start_task("task-1", "Add a new API route")
        recorder.record_step("task-1", TrajectoryStep(
            step_id="s1", node_name="planning",
            step_type="llm_call",
            timestamp=datetime.now(),
            input_summary="planning input",
            output_summary="planning output",
            success=True,
        ))
        recorder.complete_task("task-1", status="success", repair_rounds=0, validation_passed=True)

        recorder.start_task("task-2", "Fix failing test")
        recorder.record_step("task-2", TrajectoryStep(
            step_id="s1", node_name="execution",
            step_type="tool_call",
            timestamp=datetime.now(),
            input_summary="execution input",
            output_summary="execution output",
            success=True,
        ))
        recorder.complete_task("task-2", status="success", repair_rounds=1, validation_passed=True)

        # 验证轨迹已持久化
        trajectories = recorder.load_recent(n=10)
        assert len(trajectories) == 2
        task_ids = {t.task_id for t in trajectories}
        assert "task-1" in task_ids
        assert "task-2" in task_ids

    def test_strategy_extractor_to_store(self, tmp_path: Path) -> None:
        """验证策略提炼后可被保存到 StrategyStore。"""
        store = StrategyStore(base_path=str(tmp_path / "strategies"))
        mock_llm = make_mock_llm(json.dumps({
            "strategies": [
                {
                    "title": "Fix imports when moving files",
                    "condition": "when moving Python files between directories",
                    "action": "update all import statements referencing the moved file",
                    "rationale": "forgotten imports cause 80% of CI failures after refactoring",
                    "category": "workflow",
                    "confidence": 0.8,
                },
            ],
        }))
        recorder = TrajectoryRecorder(base_path=str(tmp_path / "trajectories"))
        extractor = StrategyExtractor(
            llm_client=mock_llm,
            recorder=recorder,
            store=store,
            min_confidence=0.0,
        )

        # 准备模拟轨迹
        trajectories = [
            Trajectory(
                task_id="t1",
                user_request="Move utils.py to lib/",
                started_at=datetime.now(),
                steps=[
                    TrajectoryStep(
                        step_id="s1", node_name="execution",
                        step_type="tool_call",
                        timestamp=datetime.now(),
                        input_summary="move file",
                        output_summary="moved successfully",
                        success=True,
                    ),
                ],
                final_status="success",
            ),
        ]

        # 提炼策略
        strategies = extractor.extract.__wrapped__(extractor, trajectories) if hasattr(extractor.extract, "__wrapped__") else None
        # direct call
        import asyncio
        strategies = asyncio.run(extractor.extract(trajectories))
        assert len(strategies) > 0

        # 保存到 Store
        strategy = strategies[0]
        saved_id = store.save(strategy)
        assert saved_id == strategy.strategy_id

        # 验证可从 Store 检索到
        loaded = store.get(strategy.strategy_id)
        assert loaded is not None
        assert loaded.title == strategy.title
        assert loaded.condition == strategy.condition

    def test_store_to_applier(self, tmp_path: Path) -> None:
        """验证存储的策略可通过 StrategyApplier 检索。"""
        store = StrategyStore(base_path=str(tmp_path / "strategies"))

        # 保存策略
        strategy = make_strategy(
            strategy_id="route-test-rule",
            title="Add tests with routes",
            condition="when adding new API routes",
            action="add corresponding test files covering normal and error paths",
            rationale="prevents CI failures",
        )
        store.save(strategy)

        # 通过 Applier 检索
        applier = StrategyApplier(store=store, confidence_threshold=0.0)
        import asyncio
        results = asyncio.run(applier.get_relevant_strategies(
            task_description="Add a new GET endpoint to the API",
        ))

        assert len(results) >= 1
        assert results[0].strategy_id == "route-test-rule"

    def test_full_pipeline_end_to_end(self, tmp_path: Path) -> None:
        """完整管道：记录轨迹 → 提炼策略 → 存储 → 检索 → 格式化。"""
        traj_base = str(tmp_path / "trajectories")
        strat_base = str(tmp_path / "strategies")

        recorder = TrajectoryRecorder(base_path=traj_base)
        store = StrategyStore(base_path=strat_base)

        # Step 1: 记录 3 条轨迹
        for i in range(3):
            tid = f"task-{i}"
            recorder.start_task(tid, f"Task {i} description")
            recorder.record_step(tid, TrajectoryStep(
                step_id=f"s1-{i}", node_name="execution",
                step_type="tool_call",
                timestamp=datetime.now(),
                input_summary=f"input {i}",
                output_summary=f"output {i}",
                success=True,
            ))
            recorder.complete_task(tid, status="success")

        # Step 2: 提炼策略
        mock_llm = make_mock_llm(json.dumps({
            "strategies": [
                {
                    "title": "Always run lint before commit",
                    "condition": "when finishing code changes",
                    "action": "run lint to catch style issues before finalizing",
                    "rationale": "lint catches 60% of common mistakes",
                    "category": "workflow",
                    "confidence": 0.75,
                },
            ],
        }))
        extractor = StrategyExtractor(
            llm_client=mock_llm,
            recorder=recorder,
            store=store,
            min_confidence=0.0,
        )
        import asyncio
        trajectories = recorder.load_recent(n=10)
        strategies = asyncio.run(extractor.extract(trajectories))
        assert len(strategies) >= 1

        # Step 3: 存储策略
        strategy = strategies[0]
        store.save(strategy)

        # Step 4: 检索和格式化
        applier = StrategyApplier(store=store, confidence_threshold=0.0)
        results = asyncio.run(applier.get_relevant_strategies(
            task_description="Finish code changes and commit",
        ))
        assert len(results) >= 1
        xml = applier.format_for_prompt(results)
        assert "<strategies>" in xml
        assert "</strategies>" in xml
        assert "run lint to catch style issues" in xml


# ══════════════════════════════════════════════════════════════════════════
# Scene B：置信度衰减
# ══════════════════════════════════════════════════════════════════════════


class TestSceneB_ConfidenceDecay:
    """Scene B：置信度衰减和归档。"""

    def test_decay_after_30_days(self, tmp_path: Path) -> None:
        """超过 30 天未应用的策略置信度应衰减 0.1。"""
        store = StrategyStore(base_path=str(tmp_path / "strategies"))

        old_date = datetime.now() - timedelta(days=40)
        strategy = make_strategy(
            strategy_id="decay-test",
            confidence=0.8,
        )
        strategy.last_applied = old_date
        store.save(strategy)

        affected = store.apply_decay()
        assert affected >= 1

        loaded = store.get("decay-test")
        assert loaded is not None
        assert loaded.confidence == 0.7  # 0.8 - 0.1

    def test_archive_after_90_days(self, tmp_path: Path) -> None:
        """超过 90 天未应用的策略应直接归档。"""
        store = StrategyStore(base_path=str(tmp_path / "strategies"))

        old_date = datetime.now() - timedelta(days=100)
        strategy = make_strategy(
            strategy_id="archive-test",
            confidence=0.9,
        )
        strategy.last_applied = old_date
        store.save(strategy)

        affected = store.apply_decay()
        assert affected >= 1

        # 不再在活跃列表中
        active = store.list_active()
        ids = [s.strategy_id for s in active]
        assert "archive-test" not in ids

    def test_archive_when_below_threshold(self, tmp_path: Path) -> None:
        """置信度低于 0.2 的策略应自动归档。"""
        store = StrategyStore(base_path=str(tmp_path / "strategies"))

        strategy = make_strategy(
            strategy_id="low-conf-strategy",
            confidence=0.15,
        )
        store.save(strategy)

        # 更新置信度为失败 → 应触发归档（已在 update_confidence 中处理）
        store.update_confidence("low-conf-strategy", success=False)

        active = store.list_active()
        assert "low-conf-strategy" not in [s.strategy_id for s in active]

    def test_confidence_update_success(self, tmp_path: Path) -> None:
        """成功应用后置信度应 +0.1。"""
        store = StrategyStore(base_path=str(tmp_path / "strategies"))

        strategy = make_strategy(strategy_id="conf-up", confidence=0.5)
        store.save(strategy)

        store.update_confidence("conf-up", success=True)
        loaded = store.get("conf-up")
        assert loaded is not None
        assert loaded.confidence == 0.6

    def test_confidence_update_failure(self, tmp_path: Path) -> None:
        """失败应用后置信度应 -0.15。"""
        store = StrategyStore(base_path=str(tmp_path / "strategies"))

        strategy = make_strategy(strategy_id="conf-down", confidence=0.7)
        store.save(strategy)

        store.update_confidence("conf-down", success=False)
        loaded = store.get("conf-down")
        assert loaded is not None
        assert loaded.confidence == 0.55  # 0.7 - 0.15

    def test_applied_count_tracking(self, tmp_path: Path) -> None:
        """应用次数应正确递增。"""
        store = StrategyStore(base_path=str(tmp_path / "strategies"))

        strategy = make_strategy(strategy_id="count-test")
        store.save(strategy)

        store.increment_applied("count-test", success=True)
        store.increment_applied("count-test", success=True)
        store.increment_applied("count-test", success=False)

        loaded = store.get("count-test")
        assert loaded is not None
        assert loaded.applied_count == 3
        assert loaded.success_count == 2


# ══════════════════════════════════════════════════════════════════════════
# Scene C：异常降级
# ══════════════════════════════════════════════════════════════════════════


class TestSceneC_ExceptionDegradation:
    """Scene C：异常降级和错误隔离。"""

    def test_recorder_creates_directory_automatically(self, tmp_path: Path) -> None:
        """TrajectoryRecorder 的存储路径不存在时应自动创建。"""
        base_path = tmp_path / "nonexistent" / "deep" / "trajectories"
        assert not base_path.exists()

        recorder = TrajectoryRecorder(base_path=str(base_path))
        recorder.start_task("auto-create-task", "test")
        recorder.complete_task("auto-create-task", status="success")

        assert base_path.exists()
        assert base_path.is_dir()

    def test_extractor_llm_invalid_json_returns_empty(self, tmp_path: Path) -> None:
        """StrategyExtractor 收到非法 JSON 时应返回空列表。"""
        store = StrategyStore(base_path=str(tmp_path / "strategies"))

        # LLM 返回非法 JSON
        mock_llm = make_mock_llm("not valid json at all {{{")
        recorder = TrajectoryRecorder(base_path=str(tmp_path / "trajectories"))
        extractor = StrategyExtractor(
            llm_client=mock_llm,
            recorder=recorder,
            store=store,
        )

        traj = Trajectory(
            task_id="t1",
            user_request="test",
            started_at=datetime.now(),
            final_status="success",
        )

        import asyncio
        strategies = asyncio.run(extractor.extract([traj]))
        assert strategies == []

    def test_extractor_empty_trajectories_returns_empty(self, tmp_path: Path) -> None:
        """空的轨迹列表应返回空策略列表。"""
        store = StrategyStore(base_path=str(tmp_path / "strategies"))
        recorder = TrajectoryRecorder(base_path=str(tmp_path / "trajectories"))
        extractor = StrategyExtractor(
            llm_client=make_mock_llm("{}"),
            recorder=recorder,
            store=store,
        )

        import asyncio
        strategies = asyncio.run(extractor.extract([]))
        assert strategies == []

    def test_manager_submodule_exception_isolation(self, tmp_path: Path) -> None:
        """SelfEvolutionManager 的子模块异常不应传播。"""
        store = StrategyStore(base_path=str(tmp_path / "strategies"))
        recorder = TrajectoryRecorder(base_path=str(tmp_path / "trajectories"))
        applier = StrategyApplier(store=store)
        extractor = StrategyExtractor(
            llm_client=make_mock_llm("{}"),
            recorder=recorder,
            store=store,
        )

        manager = SelfEvolutionManager(
            recorder=recorder,
            extractor=extractor,
            store=store,
            applier=applier,
        )

        # 即使 recorder 的 base_path 不可写，on_task_start 也不应抛异常
        import asyncio
        result = asyncio.run(manager.on_task_start(
            task_id="test-task",
            user_request="test",
        ))
        # 应返回字符串（可能为空）
        assert isinstance(result, str)

    def test_applier_store_search_failure_returns_empty(self, tmp_path: Path) -> None:
        """StrategyStore search 失败时 StrategyApplier 应返回空列表。"""
        store = StrategyStore(base_path=str(tmp_path / "strategies"))
        applier = StrategyApplier(store=store)

        # 破坏 store 使其 search 失败
        store._base_path = Path("/nonexistent/path/that/does/not/exist")

        import asyncio
        results = asyncio.run(applier.get_relevant_strategies("test query"))
        assert results == []

    def test_manager_record_step_exception_isolation(self, tmp_path: Path) -> None:
        """record_step 抛异常时不应传播到调用方。"""
        store = StrategyStore(base_path=str(tmp_path / "strategies"))
        recorder = TrajectoryRecorder(base_path=str(tmp_path / "trajectories"))
        applier = StrategyApplier(store=store)
        extractor = StrategyExtractor(
            llm_client=make_mock_llm("{}"),
            recorder=recorder,
            store=store,
        )
        manager = SelfEvolutionManager(
            recorder=recorder,
            extractor=extractor,
            store=store,
            applier=applier,
        )

        # 记录步骤时 task_id 不存在（recorder 内部处理），不应抛异常
        step = TrajectoryStep(
            step_id="bad", node_name="execution",
            step_type="tool_call",
            timestamp=datetime.now(),
            input_summary="in", output_summary="out",
        )
        # Should not raise
        manager.record_step("nonexistent-task", step)

    def test_self_evolution_manager_stats_with_data(self, tmp_path: Path) -> None:
        """SelfEvolutionManager.get_stats 应正确聚合数据。"""
        store = StrategyStore(base_path=str(tmp_path / "strategies"))
        recorder = TrajectoryRecorder(base_path=str(tmp_path / "trajectories"))
        applier = StrategyApplier(store=store)
        extractor = StrategyExtractor(
            llm_client=make_mock_llm("{}"),
            recorder=recorder,
            store=store,
        )

        # 保存一条策略
        strategy = make_strategy(strategy_id="stats-test")
        store.save(strategy)

        manager = SelfEvolutionManager(
            recorder=recorder,
            extractor=extractor,
            store=store,
            applier=applier,
            task_count=5,
        )

        stats = manager.get_stats()
        assert stats["task_count"] == 5
        assert stats["total"] >= 1
        assert stats["active"] >= 1
