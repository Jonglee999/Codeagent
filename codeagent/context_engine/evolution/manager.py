"""SelfEvolutionManager — 自进化系统统一 Facade。

整合 TrajectoryRecorder、StrategyExtractor、StrategyStore、StrategyApplier
四个组件，提供统一对外接口，管理触发时机和异常降级。

职责：
- 任务开始时：启动轨迹记录 + 检索相关策略
- 任务执行中：记录每一步轨迹
- 任务完成时：异步触发策略提炼（不阻塞）
- 定时维护：置信度衰减
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

from codeagent.context_engine.evolution.strategy_applier import StrategyApplier
from codeagent.context_engine.evolution.strategy_extractor import StrategyExtractor
from codeagent.context_engine.evolution.strategy_store import StrategyStore
from codeagent.context_engine.evolution.trajectory_recorder import (
    TrajectoryRecorder,
    TrajectoryStep,
)

logger = logging.getLogger(__name__)


class SelfEvolutionManager:
    """自进化系统统一 Facade——整合 Recorder / Extractor / Store / Applier。

    职责：
    - 任务开始时：启动轨迹记录 + 检索相关策略
    - 任务执行中：记录每一步轨迹
    - 任务完成时：异步触发策略提炼（不阻塞）
    - 定时维护：置信度衰减

    Args:
        recorder: TrajectoryRecorder 实例
        extractor: StrategyExtractor 实例
        store: StrategyStore 实例
        applier: StrategyApplier 实例
        task_count: 累计完成任务数（用于触发判断）
    """

    def __init__(
        self,
        recorder: TrajectoryRecorder,
        extractor: StrategyExtractor,
        store: StrategyStore,
        applier: StrategyApplier,
        task_count: int = 0,
    ) -> None:
        self._recorder = recorder
        self._extractor = extractor
        self._store = store
        self._applier = applier
        self._task_count = task_count

    # ── 公开接口 ──────────────────────────────────────────────

    async def on_task_start(
        self,
        task_id: str,
        user_request: str,
    ) -> str:
        """任务开始时调用。

        1. recorder.start_task(task_id, user_request)
        2. applier.get_relevant_strategies(user_request)
        3. 返回策略 XML 块（供 System Prompt 注入）

        Args:
            task_id: 任务唯一 ID
            user_request: 用户请求描述

        Returns:
            策略 XML 字符串，无策略时返回空字符串
        """
        try:
            self._recorder.start_task(task_id, user_request)
        except Exception as exc:
            logger.warning("SelfEvolutionManager: recorder.start_task failed: %s", exc)

        try:
            strategies = await self._applier.get_relevant_strategies(
                task_description=user_request,
            )
            if not strategies:
                return ""
            return self._applier.format_for_prompt(strategies)
        except Exception as exc:
            logger.warning(
                "SelfEvolutionManager: failed to get strategies at task start: %s",
                exc,
            )
            return ""

    def record_step(self, task_id: str, step: TrajectoryStep) -> None:
        """记录执行步骤。

        委托 recorder.record_step()。
        异常时只记录日志，不抛异常。

        Args:
            task_id: 任务唯一 ID
            step: 轨迹步骤
        """
        try:
            self._recorder.record_step(task_id, step)
        except Exception as exc:
            logger.warning(
                "SelfEvolutionManager: recorder.record_step failed: %s", exc,
            )

    async def on_task_complete(
        self,
        task_id: str,
        success: bool,
        repair_rounds: int = 0,
        validation_passed: bool = False,
        applied_strategy_ids: Optional[list[str]] = None,
    ) -> None:
        """任务完成时调用。

        1. recorder.complete_task()
        2. 如果有应用的策略：applier.record_outcome()
        3. 异步触发策略提炼（不阻塞）

        Args:
            task_id: 任务唯一 ID
            success: 任务是否成功
            repair_rounds: 修复轮数
            validation_passed: 验证是否通过
            applied_strategy_ids: 本次应用的策略 ID 列表
        """
        # Step 1: 完成轨迹记录
        status = "success" if success else "failure"
        try:
            self._recorder.complete_task(
                task_id=task_id,
                status=status,  # type: ignore[arg-type]
                repair_rounds=repair_rounds,
                validation_passed=validation_passed,
            )
        except Exception as exc:
            logger.warning(
                "SelfEvolutionManager: recorder.complete_task failed: %s", exc,
            )

        # Step 2: 更新策略置信度（如果有应用的策略）
        if applied_strategy_ids:
            try:
                await self._applier.record_outcome(
                    strategy_ids=applied_strategy_ids,
                    success=success,
                )
            except Exception as exc:
                logger.warning(
                    "SelfEvolutionManager: applier.record_outcome failed: %s",
                    exc,
                )

        # Step 3: 检查是否应触发策略提炼
        self._task_count += 1
        try:
            should_extract = await self._extractor.should_extract(
                task_count=self._task_count,
                last_had_repair=repair_rounds > 0,
            )
        except Exception as exc:
            logger.warning(
                "SelfEvolutionManager: extractor.should_extract failed: %s", exc,
            )
            return

        if should_extract:
            logger.info(
                "SelfEvolutionManager: triggering async strategy extraction "
                "(task_count=%d, repairs=%d)",
                self._task_count, repair_rounds,
            )
            asyncio.create_task(self._do_extract())

    def run_maintenance(self) -> dict:
        """执行定时维护：置信度衰减。

        Returns:
            维护报告 {"decayed": int, "archived": int}
        """
        try:
            decayed = self._store.apply_decay()
            return {"decayed": decayed, "archived": 0}
        except Exception as exc:
            logger.warning(
                "SelfEvolutionManager: store.apply_decay failed: %s", exc,
            )
            return {"decayed": 0, "archived": 0, "error": str(exc)}

    def get_stats(self) -> dict:
        """返回自进化系统统计信息。

        Returns:
            包含 store 统计和 task_count 的字典
        """
        stats: dict[str, Any] = {"task_count": self._task_count}
        try:
            store_stats = self._store.get_stats()
            stats.update(store_stats)
        except Exception as exc:
            logger.warning(
                "SelfEvolutionManager: store.get_stats failed: %s", exc,
            )
        return stats

    # ── 内部方法 ──────────────────────────────────────────────

    async def _do_extract(self) -> None:
        """后台执行策略提取。

        被 on_task_complete 通过 asyncio.create_task 调用，
        不阻塞主流程返回。
        异常时只记录日志。
        """
        try:
            strategies = await self._extractor.extract_from_recent()
            if not strategies:
                logger.debug("SelfEvolutionManager: no new strategies extracted")
                return

            saved_count = 0
            for strategy in strategies:
                try:
                    self._store.save(strategy)
                    saved_count += 1
                except Exception as exc:
                    logger.warning(
                        "SelfEvolutionManager: failed to save strategy %s: %s",
                        getattr(strategy, "strategy_id", "unknown"), exc,
                    )

            logger.info(
                "SelfEvolutionManager: extracted and saved %d/%d strategies",
                saved_count, len(strategies),
            )
        except Exception as exc:
            logger.warning(
                "SelfEvolutionManager: async extraction failed: %s", exc,
            )
