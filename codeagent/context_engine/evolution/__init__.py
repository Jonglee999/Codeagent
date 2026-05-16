"""自进化系统模块 — 轨迹记录与策略学习。

提供 TrajectoryRecorder 用于记录任务执行轨迹，
StrategyExtractor 用于从轨迹中提炼可复用策略，
StrategyStore 用于策略持久化存储和管理，
StrategyApplier 用于在任务中注入和应用策略。
"""

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

__all__ = [
    "SelfEvolutionManager",
    "Strategy",
    "StrategyApplier",
    "StrategyExtractor",
    "StrategyStore",
    "Trajectory",
    "TrajectoryRecorder",
    "TrajectoryStep",
]
