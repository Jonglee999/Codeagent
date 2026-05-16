"""E2E 测试：自进化系统验证（Phase 7.7）。

Scene M — 策略积累：
  执行 5 次类似"修复 Python bug"的任务，验证轨迹可被记录，
  策略可从轨迹中提炼，且提炼出的策略可被后续任务检索。

Scene N — 策略持久化：
  验证策略在会话间持久化，第二次会话能检索到第一次会话的策略。

前置条件：LLM_API_KEY 环境变量已设置。
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from codeagent.context_engine.evolution.manager import SelfEvolutionManager
from codeagent.context_engine.evolution.strategy_applier import StrategyApplier
from codeagent.context_engine.evolution.strategy_extractor import (
    StrategyExtractor,
)
from codeagent.context_engine.evolution.strategy_store import StrategyStore
from codeagent.context_engine.evolution.trajectory_recorder import (
    TrajectoryRecorder,
    TrajectoryStep,
)
from codeagent.gateway.validation_gateway import (
    IValidationGateway,
    ValidationError,
    ValidationResult,
)
from codeagent.orchestration.nodes.execution_node import ExecutionNode
from codeagent.orchestration.state import AgentState
from codeagent.tools.gateway import ToolGateway
from codeagent.tools.registry import ToolRegistry

# ── API Key 检测 ──────────────────────────────────────────────────────────

_HAS_API_KEY = bool(os.environ.get("LLM_API_KEY"))

requires_api_key = pytest.mark.skipif(
    not _HAS_API_KEY,
    reason="LLM_API_KEY environment variable not set — E2E test requires real LLM call",
)


# ── 辅助函数 ──────────────────────────────────────────────────────────────


def _build_tool_gateway(project_root: str) -> ToolGateway:
    """构建包含 ReadFileTool 和 WriteFileTool 的工具 Gateway。"""
    from codeagent.tools.file.read_file import ReadFileTool
    from codeagent.tools.file.write_file import WriteFileTool

    registry = ToolRegistry()
    registry.register(ReadFileTool(project_root=project_root))
    registry.register(WriteFileTool(project_root=project_root))
    return ToolGateway(registry)


def _build_validation_gateway() -> IValidationGateway:
    """构建验证 Gateway（支持 syntax_check）。"""
    from codeagent.validation.syntax_validator import SyntaxValidator

    class _ValidationGateway(IValidationGateway):
        def __init__(self) -> None:
            self._validator = SyntaxValidator()

        async def run_syntax_check(self, file_path: str) -> ValidationResult:
            return await self._validator.check_file(file_path)

        async def run_lint(self, files: list[str]) -> ValidationResult:
            return ValidationResult(passed=True)

        async def run_tests(self, project_root: str) -> ValidationResult:
            return ValidationResult(passed=True)

        async def run_runtime_check(self, file_path: str) -> ValidationResult:
            return ValidationResult(passed=True)

    return _ValidationGateway()


def _build_llm(model_name: str | None = None) -> Any:
    """构建 LLM 调用函数（litellm），含网络重试。"""
    import litellm

    api_key = os.environ.get("LLM_API_KEY", "")
    api_base = os.environ.get("LLM_API_BASE", "")
    timeout = int(os.environ.get("LLM_TIMEOUT", "120"))
    resolved_model = model_name or os.environ.get("LLM_MODEL", "openai/deepseek-v4-flash")

    async def llm_call(**kwargs: object) -> object:
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                call_kwargs: dict[str, object] = {
                    **{k: v for k, v in kwargs.items() if v is not None},
                    "timeout": timeout,
                }
                if api_key:
                    call_kwargs["api_key"] = api_key
                if api_base:
                    call_kwargs["api_base"] = api_base
                return await litellm.acompletion(**call_kwargs)
            except Exception as e:
                last_exc = e
                if attempt < 2:
                    await asyncio.sleep(1.5 * (attempt + 1))
        raise last_exc  # type: ignore[misc]

    return llm_call


# 有 bug 的 Python 函数列表（混合语法错误和逻辑错误）
BUG_TYPES: list[tuple[str, str, str]] = [
    ("语法错误", "missing_colon", "def add(a, b)\n    return a + b"),
    ("语法错误", "missing_colon_if", "def check(x):\n    if x > 0\n        return True\n    return False"),
    ("语法错误", "missing_colon_for", "def process(items):\n    for item in items\n        print(item)"),
    ("逻辑错误", "subtract_order", "def subtract(a, b):\n    return b - a"),
    ("逻辑错误", "xor_vs_power", "def power(a, b):\n    return a ^ b"),
]


# ══════════════════════════════════════════════════════════════════════════
# Scene M: 策略积累
# ══════════════════════════════════════════════════════════════════════════


@pytest.mark.e2e
@requires_api_key
class TestSceneM_StrategyAccumulation:
    """Scene M: 策略积累——执行多次 bug 修复任务，验证自进化管道完整运行。

    注意：此测试使用 ExecutionNode 直接调用 LLM 修复 bug，
    验证轨迹记录 → 策略提炼 → 策略存储的完整管道可用。
    完整的"修复轮数递减"验证需要集成 PlanningNode + 策略注入的全流程。
    """

    @pytest.mark.asyncio
    async def test_strategy_accumulation(self, tmp_path: Path) -> None:
        """执行 5 次 bug 修复任务，验证自进化管道。"""
        traj_base = str(tmp_path / ".codeagent" / "trajectories")
        strat_base = str(tmp_path / ".codeagent" / "strategies")

        recorder = TrajectoryRecorder(base_path=traj_base)
        store = StrategyStore(base_path=strat_base)
        llm = _build_llm()
        extractor = StrategyExtractor(
            llm_client=llm,
            recorder=recorder,
            store=store,
            min_confidence=0.0,
        )
        applier = StrategyApplier(store=store, confidence_threshold=0.0)
        evolution = SelfEvolutionManager(
            recorder=recorder,
            extractor=extractor,
            store=store,
            applier=applier,
            task_count=0,
        )

        workspace = tmp_path / "workspace"
        workspace.mkdir(parents=True, exist_ok=True)

        tool_gateway = _build_tool_gateway(str(workspace))
        validation_gateway = _build_validation_gateway()
        execution_node = ExecutionNode(
            llm=llm,
            tool_gateway=tool_gateway,
            validation_gateway=validation_gateway,
            max_retries=2,
        )

        task_results: list[dict[str, Any]] = []

        for i, (bug_type, bug_name, code) in enumerate(BUG_TYPES):
            task_id = f"acc-{i}"
            bug_file = workspace / f"{bug_name}.py"
            bug_file.write_text(code)

            # ── 1. 启动演化跟踪 ──
            await evolution.on_task_start(
                task_id=task_id,
                user_request=f"修复 {bug_name}.py 中的{bug_type}",
            )

            # ── 2. 执行 bug 修复 ──
            state = AgentState(
                user_request=f"修复 {bug_name}.py 中的{bug_type}",
                project_root=str(workspace),
                task_id=task_id,
                evolution_enabled=True,
            )

            result = await execution_node(state)
            errors = result.get("errors", [])
            tool_call_count = len(result.get("trajectory_steps", []))

            # ── 3. 记录轨迹步骤 ──
            evolution.record_step(task_id, TrajectoryStep(
                step_id=f"exec-{i}",
                node_name="execution",
                step_type="tool_call",
                timestamp=datetime.now(),
                input_summary=f"Fix {bug_type} in {bug_name}.py",
                output_summary=f"Tool calls: {tool_call_count}, errors: {len(errors)}",
                success=len(errors) == 0,
            ))

            # ── 4. 完成任务 ──
            await evolution.on_task_complete(
                task_id=task_id,
                success=len(errors) == 0,
                repair_rounds=0,
                validation_passed=True,
            )

            # 等待异步策略提炼完成
            await asyncio.sleep(2)

            # ── 5. 手动触发策略提炼（确保策略可用） ──
            try:
                trajectories = recorder.load_recent(n=10)
                strategies = await extractor.extract(trajectories)
                for s in strategies:
                    store.save(s)
            except Exception:
                pass

            task_results.append({
                "task_id": task_id,
                "bug_type": bug_type,
                "success": len(errors) == 0,
                "tool_call_count": tool_call_count,
                "errors": errors,
            })

        # ══ 断言 ═══════════════════════════════════════════════

        # 1. 轨迹已持久化
        loaded_trajs = recorder.load_recent(n=10)
        assert len(loaded_trajs) >= 3, (
            f"Should have persisted at least 3 trajectories, got {len(loaded_trajs)}"
        )

        # 2. 至少提炼出 1 条策略
        active_strategies = store.list_active()
        assert len(active_strategies) >= 1, (
            "Should have extracted at least 1 strategy from 5 bug-fix tasks"
        )

        # 3. 策略有合理的结构
        strategy = active_strategies[0]
        assert strategy.title, "Strategy title should not be empty"
        assert strategy.condition, "Strategy condition should not be empty"
        assert strategy.action, "Strategy action should not be empty"

        # 4. StrategyApplier 能检索到策略
        relevant = await applier.get_relevant_strategies(
            task_description="修复 Python 函数中的语法错误",
        )
        assert len(relevant) >= 1, "Applier should find relevant strategies"

        # 5. 统计信息完整
        stats = evolution.get_stats()
        assert stats["task_count"] == 5, f"Expected task_count=5, got {stats['task_count']}"
        assert stats.get("total", 0) >= 1

        # 6. 报告每次任务的结果
        success_count = sum(1 for r in task_results if r["success"])
        assert success_count >= 3, (
            f"At least 3 of 5 tasks should succeed, got {success_count}/5. "
            f"Results: {[(r['task_id'], r['bug_type'], r['success']) for r in task_results]}"
        )


# ══════════════════════════════════════════════════════════════════════════
# Scene N: 策略持久化
# ══════════════════════════════════════════════════════════════════════════


@pytest.mark.e2e
@requires_api_key
class TestSceneN_StrategyPersistence:
    """Scene N: 策略持久化——策略在会话间持续存在。"""

    @pytest.mark.asyncio
    async def test_strategy_persistence(self, tmp_path: Path) -> None:
        """验证：第一次会话执行 5 次任务 → 策略被提炼并存储；
        第二次会话构建新组件 → 能检索到第一次会话存储的策略。"""
        traj_base = str(tmp_path / ".codeagent" / "trajectories")
        strat_base = str(tmp_path / ".codeagent" / "strategies")

        # ── Session 1: 执行 5 次任务 ─────────────────────────────
        recorder1 = TrajectoryRecorder(base_path=traj_base)
        store1 = StrategyStore(base_path=strat_base)
        llm1 = _build_llm()
        extractor1 = StrategyExtractor(
            llm_client=llm1,
            recorder=recorder1,
            store=store1,
            min_confidence=0.0,
        )
        applier1 = StrategyApplier(store=store1, confidence_threshold=0.0)
        evolution1 = SelfEvolutionManager(
            recorder=recorder1,
            extractor=extractor1,
            store=store1,
            applier=applier1,
            task_count=0,
        )

        workspace = tmp_path / "workspace"
        workspace.mkdir(parents=True, exist_ok=True)

        tool_gateway = _build_tool_gateway(str(workspace))
        validation_gateway = _build_validation_gateway()
        execution_node = ExecutionNode(
            llm=llm1,
            tool_gateway=tool_gateway,
            validation_gateway=validation_gateway,
            max_retries=2,
        )

        for i, (bug_type, bug_name, code) in enumerate(BUG_TYPES):
            task_id = f"persist-{i}"
            bug_file = workspace / f"{bug_name}.py"
            bug_file.write_text(code)

            await evolution1.on_task_start(
                task_id=task_id,
                user_request=f"修复 {bug_name}.py 中的{bug_type}",
            )

            state = AgentState(
                user_request=f"修复 {bug_name}.py 中的{bug_type}",
                project_root=str(workspace),
                task_id=task_id,
                evolution_enabled=True,
            )

            result = await execution_node(state)

            evolution1.record_step(task_id, TrajectoryStep(
                step_id=f"exec-{i}",
                node_name="execution",
                step_type="tool_call",
                timestamp=datetime.now(),
                input_summary=f"Fix {bug_type}",
                output_summary=f"Errors: {len(result.get('errors', []))}",
                success=len(result.get("errors", [])) == 0,
            ))

            await evolution1.on_task_complete(
                task_id=task_id,
                success=len(result.get("errors", [])) == 0,
                repair_rounds=0,
                validation_passed=True,
            )

            # 等待异步提炼
            await asyncio.sleep(2)

            # 手动触发策略提炼
            try:
                trajectories = recorder1.load_recent(n=10)
                strategies = await extractor1.extract(trajectories)
                for s in strategies:
                    store1.save(s)
            except Exception:
                pass

        # Session 1 验证：策略已存储
        active1 = store1.list_active()
        assert len(active1) >= 1, "Session 1 should have extracted and stored strategies"

        # ── Session 2: 新组件，同一存储路径 ───────────────────────
        store2 = StrategyStore(base_path=strat_base)
        recorder2 = TrajectoryRecorder(base_path=traj_base)
        applier2 = StrategyApplier(store=store2, confidence_threshold=0.0)

        # 验证 1：Store 在第二会话中能检索到策略
        active2 = store2.list_active()
        assert len(active2) >= 1, (
            "Session 2 should retrieve strategies stored by Session 1"
        )

        # 验证 2：策略 ID 一致
        session1_ids = {s.strategy_id for s in active1}
        session2_ids = {s.strategy_id for s in active2}
        assert session1_ids == session2_ids, (
            "Strategy IDs should match across sessions"
        )

        # 验证 3：策略内容一致
        s1 = active1[0]
        s2 = active2[0]
        assert s1.title == s2.title
        assert s1.condition == s2.condition

        # 验证 4：Applier 在第二会话中能检索到策略
        relevant = await applier2.get_relevant_strategies(
            task_description="修复 Python 函数中的语法错误",
        )
        assert len(relevant) >= 1, (
            "Applier in Session 2 should retrieve strategies"
        )

        # 验证 5：TrajectoryRecorder 在第二会话中能加载历史轨迹
        loaded = recorder2.load_recent(n=10)
        assert len(loaded) >= 1, (
            "Session 2 should load trajectories recorded in Session 1"
        )
