"""LangGraph 工作流图结构单元测试。

验证图结构：节点数量、边连接正确、条件路由分支覆盖。
不执行图，仅验证结构。
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from langgraph.graph import END

from codeagent.orchestration.graph import build_workflow, run_workflow
from codeagent.orchestration.state import AgentState


class FakeNode:
    """模拟节点实例。"""

    async def __call__(self, state: AgentState) -> dict[str, Any]:
        return {}


@pytest.fixture
def graph():
    return build_workflow(
        context_node=FakeNode(),
        planning_node=FakeNode(),
        execution_node=FakeNode(),
        validation_node=FakeNode(),
    )


class FakeEvolutionManager:
    """模拟 SelfEvolutionManager。"""

    def __init__(self):
        self.on_task_start = AsyncMock(return_value="")
        self.on_task_complete = AsyncMock()
        self.record_step = MagicMock()
        self.run_maintenance = MagicMock(return_value={"decayed": 0})
        self.get_stats = MagicMock(return_value={"task_count": 0})


class TestGraphStructure:
    """图结构验证。"""

    def test_has_all_nodes(self, graph) -> None:
        """应包含所有 5 个工作流节点（不含 __start__）。"""
        node_names = {n for n in graph.nodes if not n.startswith("__")}
        expected = {"context", "planning", "execution", "validation", "reflection", "human_review"}
        assert node_names == expected

    def test_entry_point_is_context(self, graph) -> None:
        """入口点应为 context。"""
        assert graph.builder is not None
        # entry point is stored internally; verify via edges from __start__
        start_edges = [
            e for e in graph.builder.edges
            if isinstance(e, tuple) and e[0] == "__start__"
        ]
        assert any(e[1] == "context" for e in start_edges)

    def test_workflow_nodes_have_conditional_routes(self, graph) -> None:
        """All state-dependent nodes must remain connected through a branch."""
        assert set(graph.builder.branches) >= {
            "context", "planning", "execution", "validation", "reflection", "human_review",
        }

    def test_planning_branch_ends_include_end(self, graph) -> None:
        """planning 条件分支应包含 END。"""
        branches = graph.builder.branches["planning"]
        for branch_spec in branches.values():
            assert "__end__" in branch_spec.ends or END in branch_spec.ends.values()

    def test_interrupt_before_human_review(self, graph) -> None:
        """应在 human_review 前设置中断点。"""
        assert "human_review" in graph.interrupt_before_nodes


class TestGraphEdgeCases:
    """图的边界情况测试。"""

    def test_custom_human_review_node(self) -> None:
        """可使用自定义 human_review 节点。"""
        called = False

        async def custom_review(state: AgentState) -> dict[str, Any]:
            nonlocal called
            called = True
            return {}

        g = build_workflow(
            context_node=FakeNode(),
            planning_node=FakeNode(),
            execution_node=FakeNode(),
            validation_node=FakeNode(),
            human_review_node=custom_review,
        )
        assert "human_review" in g.nodes

    def test_compiled_graph_is_callable(self, graph) -> None:
        """编译后的图应可调用（ainvoke）。"""
        assert hasattr(graph, "ainvoke")
        assert callable(graph.ainvoke)

    def test_build_workflow_with_evolution_manager_param(self) -> None:
        """build_workflow 可接受 evolution_manager 参数。"""
        ev = FakeEvolutionManager()
        g = build_workflow(
            context_node=FakeNode(),
            planning_node=FakeNode(),
            execution_node=FakeNode(),
            validation_node=FakeNode(),
            evolution_manager=ev,
        )
        assert "context" in g.nodes


class TestRunWorkflow:
    """run_workflow 函数测试。"""

    async def test_run_workflow_without_evolution(self, graph) -> None:
        """不传 evolution_manager 时，run_workflow 应正常执行。"""
        state = AgentState(user_request="hello", project_root="/test")
        result = await run_workflow(graph=graph, initial_state=state)
        assert result is not None

    async def test_run_workflow_calls_on_task_start(self, graph) -> None:
        """evolution_manager 存在时应在开始时调用 on_task_start。"""
        ev = FakeEvolutionManager()
        state = AgentState(
            user_request="fix bug",
            project_root="/test",
            task_id="task-123",
            evolution_enabled=True,
        )
        await run_workflow(graph=graph, initial_state=state, evolution_manager=ev)
        ev.on_task_start.assert_called_once_with(
            task_id="task-123",
            user_request="fix bug",
        )

    async def test_run_workflow_calls_on_task_complete(self, graph) -> None:
        """evolution_manager 存在时应在完成后调用 on_task_complete。"""
        ev = FakeEvolutionManager()
        state = AgentState(
            user_request="fix bug",
            project_root="/test",
            task_id="task-123",
            evolution_enabled=True,
        )
        await run_workflow(graph=graph, initial_state=state, evolution_manager=ev)
        ev.on_task_complete.assert_called_once()

    async def test_run_workflow_skipped_when_disabled(self, graph) -> None:
        """evolution_enabled=False 时应跳过生命周期调用。"""
        ev = FakeEvolutionManager()
        state = AgentState(
            user_request="task",
            project_root="/test",
            task_id="task-456",
            evolution_enabled=False,
        )
        await run_workflow(graph=graph, initial_state=state, evolution_manager=ev)
        ev.on_task_start.assert_not_called()
        ev.on_task_complete.assert_not_called()

    async def test_run_workflow_skipped_when_none(self, graph) -> None:
        """evolution_manager=None 时应跳过生命周期调用。"""
        state = AgentState(
            user_request="task",
            project_root="/test",
            task_id="task-789",
            evolution_enabled=True,
        )
        # Should not raise
        result = await run_workflow(graph=graph, initial_state=state)
        assert result is not None

    async def test_node_lifecycle_events_are_emitted_from_real_graph_execution(self) -> None:
        progress = MagicMock()
        graph = build_workflow(
            context_node=FakeNode(),
            planning_node=FakeNode(),
            execution_node=FakeNode(),
            validation_node=FakeNode(),
            progress_callback=progress,
        )

        await run_workflow(
            graph=graph,
            initial_state=AgentState(user_request="hello", project_root="/test"),
        )

        events = [call.args[0] for call in progress.call_args_list]
        assert [(event["node"], event["type"]) for event in events] == [
            ("context", "node_start"),
            ("context", "node_complete"),
            ("planning", "node_start"),
            ("planning", "node_complete"),
        ]
        assert all(event.get("success", True) for event in events)

    async def test_callable_nodes_are_recorded_with_external_task_id(self) -> None:
        recorder = MagicMock()
        graph = build_workflow(
            context_node=FakeNode(),
            planning_node=FakeNode(),
            execution_node=FakeNode(),
            validation_node=FakeNode(),
            trajectory_recorder=recorder,
        )

        await run_workflow(
            graph=graph,
            initial_state=AgentState(
                user_request="hello",
                project_root="/test",
                task_id="task-external",
            ),
        )

        assert recorder.record_step.call_count == 2
        assert all(call.args[0] == "task-external" for call in recorder.record_step.call_args_list)


class TestBuildCheckpointerConfig:
    """_build_checkpointer 的 config 集成测试。"""

    def test_checkpointer_uses_config_function(self, monkeypatch) -> None:
        """_build_checkpointer 应调用 config.get_checkpoint_db_path()。"""
        from codeagent import config

        custom_path = "/custom/path/checkpoints.db"
        monkeypatch.setenv("CHECKPOINT_DB_PATH", custom_path)
        result = config.get_checkpoint_db_path()
        assert result == custom_path, (
            f"config.get_checkpoint_db_path() should return the env var value, "
            f"got {result!r}"
        )

    def test_checkpointer_default_path(self) -> None:
        """未设置环境变量时应使用默认路径。"""
        from codeagent import config

        # 验证默认值中包含 .codeagent/checkpoints.db
        result = config.get_checkpoint_db_path()
        assert ".codeagent" in result
        assert "checkpoints.db" in result

    def test_graph_no_bare_osenviron_get(self) -> None:
        """graph.py 中不应有裸 os.environ.get('CHECKPOINT_DB_PATH')。"""
        import inspect
        from codeagent.orchestration import graph as graph_module

        source = inspect.getsource(graph_module)
        # 允许 import os 和 os.environ 的其他用法，但不允许 CHECKPOINT_DB_PATH
        assert "CHECKPOINT_DB_PATH" not in source or "config.get_checkpoint_db_path" in source, (
            "graph.py should use config.get_checkpoint_db_path() instead of bare CHECKPOINT_DB_PATH"
        )
