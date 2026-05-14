"""LangGraph 工作流图结构单元测试。

验证图结构：节点数量、边连接正确、条件路由分支覆盖。
不执行图，仅验证结构。
"""

from __future__ import annotations

from typing import Any

import pytest
from langgraph.graph import END

from codeagent.orchestration.graph import build_workflow
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


class TestGraphStructure:
    """图结构验证。"""

    def test_has_all_nodes(self, graph) -> None:
        """应包含所有 5 个工作流节点（不含 __start__）。"""
        node_names = {n for n in graph.nodes if not n.startswith("__")}
        expected = {"context", "planning", "execution", "validation", "human_review"}
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

    def test_context_to_planning_edge(self, graph) -> None:
        """context → planning 应为普通边。"""
        assert graph.builder is not None
        has_edge = any(
            isinstance(e, tuple) and e[0] == "context" and e[1] == "planning"
            for e in graph.builder.edges
        )
        assert has_edge

    def test_planning_has_conditional_edges(self, graph) -> None:
        """planning 节点应有条件路由。"""
        assert graph.builder is not None
        assert "planning" in graph.builder.branches

    def test_execution_has_conditional_edges(self, graph) -> None:
        """execution 节点应有条件路由。"""
        assert graph.builder is not None
        assert "execution" in graph.builder.branches

    def test_validation_has_conditional_edges(self, graph) -> None:
        """validation 节点应有条件路由。"""
        assert graph.builder is not None
        assert "validation" in graph.builder.branches

    def test_human_review_has_conditional_edges(self, graph) -> None:
        """human_review 节点应有条件路由。"""
        assert graph.builder is not None
        assert "human_review" in graph.builder.branches

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
        import inspect
        assert hasattr(graph, "ainvoke")
        assert callable(graph.ainvoke)
