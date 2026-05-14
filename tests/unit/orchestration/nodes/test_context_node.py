"""ContextNode 单元测试。

Mock IContextGateway，覆盖正常流程、降级流程、空上下文等场景。
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from codeagent.gateway.context_gateway import (
    CodeSnippet,
    ContextPackage,
    IContextGateway,
)
from codeagent.orchestration.nodes.context_node import ContextNode
from codeagent.orchestration.state import AgentState


@pytest.fixture
def mock_gateway() -> MagicMock:
    """创建 Mock IContextGateway。"""
    gateway = MagicMock(spec=IContextGateway)
    gateway.build_context = AsyncMock(
        return_value=ContextPackage(
            file_tree={
                "name": "project",
                "type": "directory",
                "children": [
                    {
                        "name": "main.py",
                        "type": "file",
                        "path": "main.py",
                        "size": 100,
                    }
                ],
            },
            related_code=[
                CodeSnippet(
                    file_path="main.py",
                    start_line=1,
                    end_line=3,
                    code="x = 1\ny = 2\nz = 3",
                    score=0.95,
                )
            ],
            dependency_info={"python": ["requests>=2.0"]},
        )
    )
    return gateway


@pytest.fixture
def state() -> AgentState:
    return AgentState(
        user_request="Show me the project",
        project_root="/test/project",
    )


@pytest.fixture
def node(mock_gateway: MagicMock) -> ContextNode:
    return ContextNode(context_gateway=mock_gateway)


class TestContextNodeNormalFlow:
    """正常执行流程。"""

    @pytest.mark.asyncio
    async def test_calls_build_context(
        self, node: ContextNode, state: AgentState, mock_gateway: MagicMock
    ) -> None:
        """应调用 gateway.build_context。"""
        await node(state)
        mock_gateway.build_context.assert_called_once_with(
            project_root="/test/project",
            query="Show me the project",
        )

    @pytest.mark.asyncio
    async def test_returns_file_tree(
        self, node: ContextNode, state: AgentState
    ) -> None:
        """应返回 file_tree。"""
        result = await node(state)
        assert result["file_tree"] is not None
        assert result["file_tree"]["name"] == "project"

    @pytest.mark.asyncio
    async def test_returns_context_string(
        self, node: ContextNode, state: AgentState
    ) -> None:
        """应返回格式化的上下文字符串（Phase 1a 兼容）。"""
        result = await node(state)
        context = result["context"]
        assert isinstance(context, str)
        assert len(context) > 0
        assert "<project_tree>" in context

    @pytest.mark.asyncio
    async def test_returns_semantic_context(
        self, node: ContextNode, state: AgentState
    ) -> None:
        """应返回语义上下文。"""
        result = await node(state)
        semantic = result["semantic_context"]
        assert isinstance(semantic, str)
        assert "main.py:1-3" in semantic
        assert "x = 1" in semantic

    @pytest.mark.asyncio
    async def test_returns_current_file_context(
        self, node: ContextNode, state: AgentState
    ) -> None:
        """应返回当前文件上下文（文件树文本）。"""
        result = await node(state)
        current = result["current_file_context"]
        assert isinstance(current, str)
        assert "main.py" in current

    @pytest.mark.asyncio
    async def test_returns_dependency_graph(
        self, node: ContextNode, state: AgentState
    ) -> None:
        """应返回依赖信息。"""
        result = await node(state)
        assert result["dependency_graph"] == {"python": ["requests>=2.0"]}

    @pytest.mark.asyncio
    async def test_no_degraded_in_normal_flow(
        self, node: ContextNode, state: AgentState
    ) -> None:
        """正常流程不应设置 degraded_mode。"""
        result = await node(state)
        assert result.get("degraded_mode") is None
        assert result.get("errors") is None or result.get("errors") == []


class TestContextNodeDegradation:
    """降级流程。"""

    @pytest.mark.asyncio
    async def test_build_context_failure(
        self, mock_gateway: MagicMock, state: AgentState
    ) -> None:
        """build_context 失败应进入降级模式。"""
        mock_gateway.build_context = AsyncMock(
            side_effect=RuntimeError("Cannot connect to index")
        )
        node = ContextNode(context_gateway=mock_gateway)
        result = await node(state)
        assert result["degraded_mode"] is True
        assert len(result["errors"]) >= 1
        assert "Context build failed" in result["errors"][0]

    @pytest.mark.asyncio
    async def test_timeout_degradation(
        self, mock_gateway: MagicMock, state: AgentState
    ) -> None:
        """超时也应进入降级模式。"""
        mock_gateway.build_context = AsyncMock(
            side_effect=TimeoutError("build_context timed out")
        )
        node = ContextNode(context_gateway=mock_gateway)
        result = await node(state)
        assert result["degraded_mode"] is True

    @pytest.mark.asyncio
    async def test_empty_gateway_response(
        self, mock_gateway: MagicMock, state: AgentState
    ) -> None:
        """空响应不应触发降级，应返回空上下文。"""
        mock_gateway.build_context = AsyncMock(
            return_value=ContextPackage()
        )
        node = ContextNode(context_gateway=mock_gateway)
        result = await node(state)
        assert result.get("degraded_mode") is None
        # 空上下文应返回空字符串
        assert result["semantic_context"] == ""
        assert result["current_file_context"] == ""


class TestContextNodeEdgeCases:
    """边界情况。"""

    @pytest.mark.asyncio
    async def test_empty_related_code(
        self, mock_gateway: MagicMock, state: AgentState
    ) -> None:
        """无相关代码时 semantic_context 应回落为 assembled context。"""
        mock_gateway.build_context = AsyncMock(
            return_value=ContextPackage(
                file_tree={"name": "root", "type": "directory"},
                related_code=[],
                dependency_info={},
            )
        )
        node = ContextNode(context_gateway=mock_gateway)
        result = await node(state)
        assert result["semantic_context"] != ""
        # 当 related_code 为空时，semantic_context 应等于 assembled context
        assert result["semantic_context"] == result["context"]

    @pytest.mark.asyncio
    async def test_returns_no_errors_on_success(
        self, node: ContextNode, state: AgentState, mock_gateway: MagicMock
    ) -> None:
        """成功时不应返回 errors。"""
        result = await node(state)
        assert result.get("errors") is None or result.get("errors") == []

    @pytest.mark.asyncio
    async def test_project_root_passed_correctly(
        self, node: ContextNode, state: AgentState, mock_gateway: MagicMock
    ) -> None:
        """project_root 应正确传递给 gateway。"""
        state.project_root = "/another/project"
        await node(state)
        mock_gateway.build_context.assert_called_once_with(
            project_root="/another/project",
            query="Show me the project",
        )

    @pytest.mark.asyncio
    async def test_user_request_passed_correctly(
        self, node: ContextNode, state: AgentState, mock_gateway: MagicMock
    ) -> None:
        """user_request 应正确传递给 gateway。"""
        state.user_request = "Find all Python files"
        await node(state)
        mock_gateway.build_context.assert_called_once_with(
            project_root="/test/project",
            query="Find all Python files",
        )

    @pytest.mark.asyncio
    async def test_returns_duration(
        self, node: ContextNode, state: AgentState
    ) -> None:
        """虽然没有显式 duration 返回，但方法应正常完成。"""
        result = await node(state)
        # 正常流程应返回所有期望字段
        assert "semantic_context" in result
        assert "current_file_context" in result
        assert "dependency_graph" in result
        assert "file_tree" in result
        assert "context" in result


class TestContextNodeWithCustomAssembler:
    """自定义 ContextAssembler。"""

    @pytest.mark.asyncio
    async def test_custom_assembler(
        self, mock_gateway: MagicMock, state: AgentState
    ) -> None:
        """应使用自定义 assembler 而非默认的。"""
        assembler = MagicMock()
        assembler.assemble.return_value = "custom assembled"

        node = ContextNode(
            context_gateway=mock_gateway,
            context_assembler=assembler,
        )
        result = await node(state)
        assert result["context"] == "custom assembled"
        assembler.assemble.assert_called_once()
