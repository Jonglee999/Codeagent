"""ContextAssembler Memory 层注入单元测试。

覆盖：有记忆时 Memory 层出现在 prompt 中、无记忆时层缺失、
recall 异常时降级、memory_gateway=None 兼容、token_budget 传递。
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from codeagent.context_engine.context_assembler import ContextAssembler
from codeagent.gateway.memory_gateway import IMemoryGateway


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def mock_memory_gateway() -> AsyncMock:
    """创建一个模拟 IMemoryGateway。"""
    gw = AsyncMock(spec=IMemoryGateway)
    gw.recall.return_value = (
        "<relevant_memories>"
        '  <memory type="feedback" name="no-mock" score="0.90" scope="global">'
        "    Never mock the database"
        "  </memory>"
        "</relevant_memories>"
    )
    return gw


# ══════════════════════════════════════════════════════════════════════════
# 1. 有记忆时 Memory 层注入
# ══════════════════════════════════════════════════════════════════════════


class TestMemoryLayerPresent:
    """有记忆时 Memory 层应正确注入。"""

    @pytest.mark.asyncio
    async def test_memory_section_included(self, mock_memory_gateway: AsyncMock) -> None:
        """有记忆时 assemble_memory_section 应返回包含记忆的文本。"""
        assembler = ContextAssembler(memory_gateway=mock_memory_gateway)
        section = await assembler.assemble_memory_section("test query")

        assert "## Relevant Memories" in section
        assert "<relevant_memories>" in section
        assert "no-mock" in section

    @pytest.mark.asyncio
    async def test_memory_section_format(self, mock_memory_gateway: AsyncMock) -> None:
        """assemble_memory_section 的格式应包含正确的标题和 XML。"""
        assembler = ContextAssembler(memory_gateway=mock_memory_gateway)
        section = await assembler.assemble_memory_section("test query")

        assert section.startswith("## Relevant Memories")
        assert "no-mock" in section

    @pytest.mark.asyncio
    async def test_recall_called_with_query(self, mock_memory_gateway: AsyncMock) -> None:
        """recall 应使用 task_description 作为查询。"""
        assembler = ContextAssembler(memory_gateway=mock_memory_gateway)
        await assembler.assemble_memory_section("test query")

        mock_memory_gateway.recall.assert_awaited_once()
        kwargs = mock_memory_gateway.recall.call_args.kwargs
        assert kwargs["query"] == "test query"

    @pytest.mark.asyncio
    async def test_recall_called_with_token_budget(self, mock_memory_gateway: AsyncMock) -> None:
        """recall 应使用指定的 token_budget。"""
        assembler = ContextAssembler(memory_gateway=mock_memory_gateway, memory_token_budget=500)
        await assembler.assemble_memory_section("test")

        kwargs = mock_memory_gateway.recall.call_args.kwargs
        assert kwargs["token_budget"] == 500


# ══════════════════════════════════════════════════════════════════════════
# 2. 无记忆时层缺失
# ══════════════════════════════════════════════════════════════════════════


class TestMemoryLayerMissing:
    """无记忆时 Memory 层应返回空字符串。"""

    @pytest.mark.asyncio
    async def test_empty_recall_returns_empty(self, mock_memory_gateway: AsyncMock) -> None:
        """recall 返回空字符串时应返回空字符串。"""
        mock_memory_gateway.recall.return_value = ""
        assembler = ContextAssembler(memory_gateway=mock_memory_gateway)
        section = await assembler.assemble_memory_section("test")
        assert section == ""

    @pytest.mark.asyncio
    async def test_no_gateway_returns_empty(self) -> None:
        """memory_gateway=None 时应返回空字符串。"""
        assembler = ContextAssembler(memory_gateway=None)
        section = await assembler.assemble_memory_section("test")
        assert section == ""

    @pytest.mark.asyncio
    async def test_default_constructor_no_memory(self) -> None:
        """默认构造的 ContextAssembler 不应有 memory_gateway。"""
        assembler = ContextAssembler()
        section = await assembler.assemble_memory_section("test")
        assert section == ""


# ══════════════════════════════════════════════════════════════════════════
# 3. recall 异常降级
# ══════════════════════════════════════════════════════════════════════════


class TestMemoryLayerDegradation:
    """recall 异常时应优雅降级。"""

    @pytest.mark.asyncio
    async def test_recall_exception_returns_empty(self, mock_memory_gateway: AsyncMock) -> None:
        """recall 抛异常时应返回空字符串。"""
        mock_memory_gateway.recall.side_effect = RuntimeError("recall failed")
        assembler = ContextAssembler(memory_gateway=mock_memory_gateway)
        section = await assembler.assemble_memory_section("test")
        assert section == ""


# ══════════════════════════════════════════════════════════════════════════
# 4. ContextAssembler 构造向后兼容
# ══════════════════════════════════════════════════════════════════════════


class TestBackwardCompatibility:
    """memory_gateway 参数应完全可选，不影响现有功能。"""

    def test_no_memory_gateway_assembly_works(self) -> None:
        """未设置 memory_gateway 时 assemble 应正常工作。"""
        assembler = ContextAssembler()
        assert assembler.total_budget == 8000

    def test_existing_assemble_unaffected(self) -> None:
        """现有的 assemble 方法应不受 memory 参数影响。"""
        assembler = ContextAssembler(total_budget=8000, memory_gateway=None)
        assert assembler.total_budget == 8000
