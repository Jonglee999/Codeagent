"""ToolRegistry 注册中心单元测试。"""

from __future__ import annotations

import threading

import pytest

from codeagent.gateway.tool_gateway import ToolDefinition, ToolResult
from codeagent.tools.base import BaseTool
from codeagent.tools.registry import (
    ToolAlreadyRegisteredError,
    ToolNotFoundError,
    ToolRegistry,
)


# ── 测试辅助工具 ──────────────────────────────────────────────────────────


class DummyTool(BaseTool):
    name = "dummy"
    description = "A dummy tool for testing"
    parameters = {
        "type": "object",
        "properties": {"x": {"type": "integer"}},
    }

    async def execute(self, **kwargs: object) -> ToolResult:
        return ToolResult(success=True)


class AnotherTool(BaseTool):
    name = "another"
    description = "Another test tool"
    parameters = {}

    async def execute(self, **kwargs: object) -> ToolResult:
        return ToolResult(success=True)


# ── TestRegistry ───────────────────────────────────────────────────────────


class TestRegistry:
    """验证注册中心基本操作。"""

    def setup_method(self) -> None:
        self.registry = ToolRegistry()

    def test_register_and_get(self) -> None:
        tool = DummyTool()
        self.registry.register(tool)
        retrieved = self.registry.get("dummy")
        assert retrieved is tool

    def test_register_multiple_tools(self) -> None:
        self.registry.register(DummyTool())
        self.registry.register(AnotherTool())
        assert self.registry.get("dummy") is not None
        assert self.registry.get("another") is not None

    def test_get_nonexistent_raises(self) -> None:
        with pytest.raises(ToolNotFoundError, match="not_found"):
            self.registry.get("not_found")

    def test_duplicate_register_raises(self) -> None:
        self.registry.register(DummyTool())
        with pytest.raises(ToolAlreadyRegisteredError, match="dummy"):
            self.registry.register(DummyTool())

    def test_unregister(self) -> None:
        self.registry.register(DummyTool())
        self.registry.unregister("dummy")
        with pytest.raises(ToolNotFoundError):
            self.registry.get("dummy")

    def test_unregister_nonexistent_raises(self) -> None:
        with pytest.raises(ToolNotFoundError, match="not_found"):
            self.registry.unregister("not_found")

    def test_list_tools(self) -> None:
        self.registry.register(DummyTool())
        self.registry.register(AnotherTool())
        tool_defs = self.registry.list_tools()
        assert len(tool_defs) == 2
        names = {t.name for t in tool_defs}
        assert names == {"dummy", "another"}

    def test_list_tools_returns_definitions(self) -> None:
        self.registry.register(DummyTool())
        tool_defs = self.registry.list_tools()
        td = tool_defs[0]
        assert isinstance(td, ToolDefinition)
        assert td.name == "dummy"
        assert td.description == "A dummy tool for testing"
        assert td.parameters_schema == DummyTool.parameters

    def test_get_langchain_tools(self) -> None:
        from langchain_core.tools import StructuredTool

        self.registry.register(DummyTool())
        lc_tools = self.registry.get_langchain_tools()
        assert len(lc_tools) == 1
        assert isinstance(lc_tools[0], StructuredTool)
        assert lc_tools[0].name == "dummy"

    def test_empty_registry_list_tools(self) -> None:
        assert self.registry.list_tools() == []

    def test_empty_registry_langchain_tools(self) -> None:
        assert self.registry.get_langchain_tools() == []

    def test_register_after_unregister(self) -> None:
        """注销后可以重新注册同名工具。"""
        self.registry.register(DummyTool())
        self.registry.unregister("dummy")
        # 再次注册应成功
        self.registry.register(DummyTool())
        assert self.registry.get("dummy") is not None


class TestRegistryThreadSafety:
    """验证注册中心的并发安全性。"""

    def test_concurrent_register(self) -> None:
        registry = ToolRegistry()

        def _register(tool_name: str) -> None:
            async def execute(self: BaseTool, **kwargs: object) -> ToolResult:
                return ToolResult(success=True)

            tool_cls = type(
                f"TempTool_{tool_name}",
                (BaseTool,),
                {
                    "name": tool_name,
                    "description": "",
                    "parameters": {},
                    "execute": execute,
                },
            )
            registry.register(tool_cls())

        threads = [
            threading.Thread(target=_register, args=(f"tool_{i}",))
            for i in range(10)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(registry.list_tools()) == 10

    def test_concurrent_read_write(self) -> None:
        registry = ToolRegistry()
        registry.register(DummyTool())

        results: list[bool] = []
        results_lock = threading.Lock()

        def _reader() -> None:
            try:
                registry.get("dummy")
                with results_lock:
                    results.append(True)
            except ToolNotFoundError:
                with results_lock:
                    results.append(False)

        def _writer() -> None:
            try:
                registry.unregister("dummy")
                registry.register(DummyTool())
            except ToolNotFoundError:
                registry.register(DummyTool())

        threads = [threading.Thread(target=_reader) for _ in range(20)]
        threads += [threading.Thread(target=_writer) for _ in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # 至少大部分读取操作应成功
        assert sum(results) > 10
