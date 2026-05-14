"""工具注册中心 — 支持动态注册和发现。"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from codeagent.gateway.tool_gateway import ToolDefinition

if TYPE_CHECKING:
    from codeagent.tools.base import BaseTool


class ToolAlreadyRegisteredError(Exception):
    """工具重复注册时抛出。"""

    def __init__(self, tool_name: str) -> None:
        super().__init__(f"Tool '{tool_name}' is already registered")


class ToolNotFoundError(Exception):
    """工具不存在时抛出。"""

    def __init__(self, tool_name: str) -> None:
        super().__init__(f"Tool '{tool_name}' not found")


class ToolRegistry:
    """工具注册中心 — 支持动态注册、注销和发现。

    线程安全：register / unregister / get 均通过锁保护。
    """

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}
        self._lock = threading.Lock()

    def register(self, tool: BaseTool) -> None:
        """注册一个工具。

        Args:
            tool: 工具实例

        Raises:
            ToolAlreadyRegisteredError: 同名工具已注册
        """
        with self._lock:
            if tool.name in self._tools:
                raise ToolAlreadyRegisteredError(tool.name)
            self._tools[tool.name] = tool

    def unregister(self, tool_name: str) -> None:
        """注销一个工具。

        Args:
            tool_name: 工具名称

        Raises:
            ToolNotFoundError: 工具不存在
        """
        with self._lock:
            if tool_name not in self._tools:
                raise ToolNotFoundError(tool_name)
            del self._tools[tool_name]

    def get(self, tool_name: str) -> BaseTool:
        """获取已注册的工具实例。

        Args:
            tool_name: 工具名称

        Returns:
            BaseTool: 工具实例

        Raises:
            ToolNotFoundError: 工具不存在
        """
        with self._lock:
            if tool_name not in self._tools:
                raise ToolNotFoundError(tool_name)
            return self._tools[tool_name]

    def list_tools(self) -> list[ToolDefinition]:
        """返回所有已注册工具的定义（供 LLM function calling 使用）。

        Returns:
            list[ToolDefinition]: 所有工具的定义列表
        """
        with self._lock:
            return [
                ToolDefinition(
                    name=t.name,
                    description=t.description,
                    parameters_schema=t.parameters,
                )
                for t in self._tools.values()
            ]

    def get_langchain_tools(self) -> list:
        """返回 LangChain 兼容的工具列表。

        Returns:
            list[StructuredTool]: LangChain 工具列表
        """
        with self._lock:
            return [t.get_langchain_tool() for t in self._tools.values()]
