"""Tool Gateway — 工具系统的统一抽象接口。"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolResult:
    """工具执行结果的数据传输对象。

    Attributes:
        success: 执行是否成功
        data: 执行返回的数据
        error_message: 错误消息（失败时）
        error_code: 错误码（失败时）
        duration_ms: 执行耗时（毫秒）
        tokens_consumed: 消耗的 token 数
    """

    success: bool
    data: Any = None
    error_message: str | None = None
    error_code: str | None = None
    duration_ms: float = 0.0
    tokens_consumed: int = 0


@dataclass
class ToolDefinition:
    """工具元数据定义，用于向 LLM 描述工具。

    Attributes:
        name: 工具名称，也是 LLM function name
        description: 工具功能描述
        parameters_schema: JSON Schema 格式的参数定义
    """

    name: str
    description: str
    parameters_schema: dict = field(default_factory=dict)


class IToolGateway(ABC):
    """工具系统 Gateway 接口。

    编排核心层通过此接口调用具体工具，不直接依赖工具实现。
    """

    @abstractmethod
    async def execute_tool(self, tool_name: str, params: dict) -> ToolResult:
        """执行指定工具。

        Args:
            tool_name: 工具名称（如 "read_file"、"write_file"）
            params: 工具参数，与 ToolDefinition.parameters_schema 对应

        Returns:
            ToolResult: 工具执行结果

        Raises:
            ToolNotFoundError: 工具不存在时抛出
        """
        ...

    @abstractmethod
    def list_tools(self) -> list[ToolDefinition]:
        """列出所有已注册的工具。

        Returns:
            list[ToolDefinition]: 所有可用工具的定义列表
        """
        ...

    @abstractmethod
    async def validate_tool_params(self, tool_name: str, params: dict) -> bool:
        """验证工具参数是否合法。

        Args:
            tool_name: 工具名称
            params: 待验证的参数

        Returns:
            bool: 参数是否有效
        """
        ...
