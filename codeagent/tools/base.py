"""工具系统抽象基类 — 所有工具的基础契约。"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

try:
    from jsonschema import validate as jsonschema_validate
    from jsonschema.exceptions import ValidationError as JsonSchemaValidationError
except ImportError:  # pragma: no cover
    jsonschema_validate = None  # type: ignore[assignment]
    JsonSchemaValidationError = None  # type: ignore[assignment]

# 复用 Gateway 层定义的 DTO
from codeagent.gateway.tool_gateway import ToolResult  # noqa: F401


class BaseTool(ABC):
    """所有工具的抽象基类。

    子类必须定义以下类属性：
        name: 工具名称（LLM function name）
        description: 工具描述（供 LLM 理解用途）
        parameters: JSON Schema 格式的参数定义
    """

    # 工具元数据（子类必须重新定义）
    name: str = ""
    description: str = ""
    parameters: dict = {}

    # 安全边界配置
    requires_sandbox: bool = False
    allowed_paths: list[str] = []
    max_timeout_seconds: int = 30

    @abstractmethod
    async def execute(self, **kwargs: Any) -> ToolResult:
        """执行工具，返回统一结果。

        Args:
            **kwargs: 工具参数，与 parameters JSON Schema 对应

        Returns:
            ToolResult: 执行结果
        """
        ...

    def validate_params(self, **kwargs: Any) -> bool:
        """验证参数是否符合 parameters JSON Schema。

        Args:
            **kwargs: 待验证的参数

        Returns:
            bool: 参数是否有效
        """
        if not self.parameters or jsonschema_validate is None:
            return True

        try:
            jsonschema_validate(instance=kwargs, schema=self.parameters)
            return True
        except JsonSchemaValidationError:
            return False

    def get_langchain_tool(self) -> Any:
        """生成 LangChain/LangGraph 兼容的 Tool 对象。

        返回的 StructuredTool 可以直接用于 LLM.bind_tools()。

        Returns:
            StructuredTool: LangChain 兼容的工具包装
        """
        from langchain_core.tools import StructuredTool

        return StructuredTool(
            name=self.name,
            description=self.description,
            args_schema=self.parameters,
            func=self._sync_execute_wrapper,
            coroutine=self.execute,
        )

    def _sync_execute_wrapper(self, **kwargs: Any) -> ToolResult:
        """同步包装器，用于 LangChain StructuredTool 的 func 参数。

        注意：同步模式下无法执行 async execute，会抛出异常。
        """
        raise RuntimeError(
            f"Tool '{self.name}' is async-only. "
            "Use await or pass coroutine to LangChain."
        )


class ConcreteTool(BaseTool):
    """测试用的具体工具实现 — 仅用于验证 BaseTool 抽象机制。"""

    name = "concrete_tool"
    description = "A concrete tool for testing"
    parameters = {
        "type": "object",
        "properties": {
            "msg": {"type": "string", "description": "A message"},
        },
        "required": ["msg"],
    }

    async def execute(self, **kwargs: Any) -> ToolResult:
        start = time.monotonic()
        data = {"echo": kwargs.get("msg", "")}
        return ToolResult(
            success=True,
            data=data,
            duration_ms=(time.monotonic() - start) * 1000,
        )
