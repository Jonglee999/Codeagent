"""工具系统抽象基类 — 所有工具的基础契约。"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

try:
    from jsonschema.validators import validator_for
    from jsonschema.exceptions import ValidationError as JsonSchemaValidationError
except ImportError:  # pragma: no cover
    validator_for = None  # type: ignore[assignment]
    JsonSchemaValidationError = None  # type: ignore[assignment]

# 复用 Gateway 层定义的 DTO
from codeagent.gateway.tool_gateway import ToolResult  # noqa: F401


@dataclass(frozen=True)
class ToolParamValidationError:
    """One model-actionable JSON Schema validation error."""

    path: str
    message: str
    validator: str | None = None


@dataclass(frozen=True)
class ToolParamValidationResult:
    """Detailed parameter validation result shared by tools and the gateway."""

    valid: bool
    errors: tuple[ToolParamValidationError, ...] = ()


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

    # Orchestration metadata. Subclasses override only what differs.
    category: str = "general"
    risk_level: str = "low"
    source: str = "core"
    read_only: bool = False
    external: bool = False
    latency_hint: str = "fast"
    cost_hint: str = "free"
    idempotent: bool = True
    reversible: bool = True

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
        return self.validate_params_detailed(**kwargs).valid

    def validate_params_detailed(self, **kwargs: Any) -> ToolParamValidationResult:
        """Validate parameters and preserve exact JSON paths for correction."""

        if not self.parameters or validator_for is None:
            return ToolParamValidationResult(valid=True)

        validator_class = validator_for(self.parameters)
        validator_class.check_schema(self.parameters)
        errors = sorted(
            validator_class(self.parameters).iter_errors(kwargs),
            key=lambda error: tuple(str(part) for part in error.absolute_path),
        )
        if not errors:
            return ToolParamValidationResult(valid=True)
        return ToolParamValidationResult(
            valid=False,
            errors=tuple(
                ToolParamValidationError(
                    path=(
                        "$"
                        + "".join(
                            f"[{part}]" if isinstance(part, int) else f".{part}"
                            for part in error.absolute_path
                        )
                    ),
                    message=error.message,
                    validator=str(error.validator) if error.validator else None,
                )
                for error in errors[:10]
            ),
        )

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
