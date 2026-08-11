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
    retryable: bool | None = None
    suggested_recovery: str | None = None

    def __post_init__(self) -> None:
        """Fill a consistent recovery contract for every failed tool call."""
        if self.success:
            self.retryable = False
            return
        default_retryable, default_recovery = classify_tool_error(self.error_code)
        if self.retryable is None:
            self.retryable = default_retryable
        if self.suggested_recovery is None:
            self.suggested_recovery = default_recovery


def classify_tool_error(error_code: str | None) -> tuple[bool, str]:
    """Map heterogeneous tool codes to a small, model-actionable protocol."""
    code = (error_code or "UNKNOWN").upper()
    if any(part in code for part in ("TIMEOUT", "UNAVAILABLE", "NETWORK", "CIRCUIT")):
        return True, "Retry once with a narrower request, then use a local alternative."
    if code in {"MCP_ERROR", "SEARCH_ERROR", "DIAGNOSTICS_ERROR", "EXECUTION_ERROR"}:
        return True, "Retry once; if it fails again, switch tools or reduce the operation scope."
    if any(part in code for part in ("PATCH_CONFLICT", "FILE_EXISTS")):
        return False, "Re-read the current file state and regenerate the change against it."
    if any(part in code for part in ("INVALID", "EMPTY", "REQUIRED", "DECODE")):
        return False, "Correct the arguments or input format before calling the tool again."
    if any(part in code for part in ("NOT_FOUND", "NOT_A_FILE", "NOT_A_GIT_REPO")):
        return False, "List or search the workspace to verify the target, then choose an existing path."
    if any(
        part in code
        for part in ("PERMISSION", "PROTECTED", "SENSITIVE", "SAFETY", "BLOCKED", "TRAVERSAL")
    ):
        return False, "Choose an authorized, non-destructive operation or request explicit approval."
    if code in {"NON_ZERO_EXIT", "GIT_ERROR", "RG_ERROR"}:
        return False, "Inspect the bounded stderr/stdout details, fix the cause, then run a targeted retry."
    return False, "Inspect the error details and choose a different bounded action."


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
    category: str = "general"
    risk_level: str = "low"
    source: str = "core"
    read_only: bool = False
    external: bool = False
    latency_hint: str = "fast"
    cost_hint: str = "free"
    idempotent: bool = True
    reversible: bool = True

    def public_metadata(self) -> dict[str, Any]:
        """Return schema-free metadata safe for reports and UI."""
        return {
            "name": self.name,
            "category": self.category,
            "risk_level": self.risk_level,
            "source": self.source,
            "read_only": self.read_only,
            "external": self.external,
            "latency_hint": self.latency_hint,
            "cost_hint": self.cost_hint,
            "idempotent": self.idempotent,
            "reversible": self.reversible,
        }


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
