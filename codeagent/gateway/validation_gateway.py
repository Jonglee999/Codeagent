"""Validation Gateway — 验证闭环的抽象接口。"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ValidationError:
    """验证错误/警告项。

    Attributes:
        file_path: 文件路径（相对项目根目录）
        line: 行号
        column: 列号
        message: 错误/警告消息
        code: 错误码
        severity: 严重级别 ("error" / "warning")
    """

    file_path: str
    line: int = 0
    column: int = 0
    message: str = ""
    code: str = ""
    severity: str = "error"


@dataclass
class ValidationResult:
    """验证结果。

    Attributes:
        passed: 是否通过验证（无 error 级别的错误）
        errors: 错误列表
        warnings: 警告列表
        duration_ms: 验证耗时（毫秒）
        sandboxed: 是否在沙箱中执行验证
    """

    passed: bool = True
    errors: list[ValidationError] = field(default_factory=list)
    warnings: list[ValidationError] = field(default_factory=list)
    duration_ms: float = 0.0
    sandboxed: bool = False
    output: str = ""


class IValidationGateway(ABC):
    """验证闭环 Gateway 接口。

    编排核心层通过此接口执行语法检查、静态分析、运行测试等验证操作。
    """

    @abstractmethod
    async def run_syntax_check(self, file_path: str) -> ValidationResult:
        """对单个文件进行语法检查。

        Args:
            file_path: 文件路径

        Returns:
            ValidationResult: 验证结果
        """
        ...

    @abstractmethod
    async def run_lint(self, files: list[str]) -> ValidationResult:
        """对文件列表进行静态分析/lint。

        Args:
            files: 文件路径列表

        Returns:
            ValidationResult: 验证结果
        """
        ...

    @abstractmethod
    async def run_tests(self, project_root: str) -> ValidationResult:
        """运行项目测试套件。

        Args:
            project_root: 项目根目录路径

        Returns:
            ValidationResult: 验证结果
        """
        ...

    @abstractmethod
    async def run_runtime_check(self, file_path: str) -> ValidationResult:
        """对文件进行运行时验证（如导入检查、执行检查）。

        Args:
            file_path: 文件路径

        Returns:
            ValidationResult: 验证结果
        """
        ...
