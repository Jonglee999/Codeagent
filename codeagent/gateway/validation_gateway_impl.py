"""ValidationGateway — 三层验证 Gateway 具体实现。

整合 SyntaxValidator + StaticAnalyzer + RuntimeValidator，
为 IValidationGateway 提供完整的 Phase 4 实现。
"""

from __future__ import annotations

import logging
from pathlib import Path
from codeagent import config
from codeagent.gateway.validation_gateway import (
    IValidationGateway,
    ValidationError,
    ValidationResult,
)
from codeagent.validation.runtime_validator import RuntimeValidator
from codeagent.validation.static_analyzer import StaticAnalyzer
from codeagent.validation.syntax_validator import SyntaxValidator

logger = logging.getLogger(__name__)


class ValidationGateway(IValidationGateway):
    """三层验证 Gateway 具体实现。

    将 SyntaxValidator、StaticAnalyzer、RuntimeValidator 适配为
    IValidationGateway 接口，供 ValidationNode 和 Orchestrator 使用。

    用法:
        gateway = ValidationGateway(project_root="/path/to/project")
        result = await gateway.run_lint(["main.py"])
    """

    def __init__(
        self,
        project_root: str = "",
        syntax_validator: SyntaxValidator | None = None,
        static_analyzer: StaticAnalyzer | None = None,
        runtime_validator: RuntimeValidator | None = None,
    ) -> None:
        """初始化 ValidationGateway。

        Args:
            project_root: 项目根目录路径
            syntax_validator: 语法验证器（可选，默认创建）
            static_analyzer: 静态分析器（可选，默认创建）
            runtime_validator: 运行时验证器（可选，默认创建）
        """
        self._project_root = project_root
        self._syntax_validator = syntax_validator or SyntaxValidator()
        self._static_analyzer = static_analyzer or StaticAnalyzer(
            project_root=project_root or None
        )
        self._runtime_validator = runtime_validator
        # RuntimeValidator 需要 project_root，延迟创建
        self._lazy_rv: RuntimeValidator | None = None

    def _get_runtime_validator(self) -> RuntimeValidator | None:
        """获取或创建 RuntimeValidator。"""
        if self._runtime_validator:
            return self._runtime_validator
        if self._project_root:
            if self._lazy_rv is None:
                executor = None
                if config.get_sandbox_enabled():
                    from codeagent.sandbox.docker_executor import DockerExecutor

                    executor = DockerExecutor(
                        image=config.get_env("SANDBOX_IMAGE", "codeagent-sandbox:latest"),
                        memory_mb=config.get_sandbox_memory_mb(),
                        timeout_s=config.get_sandbox_timeout(),
                    )
                self._lazy_rv = RuntimeValidator(self._project_root, executor=executor)
            return self._lazy_rv
        return None

    @property
    def static_analyzer(self) -> StaticAnalyzer:
        """Expose the project-rooted analyzer to orchestration."""
        return self._static_analyzer

    @property
    def runtime_validator(self) -> RuntimeValidator | None:
        """Expose the project-rooted runtime validator to orchestration."""
        return self._get_runtime_validator()

    async def run_syntax_check(self, file_path: str) -> ValidationResult:
        """Layer 1: 语法检查。"""
        path = Path(file_path)
        if not path.is_absolute() and self._project_root:
            path = Path(self._project_root) / path
        return await self._syntax_validator.check_file(str(path))

    async def run_lint(self, files: list[str]) -> ValidationResult:
        """Layer 2: 静态分析 — 使用 StaticAnalyzer.run_all()。

        聚合 lint + typecheck + security 结果为单个 ValidationResult。
        """
        results = await self._static_analyzer.run_all(files)

        all_errors: list[ValidationError] = []
        all_warnings: list[ValidationError] = []
        total_duration = 0.0

        labels = ["lint", "typecheck", "security"]
        sub_errors: dict[str, int] = {}
        for label, r in zip(labels, results):
            all_errors.extend(r.errors)
            all_warnings.extend(r.warnings)
            total_duration += r.duration_ms
            if not r.passed:
                sub_errors[label] = len(r.errors)

        if sub_errors:
            parts = [f"{k}={v}" for k, v in sub_errors.items()]
            all_warnings.append(
                ValidationError(
                    file_path="",
                    message=f"Static analysis breakdown: {', '.join(parts)}",
                    code="STATIC_ANALYSIS_BREAKDOWN",
                    severity="warning",
                )
            )

        return ValidationResult(
            passed=len(all_errors) == 0,
            errors=all_errors,
            warnings=all_warnings,
            duration_ms=total_duration,
        )

    async def run_tests(self, project_root: str) -> ValidationResult:
        """Layer 3: 运行测试套件。"""
        rv = self._get_runtime_validator()
        if rv is None:
            return ValidationResult(passed=True)
        result = await rv.run_tests()
        result.output = rv.last_test_output or ""
        return result

    async def run_runtime_check(self, file_path: str) -> ValidationResult:
        """运行时验证（单个文件降级检查）。"""
        rv = self._get_runtime_validator()
        if rv is None:
            return ValidationResult(passed=True)
        return await rv.run_fallback_check([file_path])
