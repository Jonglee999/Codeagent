"""ValidationNode — 三层验证编排节点（Phase 4.A.4 升级版）。

编排语法检查、静态分析（StaticAnalyzer）、运行时验证（RuntimeValidator）
三层验证流程，集成错误分析和修复建议。
"""

from __future__ import annotations

import logging
import time
from typing import Any

from codeagent.gateway.validation_gateway import (
    IValidationGateway,
    ValidationError,
    ValidationResult,
)
from codeagent.orchestration.state import AgentState
from codeagent.validation.error_analyzer import ErrorAnalyzer, FixSuggestion
from codeagent.validation.runtime_validator import RuntimeValidator
from codeagent.validation.static_analyzer import StaticAnalyzer

logger = logging.getLogger(__name__)


class ValidationNode:
    """验证节点 — 三层验证编排。

    执行流程：
    1. Layer 1: 语法验证 (SyntaxValidator)
    2. Layer 2: 静态分析 (StaticAnalyzer.run_all → Lint + TypeCheck + Security)
    3. Layer 3: 运行时验证 (RuntimeValidator → tests 或 fallback check)
    4. 错误分析 (ErrorAnalyzer → FixSuggestion + pre_existing 标记)

    每层失败后仍继续执行下一层（收集尽可能多的信息），
    但最终是否通过由所有层综合决定。
    """

    def __init__(
        self,
        validation_gateway: IValidationGateway,
        static_analyzer: StaticAnalyzer | None = None,
        runtime_validator: RuntimeValidator | None = None,
        error_analyzer: ErrorAnalyzer | None = None,
    ) -> None:
        """初始化 ValidationNode。

        Args:
            validation_gateway: 验证 Gateway（始终需要，用于语法检查）
            static_analyzer: StaticAnalyzer 实例（可选，提供后替代简单 lint）
            runtime_validator: RuntimeValidator 实例（可选，提供后替代简单 runtime check）
            error_analyzer: ErrorAnalyzer 实例（可选，默认创建新实例）
        """
        self._gateway = validation_gateway
        self._static_analyzer = static_analyzer
        self._runtime_validator = runtime_validator
        self._error_analyzer = error_analyzer or ErrorAnalyzer()

    # ── 主入口 ─────────────────────────────────────────────────────

    async def __call__(self, state: AgentState) -> dict[str, Any]:
        start_time = time.monotonic()
        file_paths = self._collect_file_paths(state)

        # 三层验证
        layers: list[tuple[str, ValidationResult]] = [
            ("syntax", await self._run_syntax_layer(file_paths)),
            ("static_analysis", await self._run_static_layer(file_paths)),
        ]

        # Layer 3: runtime — 分离结果和原始输出用于错误分析
        runtime_result, runtime_output = await self._run_runtime_layer(
            file_paths, state
        )
        layers.append(("runtime", runtime_result))

        validation_results = [r for _, r in layers]
        layers_status = {k: r.passed for k, r in layers}

        # FixSuggestion 分析
        fix_suggestions = self._collect_fix_suggestions(
            runtime_result, runtime_output, file_paths
        )

        report = self._build_report(layers_status, layers, fix_suggestions)

        total_duration = (time.monotonic() - start_time) * 1000
        logger.info(
            "ValidationNode completed in %.1fms (%d files, %d/%d layers passed)",
            total_duration, len(file_paths),
            sum(1 for v in layers_status.values() if v), len(layers_status),
        )

        return {
            "validation_results": validation_results,
            "fix_suggestions": fix_suggestions,
            "execution_log": [
                *state.execution_log,
                {
                    "type": "validation_report",
                    "report": report,
                    "fix_suggestion_count": len(fix_suggestions),
                    "timestamp": time.time(),
                },
            ],
        }

    # ── 文件收集 ──────────────────────────────────────────────────

    def _collect_file_paths(self, state: AgentState) -> list[str]:
        """从 accumulated_changes 和 execution_log 中提取修改过的文件路径。"""
        file_paths: list[str] = []
        seen: set[str] = set()

        for change in state.accumulated_changes:
            fp = change.get("file_path", "")
            if fp and fp not in seen:
                file_paths.append(fp)
                seen.add(fp)

        for entry in state.execution_log:
            if (
                entry.get("type") == "tool_call"
                and entry.get("tool_name") == "write_file"
            ):
                fp = entry.get("arguments", {}).get("file_path", "")
                if fp and fp not in seen:
                    file_paths.append(fp)
                    seen.add(fp)

        return file_paths

    # ── Layer 1: 语法检查 ─────────────────────────────────────────

    async def _run_syntax_layer(self, file_paths: list[str]) -> ValidationResult:
        """Layer 1: 对所有修改过的文件运行语法检查。"""
        if not file_paths:
            return ValidationResult(passed=True, duration_ms=0.0)

        all_errors: list[Any] = []
        all_warnings: list[Any] = []
        total_duration = 0.0

        for fp in file_paths:
            try:
                result = await self._gateway.run_syntax_check(fp)
                total_duration += result.duration_ms
                all_errors.extend(result.errors)
                all_warnings.extend(result.warnings)
            except Exception as e:
                logger.warning("Syntax check failed for %s: %s", fp, e)
                all_errors.append(
                    ValidationError(
                        file_path=fp,
                        message=f"Syntax check error: {e}",
                        severity="error",
                    )
                )

        return ValidationResult(
            passed=len(all_errors) == 0,
            errors=all_errors,
            warnings=all_warnings,
            duration_ms=total_duration,
        )

    # ── Layer 2: 静态分析 ─────────────────────────────────────────

    async def _run_static_layer(self, file_paths: list[str]) -> ValidationResult:
        """Layer 2: 静态分析 — 整合 StaticAnalyzer.run_all()。

        当 static_analyzer 可用时，运行完整的 Lint + TypeCheck + Security 扫描。
        否则回退到 gateway.run_lint()。
        """
        if not file_paths:
            return ValidationResult(passed=True, duration_ms=0.0)

        if self._static_analyzer:
            return await self._run_full_static_analysis(file_paths)

        # 回退：使用 gateway
        try:
            return await self._gateway.run_lint(file_paths)
        except Exception as e:
            logger.warning("Lint check failed: %s", e)
            return ValidationResult(
                passed=False,
                errors=[
                    ValidationError(
                        file_path="",
                        message=f"Static analysis error: {e}",
                        severity="error",
                    )
                ],
            )

    async def _run_full_static_analysis(
        self, file_paths: list[str]
    ) -> ValidationResult:
        """完整静态分析：lint + typecheck + security。

        合并三个子结果为一个 ValidationResult，同时保留子结果明细。
        """
        results = await self._static_analyzer.run_all(file_paths)  # type: ignore[misc]

        all_errors: list[ValidationError] = []
        all_warnings: list[ValidationError] = []
        total_duration = 0.0
        sub_category_errors: dict[str, int] = {}

        labels = ["lint", "typecheck", "security"]
        for label, r in zip(labels, results):
            all_errors.extend(r.errors)
            all_warnings.extend(r.warnings)
            total_duration += r.duration_ms
            if not r.passed:
                sub_category_errors[label] = len(r.errors)

        # 在 warnings 中嵌入子结果摘要
        if sub_category_errors:
            parts = [f"{k}={v}" for k, v in sub_category_errors.items()]
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

    # ── Layer 3: 运行时验证 ──────────────────────────────────────

    async def _run_runtime_layer(
        self,
        file_paths: list[str],
        state: AgentState,
    ) -> tuple[ValidationResult, str | None]:
        """Layer 3: 运行时验证。

        当 runtime_validator 可用时:
          a. 先尝试 run_tests()
          b. 若无测试框架，降级为 run_fallback_check()
        否则回退到 gateway.run_runtime_check()。

        Returns:
            tuple: (ValidationResult, 原始输出用于错误分析)
        """
        if not file_paths:
            return ValidationResult(passed=True, duration_ms=0.0), None

        if self._runtime_validator:
            return await self._run_full_runtime_validation(file_paths)

        # 回退：使用 gateway
        all_errors: list[Any] = []
        total_duration = 0.0
        for fp in file_paths:
            try:
                result = await self._gateway.run_runtime_check(fp)
                total_duration += result.duration_ms
                all_errors.extend(result.errors)
            except Exception as e:
                logger.warning("Runtime check failed for %s: %s", fp, e)
                all_errors.append(
                    ValidationError(
                        file_path=fp,
                        message=f"Runtime check error: {e}",
                        severity="error",
                    )
                )

        return (
            ValidationResult(
                passed=len(all_errors) == 0,
                errors=all_errors,
                duration_ms=total_duration,
            ),
            None,
        )

    async def _run_full_runtime_validation(
        self,
        file_paths: list[str],
    ) -> tuple[ValidationResult, str | None]:
        """完整运行时验证：先尝试测试，再降级检查。"""
        # a. 先尝试运行测试
        test_result = await self._runtime_validator.run_tests()  # type: ignore[misc]

        # 获取原始测试输出用于 FixSuggestion 分析
        runtime_output = self._runtime_validator.last_test_output  # type: ignore[misc]

        # b. 无测试框架 → 降级检查
        has_no_framework = any(
            "NO_TEST_FRAMEWORK" in w.code
            for w in test_result.warnings
        )
        if has_no_framework:
            fallback_result = await self._runtime_validator.run_fallback_check(  # type: ignore[misc]
                file_paths
            )
            return fallback_result, None

        # c. 测试已执行，返回结果及原始输出用于 ErrorAnalyzer
        return test_result, runtime_output

    # ── FixSuggestion 分析 ────────────────────────────────────────

    def _collect_fix_suggestions(
        self,
        runtime_result: ValidationResult,
        runtime_output: str | None,
        file_paths: list[str],
    ) -> list[FixSuggestion]:
        """从运行时结果收集修复建议。

        当 ErrorAnalyzer 可用且运行时验证失败时，尝试分析失败原因。
        """
        if not runtime_result.passed and runtime_output:
            return self._error_analyzer.analyze(
                runtime_output, file_paths
            )
        return []

    # ── 报告生成 ──────────────────────────────────────────────────

    def _build_report(
        self,
        layers_status: dict[str, bool],
        layers: list[tuple[str, ValidationResult]],
        fix_suggestions: list[FixSuggestion] | None = None,
    ) -> str:
        """生成 Markdown 格式验证报告（含 FixSuggestion 和 pre_existing 标记）。"""
        display_names = {
            "syntax": "Syntax",
            "static_analysis": "Static Analysis",
            "runtime": "Runtime Tests",
        }

        lines = ["### Validation Report", ""]

        for key, result in layers:
            display = display_names.get(key, key)
            passed = layers_status.get(key, False)
            status_str = "✅ PASSED" if passed else "❌ FAILED"

            parts = []
            if result.errors:
                parts.append(f"{len(result.errors)} error(s)")
            if result.warnings:
                parts.append(f"{len(result.warnings)} warning(s)")

            line = f"- {display}: {status_str}"
            if parts:
                line += f" ({', '.join(parts)})"
            lines.append(line)

            # 列出前 10 个错误
            for err in result.errors[:10]:
                loc = f"{err.file_path}" if err.file_path else ""
                if hasattr(err, "line") and err.line:
                    loc += f":{err.line}"
                msg = err.message if hasattr(err, "message") else str(err)
                if loc:
                    lines.append(f"  - {loc}: {msg}")
                else:
                    lines.append(f"  - {msg}")

            if len(result.errors) > 10:
                lines.append(f"  - ... and {len(result.errors) - 10} more")

            lines.append("")

        # FixSuggestion 部分
        if fix_suggestions:
            lines.append("### Fix Suggestions")
            for i, s in enumerate(fix_suggestions, 1):
                marker = " 🔄 pre-existing" if s.is_pre_existing else " 🆕 new"
                lines.append(f"{i}. **{s.error_summary}**{marker}")
                lines.append(f"   - Files: {', '.join(s.affected_files)}")
                lines.append(f"   - Cause: {s.likely_cause}")
                lines.append(f"   - Fix: {s.suggested_fix}")
            lines.append("")

        failed = sum(1 for v in layers_status.values() if not v)
        total = len(layers_status)
        lines.append(f"**Summary:** {failed}/{total} layer(s) failed")

        return "\n".join(lines)
