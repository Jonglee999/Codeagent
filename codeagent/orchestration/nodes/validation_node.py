"""ValidationNode — 三层验证编排节点（Phase 4.A.4 升级版）。

编排语法检查、静态分析（StaticAnalyzer）、运行时验证（RuntimeValidator）
三层验证流程，集成错误分析和修复建议。
"""

from __future__ import annotations

import difflib
import logging
import time
from pathlib import Path
from typing import Any

from codeagent.gateway.validation_gateway import (
    IValidationGateway,
    ValidationError,
    ValidationResult,
)
from codeagent.tracing import trace_node
from codeagent.interaction.api.metrics import observe_validation
from codeagent.orchestration.state import AgentState
from codeagent.validation.error_analyzer import ErrorAnalyzer, FixSuggestion
from codeagent.validation.runtime_validator import RuntimeValidator
from codeagent.validation.static_analyzer import StaticAnalyzer

logger = logging.getLogger(__name__)


def benchmark_runtime_environment_unavailable(
    state: AgentState,
    results: list[ValidationResult],
) -> bool:
    """Recognize benchmark test startup failures caused by missing tooling.

    The official SWE-bench image is authoritative. The generic local sandbox
    intentionally lacks some upstream test dependencies, so collection/setup
    failures must not be treated as evidence that the candidate patch is wrong.
    """

    if not state.benchmark_instance_id or len(results) != 3:
        return False
    if not all(result.passed for result in results[:2]) or results[2].passed:
        return False
    runtime = results[2]
    evidence = "\n".join([
        runtime.output or "",
        *(error.message for error in (runtime.errors or [])),
    ]).lower()
    missing_module = (
        "modulenotfounderror" in evidence or "no module named" in evidence
    )
    before_tests = any(marker in evidence for marker in (
        "error collecting",
        "collected 0 items",
        "while loading conftest",
    ))
    missing_runtime = any(marker in evidence for marker in (
        "command not found",
        "is not recognized as an internal or external command",
        "could not find a version that satisfies the requirement",
    ))
    return (missing_module and before_tests) or missing_runtime


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

    @trace_node("validation")
    async def __call__(self, state: AgentState) -> dict[str, Any]:
        start_time = time.monotonic()
        file_paths = self._collect_file_paths(state)

        # 三层验证
        syntax_result = await self._run_syntax_layer(file_paths)
        static_result = await self._run_static_layer(file_paths)
        static_result = self._filter_pre_existing_benchmark_static_errors(
            state, static_result
        )
        layers: list[tuple[str, ValidationResult]] = [
            ("syntax", syntax_result),
            ("static_analysis", static_result),
        ]

        # Layer 3: runtime — 分离结果和原始输出用于错误分析
        runtime_result, runtime_output = await self._run_runtime_layer(
            file_paths, state
        )
        layers.append(("runtime", runtime_result))

        # Prometheus 验证指标记录 (non-blocking)
        _layer_short = {"syntax": "syntax", "static_analysis": "static", "runtime": "runtime"}
        for layer_name, result in layers:
            observe_validation(_layer_short.get(layer_name, layer_name), result.passed)

        validation_results = [r for _, r in layers]
        layers_status = {k: r.passed for k, r in layers}
        validation_degraded = benchmark_runtime_environment_unavailable(
            state, validation_results
        )

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

        # Phase 7.6: 记录验证结果到轨迹
        trajectory_steps = list(state.trajectory_steps)
        trajectory_steps.append({
            "node_name": "validation",
            "step_type": "validation",
            "file_count": len(file_paths),
            "layers_passed": sum(1 for v in layers_status.values() if v),
            "layers_total": len(layers_status),
            "validation_passed": all(r.passed for _, r in layers),
            "duration_ms": int(total_duration),
            "fix_suggestion_count": len(fix_suggestions),
        })

        warnings = list(state.warnings)
        if validation_degraded:
            warning = (
                "Local benchmark validation was unavailable because a required "
                "test dependency or runtime was missing; official evaluation "
                "remains authoritative."
            )
            if warning not in warnings:
                warnings.append(warning)

        return {
            "validation_results": validation_results,
            "validation_state": (
                "validation_degraded"
                if validation_degraded
                else "validation_passed"
                if all(result.passed for result in validation_results)
                else "validation_failed"
            ),
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
            "trajectory_steps": trajectory_steps,
            "warnings": warnings,
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
                and entry.get("tool_name") in {"write_file", "apply_patch"}
            ):
                fp = entry.get("arguments", {}).get("file_path", "")
                if fp and fp not in seen:
                    file_paths.append(fp)
                    seen.add(fp)

        return file_paths

    @staticmethod
    def _filter_pre_existing_benchmark_static_errors(
        state: AgentState,
        result: ValidationResult,
    ) -> ValidationResult:
        """Do not fail a benchmark patch for findings on unchanged source lines."""

        if not state.benchmark_instance_id or result.passed or not result.errors:
            return result

        changed_lines: dict[Path, set[int]] = {}
        seen: set[Path] = set()
        root = Path(state.project_root)
        for change in state.accumulated_changes:
            raw_path = str(change.get("file_path", ""))
            if not raw_path or "original_content" not in change:
                continue
            path = Path(raw_path)
            if not path.is_absolute():
                path = root / path
            path = path.resolve()
            if path in seen or not path.is_file():
                continue
            seen.add(path)
            original = change.get("original_content")
            current_lines = path.read_text(encoding="utf-8").splitlines()
            if original is None:
                changed_lines[path] = set(range(1, len(current_lines) + 1))
                continue
            original_lines = str(original).splitlines()
            lines: set[int] = set()
            matcher = difflib.SequenceMatcher(None, original_lines, current_lines)
            for tag, _old_start, _old_end, new_start, new_end in matcher.get_opcodes():
                if tag != "equal":
                    lines.update(range(new_start + 1, new_end + 1))
            changed_lines[path] = lines

        if not changed_lines:
            return result

        active: list[ValidationError] = []
        pre_existing: list[ValidationError] = []
        for error in result.errors:
            error_path = Path(error.file_path)
            if not error_path.is_absolute():
                error_path = root / error_path
            lines = changed_lines.get(error_path.resolve())
            if lines is not None and error.line > 0 and error.line not in lines:
                pre_existing.append(error)
            else:
                active.append(error)

        if not pre_existing:
            return result
        warning = ValidationError(
            file_path="",
            message=(
                f"Ignored {len(pre_existing)} pre-existing static finding(s) on "
                "unchanged benchmark lines"
            ),
            code="PRE_EXISTING_STATIC",
            severity="warning",
        )
        return ValidationResult(
            passed=not active,
            errors=active,
            warnings=[*result.warnings, warning],
            duration_ms=result.duration_ms,
            sandboxed=result.sandboxed,
            output=result.output,
        )

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
            benchmark_targets = [
                *state.benchmark_fail_to_pass,
                *state.benchmark_pass_to_pass,
            ]
            return await self._run_full_runtime_validation(
                file_paths,
                benchmark_targets or file_paths,
            )

        # 回退：使用 gateway
        all_errors: list[Any] = []
        all_warnings: list[Any] = []
        outputs: list[str] = []
        all_passed = True
        total_duration = 0.0
        for fp in file_paths:
            try:
                result = await self._gateway.run_runtime_check(fp)
                total_duration += result.duration_ms
                all_errors.extend(result.errors)
                all_warnings.extend(result.warnings)
                if result.output:
                    outputs.append(result.output)
                if not result.passed:
                    all_passed = False
                    if not result.errors:
                        all_errors.append(
                            ValidationError(
                                file_path=fp,
                                message=(
                                    result.output
                                    or "Runtime validation failed without structured diagnostics"
                                ),
                                severity="error",
                            )
                        )
            except Exception as e:
                logger.warning("Runtime check failed for %s: %s", fp, e)
                all_passed = False
                all_errors.append(
                    ValidationError(
                        file_path=fp,
                        message=f"Runtime check error: {e}",
                        severity="error",
                    )
                )

        return (
            ValidationResult(
                passed=all_passed and len(all_errors) == 0,
                errors=all_errors,
                warnings=all_warnings,
                duration_ms=total_duration,
                output="\n".join(outputs),
            ),
            "\n".join(outputs) or None,
        )

    async def _run_full_runtime_validation(
        self,
        file_paths: list[str],
        test_targets: list[str] | None = None,
    ) -> tuple[ValidationResult, str | None]:
        """完整运行时验证：先尝试测试，再降级检查。"""
        # a. 先尝试运行测试
        test_result = await self._runtime_validator.run_tests(  # type: ignore[misc]
            test_targets=test_targets or file_paths
        )

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
        test_result.output = runtime_output or ""
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
