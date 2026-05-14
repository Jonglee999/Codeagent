"""ValidationNode — 三层验证编排节点（Phase 1b）。

编排语法检查、静态分析、运行时验证三层验证流程。
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

logger = logging.getLogger(__name__)


class ValidationNode:
    """验证节点 — 三层验证编排。

    执行流程：
    1. Layer 1: 语法验证 (Syntax Check)
    2. Layer 2: 静态分析 (Lint)
    3. Layer 3: 运行时验证 (Runtime Check)

    每层失败后仍继续执行下一层（收集尽可能多的信息），
    但最终是否通过由所有层综合决定。
    """

    def __init__(self, validation_gateway: IValidationGateway) -> None:
        self._gateway = validation_gateway

    async def __call__(self, state: AgentState) -> dict[str, Any]:
        start_time = time.monotonic()
        file_paths = self._collect_file_paths(state)

        # 三层验证
        layers: list[tuple[str, ValidationResult]] = [
            ("syntax", await self._run_syntax_layer(file_paths)),
            ("static_analysis", await self._run_lint_layer(file_paths)),
            ("runtime", await self._run_runtime_layer(file_paths)),
        ]

        validation_results = [r for _, r in layers]
        layers_status = {k: r.passed for k, r in layers}

        report = self._build_report(layers_status, layers)

        total_duration = (time.monotonic() - start_time) * 1000
        logger.info(
            "ValidationNode completed in %.1fms (%d files, %d/%d layers passed)",
            total_duration, len(file_paths),
            sum(1 for v in layers_status.values() if v), len(layers_status),
        )

        return {
            "validation_results": validation_results,
            "execution_log": [
                *state.execution_log,
                {
                    "type": "validation_report",
                    "report": report,
                    "timestamp": time.time(),
                },
            ],
        }

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

    async def _run_lint_layer(self, file_paths: list[str]) -> ValidationResult:
        """Layer 2: 对修改过的文件运行静态分析/lint。"""
        if not file_paths:
            return ValidationResult(passed=True, duration_ms=0.0)

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

    async def _run_runtime_layer(self, file_paths: list[str]) -> ValidationResult:
        """Layer 3: 运行时验证（对每个文件做 import/执行检查）。"""
        if not file_paths:
            return ValidationResult(passed=True, duration_ms=0.0)

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

        return ValidationResult(
            passed=len(all_errors) == 0,
            errors=all_errors,
            duration_ms=total_duration,
        )

    def _build_report(
        self,
        layers_status: dict[str, bool],
        layers: list[tuple[str, ValidationResult]],
    ) -> str:
        """生成 Markdown 格式验证报告。"""
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

        failed = sum(1 for v in layers_status.values() if not v)
        total = len(layers_status)
        lines.append(f"**Summary:** {failed}/{total} layer(s) failed")

        return "\n".join(lines)


