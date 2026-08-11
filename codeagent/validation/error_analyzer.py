"""ErrorAnalyzer — 测试错误分析器。

参考 SRS §6.3.3。
解析 pytest 测试输出，提取结构化错误信息，生成修复建议。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TestFailure:
    """单个测试失败的详细信息。"""

    test_name: str
    file_path: str
    line: int
    error_type: str
    message: str
    traceback_lines: list[str] = field(default_factory=list)


@dataclass
class FixSuggestion:
    """修复建议。

    Attributes:
        error_summary: 错误摘要
        affected_files: 相关文件列表
        likely_cause: 可能原因
        suggested_fix: 建议修复方向
        is_pre_existing: 是否为已有错误（非 Agent 引入）
    """

    error_summary: str
    affected_files: list[str]
    likely_cause: str
    suggested_fix: str
    is_pre_existing: bool = False


class ErrorAnalyzer:
    """错误分析器 — 解析 traceback，定位失败根因。"""

    # 错误类型 → 可能原因
    _CAUSE_MAP: dict[str, str] = {
        "AssertionError": "Test assertion failed — expected value does not match actual value",
        "ImportError": "Missing import — a required module or symbol could not be found",
        "ModuleNotFoundError": "Missing module — a required module is not installed or not in path",
        "TypeError": "Type mismatch — function called with wrong argument type or count",
        "NameError": "Name not defined — variable or function name not found in scope",
        "AttributeError": "Attribute not found — object does not have the accessed attribute",
        "ValueError": "Invalid value — function received correct type but unexpected value",
        "KeyError": "Missing dictionary key — accessing a key that does not exist",
        "IndexError": "List index out of range — accessing an index that does not exist",
        "SyntaxError": "Syntax error — invalid Python syntax",
        "ZeroDivisionError": "Division by zero — mathematical operation caused division by zero",
        "StopIteration": "Iterator exhausted — next() called on an empty iterator",
        "RuntimeError": "Runtime error — unexpected runtime condition",
        "TimeoutError": "Operation timed out — code took too long to complete",
    }

    # 错误类型 → 建议修复
    _FIX_MAP: dict[str, str] = {
        "AssertionError": "Review the assertion logic and verify expected vs actual values",
        "ImportError": "Check that the module is installed and import paths are correct",
        "ModuleNotFoundError": "Verify the module is installed or add it to dependencies",
        "TypeError": "Check function signature and verify argument types and count",
        "NameError": "Verify the variable or function is defined before use",
        "AttributeError": "Verify the attribute or method exists on the object",
        "ValueError": "Add input validation or check the value being passed",
        "KeyError": "Use dict.get() or check key existence before access",
        "IndexError": "Verify list bounds or use a bounds check before access",
        "SyntaxError": "Review Python syntax at the indicated line",
        "ZeroDivisionError": "Add a zero check before the division operation",
        "StopIteration": "Use a for loop instead of manual next() calls",
        "RuntimeError": "Review the runtime logic and add proper error handling",
        "TimeoutError": "Optimize performance or increase timeout value",
    }

    def analyze(
        self,
        test_output: str,
        changed_files: list[str],
    ) -> list[FixSuggestion]:
        """解析 pytest 输出，提取结构化错误信息，生成修复建议。

        Args:
            test_output: pytest 原始输出文本
            changed_files: Agent 本次修改的文件列表

        Returns:
            list[FixSuggestion]: 修复建议列表
        """
        failures = self._parse_traceback(test_output)
        return self._map_to_changed_files(failures, changed_files)

    # ── Traceback 解析 ──────────────────────────────────────────

    def _parse_traceback(self, output: str) -> list[TestFailure]:
        """解析 pytest 输出中的测试失败信息。

        Args:
            output: pytest 原始输出

        Returns:
            list[TestFailure]: 解析出的失败信息列表
        """
        failures: list[TestFailure] = []

        # 1. 按 ____ test_name ____ 分隔符拆分失败块
        failures.extend(self._parse_failure_blocks(output))

        # 2. 如果块解析失败，回退到从 FAILED 摘要行解析
        if not failures:
            failures.extend(self._parse_failed_summary(output))

        return failures

    def _parse_failure_blocks(self, output: str) -> list[TestFailure]:
        """从 ____ test_name ____ 格式块中解析失败信息。"""
        failures: list[TestFailure] = []

        # 分隔符: 至少 3 个下划线，中间是测试名称，再至少 3 个下划线
        header_re = re.compile(
            r'^_{3,}\s*(.+?)\s*_{3,}\s*$', re.MULTILINE
        )
        parts = header_re.split(output)

        for i in range(1, len(parts), 2):
            if i + 1 >= len(parts):
                break

            test_name = parts[i].strip()
            block = parts[i + 1]

            # 跳过 FAILURES 标题
            if not test_name or test_name.upper() == "FAILURES":
                continue

            failure = self._parse_single_block(test_name, block)
            if failure:
                failures.append(failure)

        return failures

    def _parse_single_block(
        self, test_name: str, block: str
    ) -> TestFailure | None:
        """从单个失败块中提取错误信息。"""
        lines = block.strip().split("\n")

        error_type = ""
        message = ""
        file_path = ""
        line_no = 0

        # 1. 从 E  行提取错误类型和消息
        for line_text in lines:
            if line_text.startswith("E   "):
                content = line_text[4:].strip()
                ematch = re.match(
                    r"^(\w+(?:Error|Exception|Warning|StopIteration|Exit))"
                    r"(?::\s*(.*))?$",
                    content,
                )
                if ematch:
                    if not error_type:
                        error_type = ematch.group(1)
                    message = (ematch.group(2) or "").strip()
                elif not error_type and content.startswith("assert"):
                    error_type = "AssertionError"
                    if not message:
                        message = content
                elif not message:
                    message = content

        # 2. 从 file:line: 引用中提取文件路径和行号
        for line_text in reversed(lines):
            stripped = line_text.strip()
            if stripped.startswith("E "):
                continue
            match = re.search(r"^(.+?):(\d+):", stripped)
            if match:
                file_path = match.group(1)
                line_no = int(match.group(2))
                break

        # 3. 尝试从最后一行提取错误类型
        if not error_type:
            for line_text in lines:
                match = re.match(
                    r"^(.+?):(\d+):\s*"
                    r"(\w+(?:Error|Exception|Warning))\s*$",
                    line_text.strip(),
                )
                if match:
                    error_type = match.group(3)
                    if not file_path:
                        file_path = match.group(1)
                    break

        if not test_name:
            return None

        return TestFailure(
            test_name=test_name,
            file_path=file_path or "",
            line=line_no,
            error_type=error_type or "UnknownError",
            message=message or block.strip()[:200],
            traceback_lines=[
                line.rstrip()
                for line in lines
                if not line.startswith("E   ")
            ],
        )

    def _parse_failed_summary(self, output: str) -> list[TestFailure]:
        """从 FAILED 摘要行解析失败信息。"""
        failures: list[TestFailure] = []

        pattern = re.compile(
            r"^FAILED\s+(.+?)\s+-\s+(.+)$", re.MULTILINE
        )
        for match in pattern.finditer(output):
            test_path = match.group(1).strip()
            error_info = match.group(2).strip()

            if "::" in test_path:
                file_path, test_name = test_path.rsplit("::", 1)
            else:
                file_path = test_path
                test_name = test_path

            ematch = re.match(
                r"^(\w+(?:Error|Exception|Warning|StopIteration|Exit))"
                r"(?::\s*(.*))?$",
                error_info,
            )
            if ematch:
                error_type = ematch.group(1)
                msg = (ematch.group(2) or "").strip()
            else:
                error_type = "UnknownError"
                msg = error_info

            failures.append(
                TestFailure(
                    test_name=test_name,
                    file_path=file_path,
                    line=0,
                    error_type=error_type,
                    message=msg,
                )
            )

        return failures

    # ── 错误归属 ────────────────────────────────────────────────

    def _map_to_changed_files(
        self,
        failures: list[TestFailure],
        changed_files: list[str],
    ) -> list[FixSuggestion]:
        """将失败映射到 Agent 修改的文件。

        判断逻辑:
        - 失败文件的绝对路径在 changed_files 中 → Agent 引入的错误
        - 失败文件的绝对路径不在 changed_files 中 → 已有错误
        - 也检查 traceback 中是否涉及 changed_files
        """
        resolved_changed = {
            str(Path(f).resolve()) for f in changed_files
        }

        suggestions: list[FixSuggestion] = []
        for failure in failures:
            is_new = False
            affected = set()

            # 检查主失败文件
            if failure.file_path:
                resolved = str(Path(failure.file_path).resolve())
                affected.add(resolved)
                if resolved in resolved_changed:
                    is_new = True

            # 检查 traceback 中的文件
            if not is_new:
                for tb_line in failure.traceback_lines:
                    tb_match = re.match(r"^(.+?):(\d+):", tb_line.strip())
                    if tb_match:
                        tb_path = str(Path(tb_match.group(1)).resolve())
                        affected.add(tb_path)
                        if tb_path in resolved_changed:
                            is_new = True
                            break

            suggestions.append(
                FixSuggestion(
                    error_summary=(
                        f"{failure.error_type}: {failure.message}"
                        if failure.message
                        else failure.error_type
                    ),
                    affected_files=sorted(affected),
                    likely_cause=self._infer_cause(failure),
                    suggested_fix=self._generate_suggested_fix(failure),
                    is_pre_existing=not is_new,
                )
            )

        return suggestions

    # ── 原因推断与修复建议 ───────────────────────────────────────

    def _infer_cause(self, failure: TestFailure) -> str:
        """推断错误的可能原因。"""
        return self._CAUSE_MAP.get(
            failure.error_type,
            f"Unhandled error of type {failure.error_type}: "
            f"{failure.message}",
        )

    def _generate_suggested_fix(self, failure: TestFailure) -> str:
        """生成建议修复方向。"""
        return self._FIX_MAP.get(
            failure.error_type,
            "Review the error and fix the issue at the indicated location",
        )

    # ── 摘要提取 ────────────────────────────────────────────────

    def extract_summary(self, output: str) -> dict[str, Any]:
        """从 pytest 输出中提取测试摘要信息。

        Returns:
            dict: {total, passed, failed, errors, failed_tests}
        """
        summary: dict[str, Any] = {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "errors": 0,
            "failed_tests": [],
        }

        # 提取数值: === X passed, Y failed, Z errors in Ws ===
        summary_line_re = re.compile(
            r"={3,}\s*(.*?)\s*={3,}", re.MULTILINE
        )
        for match in summary_line_re.finditer(output):
            line = match.group(1)

            pm = re.search(r"(\d+)\s+passed", line)
            fm = re.search(r"(\d+)\s+failed", line)
            em = re.search(r"(\d+)\s+errors?", line)

            if pm:
                summary["passed"] = int(pm.group(1))
            if fm:
                summary["failed"] = int(fm.group(1))
            if em:
                summary["errors"] = int(em.group(1))

        summary["total"] = (
            summary["passed"] + summary["failed"] + summary["errors"]
        )

        # 提取 FAILED 测试列表
        failed_re = re.compile(
            r"^FAILED\s+(.+?::.+?)(?:\s+-|\s*$)", re.MULTILINE
        )
        summary["failed_tests"] = [
            m.group(1) for m in failed_re.finditer(output)
        ]

        return summary
