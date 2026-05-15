"""RuntimeValidator — 运行时验证器（测试执行 + 降级检查）。

参考 SRS §6.3.3。
结合 TestDetector 和 ErrorAnalyzer，执行测试并分析失败原因。
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

from codeagent.gateway.validation_gateway import ValidationError, ValidationResult
from codeagent.validation.error_analyzer import ErrorAnalyzer, FixSuggestion
from codeagent.validation.test_detector import TestDetector

# 子进程超时（秒）
_DEFAULT_TEST_TIMEOUT = 300
_FALLBACK_TIMEOUT = 60


def _run_subprocess(
    cmd: list[str],
    timeout: int = _DEFAULT_TEST_TIMEOUT,
    cwd: str | None = None,
) -> subprocess.CompletedProcess:
    """执行子进程，自动处理 Windows 编码问题。

    Windows 中文环境下可能输出 GBK 编码文本，
    使用 bytes 手动解码避免 UnicodeDecodeError。
    """
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=False,
            timeout=timeout,
            cwd=cwd,
        )
        for enc in ("utf-8", "gbk", "gb18030"):
            try:
                stdout = (
                    proc.stdout.decode(enc, errors="replace")
                    if proc.stdout
                    else ""
                )
                stderr = (
                    proc.stderr.decode(enc, errors="replace")
                    if proc.stderr
                    else ""
                )
                break
            except (UnicodeDecodeError, LookupError):
                continue
        else:
            stdout = (
                proc.stdout.decode("utf-8", errors="replace")
                if proc.stdout
                else ""
            )
            stderr = (
                proc.stderr.decode("utf-8", errors="replace")
                if proc.stderr
                else ""
            )

        proc.stdout = stdout  # type: ignore[assignment]
        proc.stderr = stderr  # type: ignore[assignment]
        return proc
    except FileNotFoundError:
        raise
    except OSError:
        raise


class RuntimeValidator:
    """运行时验证器 — 测试执行 + 结果分析。

    流程:
    1. 委托 TestDetector 检测测试框架
    2. 执行测试命令（pytest / unittest）
    3. 委托 ErrorAnalyzer 分析失败原因
    4. 无测试框架时执行降级检查
    """

    def __init__(self, project_root: str) -> None:
        """初始化 RuntimeValidator。

        Args:
            project_root: 项目根目录路径
        """
        self._project_root = Path(project_root).resolve()
        self._test_detector = TestDetector(project_root)
        self._error_analyzer = ErrorAnalyzer()
        self._last_raw_output: str | None = None

    # ── 测试执行 ────────────────────────────────────────────────

    async def run_tests(
        self, timeout: int = _DEFAULT_TEST_TIMEOUT
    ) -> ValidationResult:
        """运行测试套件。

        Args:
            timeout: 超时秒数，默认 300

        Returns:
            ValidationResult: 测试执行结果
        """
        start = time.monotonic()

        info = await self._test_detector.detect()

        if not info.has_tests:
            return ValidationResult(
                passed=True,
                warnings=[
                    ValidationError(
                        file_path="",
                        message=(
                            "No test framework detected — "
                            "tests skipped"
                        ),
                        code="NO_TEST_FRAMEWORK",
                        severity="warning",
                    )
                ],
                duration_ms=(time.monotonic() - start) * 1000,
            )

        cmd = info.test_command.split()
        if not cmd:
            return ValidationResult(
                passed=True,
                warnings=[
                    ValidationError(
                        file_path="",
                        message="Empty test command — tests skipped",
                        code="EMPTY_TEST_COMMAND",
                        severity="warning",
                    )
                ],
                duration_ms=(time.monotonic() - start) * 1000,
            )

        try:
            proc = _run_subprocess(
                cmd, timeout=timeout, cwd=str(self._project_root)
            )

            output = str(proc.stdout) + "\n" + str(proc.stderr)
            self._last_raw_output = output
            passed = proc.returncode == 0

            if not passed:
                errors = self._parse_test_errors(output)
            else:
                errors = []

            # 提取摘要统计
            summary = self._error_analyzer.extract_summary(output)

            return ValidationResult(
                passed=passed,
                errors=errors,
                duration_ms=(time.monotonic() - start) * 1000,
            )

        except subprocess.TimeoutExpired:
            return ValidationResult(
                passed=False,
                errors=[
                    ValidationError(
                        file_path="",
                        message=(
                            f"Test execution timed out after "
                            f"{timeout}s"
                        ),
                        code="TEST_TIMEOUT",
                    )
                ],
                duration_ms=(time.monotonic() - start) * 1000,
            )
        except FileNotFoundError:
            return ValidationResult(
                passed=False,
                errors=[
                    ValidationError(
                        file_path="",
                        message=(
                            "Test command not found — "
                            "is pytest/unittest installed?"
                        ),
                        code="TEST_COMMAND_NOT_FOUND",
                    )
                ],
                duration_ms=(time.monotonic() - start) * 1000,
            )
        except OSError as e:
            return ValidationResult(
                passed=False,
                errors=[
                    ValidationError(
                        file_path="",
                        message=f"Test execution error: {e}",
                        code="TEST_EXECUTION_ERROR",
                    )
                ],
                duration_ms=(time.monotonic() - start) * 1000,
            )

    def _parse_test_errors(
        self, output: str
    ) -> list[ValidationError]:
        """解析测试输出中的失败信息为 ValidationError 列表。"""
        failures = self._error_analyzer._parse_traceback(output)  # noqa: SLF001

        errors: list[ValidationError] = []
        for f in failures:
            errors.append(
                ValidationError(
                    file_path=f.file_path,
                    line=f.line,
                    message=(
                        f"{f.error_type}: {f.message}"
                        if f.message
                        else f.error_type
                    ),
                    code=f.error_type,
                )
            )

        # 如果没有解析到具体失败，但返回码非零，生成通用错误
        if not errors:
            summary = self._error_analyzer.extract_summary(output)
            if summary["failed_tests"]:
                for ft in summary["failed_tests"][:10]:
                    errors.append(
                        ValidationError(
                            file_path=ft.split("::")[0]
                            if "::" in ft
                            else ft,
                            message=f"Test failed: {ft}",
                            code="TEST_FAILURE",
                        )
                    )
            else:
                errors.append(
                    ValidationError(
                        file_path="",
                        message=(
                            "Tests failed with non-zero exit code"
                        ),
                        code="TEST_FAILURE",
                    )
                )

        return errors

    # ── 降级检查 ────────────────────────────────────────────────

    async def run_fallback_check(
        self, changed_files: list[str]
    ) -> ValidationResult:
        """无测试框架时的降级检查策略。

        对每个修改过的文件按优先级执行:
        a. 脚本模式: 有 __name__ == "__main__" → python file.py
        b. 导入模式: python -c "import module"
        c. 主入口: 有 main() 函数 → python file.py --help
        d. 不适用 → 跳过

        Args:
            changed_files: 本次修改的文件路径列表

        Returns:
            ValidationResult: 降级检查结果
        """
        start = time.monotonic()
        errors: list[ValidationError] = []
        warnings: list[ValidationError] = []
        checked_any = False

        for file_path in changed_files:
            if not file_path.endswith(".py"):
                continue

            abs_path = self._project_root / file_path
            if not abs_path.is_file():
                abs_path = Path(file_path).resolve()
                if not abs_path.is_file():
                    continue

            source = self._read_file_safe(abs_path)
            if source is None:
                continue

            # a. 脚本模式
            if self._has_main_block(source):
                checked_any = True
                result = self._try_run_script(
                    str(abs_path), file_path
                )
                if result:
                    errors.append(result)

            # b. 导入模式
            module_path = self._file_to_module(
                str(abs_path), file_path
            )
            if module_path:
                checked_any = True
                result = self._try_import_module(
                    module_path, file_path
                )
                if result:
                    errors.append(result)

            # c. 主入口
            if self._has_main_function(source):
                checked_any = True
                result = self._try_run_help(
                    str(abs_path), file_path
                )
                if result:
                    errors.append(result)

        if not checked_any:
            warnings.append(
                ValidationError(
                    file_path="",
                    message=(
                        "No runtime check available — "
                        "files have no executable entry point"
                    ),
                    code="NO_RUNTIME_CHECK",
                    severity="warning",
                )
            )

        return ValidationResult(
            passed=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            duration_ms=(time.monotonic() - start) * 1000,
        )

    # ── 降级检查辅助方法 ───────────────────────────────────────

    def _has_main_block(self, source: str) -> bool:
        """检查文件是否有 __name__ == "__main__" 入口。"""
        return (
            'if __name__ == "__main__"' in source
            or "if __name__ == '__main__'" in source
        )

    def _has_main_function(self, source: str) -> bool:
        """检查文件是否有 main() 函数定义。"""
        return "def main(" in source or "def main()" in source

    def _file_to_module(
        self, abs_path: str, file_path: str
    ) -> str | None:
        """将文件路径转换为模块导入路径。"""
        # 尝试基于项目根目录
        try:
            rel = Path(abs_path).relative_to(self._project_root)
            parts = list(rel.parts)
            if parts[-1] == "__init__.py":
                parts = parts[:-1]
            else:
                parts[-1] = parts[-1].replace(".py", "")
            return ".".join(parts) if parts else None
        except ValueError:
            pass

        # 尝试基于 file_path 参数
        p = Path(file_path)
        if p.suffix == ".py":
            module = str(p.with_suffix("")).replace(os.sep, ".")
            return module

        return None

    def _try_run_script(
        self, abs_path: str, file_path: str
    ) -> ValidationError | None:
        """尝试以脚本模式运行 Python 文件。"""
        try:
            proc = _run_subprocess(
                ["python", abs_path],
                timeout=_FALLBACK_TIMEOUT,
            )
            if proc.returncode != 0:
                stderr = str(proc.stderr).strip()
                return ValidationError(
                    file_path=file_path,
                    message=(
                        f"Script execution failed "
                        f"(exit {proc.returncode}): {stderr}"
                    ),
                    code="FALLBACK_SCRIPT_ERROR",
                    severity="error" if proc.returncode != 0 else "warning",
                )
        except subprocess.TimeoutExpired:
            return ValidationError(
                file_path=file_path,
                message="Script execution timed out",
                code="FALLBACK_TIMEOUT",
                severity="warning",
            )
        except (FileNotFoundError, OSError) as e:
            return ValidationError(
                file_path=file_path,
                message=f"Script execution error: {e}",
                code="FALLBACK_EXECUTION_ERROR",
                severity="warning",
            )
        return None

    def _try_import_module(
        self, module_path: str, file_path: str
    ) -> ValidationError | None:
        """尝试导入模块。"""
        try:
            proc = _run_subprocess(
                ["python", "-c", f"import {module_path}"],
                timeout=_FALLBACK_TIMEOUT,
            )
            if proc.returncode != 0:
                stderr = str(proc.stderr).strip()
                return ValidationError(
                    file_path=file_path,
                    message=(
                        f"Import check failed: {stderr}"
                    ),
                    code="FALLBACK_IMPORT_ERROR",
                    severity="error",
                )
        except subprocess.TimeoutExpired:
            return ValidationError(
                file_path=file_path,
                message="Import check timed out",
                code="FALLBACK_TIMEOUT",
                severity="warning",
            )
        except (FileNotFoundError, OSError) as e:
            return ValidationError(
                file_path=file_path,
                message=f"Import check error: {e}",
                code="FALLBACK_EXECUTION_ERROR",
                severity="warning",
            )
        return None

    def _try_run_help(
        self, abs_path: str, file_path: str
    ) -> ValidationError | None:
        """尝试运行 Python 文件 --help。"""
        try:
            proc = _run_subprocess(
                ["python", abs_path, "--help"],
                timeout=_FALLBACK_TIMEOUT,
            )
            if proc.returncode != 0:
                stderr = str(proc.stderr).strip()
                return ValidationError(
                    file_path=file_path,
                    message=(
                        f"Entry point check failed "
                        f"(exit {proc.returncode}): {stderr}"
                    ),
                    code="FALLBACK_HELP_ERROR",
                    severity="warning",
                )
        except subprocess.TimeoutExpired:
            return ValidationError(
                file_path=file_path,
                message="Entry point check timed out",
                code="FALLBACK_TIMEOUT",
                severity="warning",
            )
        except (FileNotFoundError, OSError) as e:
            return ValidationError(
                file_path=file_path,
                message=f"Entry point check error: {e}",
                code="FALLBACK_EXECUTION_ERROR",
                severity="warning",
            )
        return None

    # ── 测试输出获取 ────────────────────────────────────────────

    @property
    def last_test_output(self) -> str | None:
        """获取最近一次测试执行的原始输出（用于 ErrorAnalyzer 分析）。"""
        return self._last_raw_output

    # ── 失败分析 ────────────────────────────────────────────────

    async def analyze_failures(
        self,
        test_output: str,
        changed_files: list[str],
    ) -> list[FixSuggestion]:
        """分析测试失败，生成修复建议。

        委托给 ErrorAnalyzer。

        Args:
            test_output: pytest 原始输出
            changed_files: Agent 本次修改的文件列表

        Returns:
            list[FixSuggestion]: 修复建议列表
        """
        return self._error_analyzer.analyze(test_output, changed_files)

    # ── 工具方法 ────────────────────────────────────────────────

    @staticmethod
    def _read_file_safe(path: Path) -> str | None:
        """安全读取文件内容。"""
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except (FileNotFoundError, PermissionError, OSError):
            return None
