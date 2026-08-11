"""StaticAnalyzer — 静态分析器（Lint + TypeCheck + 安全扫描）。

参考 SRS §6.3.2。
对修改后的代码执行 Lint 检查（ruff）、类型检查（mypy）和安全扫描。
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from codeagent.gateway.validation_gateway import ValidationError, ValidationResult

if TYPE_CHECKING:
    pass

# ruff / mypy 子进程超时（秒）
_TOOL_TIMEOUT = 60

# 安全扫描检测的正则模式
_SECURITY_PATTERNS: list[dict[str, Any]] = [
    {
        "name": "eval_call",
        "regex": re.compile(r"\beval\s*\("),
        "message": "Dynamic code execution via eval() — potential code injection risk",
        "code": "SEC_EVAL",
    },
    {
        "name": "exec_call",
        "regex": re.compile(r"\bexec\s*\("),
        "message": "Dynamic code execution via exec() — potential code injection risk",
        "code": "SEC_EXEC",
    },
    {
        "name": "compile_call",
        "regex": re.compile(r"\bcompile\s*\("),
        "message": "Dynamic code execution via compile() — potential code injection risk",
        "code": "SEC_COMPILE",
    },
    {
        "name": "os_system",
        "regex": re.compile(r"\bos\.system\s*\("),
        "message": "Shell command execution via os.system() — potential command injection risk",
        "code": "SEC_OS_SYSTEM",
    },
    {
        "name": "subprocess_shell",
        "regex": re.compile(r"(?:subprocess|sp)\.\w+\s*\([\s\S]*?shell\s*=\s*True"),
        "message": "Shell execution with shell=True — potential command injection risk",
        "code": "SEC_SUBPROCESS_SHELL",
    },
    {
        "name": "sql_injection_concat",
        "regex": re.compile(
            r"""(?:execute|executemany|query)\s*\([^)]*["']\s*\+|["'].*(?:SELECT|INSERT|UPDATE|DELETE)\s+.*["']\s*\+"""
        ),
        "message": "Possible SQL injection — string concatenation in SQL query",
        "code": "SEC_SQL_INJECTION",
    },
    {
        "name": "aws_key",
        "regex": re.compile(r"""(?:AKIA[0-9A-Z]{16}|aws[_-]?access[_-]?key[_-]?id|aws[_-]?secret[_-]?access[_-]?key)\s*[:=]\s*["']"""),
        "message": "Possible hardcoded AWS credential detected",
        "code": "SEC_AWS_CREDENTIAL",
    },
    {
        "name": "password_hardcode",
        "regex": re.compile(r"""(?:password|passwd|pwd|secret)\s*[:=]\s*["'][^"']{4,}["']""", re.IGNORECASE),
        "message": "Possible hardcoded password/secret — consider using environment variables",
        "code": "SEC_HARDCODED_SECRET",
    },
    {
        "name": "jwt_secret_hardcode",
        "regex": re.compile(r"""(?:jwt[_-]?secret|token[_-]?secret|api[_-]?key)\s*[:=]\s*["'][^"']{8,}["']""", re.IGNORECASE),
        "message": "Possible hardcoded JWT secret or API key",
        "code": "SEC_JWT_SECRET",
    },
    {
        "name": "requests_get",
        "regex": re.compile(r"requests\.(get|post|put|delete)\s*\("),
        "message": "Usage of requests library — ensure proper error handling and timeouts",
        "code": "SEC_REQUESTS_USAGE",
        "severity": "warning",
    },
]


def _is_python_file(file_path: str) -> bool:
    """判断文件是否为 Python 文件。"""
    return Path(file_path).suffix.lower() == ".py"


def _resolve_path(file_path: str) -> str:
    """将路径统一为正斜杠格式。"""
    return Path(file_path).as_posix()


def _run_subprocess(
    cmd: list[str],
    timeout: int = _TOOL_TIMEOUT,
) -> subprocess.CompletedProcess:
    """执行子进程，自动处理 Windows 编码问题。

    Windows 中文环境下 ruff/mypy 可能输出 GBK 编码文本，
    使用 bytes 手动解码避免 UnicodeDecodeError。
    """
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=False,  # 用 bytes 而非 text=True
            timeout=timeout,
        )
        # 尝试 UTF-8 解码，失败则用系统编码
        for enc in ("utf-8", "gbk", "gb18030"):
            try:
                stdout = proc.stdout.decode(enc, errors="replace") if proc.stdout else ""
                stderr = proc.stderr.decode(enc, errors="replace") if proc.stderr else ""
                break
            except (UnicodeDecodeError, LookupError):
                continue
        else:
            stdout = proc.stdout.decode("utf-8", errors="replace") if proc.stdout else ""
            stderr = proc.stderr.decode("utf-8", errors="replace") if proc.stderr else ""

        proc.stdout = stdout  # type: ignore[assignment]
        proc.stderr = stderr  # type: ignore[assignment]
        return proc
    except FileNotFoundError:
        raise
    except OSError:
        raise


class StaticAnalyzer:
    """静态分析器 — Lint + TypeCheck + 安全扫描。

    对修改后的代码文件执行三层静态分析，返回结构化的验证结果。
    各子分析器独立运行，一个失败不影响其他分析器。
    """

    def __init__(self, project_root: str | None = None) -> None:
        """初始化 StaticAnalyzer。

        Args:
            project_root: 项目根目录（用于相对路径解析）。为 None 时使用当前目录。
        """
        self._project_root = project_root or os.getcwd()
        self._ruff_available: bool | None = None  # 懒检测
        self._mypy_available: bool | None = None

    def _project_files(self, files: list[str]) -> list[str]:
        """Resolve task-relative files against the configured project root."""
        root = Path(self._project_root)
        resolved: list[str] = []
        for item in files:
            path = Path(item)
            resolved.append(str(path if path.is_absolute() else root / path))
        return resolved

    # ── 工具可用性检测 ────────────────────────────────────────

    def _check_ruff_available(self) -> bool:
        """检测 ruff 是否可用。"""
        if self._ruff_available is None:
            self._ruff_available = shutil.which("ruff") is not None
        return self._ruff_available

    def _check_mypy_available(self) -> bool:
        """检测 mypy 是否可用。"""
        if self._mypy_available is None:
            self._mypy_available = shutil.which("mypy") is not None
        return self._mypy_available

    # ── Lint 检查 ──────────────────────────────────────────────

    async def run_lint(
        self,
        files: list[str],
        config: str | None = None,
    ) -> ValidationResult:
        """运行 ruff Lint 检查。

        Args:
            files: 文件路径列表（仅 .py 文件会实际检查）
            config: ruff 配置文件路径。为 None 时自动发现。

        Returns:
            ValidationResult: Lint 检查结果
        """
        start = time.monotonic()

        if not self._check_ruff_available():
            return ValidationResult(
                passed=True,
                warnings=[
                    ValidationError(
                        file_path="",
                        message="ruff is not installed — Lint check skipped",
                        code="RUFF_NOT_FOUND",
                        severity="warning",
                    )
                ],
                duration_ms=(time.monotonic() - start) * 1000,
            )

        py_files = self._project_files([f for f in files if _is_python_file(f)])
        if not py_files:
            return ValidationResult(
                passed=True,
                duration_ms=(time.monotonic() - start) * 1000,
            )

        try:
            cmd = ["ruff", "check", "--output-format", "json"]
            if config:
                cmd.extend(["--config", config])
            cmd.extend(py_files)

            result = _run_subprocess(cmd)

            # ruff 存在就返回解析结果（exit code 1 = 有 lint 错误，0 = 无，其他=执行错误）
            if result.returncode not in (0, 1):
                return ValidationResult(
                    passed=False,
                    errors=[
                        ValidationError(
                            file_path="",
                            message=f"ruff execution error: {result.stderr.strip()}",
                            code="RUFF_EXECUTION_ERROR",
                        )
                    ],
                    duration_ms=(time.monotonic() - start) * 1000,
                )

            if not result.stdout:
                return ValidationResult(
                    passed=True,
                    duration_ms=(time.monotonic() - start) * 1000,
                )

            return ValidationResult(
                passed=result.returncode == 0,
                errors=self._parse_ruff_output(str(result.stdout)),
                duration_ms=(time.monotonic() - start) * 1000,
            )

        except subprocess.TimeoutExpired:
            return ValidationResult(
                passed=False,
                errors=[
                    ValidationError(
                        file_path="",
                        message="ruff check timed out",
                        code="RUFF_TIMEOUT",
                    )
                ],
                duration_ms=(time.monotonic() - start) * 1000,
            )
        except FileNotFoundError:
            self._ruff_available = False
            return ValidationResult(
                passed=True,
                warnings=[
                    ValidationError(
                        file_path="",
                        message="ruff is not installed — Lint check skipped",
                        code="RUFF_NOT_FOUND",
                        severity="warning",
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
                        message=f"ruff execution error: {e}",
                        code="RUFF_EXECUTION_ERROR",
                    )
                ],
                duration_ms=(time.monotonic() - start) * 1000,
            )

    def _parse_ruff_output(self, stdout: str) -> list[ValidationError]:
        """解析 ruff JSON 输出为 ValidationError 列表。"""
        errors: list[ValidationError] = []
        try:
            diagnostics = json.loads(stdout)
            for d in diagnostics:
                severity = d.get("severity", "error")
                errors.append(
                    ValidationError(
                        file_path=_resolve_path(d.get("filename", "")),
                        line=d.get("location", {}).get("row", 0),
                        column=d.get("location", {}).get("column", 0),
                        message=d.get("message", ""),
                        code=d.get("code", ""),
                        severity=severity,
                    )
                )
        except (json.JSONDecodeError, KeyError, TypeError):
            errors.append(
                ValidationError(
                    file_path="",
                    message="Failed to parse ruff output",
                    code="RUFF_PARSE_ERROR",
                )
            )
        return errors

    # ── 类型检查 ──────────────────────────────────────────────

    async def run_typecheck(
        self,
        files: list[str],
        config: str | None = None,
    ) -> ValidationResult:
        """运行 mypy 类型检查。

        Args:
            files: 文件路径列表（仅 .py 文件会实际检查）
            config: mypy 配置文件路径。为 None 时自动发现。

        Returns:
            ValidationResult: 类型检查结果
        """
        start = time.monotonic()

        if not self._check_mypy_available():
            return ValidationResult(
                passed=True,
                warnings=[
                    ValidationError(
                        file_path="",
                        message="mypy is not installed — type check skipped",
                        code="MYPY_NOT_FOUND",
                        severity="warning",
                    )
                ],
                duration_ms=(time.monotonic() - start) * 1000,
            )

        py_files = self._project_files([f for f in files if _is_python_file(f)])
        if not py_files:
            return ValidationResult(
                passed=True,
                duration_ms=(time.monotonic() - start) * 1000,
            )

        try:
            cmd = ["mypy", "--show-error-codes"]
            if config:
                cmd.extend(["--config-file", config])
            cmd.extend(py_files)

            result = _run_subprocess(cmd)

            if result.returncode == 0:
                return ValidationResult(
                    passed=True,
                    duration_ms=(time.monotonic() - start) * 1000,
                )

            # 解析 mypy 输出
            errors = self._parse_mypy_output(str(result.stdout))
            return ValidationResult(
                passed=False,
                errors=errors,
                duration_ms=(time.monotonic() - start) * 1000,
            )

        except subprocess.TimeoutExpired:
            return ValidationResult(
                passed=False,
                errors=[
                    ValidationError(
                        file_path="",
                        message="mypy type check timed out",
                        code="MYPY_TIMEOUT",
                    )
                ],
                duration_ms=(time.monotonic() - start) * 1000,
            )
        except FileNotFoundError:
            self._mypy_available = False
            return ValidationResult(
                passed=True,
                warnings=[
                    ValidationError(
                        file_path="",
                        message="mypy is not installed — type check skipped",
                        code="MYPY_NOT_FOUND",
                        severity="warning",
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
                        message=f"mypy execution error: {e}",
                        code="MYPY_EXECUTION_ERROR",
                    )
                ],
                duration_ms=(time.monotonic() - start) * 1000,
            )

    def _parse_mypy_output(self, stdout: str) -> list[ValidationError]:
        """解析 mypy 文本输出为 ValidationError 列表。

        mypy 输出格式: file:line:col: error: message [error-code]
        """
        errors: list[ValidationError] = []
        pattern = re.compile(
            r"^(.+?):(\d+):(?:\d+:)?\s*(error|warning):\s*(.+?)(?:\s+\[(.+?)\])?\s*$",
            re.MULTILINE,
        )
        for match in pattern.finditer(stdout):
            errors.append(
                ValidationError(
                    file_path=_resolve_path(match.group(1)),
                    line=int(match.group(2)),
                    message=match.group(4).strip(),
                    code=match.group(5) or "MYPY_ERROR",
                    severity=match.group(3),
                )
            )

        # fallback: 如果没有匹配到，尝试 simpler pattern for edge cases
        if not errors:
            for line in stdout.strip().split("\n"):
                line = line.strip()
                if ": error:" in line or ": warning:" in line:
                    parts = line.split(":", 3)
                    if len(parts) >= 3:
                        file_path = parts[0]
                        try:
                            line_no = int(parts[1])
                        except ValueError:
                            continue
                        errors.append(
                            ValidationError(
                                file_path=_resolve_path(file_path),
                                line=line_no,
                                message=parts[-1].strip() if len(parts) > 3 else line,
                                code="MYPY_ERROR",
                            )
                        )

        return errors

    # ── 安全扫描 ──────────────────────────────────────────────

    async def run_security_scan(
        self,
        files: list[str],
    ) -> ValidationResult:
        """对文件列表执行通用安全检查。

        优先使用 bandit（AST 分析），不可用时降级为正则检测。
        检测模式：
        - eval() / exec() / compile() 动态代码执行
        - SQL 注入模式（字符串拼接 SQL）
        - 硬编码密钥/密码（AWS Key, JWT Secret, password）
        - os.system() / subprocess shell=True

        Args:
            files: 文件路径列表（仅 .py 文件会实际检查）

        Returns:
            ValidationResult: 安全扫描结果
        """
        start = time.monotonic()

        py_files = self._project_files([f for f in files if _is_python_file(f)])
        if not py_files:
            return ValidationResult(
                passed=True,
                duration_ms=(time.monotonic() - start) * 1000,
            )

        # 优先使用 bandit
        bandit_result = self._run_bandit(py_files)
        if bandit_result is not None:
            bandit_result.duration_ms = (time.monotonic() - start) * 1000
            return bandit_result

        # 降级：正则检测
        result = await self._run_security_check_regex(py_files)
        result.duration_ms = (time.monotonic() - start) * 1000
        return result

    def _run_bandit(self, files: list[str]) -> ValidationResult | None:
        """运行 bandit 安全扫描。

        bandit 可用时返回 ValidationResult，不可用或出错时返回 None（降级）。

        bandit 返回码：0=无问题，1=有问题，2=自身错误。
        HIGH severity → errors，MEDIUM → warnings，LOW → 忽略。
        """
        if not shutil.which("bandit"):
            return None

        try:
            cmd = ["bandit", "-f", "json", "--quiet", "-ll"] + files
            result = _run_subprocess(cmd, timeout=_TOOL_TIMEOUT)
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            return None

        if result.returncode == 2:
            return None

        try:
            data = json.loads(result.stdout or "{}")
        except json.JSONDecodeError:
            return None

        errors: list[ValidationError] = []
        warnings: list[ValidationError] = []
        blocking_test_ids = {"B102", "B307", "B602", "B604", "B605", "B606"}
        for issue in data.get("results", []):
            severity = issue.get("issue_severity", "LOW")
            test_id = issue.get("test_id", "")
            err = ValidationError(
                file_path=issue.get("filename", ""),
                line=issue.get("line_number", 0),
                message=issue.get("issue_text", ""),
                code=test_id,
                severity="error" if severity == "HIGH" or test_id in blocking_test_ids else "warning",
            )
            if severity == "HIGH" or test_id in blocking_test_ids:
                errors.append(err)
            elif severity == "MEDIUM":
                warnings.append(err)
            # LOW severity: 忽略，避免噪音

        return ValidationResult(
            passed=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    async def _run_security_check_regex(self, files: list[str]) -> ValidationResult:
        """降级安全扫描：使用正则检测。"""
        errors: list[ValidationError] = []
        warnings: list[ValidationError] = []

        for file_path in files:
            try:
                with open(file_path, encoding="utf-8", errors="ignore") as f:
                    source = f.read()
            except (FileNotFoundError, PermissionError, OSError):
                continue

            if not source:
                continue

            file_errors, file_warnings = self._scan_source(
                source, file_path
            )
            errors.extend(file_errors)
            warnings.extend(file_warnings)

        return ValidationResult(
            passed=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    def _scan_source(
        self,
        source: str,
        file_path: str,
    ) -> tuple[list[ValidationError], list[ValidationError]]:
        """扫描源代码中的安全风险。

        Returns:
            tuple: (errors, warnings)
        """
        errors: list[ValidationError] = []
        warnings: list[ValidationError] = []

        lines = source.split("\n")

        for pattern_def in _SECURITY_PATTERNS:
            matches = pattern_def["regex"].finditer(source)
            found_lines: set[int] = set()

            for match in matches:
                # 找到匹配所在的行号
                pos = match.start()
                line_no = source[:pos].count("\n") + 1

                if line_no in found_lines:
                    continue
                found_lines.add(line_no)

                # 如果是注释中的模式，跳过
                line_content = lines[line_no - 1].strip() if line_no <= len(lines) else ""
                if line_content.startswith("#") or line_content.startswith("//"):
                    continue

                sev = pattern_def.get("severity", "error")
                error = ValidationError(
                    file_path=_resolve_path(file_path),
                    line=line_no,
                    message=pattern_def["message"],
                    code=pattern_def["code"],
                    severity=sev,
                )
                if sev == "warning":
                    warnings.append(error)
                else:
                    errors.append(error)

        return errors, warnings

    # ── 统一入口 ──────────────────────────────────────────────

    async def run_all(
        self,
        files: list[str],
        skip_lint: bool = False,
        skip_typecheck: bool = False,
        skip_security: bool = False,
    ) -> list[ValidationResult]:
        """运行所有静态分析检查。

        Args:
            files: 文件路径列表
            skip_lint: 跳过 Lint 检查
            skip_typecheck: 跳过类型检查
            skip_security: 跳过安全扫描

        Returns:
            list[ValidationResult]: 各分析的运行结果 [lint, typecheck, security]
        """
        results: list[ValidationResult] = []

        if skip_lint:
            results.append(ValidationResult(passed=True))
        else:
            results.append(await self.run_lint(files))

        if skip_typecheck:
            results.append(ValidationResult(passed=True))
        else:
            results.append(await self.run_typecheck(files))

        if skip_security:
            results.append(ValidationResult(passed=True))
        else:
            results.append(await self.run_security_scan(files))

        return results
