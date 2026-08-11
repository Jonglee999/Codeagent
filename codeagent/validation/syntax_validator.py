"""SyntaxValidator — 语法验证器（Phase 1b 升级版）。

Python 文件使用 ast.parse() + tree-sitter AST 双重检查，
TypeScript/JavaScript 使用 tree-sitter 解析，tsc --noEmit 作为备选。
"""

from __future__ import annotations

import ast
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING

from codeagent.gateway.validation_gateway import (
    ValidationError,
    ValidationResult,
)

if TYPE_CHECKING:
    from tree_sitter import Node, Parser

# 支持的文件扩展名及其语言
_PYTHON_EXTENSIONS = {".py"}
_TS_EXTENSIONS = {".ts"}
_TSX_EXTENSIONS = {".tsx"}
_JS_EXTENSIONS = {".js", ".jsx", ".mjs", ".cjs"}
_ALL_TS_JS_EXTENSIONS = _TS_EXTENSIONS | _TSX_EXTENSIONS | _JS_EXTENSIONS

# tsc 最大执行时间（秒）
_TSC_TIMEOUT_SECONDS = 30

# tree-sitter 可用性标记
_TREE_SITTER_AVAILABLE: bool = True
try:
    from tree_sitter import Language as _TSLanguage, Parser as _TSParser

    import tree_sitter_python as _tspy
    import tree_sitter_typescript as _tsts
    import tree_sitter_javascript as _tsjs
except ImportError:
    _TREE_SITTER_AVAILABLE = False
    _TSLanguage = None  # type: ignore[assignment]
    _TSParser = None  # type: ignore[assignment]

# 语言名称 → (language_func, file_extensions)
_LANGUAGE_MAP: dict[str, tuple[str, set[str]]] = {
    "python": ("python", {".py"}),
    "typescript": ("typescript", {".ts"}),
    "tsx": ("tsx", {".tsx"}),
    "javascript": ("javascript", {".js", ".jsx", ".mjs", ".cjs"}),
}


def _init_tree_sitter() -> bool:
    """尝试初始化 tree-sitter 环境。"""
    global _TREE_SITTER_AVAILABLE
    if not _TREE_SITTER_AVAILABLE:
        return False
    try:
        # 验证各语言包可加载
        _tspy.language()
        _tsts.language_typescript()
        _tsjs.language()
        return True
    except Exception:
        _TREE_SITTER_AVAILABLE = False
        return False


class SyntaxValidator:
    """语法验证器。

    根据文件扩展名选择验证方式：
    - Python (.py): ast.parse() + tree-sitter AST 双重检查
    - TypeScript (.ts): tree-sitter 解析，tsc --noEmit 备选
    - TSX (.tsx): tree-sitter tsx 解析
    - JavaScript (.js/.jsx/.mjs/.cjs): tree-sitter JavaScript 解析
    - 其他: 不验证（视为通过）

    tree-sitter 不可用时自动降级到各语言备选方案。
    """

    def __init__(
        self,
        enable_ts_check: bool = False,
        use_tree_sitter: bool = True,
    ) -> None:
        """初始化 SyntaxValidator。

        Args:
            enable_ts_check: 是否启用 TypeScript 备选检查（tsc --noEmit）
            use_tree_sitter: 是否启用 tree-sitter（默认自动检测可用性）
        """
        self._enable_ts_check = enable_ts_check
        self._use_tree_sitter = use_tree_sitter and _TREE_SITTER_AVAILABLE
        self._parsers: dict[str, Parser] = {}

        if self._use_tree_sitter:
            _init_tree_sitter()

    def _detect_language(self, file_path: str) -> str | None:
        """检测文件路径对应的语言名称。

        Returns:
            str: "python", "typescript", "tsx", "javascript", 或 None
        """
        ext = Path(file_path).suffix.lower()
        for lang_name, (_, exts) in _LANGUAGE_MAP.items():
            if ext in exts:
                return lang_name
        return None

    # ── tree-sitter 支持 ─────────────────────────────────────

    def _get_tree_sitter_parser(self, language: str) -> Parser | None:
        """获取（或创建）指定语言的 tree-sitter 解析器。

        Args:
            language: "python", "typescript", "tsx", "javascript"

        Returns:
            Parser 实例，失败返回 None
        """
        if not self._use_tree_sitter:
            return None
        if language in self._parsers:
            return self._parsers[language]

        try:
            parser = _TSParser()

            if language == "python":
                lang = _TSLanguage(_tspy.language())
            elif language == "typescript":
                lang = _TSLanguage(_tsts.language_typescript())
            elif language == "tsx":
                lang = _TSLanguage(_tsts.language_tsx())
            elif language == "javascript":
                lang = _TSLanguage(_tsjs.language())
            else:
                return None

            parser.language = lang
            self._parsers[language] = parser
            return parser
        except Exception:
            return None

    def _extract_tree_sitter_errors(
        self,
        node: Node,
        source_bytes: bytes,
        file_path: str,
    ) -> list[ValidationError]:
        """递归遍历 tree-sitter AST，提取 ERROR 和 MISSING 节点。

        Args:
            node: 当前 AST 节点
            source_bytes: 源代码字节（用于错误上下文提取）
            file_path: 文件路径

        Returns:
            list[ValidationError]: 提取的错误列表
        """
        errors: list[ValidationError] = []

        def visit(n: Node) -> None:
            if n.type == "ERROR":
                row, col = n.start_point
                end_row, end_col = n.end_point
                # 提取错误上下文
                lines = source_bytes.split(b"\n")
                context = ""
                if row < len(lines):
                    context = lines[row][:80].decode("utf-8", errors="replace").strip()
                message = f"Syntax error at line {row + 1}"
                if context:
                    message += f": '{context}'"
                errors.append(
                    ValidationError(
                        file_path=file_path,
                        line=row + 1,
                        column=col + 1,
                        message=message,
                        code="TS_SYNTAX_ERROR",
                        severity="error",
                    )
                )
            if n.is_missing:
                row, col = n.start_point
                errors.append(
                    ValidationError(
                        file_path=file_path,
                        line=row + 1,
                        column=col + 1,
                        message=f"Missing token: '{n.type}'",
                        code="TS_MISSING_TOKEN",
                        severity="error",
                    )
                )
            for child in n.children:
                visit(child)

        visit(node)
        return errors

    def _check_with_tree_sitter(
        self,
        source: str,
        file_path: str,
        language: str,
    ) -> list[ValidationError]:
        """使用 tree-sitter 解析源代码，返回错误列表。

        Args:
            source: 源代码字符串
            file_path: 文件路径
            language: 语言名称

        Returns:
            list[ValidationError]: 解析出的语法错误
        """
        parser = self._get_tree_sitter_parser(language)
        if parser is None:
            return []

        try:
            source_bytes = source.encode("utf-8")
            tree = parser.parse(source_bytes)
            if tree is None:
                return []
            return self._extract_tree_sitter_errors(
                tree.root_node, source_bytes, file_path
            )
        except Exception:
            return []

    # ── 主入口 ───────────────────────────────────────────────

    async def check_file(self, file_path: str) -> ValidationResult:
        """检查单个文件的语法。

        Args:
            file_path: 文件路径

        Returns:
            ValidationResult: 验证结果
        """
        start = time.monotonic()
        ext = Path(file_path).suffix.lower()

        if ext in _PYTHON_EXTENSIONS:
            result = self._check_python(file_path)
        elif ext in _TS_EXTENSIONS:
            result = self._check_typescript(file_path)
        elif ext in _TSX_EXTENSIONS:
            result = self._check_tsx(file_path)
        elif ext in _JS_EXTENSIONS:
            result = self._check_javascript(file_path)
        else:
            result = ValidationResult(passed=True)

        result.duration_ms = (time.monotonic() - start) * 1000
        return result

    async def check_files(self, file_paths: list[str]) -> list[ValidationResult]:
        """批量检查多个文件的语法。

        按语言分组执行，对 Python 文件逐个检查，对 TypeScript/JavaScript
        文件可以调用一次 tsc 做批量检查。

        Args:
            file_paths: 文件路径列表

        Returns:
            list[ValidationResult]: 验证结果列表，顺序与输入对应
        """
        results: list[ValidationResult] = []

        # 按语言分组
        py_files: list[str] = []
        ts_files: list[str] = []
        tsx_files: list[str] = []
        js_files: list[str] = []
        other_files: list[str] = []

        for f in file_paths:
            ext = Path(f).suffix.lower()
            if ext in _PYTHON_EXTENSIONS:
                py_files.append(f)
            elif ext in _TS_EXTENSIONS:
                ts_files.append(f)
            elif ext in _TSX_EXTENSIONS:
                tsx_files.append(f)
            elif ext in _JS_EXTENSIONS:
                js_files.append(f)
            else:
                other_files.append(f)

        # Python: 逐个检查
        for f in py_files:
            results.append(await self.check_file(f))

        # TypeScript: 批量检查（用 tsc 做批量）
        if ts_files:
            ts_results = self._check_typescript_batch(ts_files)
            results.extend(ts_results)

        # TSX: 逐个检查
        for f in tsx_files:
            results.append(await self.check_file(f))

        # JavaScript: 逐个检查
        for f in js_files:
            results.append(await self.check_file(f))

        # 其他文件类型：直接通过
        for _ in other_files:
            results.append(ValidationResult(passed=True))

        return results

    # ── Python 检查 ──────────────────────────────────────────

    def _check_python(self, file_path: str) -> ValidationResult:
        """使用 ast.parse() + tree-sitter 检查 Python 文件语法。"""
        try:
            with open(file_path, encoding="utf-8", errors="ignore") as f:
                source = f.read()
        except FileNotFoundError:
            return ValidationResult(
                passed=False,
                errors=[
                    ValidationError(
                        file_path=file_path,
                        message=f"File not found: {file_path}",
                        code="FILE_NOT_FOUND",
                    )
                ],
            )
        except PermissionError:
            return ValidationResult(
                passed=False,
                errors=[
                    ValidationError(
                        file_path=file_path,
                        message=f"Permission denied: {file_path}",
                        code="PERMISSION_DENIED",
                    )
                ],
            )
        except OSError as e:
            return ValidationResult(
                passed=False,
                errors=[
                    ValidationError(
                        file_path=file_path,
                        message=f"Error reading file: {e}",
                        code="READ_ERROR",
                    )
                ],
            )

        return self._check_python_source(source, file_path)

    def _check_python_source(
        self, source: str, file_path: str
    ) -> ValidationResult:
        """解析 Python 源代码字符串。"""
        if not source:
            return ValidationResult(passed=True)

        errors: list[ValidationError] = []

        # 1. ast.parse 验证（主要方式）
        try:
            ast.parse(source, filename=file_path)
        except SyntaxError as e:
            errors.append(self._parse_syntax_error(e, file_path))
        except ValueError as e:
            errors.append(
                ValidationError(
                    file_path=file_path,
                    message=str(e),
                    code="VALUE_ERROR",
                )
            )

        # 2. tree-sitter 增强检查（如果可用）
        ts_errors = self._check_with_tree_sitter(source, file_path, "python")
        errors.extend(ts_errors)

        # 额外检查常见的警告模式
        warnings: list[ValidationError] = []
        self._check_warnings(source, file_path, warnings)

        return ValidationResult(
            passed=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    def _parse_syntax_error(
        self, error: SyntaxError, file_path: str
    ) -> ValidationError:
        """解析 Python SyntaxError 为结构化 ValidationError。"""
        error_msg = str(error)
        error_type = type(error).__name__

        # 提取错误代码（如果有）
        code = error_type
        if error.msg:
            known_errors = {
                "invalid syntax": "INVALID_SYNTAX",
                "invalid decimal literal": "INVALID_DECIMAL_LITERAL",
                "unmatched ')'": "UNMATCHED_PAREN",
                "unexpected indent": "UNEXPECTED_INDENT",
                "unindent does not match": "UNINDENT_MISMATCH",
                "expected an indented block": "EXPECTED_INDENTED_BLOCK",
                "invalid character": "INVALID_CHARACTER",
                "f-string": "FSTRING_ERROR",
                "cannot assign to": "CANNOT_ASSIGN",
                "name defined as positional": "NAME_ERROR",
                "duplicate argument": "DUPLICATE_ARGUMENT",
                "positional-only argument": "POSITIONAL_ONLY_ARG",
            }
            for keyword, err_code in known_errors.items():
                if keyword in error_msg.lower():
                    code = err_code
                    break

        return ValidationError(
            file_path=file_path,
            line=error.lineno or 0,
            column=error.offset or 0,
            message=error_msg,
            code=code,
            severity="error",
        )

    def _check_warnings(
        self,
        source: str,
        file_path: str,
        warnings: list[ValidationError],
    ) -> None:
        """检查常见的代码问题（非阻塞警告）。"""
        lines = source.split("\n")

        # 检查过长的行
        for i, line in enumerate(lines, 1):
            if len(line) > 200:
                warnings.append(
                    ValidationError(
                        file_path=file_path,
                        line=i,
                        column=200,
                        message=f"Line too long ({len(line)} > 200 characters)",
                        code="LINE_TOO_LONG",
                        severity="warning",
                    )
                )
                break  # 只报告第一个超长行

    # ── TypeScript 检查 ──────────────────────────────────────

    def _check_typescript(self, file_path: str) -> ValidationResult:
        """检查 TypeScript 文件语法。

        优先使用 tree-sitter，备选 tsc --noEmit。
        """
        try:
            with open(file_path, encoding="utf-8", errors="ignore") as f:
                source = f.read()
        except (FileNotFoundError, PermissionError, OSError) as e:
            return self._file_error_result(file_path, e)

        if not source:
            return ValidationResult(passed=True)

        errors: list[ValidationError] = []

        # 1. tree-sitter 解析（主要方式）
        ts_errors = self._check_with_tree_sitter(source, file_path, "typescript")
        errors.extend(ts_errors)

        # 2. 如果不需要 tsc 或已有错误，直接返回
        if not self._enable_ts_check or errors:
            return ValidationResult(
                passed=len(errors) == 0,
                errors=errors,
            )

        # 3. tsc --noEmit 备选检查
        ts_errors = self._check_typescript_batch_inner([file_path])
        errors.extend(ts_errors)

        return ValidationResult(
            passed=len(errors) == 0,
            errors=errors,
        )

    def _check_tsx(self, file_path: str) -> ValidationResult:
        """检查 TSX 文件语法，使用 tree-sitter tsx 解析器。"""
        return self._check_with_js_like(file_path, "tsx")

    def _check_javascript(self, file_path: str) -> ValidationResult:
        """检查 JavaScript/JSX 文件语法，使用 tree-sitter JavaScript 解析器。"""
        return self._check_with_js_like(file_path, "javascript")

    def _check_with_js_like(self, file_path: str, language: str) -> ValidationResult:
        """使用 tree-sitter 检查 JS-like 文件。"""
        try:
            with open(file_path, encoding="utf-8", errors="ignore") as f:
                source = f.read()
        except (FileNotFoundError, PermissionError, OSError) as e:
            return self._file_error_result(file_path, e)

        if not source:
            return ValidationResult(passed=True)

        errors = self._check_with_tree_sitter(source, file_path, language)
        return ValidationResult(
            passed=len(errors) == 0,
            errors=errors,
        )

    def _check_typescript_batch(
        self, file_paths: list[str]
    ) -> list[ValidationResult]:
        """批量检查 TypeScript 文件。

        优先使用 tree-sitter 逐个检查。
        如果启用了 tsc 备选，对所有文件做一次 tsc 批量检查。
        """
        results: list[ValidationResult] = []

        # tree-sitter 逐个检查
        for f in file_paths:
            results.append(self._check_typescript(f))

        # tsc 批量备选检查
        if self._enable_ts_check and _TREE_SITTER_AVAILABLE:
            # tree-sitter 已覆盖，tsc 做增强（仅追加 warnings）
            tsc_result = self._check_typescript_batch_inner(file_paths)
            if tsc_result.errors or tsc_result.warnings:
                for i, f in enumerate(file_paths):
                    file_errors = [e for e in tsc_result.errors if e.file_path == f]
                    file_warnings = [e for e in tsc_result.warnings if e.file_path == f]
                    results[i].errors.extend(file_errors)
                    results[i].warnings.extend(file_warnings)
                    if file_errors:
                        results[i].passed = False
        elif self._enable_ts_check and not _TREE_SITTER_AVAILABLE:
            # tree-sitter 不可用，只用 tsc
            results = []
            ts_result = self._check_typescript_batch_inner(file_paths)
            for f in file_paths:
                file_errors = [e for e in ts_result.errors if e.file_path == f]
                file_warnings = [e for e in ts_result.warnings if e.file_path == f]
                results.append(
                    ValidationResult(
                        passed=len(file_errors) == 0,
                        errors=file_errors,
                        warnings=file_warnings,
                        duration_ms=ts_result.duration_ms,
                    )
                )

        return results

    def _check_typescript_batch_inner(
        self, file_paths: list[str]
    ) -> ValidationResult:
        """调用 tsc --noEmit 批量检查 TypeScript 文件。"""
        errors: list[ValidationError] = []

        try:
            result = subprocess.run(
                ["tsc", "--noEmit", "--pretty", "false", *file_paths],
                capture_output=True,
                text=True,
                timeout=_TSC_TIMEOUT_SECONDS,
            )

            if result.returncode != 0:
                errors = self._parse_tsc_output(result.stdout + result.stderr)

        except FileNotFoundError:
            errors.append(
                ValidationError(
                    file_path=file_paths[0] if file_paths else "",
                    message="tsc not found. Install TypeScript: npm install -g typescript",
                    code="TSC_NOT_FOUND",
                )
            )
        except subprocess.TimeoutExpired:
            errors.append(
                ValidationError(
                    file_path=file_paths[0] if file_paths else "",
                    message="tsc timed out",
                    code="TSC_TIMEOUT",
                )
            )
        except OSError as e:
            errors.append(
                ValidationError(
                    file_path=file_paths[0] if file_paths else "",
                    message=f"tsc execution error: {e}",
                    code="TSC_EXECUTION_ERROR",
                )
            )

        return ValidationResult(
            passed=len(errors) == 0,
            errors=errors,
        )

    def _parse_tsc_output(self, output: str) -> list[ValidationError]:
        """解析 tsc 输出，提取错误信息。

        tsc 错误格式示例：
        src/file.ts(10,5): error TS2322: Type 'X' is not assignable to type 'Y'
        """
        errors: list[ValidationError] = []

        for line in output.split("\n"):
            line = line.strip()
            if not line:
                continue

            import re

            match = re.match(
                r"^(.*?)\((\d+),(\d+)\):\s+(error|warning)\s+(\S+):\s+(.+)$",
                line,
            )
            if match:
                errors.append(
                    ValidationError(
                        file_path=match.group(1),
                        line=int(match.group(2)),
                        column=int(match.group(3)),
                        severity=match.group(4),
                        code=match.group(5),
                        message=match.group(6),
                    )
                )

        return errors

    # ── 工具方法 ─────────────────────────────────────────────

    def _file_error_result(self, file_path: str, error: Exception) -> ValidationResult:
        """将文件读取错误转换为 ValidationResult。"""
        if isinstance(error, FileNotFoundError):
            code = "FILE_NOT_FOUND"
            msg = f"File not found: {file_path}"
        elif isinstance(error, PermissionError):
            code = "PERMISSION_DENIED"
            msg = f"Permission denied: {file_path}"
        else:
            code = "READ_ERROR"
            msg = f"Error reading file: {error}"
        return ValidationResult(
            passed=False,
            errors=[ValidationError(file_path=file_path, message=msg, code=code)],
        )
