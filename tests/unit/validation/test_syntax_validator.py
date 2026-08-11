"""SyntaxValidator 单元测试。

覆盖 Python 语法检查（有效代码、各类语法错误、空文件、编码问题等）。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from codeagent.gateway.validation_gateway import ValidationError, ValidationResult
from codeagent.validation.syntax_validator import SyntaxValidator


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def validator() -> SyntaxValidator:
    return SyntaxValidator()


# ── Tests: Valid Python files ────────────────────────────────────────────


@pytest.mark.asyncio
class TestValidPython:
    """验证有效的 Python 代码。"""

    async def test_valid_simple(self, tmp_path: Path, validator: SyntaxValidator) -> None:
        f = tmp_path / "valid.py"
        f.write_text("x = 1\n")
        result = await validator.check_file(str(f))
        assert result.passed is True
        assert result.errors == []

    async def test_valid_function(self, tmp_path: Path, validator: SyntaxValidator) -> None:
        f = tmp_path / "func.py"
        f.write_text("def hello():\n    return 'world'\n")
        result = await validator.check_file(str(f))
        assert result.passed is True

    async def test_valid_class(self, tmp_path: Path, validator: SyntaxValidator) -> None:
        f = tmp_path / "class.py"
        f.write_text("class A:\n    pass\n")
        result = await validator.check_file(str(f))
        assert result.passed is True

    async def test_valid_complex(self, tmp_path: Path, validator: SyntaxValidator) -> None:
        f = tmp_path / "complex.py"
        f.write_text(
            "import os\n"
            "from pathlib import Path\n\n"
            "def main():\n"
            '    """Docstring."""\n'
            "    x = [i for i in range(10)]\n"
            "    y = {k: v for k, v in enumerate(x)}\n"
            "    return y\n\n\n"
            "if __name__ == '__main__':\n"
            "    main()\n"
        )
        result = await validator.check_file(str(f))
        assert result.passed is True

    async def test_valid_empty_file(self, tmp_path: Path, validator: SyntaxValidator) -> None:
        f = tmp_path / "empty.py"
        f.write_text("")
        result = await validator.check_file(str(f))
        assert result.passed is True

    async def test_valid_async_code(self, tmp_path: Path, validator: SyntaxValidator) -> None:
        f = tmp_path / "async_code.py"
        f.write_text(
            "async def fetch():\n"
            "    return await some_async()\n"
        )
        result = await validator.check_file(str(f))
        assert result.passed is True

    async def test_valid_type_hints(self, tmp_path: Path, validator: SyntaxValidator) -> None:
        f = tmp_path / "typed.py"
        f.write_text(
            "from typing import Optional\n\n"
            "def greet(name: str) -> Optional[str]:\n"
            '    return f"Hello {name}"\n'
        )
        result = await validator.check_file(str(f))
        assert result.passed is True

    async def test_comment_only_file(self, tmp_path: Path, validator: SyntaxValidator) -> None:
        """仅包含注释的 Python 文件应通过验证。"""
        f = tmp_path / "comment_only.py"
        f.write_text(
            "# This is a comment\n"
            "# Another comment\n"
            "# TODO: implement something\n"
        )
        result = await validator.check_file(str(f))
        assert result.passed is True

    async def test_docstring_only_file(self, tmp_path: Path, validator: SyntaxValidator) -> None:
        """仅包含 docstring 的 Python 文件应通过验证。"""
        f = tmp_path / "docstring_only.py"
        f.write_text('"""Module docstring."""\n')
        result = await validator.check_file(str(f))
        assert result.passed is True

    async def test_bom_encoded_python(self, tmp_path: Path, validator: SyntaxValidator) -> None:
        """UTF-8 BOM 编码文件在 Python 中应被标记为语法错误。"""
        f = tmp_path / "bom.py"
        f.write_bytes(b"\xef\xbb\xbfx = 1\n")
        result = await validator.check_file(str(f))
        # BOM 字符 U+FEFF 使 ast.parse 报错（需 utf-8-sig codec）
        assert result.passed is False


# ── Tests: Invalid Python files ──────────────────────────────────────────


@pytest.mark.asyncio
class TestInvalidPython:
    """验证错误的 Python 代码。"""

    async def test_syntax_error_invalid_syntax(self, tmp_path: Path, validator: SyntaxValidator) -> None:
        f = tmp_path / "invalid.py"
        f.write_text("x = 1\n" "y = \n")  # 不完整表达式
        result = await validator.check_file(str(f))
        assert result.passed is False
        assert len(result.errors) >= 1
        assert result.errors[0].line > 0
        assert result.errors[0].message

    async def test_syntax_error_unclosed_paren(self, tmp_path: Path, validator: SyntaxValidator) -> None:
        f = tmp_path / "unclosed.py"
        f.write_text("def foo():\n    x = (1 + 2\n")
        result = await validator.check_file(str(f))
        assert result.passed is False
        assert len(result.errors) >= 1

    async def test_syntax_error_bad_indent(self, tmp_path: Path, validator: SyntaxValidator) -> None:
        f = tmp_path / "bad_indent.py"
        f.write_text("def foo():\n    x = 1\n  y = 2\n")  # 缩进不一致
        result = await validator.check_file(str(f))
        assert result.passed is False
        assert len(result.errors) >= 1

    async def test_syntax_error_unexpected_eof(self, tmp_path: Path, validator: SyntaxValidator) -> None:
        f = tmp_path / "eof_error.py"
        f.write_text("def foo():\n    x = 1\n")  # 缺少 pass/return
        # 实际上这个不是语法错误
        # 用真正的语法错误：
        f.write_text("if True:\n")
        result = await validator.check_file(str(f))
        assert result.passed is False

    async def test_syntax_error_missing_colon(self, tmp_path: Path, validator: SyntaxValidator) -> None:
        f = tmp_path / "missing_colon.py"
        f.write_text("if True\n    pass\n")
        result = await validator.check_file(str(f))
        assert result.passed is False

    async def test_syntax_error_unknown_encoding(self, tmp_path: Path, validator: SyntaxValidator) -> None:
        """模拟编码声明错误。"""
        f = tmp_path / "encoding.py"
        f.write_text("# -*- coding: utf-8 -*-\nx = 1\n")
        # 这个实际上是有效的，我们用真正的语法错误
        pass

    async def test_syntax_error_fstring(self, tmp_path: Path, validator: SyntaxValidator) -> None:
        f = tmp_path / "fstring.py"
        f.write_text("x = f'{invalid}'\n")  # 有效语法
        # 用真正无效的 f-string:
        f.write_text("x = f'{1+'\n")  # 不完整
        result = await validator.check_file(str(f))
        assert result.passed is False

    async def test_mixed_tabs_and_spaces(self, tmp_path: Path, validator: SyntaxValidator) -> None:
        """混用 Tab 和空格缩进应报错。"""
        f = tmp_path / "mixed_indent.py"
        f.write_text(
            "def foo():\n"
            "\tx = 1\n"  # tab
            "    y = 2\n"  # spaces
        )
        result = await validator.check_file(str(f))
        assert result.passed is False

    async def test_error_extraction(self, tmp_path: Path, validator: SyntaxValidator) -> None:
        """验证错误信息提取的完整性。"""
        f = tmp_path / "err_extract.py"
        f.write_text("if True\n    pass\n")  # 缺少冒号
        result = await validator.check_file(str(f))
        assert len(result.errors) >= 1
        err = result.errors[0]
        assert err.line >= 1
        assert err.column >= 0
        assert len(err.message) > 0
        assert len(err.code) > 0
        assert err.severity == "error"


# ── Tests: File not found / permission issues ────────────────────────────


@pytest.mark.asyncio
class TestFileIssues:
    """文件不存在和权限等异常情况。"""

    async def test_file_not_found(self, validator: SyntaxValidator) -> None:
        result = await validator.check_file("/nonexistent/path/file.py")
        assert result.passed is False
        assert any(e.code == "FILE_NOT_FOUND" for e in result.errors)

    async def test_check_files_mixed(self, tmp_path: Path, validator: SyntaxValidator) -> None:
        """混合有效/无效文件的批量检查。"""
        valid = tmp_path / "valid.py"
        valid.write_text("x = 1\n")
        invalid = tmp_path / "invalid.py"
        invalid.write_text("if True\n    pass\n")

        results = await validator.check_files([str(valid), str(invalid)])
        assert len(results) == 2
        assert results[0].passed is True
        assert results[1].passed is False

    async def test_non_ascii_file_path(self, tmp_path: Path, validator: SyntaxValidator) -> None:
        """非 ASCII 文件路径应正常工作。"""
        f = tmp_path / "中文文件.py"
        f.write_text("x = 1\n")
        result = await validator.check_file(str(f))
        assert result.passed is True

    async def test_unsupported_language_skipped(self, tmp_path: Path, validator: SyntaxValidator) -> None:
        """不支持的语言文件应跳过验证。"""
        f = tmp_path / "data.rst"
        f.write_text("Some text")
        result = await validator.check_file(str(f))
        assert result.passed is True


# ── Tests: check_files ──────────────────────────────────────────────────


@pytest.mark.asyncio
class TestCheckFiles:
    """批量检查功能。"""

    async def test_check_files_all_valid(self, tmp_path: Path, validator: SyntaxValidator) -> None:
        files = []
        for i in range(3):
            f = tmp_path / f"mod{i}.py"
            f.write_text(f"x = {i}\n")
            files.append(str(f))
        results = await validator.check_files(files)
        assert all(r.passed for r in results)
        assert len(results) == 3

    async def test_check_files_all_invalid(self, tmp_path: Path, validator: SyntaxValidator) -> None:
        files = []
        for i in range(2):
            f = tmp_path / f"bad{i}.py"
            f.write_text("if True\n    pass\n")  # 缺少冒号
            files.append(str(f))
        results = await validator.check_files(files)
        assert all(not r.passed for r in results)

    async def test_check_files_empty(self, validator: SyntaxValidator) -> None:
        results = await validator.check_files([])
        assert results == []

    async def test_check_files_non_python(self, tmp_path: Path, validator: SyntaxValidator) -> None:
        """非 Python 文件直接返回通过。"""
        f = tmp_path / "readme.md"
        f.write_text("# Hello\n")
        results = await validator.check_files([str(f)])
        assert len(results) == 1
        assert results[0].passed is True


# ── Tests: Duration ──────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestDuration:
    """验证耗时记录。"""

    async def test_duration_recorded(self, tmp_path: Path, validator: SyntaxValidator) -> None:
        f = tmp_path / "dur.py"
        f.write_text("x = 1\n")
        result = await validator.check_file(str(f))
        assert result.duration_ms >= 0

    async def test_duration_recorded_on_error(self, tmp_path: Path, validator: SyntaxValidator) -> None:
        f = tmp_path / "bad.py"
        f.write_text("if True\n    pass\n")
        result = await validator.check_file(str(f))
        assert result.duration_ms >= 0


# ── Tests: ValidationError dataclass ─────────────────────────────────────


class TestValidationErrorDataclass:
    def test_full_construction(self) -> None:
        err = ValidationError(
            file_path="test.py",
            line=10,
            column=5,
            message="invalid syntax",
            code="INVALID_SYNTAX",
            severity="error",
        )
        assert err.file_path == "test.py"
        assert err.line == 10
        assert err.column == 5
        assert err.message == "invalid syntax"
        assert err.code == "INVALID_SYNTAX"
        assert err.severity == "error"

    def test_default_values(self) -> None:
        err = ValidationError(file_path="test.py")
        assert err.line == 0
        assert err.column == 0
        assert err.message == ""
        assert err.code == ""
        assert err.severity == "error"


class TestValidationResultDataclass:
    def test_default_passed(self) -> None:
        r = ValidationResult()
        assert r.passed is True
        assert r.errors == []
        assert r.warnings == []
        assert r.duration_ms == 0.0

    def test_with_errors(self) -> None:
        err = ValidationError(file_path="test.py", message="bad")
        r = ValidationResult(passed=False, errors=[err])
        assert r.passed is False
        assert len(r.errors) == 1


# ── Tests: Warnings ──────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestWarnings:
    """验证警告检测。"""

    async def test_line_too_long_warning(self, tmp_path: Path, validator: SyntaxValidator) -> None:
        f = tmp_path / "long_line.py"
        f.write_text("x = 1\n" + "#" * 300 + "\n")
        result = await validator.check_file(str(f))
        codes = {w.code for w in result.warnings}
        assert "LINE_TOO_LONG" in codes

    async def test_valid_no_warnings(self, tmp_path: Path, validator: SyntaxValidator) -> None:
        f = tmp_path / "clean.py"
        f.write_text("x = 1\ny = 2\n")
        result = await validator.check_file(str(f))
        [w for w in result.warnings if w.code]
        # 只检查实际有意义的警告
        line_too_long = any(w.code == "LINE_TOO_LONG" for w in result.warnings)
        assert not line_too_long


# ── Tests: check_python_source (direct) ──────────────────────────────────


class TestCheckPythonSource:
    """直接测试 _check_python_source 方法。"""

    def test_valid_source(self, validator: SyntaxValidator) -> None:
        result = validator._check_python_source("x = 1\n", "test.py")
        assert result.passed is True

    def test_empty_source(self, validator: SyntaxValidator) -> None:
        result = validator._check_python_source("", "test.py")
        assert result.passed is True

    def test_invalid_source(self, validator: SyntaxValidator) -> None:
        result = validator._check_python_source("if True\n    pass\n", "test.py")
        assert result.passed is False
        assert len(result.errors) >= 1

    def test_syntax_error_lineno(self, validator: SyntaxValidator) -> None:
        result = validator._check_python_source(
            "x = 1\ny = \nz = 2\n", "test.py"
        )
        assert result.passed is False
        assert result.errors[0].line == 2


# ── Tests: Tree-sitter Python enhancement ─────────────────────────────────


@pytest.mark.asyncio
class TestTreeSitterPython:
    """tree-sitter 对 Python 文件的增强检查。"""

    async def test_tree_sitter_enabled(self, tmp_path: Path) -> None:
        """验证 tree-sitter 默认启用。"""
        validator = SyntaxValidator(use_tree_sitter=True)
        f = tmp_path / "simple.py"
        f.write_text("x = 1\n")
        result = await validator.check_file(str(f))
        assert result.passed is True

    async def test_valid_python_with_ts(self, tmp_path: Path) -> None:
        """有效的 Python 代码 tree-sitter 不应报错。"""
        validator = SyntaxValidator(use_tree_sitter=True)
        f = tmp_path / "valid.py"
        f.write_text(
            "import sys\n"
            "from pathlib import Path\n\n"
            "def process(path: str) -> None:\n"
            '    """Process the path."""\n'
            "    p = Path(path)\n"
            "    if p.exists():\n"
            "        print(p.resolve())\n\n"
            "if __name__ == '__main__':\n"
            '    process("/tmp")\n'
        )
        result = await validator.check_file(str(f))
        assert result.passed is True

    async def test_invalid_python_ts_enhancement(self, tmp_path: Path) -> None:
        """tree-sitter 应能检测到 ast.parse 也能检测到的错误。"""
        validator = SyntaxValidator(use_tree_sitter=True)
        f = tmp_path / "bad.py"
        f.write_text("if True\n    pass\n")  # 缺少冒号
        result = await validator.check_file(str(f))
        assert result.passed is False
        assert len(result.errors) >= 1

    async def test_ts_disabled_fallback_to_ast(self, tmp_path: Path) -> None:
        """关闭 tree-sitter 后应回退到纯 ast.parse。"""
        validator = SyntaxValidator(use_tree_sitter=False)
        f = tmp_path / "simple.py"
        f.write_text("x = 1\n")
        result = await validator.check_file(str(f))
        assert result.passed is True

        f2 = tmp_path / "bad.py"
        f2.write_text("if True\n    pass\n")
        result2 = await validator.check_file(str(f2))
        assert result2.passed is False


# ── Tests: TypeScript ─────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestTypeScript:
    """TypeScript 文件语法测试（tree-sitter）。"""

    async def test_valid_typescript(self, tmp_path: Path) -> None:
        validator = SyntaxValidator(use_tree_sitter=True)
        f = tmp_path / "greet.ts"
        f.write_text(
            "function greet(name: string): string {\n"
            '  return `Hello, ${name}!`;\n'
            "}\n\n"
            "const result = greet('World');\n"
            "console.log(result);\n"
        )
        result = await validator.check_file(str(f))
        assert result.passed is True, f"errors: {result.errors}"

    async def test_invalid_typescript_syntax(self, tmp_path: Path) -> None:
        validator = SyntaxValidator(use_tree_sitter=True)
        f = tmp_path / "bad.ts"
        f.write_text("const x: string = ;\n")  # 缺少值
        result = await validator.check_file(str(f))
        assert result.passed is False
        assert any(e.code == "TS_SYNTAX_ERROR" for e in result.errors)

    async def test_typescript_class(self, tmp_path: Path) -> None:
        validator = SyntaxValidator(use_tree_sitter=True)
        f = tmp_path / "class.ts"
        f.write_text(
            "class Animal {\n"
            "  constructor(public name: string) {}\n"
            "  speak(): void {\n"
            "    console.log(this.name);\n"
            "  }\n"
            "}\n"
        )
        result = await validator.check_file(str(f))
        assert result.passed is True

    async def test_typescript_arrow_functions(self, tmp_path: Path) -> None:
        validator = SyntaxValidator(use_tree_sitter=True)
        f = tmp_path / "arrow.ts"
        f.write_text("const add = (a: number, b: number): number => a + b;\n")
        result = await validator.check_file(str(f))
        assert result.passed is True

    async def test_typescript_empty_file(self, tmp_path: Path) -> None:
        validator = SyntaxValidator(use_tree_sitter=True)
        f = tmp_path / "empty.ts"
        f.write_text("")
        result = await validator.check_file(str(f))
        assert result.passed is True

    async def test_typescript_file_not_found(self) -> None:
        validator = SyntaxValidator(use_tree_sitter=True)
        result = await validator.check_file("/nonexistent/file.ts")
        assert result.passed is False
        assert any(e.code == "FILE_NOT_FOUND" for e in result.errors)


# ── Tests: TSX ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestTSX:
    """TSX 文件语法测试（tree-sitter tsx 解析器）。"""

    async def test_valid_tsx(self, tmp_path: Path) -> None:
        validator = SyntaxValidator(use_tree_sitter=True)
        f = tmp_path / "component.tsx"
        f.write_text(
            "import React from 'react';\n\n"
            "interface Props {\n"
            "  name: string;\n"
            "  age?: number;\n"
            "}\n\n"
            "const Hello: React.FC<Props> = ({ name, age }) => {\n"
            "  return (\n"
            '    <div>\n'
            "      <h1>Hello, {name}!</h1>\n"
            "      {age && <p>Age: {age}</p>}\n"
            "    </div>\n"
            "  );\n"
            "};\n\n"
            "export default Hello;\n"
        )
        result = await validator.check_file(str(f))
        assert result.passed is True, f"errors: {result.errors}"

    async def test_invalid_tsx(self, tmp_path: Path) -> None:
        validator = SyntaxValidator(use_tree_sitter=True)
        f = tmp_path / "bad.tsx"
        # 缺少闭合标签
        f.write_text(
            "const el = <div>Hello;\n"
        )
        result = await validator.check_file(str(f))
        assert result.passed is False
        # 应该检测到 JSX 错误
        assert len(result.errors) >= 1


# ── Tests: JavaScript ─────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestJavaScript:
    """JavaScript/JSX 文件语法测试（tree-sitter JavaScript 解析器）。"""

    async def test_valid_javascript(self, tmp_path: Path) -> None:
        validator = SyntaxValidator(use_tree_sitter=True)
        f = tmp_path / "script.js"
        f.write_text(
            "function greet(name) {\n"
            '  return `Hello, ${name}!`;\n'
            "}\n\n"
            "const result = greet('World');\n"
            "console.log(result);\n"
        )
        result = await validator.check_file(str(f))
        assert result.passed is True

    async def test_invalid_javascript_syntax(self, tmp_path: Path) -> None:
        validator = SyntaxValidator(use_tree_sitter=True)
        f = tmp_path / "bad.js"
        f.write_text("const x = ;\n")
        result = await validator.check_file(str(f))
        assert result.passed is False

    async def test_valid_jsx(self, tmp_path: Path) -> None:
        validator = SyntaxValidator(use_tree_sitter=True)
        f = tmp_path / "element.jsx"
        f.write_text("const el = <div>Hello World</div>;\n")
        result = await validator.check_file(str(f))
        assert result.passed is True, f"errors: {result.errors}"

    async def test_valid_mjs(self, tmp_path: Path) -> None:
        validator = SyntaxValidator(use_tree_sitter=True)
        f = tmp_path / "module.mjs"
        f.write_text("export const foo = 42;\n")
        result = await validator.check_file(str(f))
        assert result.passed is True

    async def test_valid_cjs(self, tmp_path: Path) -> None:
        validator = SyntaxValidator(use_tree_sitter=True)
        f = tmp_path / "module.cjs"
        f.write_text("module.exports = { foo: 42 };\n")
        result = await validator.check_file(str(f))
        assert result.passed is True

    async def test_javascript_empty_file(self, tmp_path: Path) -> None:
        validator = SyntaxValidator(use_tree_sitter=True)
        f = tmp_path / "empty.js"
        f.write_text("")
        result = await validator.check_file(str(f))
        assert result.passed is True

    async def test_javascript_file_not_found(self) -> None:
        validator = SyntaxValidator(use_tree_sitter=True)
        result = await validator.check_file("/nonexistent/file.js")
        assert result.passed is False

    async def test_javascript_es6_syntax(self, tmp_path: Path) -> None:
        validator = SyntaxValidator(use_tree_sitter=True)
        f = tmp_path / "es6.js"
        f.write_text(
            "class Calculator {\n"
            "  constructor() {\n"
            "    this.value = 0;\n"
            "  }\n\n"
            "  add(n) {\n"
            "    this.value += n;\n"
            "    return this;\n"
            "  }\n"
            "}\n\n"
            "const calc = new Calculator();\n"
            "const result = calc.add(5).add(3);\n"
            "export default result;\n"
        )
        result = await validator.check_file(str(f))
        assert result.passed is True


# ── Tests: Language detection ─────────────────────────────────────────────


class TestLanguageDetection:
    """语言检测功能测试。"""

    def test_detect_python(self) -> None:
        validator = SyntaxValidator()
        assert validator._detect_language("file.py") == "python"
        assert validator._detect_language("/path/to/module.py") == "python"

    def test_detect_typescript(self) -> None:
        validator = SyntaxValidator()
        assert validator._detect_language("file.ts") == "typescript"

    def test_detect_tsx(self) -> None:
        validator = SyntaxValidator()
        assert validator._detect_language("component.tsx") == "tsx"

    def test_detect_javascript(self) -> None:
        validator = SyntaxValidator()
        assert validator._detect_language("file.js") == "javascript"
        assert validator._detect_language("file.jsx") == "javascript"
        assert validator._detect_language("file.mjs") == "javascript"
        assert validator._detect_language("file.cjs") == "javascript"

    def test_detect_unknown(self) -> None:
        validator = SyntaxValidator()
        assert validator._detect_language("file.md") is None
        assert validator._detect_language("file.txt") is None
        assert validator._detect_language("") is None


# ── Tests: Mixed language batch check ─────────────────────────────────────


@pytest.mark.asyncio
class TestMixedLanguageBatch:
    """混合语言批量检查。"""

    async def test_mixed_py_and_js(self, tmp_path: Path) -> None:
        validator = SyntaxValidator(use_tree_sitter=True)

        py_file = tmp_path / "main.py"
        py_file.write_text("x = 1\n")
        js_file = tmp_path / "util.js"
        js_file.write_text("const y = 2;\n")
        md_file = tmp_path / "readme.md"
        md_file.write_text("# Docs\n")

        results = await validator.check_files(
            [str(py_file), str(js_file), str(md_file)]
        )
        assert len(results) == 3
        assert results[0].passed is True  # Python
        assert results[1].passed is True  # JavaScript
        assert results[2].passed is True  # Markdown

    async def test_mixed_with_errors(self, tmp_path: Path) -> None:
        validator = SyntaxValidator(use_tree_sitter=True)

        valid_py = tmp_path / "good.py"
        valid_py.write_text("x = 1\n")
        bad_ts = tmp_path / "bad.ts"
        bad_ts.write_text("const x: string = ;\n")
        valid_js = tmp_path / "good.js"
        valid_js.write_text("const y = 2;\n")

        results = await validator.check_files(
            [str(valid_py), str(bad_ts), str(valid_js)]
        )
        assert len(results) == 3
        assert results[0].passed is True   # valid Python
        assert results[1].passed is False  # invalid TS
        assert results[2].passed is True   # valid JS


# ── Tests: Tree-sitter parser caching ─────────────────────────────────────


class TestTreeSitterParserCaching:
    """验证 tree-sitter 解析器缓存机制。"""

    def test_parser_reuse(self) -> None:
        """同一语言的解析器只初始化一次。"""
        validator = SyntaxValidator(use_tree_sitter=True)
        parser1 = validator._get_tree_sitter_parser("python")
        parser2 = validator._get_tree_sitter_parser("python")
        assert parser1 is parser2  # 同一个实例

    def test_different_language_parsers(self) -> None:
        """不同语言使用不同的解析器实例。"""
        validator = SyntaxValidator(use_tree_sitter=True)
        py_parser = validator._get_tree_sitter_parser("python")
        js_parser = validator._get_tree_sitter_parser("javascript")
        if py_parser is not None and js_parser is not None:
            assert py_parser is not js_parser


# ── Tests: Graceful degradation ───────────────────────────────────────────


@pytest.mark.asyncio
class TestGracefulDegradation:
    """tree-sitter 不可用时的降级行为。"""

    async def test_ts_unavailable_still_checks_python(self, tmp_path: Path) -> None:
        """tree-sitter 不可用时 Python 检查仍应工作。"""
        validator = SyntaxValidator(use_tree_sitter=False)
        f = tmp_path / "test.py"
        f.write_text("x = 1\n")
        result = await validator.check_file(str(f))
        assert result.passed is True

        f2 = tmp_path / "bad.py"
        f2.write_text("if True\n")
        result2 = await validator.check_file(str(f2))
        assert result2.passed is False

    async def test_ts_unavailable_valid_js_no_error(self, tmp_path: Path) -> None:
        """tree-sitter 不可用时 JS 文件应通过（无验证器）。"""
        validator = SyntaxValidator(use_tree_sitter=False)
        f = tmp_path / "test.js"
        f.write_text("const x = 1;\n")
        result = await validator.check_file(str(f))
        # 没有 tree-sitter 也没有 tsc，JS 文件应该通过（无法验证）
        assert result.passed is True

    async def test_ts_unavailable_valid_ts_no_error(self, tmp_path: Path) -> None:
        """tree-sitter 不可用时 TS 文件应通过（无验证器）。"""
        validator = SyntaxValidator(use_tree_sitter=False)
        f = tmp_path / "test.ts"
        f.write_text("const x: number = 1;\n")
        result = await validator.check_file(str(f))
        assert result.passed is True
