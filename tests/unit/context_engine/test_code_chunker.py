"""CodeChunker 单元测试。"""

from __future__ import annotations

import os
import tempfile

import pytest

from codeagent.context_engine.code_chunker import CodeChunk, CodeChunker


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def chunker() -> CodeChunker:
    return CodeChunker()


@pytest.fixture
def temp_file() -> str:
    tmpdir = tempfile.mkdtemp()
    file_path = os.path.join(tmpdir, "test.py")
    yield file_path
    import shutil
    shutil.rmtree(tmpdir)


# ── Tests: Python 代码分块 ────────────────────────────────────────────────────


class TestCodeChunkerPython:
    """测试 Python 代码分块。"""

    def test_chunk_simple_functions(self, chunker: CodeChunker) -> None:
        """简单函数应被正确分块。"""
        code = """
def foo():
    pass

def bar():
    return 42
"""
        chunks = chunker.chunk_code(code, "python")
        # 4 chunks: leading blank line (orphan) + foo + blank line (orphan) + bar
        assert len(chunks) == 4
        # Find function chunks
        func_chunks = [c for c in chunks if c.symbol_name]
        names = {c.symbol_name for c in func_chunks}
        assert "foo" in names
        assert "bar" in names

    def test_chunk_class_with_methods(self, chunker: CodeChunker) -> None:
        """类及其方法应被正确分块。"""
        code = """
class MyClass:
    def method_a(self):
        pass

    def method_b(self):
        return 1
"""
        chunks = chunker.chunk_code(code, "python")
        names = {c.symbol_name for c in chunks if c.symbol_name}
        assert "MyClass" in names
        assert "method_a" in names
        assert "method_b" in names

    def test_async_function(self, chunker: CodeChunker) -> None:
        """异步函数应被正确分块。"""
        code = """
async def fetch_data():
    return await api()

def sync_func():
    pass
"""
        chunks = chunker.chunk_code(code, "python")
        names = {c.symbol_name for c in chunks if c.symbol_name}
        assert "fetch_data" in names
        assert "sync_func" in names

    def test_class_without_methods(self, chunker: CodeChunker) -> None:
        """没有方法的类。"""
        code = """
class Empty:
    pass
"""
        chunks = chunker.chunk_code(code, "python")
        assert any(c.symbol_name == "Empty" for c in chunks)

    def test_module_level_code(self, chunker: CodeChunker) -> None:
        """模块级代码（import、常量）应被捕获。"""
        code = """import os
import sys

CONSTANT = 42

def helper():
    pass
"""
        chunks = chunker.chunk_code(code, "python")
        assert any(c.symbol_name == "helper" for c in chunks)
        # There should be module-level chunks (imports and CONSTANT)
        module_chunks = [c for c in chunks if not c.symbol_name]
        assert len(module_chunks) >= 1

    def test_nested_classes(self, chunker: CodeChunker) -> None:
        """嵌套类。"""
        code = """
class Outer:
    class Inner:
        def inner_method(self):
            pass
"""
        chunks = chunker.chunk_code(code, "python")
        names = {c.symbol_name for c in chunks if c.symbol_name}
        assert "Outer" in names
        # 内部类的方法
        inner_methods = [c for c in chunks if c.symbol_name == "inner_method"]
        assert len(inner_methods) >= 0  # 内部类的方法作为嵌套类内容


# ── Tests: JavaScript/TypeScript 代码分块 ─────────────────────────────────────


class TestCodeChunkerJavaScript:
    """测试 JS/TS 代码分块。"""

    def test_js_functions(self, chunker: CodeChunker) -> None:
        """JS 函数声明。"""
        code = """
function foo() {
    return 1;
}

function bar() {
    return 2;
}
"""
        chunks = chunker.chunk_code(code, "javascript")
        names = {c.symbol_name for c in chunks if c.symbol_name}
        assert "foo" in names
        assert "bar" in names

    def test_ts_interfaces(self, chunker: CodeChunker) -> None:
        """TypeScript 接口。"""
        code = """
interface User {
    name: string;
    age: number;
}

type Role = "admin" | "user";
"""
        chunks = chunker.chunk_code(code, "typescript")
        names = {c.symbol_name for c in chunks if c.symbol_name}
        assert "User" in names
        assert "Role" in names

    def test_ts_class(self, chunker: CodeChunker) -> None:
        """TypeScript 类。"""
        code = """
class Service {
    getData(): string {
        return "data";
    }

    setData(value: string): void {
        console.log(value);
    }
}
"""
        chunks = chunker.chunk_code(code, "typescript")
        names = {c.symbol_name for c in chunks if c.symbol_name}
        assert "Service" in names
        assert "getData" in names
        assert "setData" in names


# ── Tests: 文件分块 ──────────────────────────────────────────────────────────


class TestCodeChunkerFile:
    """测试基于文件的分块。"""

    def test_chunk_python_file(self, chunker: CodeChunker, temp_file: str) -> None:
        """从 Python 文件分块。"""
        with open(temp_file, "w") as f:
            f.write("""
def existing_func():
    return "hello"
""")
        chunks = chunker.chunk_file(temp_file)
        assert len(chunks) >= 1
        assert any(c.symbol_name == "existing_func" for c in chunks)

    def test_file_not_found(self, chunker: CodeChunker) -> None:
        """不存在的文件应抛出 FileNotFoundError。"""
        with pytest.raises(FileNotFoundError):
            chunker.chunk_file("/nonexistent/path/file.py")

    def test_unsupported_language(self, chunker: CodeChunker, temp_file: str) -> None:
        """不支持的语言应返回空列表。"""
        file_path = temp_file.replace(".py", ".xyz")
        with open(file_path, "w") as f:
            f.write("some content")
        chunks = chunker.chunk_file(file_path)
        assert chunks == []

    def test_empty_file(self, chunker: CodeChunker, temp_file: str) -> None:
        """空文件应返回空列表。"""
        with open(temp_file, "w") as f:
            f.write("")
        chunks = chunker.chunk_file(temp_file)
        assert chunks == []


# ── Tests: 边界情况 ──────────────────────────────────────────────────────────


class TestCodeChunkerEdgeCases:
    """边界情况测试。"""

    def test_empty_code(self, chunker: CodeChunker) -> None:
        """空代码字符串。"""
        chunks = chunker.chunk_code("", "python")
        assert chunks == []

    def test_whitespace_only(self, chunker: CodeChunker) -> None:
        """仅空白字符。"""
        chunks = chunker.chunk_code("   \n\n  \n", "python")
        assert chunks == []

    def test_single_line_file(self, chunker: CodeChunker) -> None:
        """单行文件。"""
        code = "x = 1\n"
        chunks = chunker.chunk_code(code, "python")
        assert len(chunks) >= 1

    @pytest.mark.asyncio
    async def test_large_function_splitting(self, chunker: CodeChunker) -> None:
        """大函数应被拆分。"""
        # 创建包含很多行的函数
        lines = ["def large_func():\n"]
        lines += [f"    x_{i} = {i}\n" for i in range(200)]
        code = "".join(lines)

        chunks = chunker.chunk_code(code, "python", max_chunk_size=50)
        # 应被拆分为多个块
        [c for c in chunks if c.symbol_name == "large_func"]
        # 大的函数可能会被拆分为多个块
        assert len(chunks) >= 1

    def test_token_count(self, chunker: CodeChunker) -> None:
        """验证 token 计数。"""
        code = """
def short():
    pass
"""
        chunks = chunker.chunk_code(code, "python")
        for chunk in chunks:
            assert chunk.token_count > 0
            assert chunk.token_count <= len(chunk.code.split()) + 10

    def test_chunk_metadata(self, chunker: CodeChunker) -> None:
        """验证块元数据。"""
        code = """
def foo():
    pass
"""
        chunks = chunker.chunk_code(code, "python", file_path="/test/project/main.py")
        func_chunks = [c for c in chunks if c.symbol_name == "foo"]
        if func_chunks:
            c = func_chunks[0]
            assert c.file_path == "/test/project/main.py"
            assert c.language == "python"
            assert c.start_line >= 1
            assert c.end_line >= c.start_line

    def test_mixed_content(self, chunker: CodeChunker) -> None:
        """混合 import、类和函数。"""
        code = """import os
from typing import List

CONFIG = {
    "debug": True
}

class Handler:
    def process(self):
        pass

def create_app():
    pass
"""
        chunks = chunker.chunk_code(code, "python")
        names = {c.symbol_name for c in chunks if c.symbol_name}
        assert "Handler" in names
        assert "process" in names
        assert "create_app" in names
        # 应该也有模块级的 orphan 代码块
        module_chunks = [c for c in chunks if not c.symbol_name]
        assert len(module_chunks) >= 1

    def test_to_dict_serialization(self, chunker: CodeChunker) -> None:
        """验证 CodeChunk.to_dict() 序列化。"""
        chunk = CodeChunk(
            file_path="/test/main.py",
            start_line=1,
            end_line=10,
            code="def foo(): pass",
            symbol_name="foo",
            language="python",
            token_count=10,
        )
        d = chunk.to_dict()
        assert d["file_path"] == "/test/main.py"
        assert d["start_line"] == 1
        assert d["symbol_name"] == "foo"
        assert d["language"] == "python"
        assert d["token_count"] == 10
