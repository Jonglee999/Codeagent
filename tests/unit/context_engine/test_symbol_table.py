"""SymbolTable 单元测试。"""

from __future__ import annotations

import os
import tempfile

import pytest

from codeagent.context_engine.symbol_table import Symbol, SymbolTable


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def temp_dir() -> str:
    """创建临时目录。"""
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    import shutil
    shutil.rmtree(tmpdir)


@pytest.fixture
def python_project(temp_dir: str) -> str:
    """创建包含 Python 代码的临时项目。"""
    # simple.py — 基本符号
    with open(os.path.join(temp_dir, "simple.py"), "w") as f:
        f.write("""
def hello(name):
    \"\"\"Say hello.\"\"\"
    return f"Hello, {name}"

def goodbye():
    return "bye"

class UserService:
    \"\"\"User service class.\"\"\"

    async def get_user(self, user_id: int):
        pass

    def create_user(self, name: str) -> None:
        pass

def _private_helper():
    pass
""")

    # empty.py — 空文件
    with open(os.path.join(temp_dir, "empty.py"), "w") as f:
        f.write("\n")

    # nogood.txt — 非 Python 文件（应被忽略）
    with open(os.path.join(temp_dir, "nogood.txt"), "w") as f:
        f.write("def not_a_python_function(): pass\n")

    return temp_dir


@pytest.fixture
def js_project(temp_dir: str) -> str:
    """创建包含 JS/TS 代码的临时项目。"""
    # app.js
    with open(os.path.join(temp_dir, "app.js"), "w") as f:
        f.write("""
function greet(name) {
    return "Hello, " + name;
}

class UserController {
    async getUser(id) {
        return { id };
    }

    static create(data) {
        return data;
    }
}

const helper = () => {
    return 42;
};

const fn = function() {
    return 0;
};
""")

    # types.ts
    with open(os.path.join(temp_dir, "types.ts"), "w") as f:
        f.write("""
interface User {
    name: string;
    age: number;
}

type ID = string;

function process(data: User): ID {
    return data.age;
}
""")

    return temp_dir


@pytest.fixture
def symbol_table() -> SymbolTable:
    return SymbolTable()


# ── Tests: Symbol dataclass ──────────────────────────────────────────────


class TestSymbol:
    def test_symbol_to_dict(self) -> None:
        sym = Symbol(
            name="my_func",
            kind="function_definition",
            file_path="/project/main.py",
            start_line=1,
            end_line=10,
            signature="def my_func():",
            docstring="\"\"\"Doc.\"\"\"",
        )
        d = sym.to_dict()
        assert d["name"] == "my_func"
        assert d["kind"] == "function_definition"
        assert d["file_path"] == "/project/main.py"
        assert d["start_line"] == 1
        assert d["end_line"] == 10
        assert d["signature"] == "def my_func():"
        assert d["docstring"] == "\"\"\"Doc.\"\"\""

    def test_symbol_to_dict_minimal(self) -> None:
        sym = Symbol(
            name="f",
            kind="function_definition",
            file_path="main.py",
            start_line=1,
            end_line=5,
        )
        d = sym.to_dict()
        assert d["name"] == "f"
        assert "signature" not in d
        assert "docstring" not in d


# ── Tests: SymbolTable — Python ──────────────────────────────────────────


class TestSymbolTablePython:
    @pytest.mark.asyncio
    async def test_build_finds_symbols(
        self, symbol_table: SymbolTable, python_project: str
    ) -> None:
        await symbol_table.build(python_project)
        symbols = symbol_table.get_all_symbols()
        # simple.py: hello, goodbye, UserService, get_user, create_user, _private_helper
        assert len(symbols) >= 6

    @pytest.mark.asyncio
    async def test_build_symbol_kinds(
        self, symbol_table: SymbolTable, python_project: str
    ) -> None:
        await symbol_table.build(python_project)
        names = {s.name: s.kind for s in symbol_table.get_all_symbols()}
        assert names.get("hello") == "function_definition"
        assert names.get("UserService") == "class_definition"
        assert names.get("goodbye") == "function_definition"

    @pytest.mark.asyncio
    async def test_async_function_kind(
        self, symbol_table: SymbolTable, python_project: str
    ) -> None:
        await symbol_table.build(python_project)
        syms = symbol_table.query("get_user")
        assert len(syms) == 1
        assert syms[0].kind == "async_function_definition"

    @pytest.mark.asyncio
    async def test_docstring_extraction(
        self, symbol_table: SymbolTable, python_project: str
    ) -> None:
        await symbol_table.build(python_project)
        syms = symbol_table.query("hello")
        assert len(syms) == 1
        assert syms[0].docstring is not None
        assert "Say hello" in syms[0].docstring

    @pytest.mark.asyncio
    async def test_class_docstring(
        self, symbol_table: SymbolTable, python_project: str
    ) -> None:
        await symbol_table.build(python_project)
        syms = symbol_table.query("UserService")
        assert len(syms) == 1
        assert syms[0].docstring is not None
        assert "User service" in syms[0].docstring

    @pytest.mark.asyncio
    async def test_query_exact_match(
        self, symbol_table: SymbolTable, python_project: str
    ) -> None:
        await symbol_table.build(python_project)
        results = symbol_table.query("hello")
        assert len(results) == 1
        assert results[0].name == "hello"

    @pytest.mark.asyncio
    async def test_query_no_match(
        self, symbol_table: SymbolTable, python_project: str
    ) -> None:
        await symbol_table.build(python_project)
        results = symbol_table.query("NonExistentSymbol")
        assert results == []

    @pytest.mark.asyncio
    async def test_fuzzy_search(
        self, symbol_table: SymbolTable, python_project: str
    ) -> None:
        await symbol_table.build(python_project)
        results = symbol_table.fuzzy_search("user")
        names = {s.name for s in results}
        assert "UserService" in names
        assert "get_user" in names
        assert "create_user" in names

    @pytest.mark.asyncio
    async def test_fuzzy_search_empty_query(
        self, symbol_table: SymbolTable, python_project: str
    ) -> None:
        await symbol_table.build(python_project)
        results = symbol_table.fuzzy_search("")
        assert results == []

    @pytest.mark.asyncio
    async def test_fuzzy_search_case_insensitive(
        self, symbol_table: SymbolTable, python_project: str
    ) -> None:
        await symbol_table.build(python_project)
        results = symbol_table.fuzzy_search("USER")
        assert any(s.name == "UserService" for s in results)

    @pytest.mark.asyncio
    async def test_get_symbols_in_file(
        self, symbol_table: SymbolTable, python_project: str
    ) -> None:
        await symbol_table.build(python_project)
        simple_path = os.path.join(python_project, "simple.py")
        symbols = symbol_table.get_symbols_in_file(simple_path)
        assert len(symbols) >= 6

    @pytest.mark.asyncio
    async def test_get_symbols_in_empty_file(
        self, symbol_table: SymbolTable, python_project: str
    ) -> None:
        await symbol_table.build(python_project)
        empty_path = os.path.join(python_project, "empty.py")
        symbols = symbol_table.get_symbols_in_file(empty_path)
        assert symbols == []

    @pytest.mark.asyncio
    async def test_to_json(
        self, symbol_table: SymbolTable, python_project: str
    ) -> None:
        await symbol_table.build(python_project)
        json_data = symbol_table.to_json()
        assert isinstance(json_data, list)
        assert len(json_data) >= 6
        for entry in json_data:
            assert "name" in entry
            assert "kind" in entry
            assert "file_path" in entry

    @pytest.mark.asyncio
    async def test_update_file(
        self, symbol_table: SymbolTable, python_project: str
    ) -> None:
        await symbol_table.build(python_project)
        simple_path = os.path.join(python_project, "simple.py")
        before = len(symbol_table.get_symbols_in_file(simple_path))

        # Append a new function
        with open(simple_path, "a") as f:
            f.write("\ndef new_function():\n    pass\n")

        await symbol_table.update_file(simple_path)
        after = len(symbol_table.get_symbols_in_file(simple_path))
        assert after == before + 1
        assert len(symbol_table.query("new_function")) == 1

    @pytest.mark.asyncio
    async def test_build_nonexistent_dir(self, symbol_table: SymbolTable) -> None:
        with pytest.raises(NotADirectoryError):
            await symbol_table.build("/nonexistent/path/xyz123")

    @pytest.mark.asyncio
    async def test_update_nonexistent_file(
        self, symbol_table: SymbolTable, python_project: str
    ) -> None:
        await symbol_table.build(python_project)
        # update_file on nonexistent file should not raise
        await symbol_table.update_file("/nonexistent/file.py")

    @pytest.mark.asyncio
    async def test_signature_extraction(
        self, symbol_table: SymbolTable, python_project: str
    ) -> None:
        await symbol_table.build(python_project)
        syms = symbol_table.query("hello")
        assert len(syms) == 1
        assert syms[0].signature is not None

    @pytest.mark.asyncio
    async def test_private_function(
        self, symbol_table: SymbolTable, python_project: str
    ) -> None:
        await symbol_table.build(python_project)
        results = symbol_table.query("_private_helper")
        assert len(results) == 1


# ── Tests: SymbolTable — JS/TS ───────────────────────────────────────────


class TestSymbolTableJavaScript:
    @pytest.mark.asyncio
    async def test_build_js_symbols(
        self, symbol_table: SymbolTable, js_project: str
    ) -> None:
        await symbol_table.build(js_project)
        symbols = symbol_table.get_all_symbols()
        # JS: greet, UserController, getUser, create, helper, fn
        # TS: User, ID, process
        assert len(symbols) >= 9

    @pytest.mark.asyncio
    async def test_js_function_declaration(
        self, symbol_table: SymbolTable, js_project: str
    ) -> None:
        await symbol_table.build(js_project)
        syms = symbol_table.query("greet")
        assert len(syms) == 1
        assert syms[0].kind == "function_declaration"

    @pytest.mark.asyncio
    async def test_js_class_declaration(
        self, symbol_table: SymbolTable, js_project: str
    ) -> None:
        await symbol_table.build(js_project)
        syms = symbol_table.query("UserController")
        assert len(syms) == 1
        assert syms[0].kind == "class_declaration"

    @pytest.mark.asyncio
    async def test_js_method_definition(
        self, symbol_table: SymbolTable, js_project: str
    ) -> None:
        await symbol_table.build(js_project)
        syms = symbol_table.query("getUser")
        assert len(syms) == 1
        assert syms[0].kind == "method_definition"

    @pytest.mark.asyncio
    async def test_js_arrow_function(
        self, symbol_table: SymbolTable, js_project: str
    ) -> None:
        await symbol_table.build(js_project)
        syms = symbol_table.query("helper")
        assert len(syms) == 1
        assert syms[0].kind == "arrow_function"

    @pytest.mark.asyncio
    async def test_ts_interface_declaration(
        self, symbol_table: SymbolTable, js_project: str
    ) -> None:
        await symbol_table.build(js_project)
        syms = symbol_table.query("User")
        assert any(s.kind == "interface_declaration" for s in syms)

    @pytest.mark.asyncio
    async def test_ts_type_alias(
        self, symbol_table: SymbolTable, js_project: str
    ) -> None:
        await symbol_table.build(js_project)
        syms = symbol_table.query("ID")
        assert any(s.kind == "type_alias_declaration" for s in syms)


# ── Tests: SymbolTable — Language configuration ──────────────────────────


class TestSymbolTableLanguageConfig:
    @pytest.mark.asyncio
    async def test_python_only(self, temp_dir: str) -> None:
        st = SymbolTable(languages=["python"])
        with open(os.path.join(temp_dir, "main.py"), "w") as f:
            f.write("def foo(): pass\n")
        with open(os.path.join(temp_dir, "app.js"), "w") as f:
            f.write("function bar() {}\n")
        await st.build(temp_dir)
        # Should only find Python symbols
        assert len(st.query("foo")) == 1
        assert len(st.query("bar")) == 0

    @pytest.mark.asyncio
    async def test_javascript_only(self, temp_dir: str) -> None:
        st = SymbolTable(languages=["javascript"])
        with open(os.path.join(temp_dir, "main.py"), "w") as f:
            f.write("def foo(): pass\n")
        with open(os.path.join(temp_dir, "app.js"), "w") as f:
            f.write("function bar() {}\n")
        await st.build(temp_dir)
        assert len(st.query("foo")) == 0
        assert len(st.query("bar")) == 1

    @pytest.mark.asyncio
    async def test_symtable_language_not_installed(self) -> None:
        # Should not crash with unsupported language
        SymbolTable(languages=["python", "rust"])
        # Should work fine (rust not supported but ignored)


# ── Tests: SymbolTable — Edge cases ──────────────────────────────────────


class TestSymbolTableEdgeCases:
    @pytest.mark.asyncio
    async def test_empty_project(self, temp_dir: str) -> None:
        st = SymbolTable()
        await st.build(temp_dir)
        assert len(st.get_all_symbols()) == 0

    @pytest.mark.asyncio
    async def test_no_source_files(self, temp_dir: str) -> None:
        st = SymbolTable()
        with open(os.path.join(temp_dir, "readme.txt"), "w") as f:
            f.write("not code\n")
        await st.build(temp_dir)
        assert len(st.get_all_symbols()) == 0

    @pytest.mark.asyncio
    async def test_binary_file_ignored(self, temp_dir: str) -> None:
        st = SymbolTable()
        with open(os.path.join(temp_dir, "data.py"), "wb") as f:
            f.write(b"\x00\x01\x02")
        await st.build(temp_dir)
        assert len(st.get_all_symbols()) == 0

    @pytest.mark.asyncio
    async def test_nested_directory(self, temp_dir: str) -> None:
        st = SymbolTable()
        nested = os.path.join(temp_dir, "a", "b", "c")
        os.makedirs(nested)
        with open(os.path.join(nested, "mod.py"), "w") as f:
            f.write("def deep_func(): pass\n")
        await st.build(temp_dir)
        assert len(st.query("deep_func")) == 1
