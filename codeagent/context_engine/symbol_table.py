"""SymbolTable — 使用 tree-sitter 解析代码 AST，提取所有符号的定义位置。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tree_sitter import Language, Node, Parser

# ── 语言解析器初始化 ──────────────────────────────────────────────────────────

_LANGUAGES: dict[str, Language] = {}
_PARSERS: dict[str, Parser] = {}


def _init_parser(language: str) -> Parser:
    """延迟初始化指定语言的 tree-sitter 解析器。"""
    if language in _PARSERS:
        return _PARSERS[language]

    if language == "python":
        from tree_sitter_python import language as _py_lang

        lang = Language(_py_lang())
    elif language == "javascript":
        from tree_sitter_javascript import language as _js_lang

        lang = Language(_js_lang())
    elif language in ("typescript", "tsx"):
        from tree_sitter_typescript import language_tsx, language_typescript

        capsule = language_tsx() if language == "tsx" else language_typescript()
        lang = Language(capsule)
    else:
        raise ValueError(f"Unsupported language: {language}")

    parser = Parser()
    parser.language = lang
    _LANGUAGES[language] = lang
    _PARSERS[language] = parser
    return parser


def _get_language(language: str) -> Language:
    """获取指定语言的 Language 实例。"""
    if language not in _LANGUAGES:
        _init_parser(language)
    return _LANGUAGES[language]


# ── 文件扩展名 → 语言映射 ─────────────────────────────────────────────────────

_EXT_TO_LANG: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".mjs": "javascript",
    ".cjs": "javascript",
}


def _detect_language(file_path: str) -> str | None:
    """根据文件扩展名检测语言。"""
    ext = Path(file_path).suffix.lower()
    return _EXT_TO_LANG.get(ext)


# ── Symbol 数据类 ────────────────────────────────────────────────────────────


@dataclass
class Symbol:
    """代码符号信息。"""

    name: str
    kind: str
    file_path: str
    start_line: int
    end_line: int
    signature: str | None = None
    docstring: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典。"""
        d: dict[str, Any] = {
            "name": self.name,
            "kind": self.kind,
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
        }
        if self.signature is not None:
            d["signature"] = self.signature
        if self.docstring is not None:
            d["docstring"] = self.docstring
        return d


# ── 符号提取逻辑 ──────────────────────────────────────────────────────────────


def _extract_python_symbols(
    file_path: str, code: bytes
) -> list[Symbol]:
    """从 Python 源码中提取符号。"""
    parser = _init_parser("python")
    tree = parser.parse(code)
    root = tree.root_node

    symbols: list[Symbol] = []
    _walk_python(root, file_path, code, symbols)
    return symbols


def _walk_python(
    node: Node, file_path: str, code: bytes, symbols: list[Symbol]
) -> None:
    """递归遍历 Python AST 提取符号。"""
    if node.type == "function_definition":
        sym = _make_python_function_symbol(node, file_path, code)
        if sym:
            symbols.append(sym)
    elif node.type == "class_definition":
        sym = _make_python_class_symbol(node, file_path, code)
        if sym:
            symbols.append(sym)
            # 递归提取类中的方法
            body = node.child_by_field_name("body")
            if body:
                for child in body.children:
                    _walk_python(child, file_path, code, symbols)
            return

    for child in node.children:
        _walk_python(child, file_path, code, symbols)


def _make_python_function_symbol(
    node: Node, file_path: str, code: bytes
) -> Symbol | None:
    """从 function_definition 节点创建 Symbol。"""
    name_node = node.child_by_field_name("name")
    if not name_node:
        return None

    name = name_node.text.decode("utf-8", errors="replace")
    start_line = node.start_point[0] + 1  # 1-indexed
    end_line = node.end_point[0] + 1

    # 检测是否是 async function
    kind = "function_definition"
    for child in node.children:
        if child.type == "async":
            kind = "async_function_definition"
            break

    # 提取签名（从 def/async 到 : 或 body 开始前）
    signature = _extract_text(code, node.start_byte, node.end_byte)

    # 提取 docstring
    docstring = _extract_python_docstring(node)

    return Symbol(
        name=name,
        kind=kind,
        file_path=file_path,
        start_line=start_line,
        end_line=end_line,
        signature=signature,
        docstring=docstring,
    )


def _make_python_class_symbol(
    node: Node, file_path: str, code: bytes
) -> Symbol | None:
    """从 class_definition 节点创建 Symbol。"""
    name_node = node.child_by_field_name("name")
    if not name_node:
        return None

    name = name_node.text.decode("utf-8", errors="replace")
    start_line = node.start_point[0] + 1
    end_line = node.end_point[0] + 1
    signature = _extract_text(code, node.start_byte, node.end_byte)
    docstring = _extract_python_docstring(node)

    return Symbol(
        name=name,
        kind="class_definition",
        file_path=file_path,
        start_line=start_line,
        end_line=end_line,
        signature=signature,
        docstring=docstring,
    )


def _extract_python_docstring(node: Node) -> str | None:
    """提取 Python 函数/类定义中的 docstring。"""
    body = node.child_by_field_name("body")
    if not body:
        return None
    for child in body.children:
        if child.type == "expression_statement":
            for sub in child.children:
                if sub.type == "string":
                    return sub.text.decode("utf-8", errors="replace").strip()
            break  # 只检查第一个 expression_statement
    return None


def _extract_js_ts_symbols(
    file_path: str, code: bytes, language: str
) -> list[Symbol]:
    """从 JS/TS 源码中提取符号。"""
    parser = _init_parser(language)
    tree = parser.parse(code)
    root = tree.root_node

    symbols: list[Symbol] = []
    _walk_js_ts(root, file_path, code, symbols)
    return symbols


def _walk_js_ts(
    node: Node, file_path: str, code: bytes, symbols: list[Symbol]
) -> None:
    """递归遍历 JS/TS AST 提取符号。"""
    if node.type == "function_declaration":
        sym = _make_js_function_symbol(node, file_path, code, "function_declaration")
        if sym:
            symbols.append(sym)
    elif node.type == "class_declaration":
        sym = _make_js_class_symbol(node, file_path, code)
        if sym:
            symbols.append(sym)
            # 递归提取类中的方法
            body = node.child_by_field_name("body")
            if body:
                for child in body.children:
                    _walk_js_ts(child, file_path, code, symbols)
            return
    elif node.type == "interface_declaration":
        sym = _make_js_interface_symbol(node, file_path, code)
        if sym:
            symbols.append(sym)
    elif node.type == "type_alias_declaration":
        sym = _make_js_type_alias_symbol(node, file_path, code)
        if sym:
            symbols.append(sym)
    elif node.type in ("lexical_declaration", "variable_declaration"):
        # const x = () => {} 或 const x = function() {} 或 var x = function() {}
        for declarator in node.children:
            if declarator.type == "variable_declarator":
                value = declarator.child_by_field_name("value")
                if value:
                    if value.type == "arrow_function":
                        sym = _make_js_arrow_function_symbol(
                            declarator, file_path, code
                        )
                        if sym:
                            symbols.append(sym)
                    elif value.type in ("function", "function_expression"):
                        sym = _make_js_function_symbol(
                            value, file_path, code, "function_declaration"
                        )
                        if sym:
                            name_node = declarator.child_by_field_name("name")
                            if name_node:
                                sym.name = name_node.text.decode(
                                    "utf-8", errors="replace"
                                )
                            symbols.append(sym)
        return
    elif node.type == "method_definition":
        sym = _make_js_method_symbol(node, file_path, code)
        if sym:
            symbols.append(sym)
            return

    for child in node.children:
        _walk_js_ts(child, file_path, code, symbols)


def _make_js_function_symbol(
    node: Node, file_path: str, code: bytes, kind: str
) -> Symbol | None:
    """从 JS/TS function_declaration/function_expression 节点创建 Symbol。

    匿名函数返回 name=""，由调用者用变量名填充。
    """
    name_node = node.child_by_field_name("name")
    name = (name_node.text.decode("utf-8", errors="replace")
            if name_node else "")
    start_line = node.start_point[0] + 1
    end_line = node.end_point[0] + 1
    signature = _extract_text(code, node.start_byte, node.end_byte)

    return Symbol(
        name=name,
        kind=kind,
        file_path=file_path,
        start_line=start_line,
        end_line=end_line,
        signature=signature,
    )


def _make_js_class_symbol(
    node: Node, file_path: str, code: bytes
) -> Symbol | None:
    """从 JS/TS class_declaration 节点创建 Symbol。"""
    name_node = node.child_by_field_name("name")
    if not name_node:
        return None

    name = name_node.text.decode("utf-8", errors="replace")
    start_line = node.start_point[0] + 1
    end_line = node.end_point[0] + 1
    signature = _extract_text(code, node.start_byte, node.end_byte)

    return Symbol(
        name=name,
        kind="class_declaration",
        file_path=file_path,
        start_line=start_line,
        end_line=end_line,
        signature=signature,
    )


def _make_js_method_symbol(
    node: Node, file_path: str, code: bytes
) -> Symbol | None:
    """从 JS/TS method_definition 节点创建 Symbol。"""
    name_node = node.child_by_field_name("name")
    if not name_node:
        return None

    name = name_node.text.decode("utf-8", errors="replace")
    start_line = node.start_point[0] + 1
    end_line = node.end_point[0] + 1
    signature = _extract_text(code, node.start_byte, node.end_byte)

    return Symbol(
        name=name,
        kind="method_definition",
        file_path=file_path,
        start_line=start_line,
        end_line=end_line,
        signature=signature,
    )


def _make_js_arrow_function_symbol(
    declarator: Node, file_path: str, code: bytes
) -> Symbol | None:
    """从 JS/TS variable_declarator 中的箭头函数创建 Symbol。"""
    name_node = declarator.child_by_field_name("name")
    if not name_node:
        return None

    name = name_node.text.decode("utf-8", errors="replace")
    arrow_node = declarator.child_by_field_name("value")
    if not arrow_node:
        return None

    start_line = declarator.start_point[0] + 1
    end_line = declarator.end_point[0] + 1
    signature = _extract_text(code, declarator.start_byte, declarator.end_byte)

    return Symbol(
        name=name,
        kind="arrow_function",
        file_path=file_path,
        start_line=start_line,
        end_line=end_line,
        signature=signature,
    )


def _make_js_interface_symbol(
    node: Node, file_path: str, code: bytes
) -> Symbol | None:
    """从 TS interface_declaration 节点创建 Symbol。"""
    name_node = node.child_by_field_name("name")
    if not name_node:
        return None

    name = name_node.text.decode("utf-8", errors="replace")
    start_line = node.start_point[0] + 1
    end_line = node.end_point[0] + 1
    signature = _extract_text(code, node.start_byte, node.end_byte)

    return Symbol(
        name=name,
        kind="interface_declaration",
        file_path=file_path,
        start_line=start_line,
        end_line=end_line,
        signature=signature,
    )


def _make_js_type_alias_symbol(
    node: Node, file_path: str, code: bytes
) -> Symbol | None:
    """从 TS type_alias_declaration 节点创建 Symbol。"""
    name_node = node.child_by_field_name("name")
    if not name_node:
        return None

    name = name_node.text.decode("utf-8", errors="replace")
    start_line = node.start_point[0] + 1
    end_line = node.end_point[0] + 1
    signature = _extract_text(code, node.start_byte, node.end_byte)

    return Symbol(
        name=name,
        kind="type_alias_declaration",
        file_path=file_path,
        start_line=start_line,
        end_line=end_line,
        signature=signature,
    )


def _extract_text(code: bytes, start: int, end: int) -> str:
    """从字节码中提取文本。"""
    return code[start:end].decode("utf-8", errors="replace")


# ── SymbolTable 主类 ─────────────────────────────────────────────────────────


class SymbolTable:
    """使用 tree-sitter 解析代码，提取所有符号的定义位置。"""

    def __init__(self, languages: list[str] | None = None) -> None:
        """初始化 SymbolTable。

        Args:
            languages: 支持的语言列表，默认 ["python", "typescript", "javascript"]
        """
        self._languages = languages or ["python", "typescript", "javascript"]
        self._symbols: list[Symbol] = []
        self._file_symbols: dict[str, list[Symbol]] = {}
        self._name_index: dict[str, list[Symbol]] = {}

        # 预初始化解析器
        for lang in self._languages:
            try:
                _init_parser(lang)
            except (ImportError, ValueError):
                pass  # 仅记录，后续解析时再报错

    async def build(self, project_root: str) -> None:
        """全量构建符号表（遍历项目所有源文件）。"""
        self._symbols.clear()
        self._file_symbols.clear()
        self._name_index.clear()

        root = Path(project_root).resolve()
        if not root.is_dir():
            raise NotADirectoryError(f"Not a directory: {root}")

        for file_path in _iter_source_files(root, self._languages):
            symbols = self._parse_file(str(file_path))
            for sym in symbols:
                self._add_symbol(sym)

    async def update_file(self, file_path: str) -> None:
        """增量更新：仅重新解析变更的文件。"""
        # 移除旧符号
        old_symbols = self._file_symbols.pop(file_path, [])
        for sym in old_symbols:
            name_symbols = self._name_index.get(sym.name, [])
            if name_symbols:
                try:
                    name_symbols.remove(sym)
                except ValueError:
                    pass
                if not name_symbols:
                    del self._name_index[sym.name]
            try:
                self._symbols.remove(sym)
            except ValueError:
                pass

        # 解析新符号
        symbols = self._parse_file(file_path)
        for sym in symbols:
            self._add_symbol(sym)

    def query(self, name: str) -> list[Symbol]:
        """精确查询符号（按名称完全匹配）。"""
        return list(self._name_index.get(name, []))

    def fuzzy_search(self, name: str) -> list[Symbol]:
        """模糊搜索符号（按名称部分匹配）。"""
        if not name:
            return []
        lower = name.lower()
        results: list[Symbol] = []
        for sym_name, symbols in self._name_index.items():
            if lower in sym_name.lower():
                results.extend(symbols)
        return results

    def get_symbols_in_file(self, file_path: str) -> list[Symbol]:
        """获取指定文件中的所有符号。"""
        return list(self._file_symbols.get(file_path, []))

    def get_all_symbols(self) -> list[Symbol]:
        """获取所有符号。"""
        return list(self._symbols)

    def to_json(self) -> list[dict]:
        """序列化为 JSON 供 LLM 消费。"""
        return [sym.to_dict() for sym in self._symbols]

    # ── 内部方法 ──────────────────────────────────────────────────────────────

    def _parse_file(self, file_path: str) -> list[Symbol]:
        """解析单个文件，提取符号。"""
        language = _detect_language(file_path)
        if language is None:
            return []
        if language not in self._languages:
            return []

        try:
            with open(file_path, "rb") as f:
                code = f.read()
        except (OSError, PermissionError):
            return []

        if not code:
            return []

        if language == "python":
            return _extract_python_symbols(file_path, code)
        elif language in ("javascript", "typescript", "tsx"):
            return _extract_js_ts_symbols(file_path, code, language)
        else:
            return []

    def _add_symbol(self, sym: Symbol) -> None:
        """添加符号到索引。"""
        self._symbols.append(sym)

        file_syms = self._file_symbols.setdefault(sym.file_path, [])
        file_syms.append(sym)

        name_syms = self._name_index.setdefault(sym.name, [])
        name_syms.append(sym)


# ── 工具函数 ──────────────────────────────────────────────────────────────────


def _iter_source_files(
    root: Path, languages: list[str]
) -> list[Path]:
    """遍历项目目录中的源文件。"""
    extensions: set[str] = set()
    lang_to_exts = {
        "python": {".py"},
        "typescript": {".ts", ".tsx"},
        "javascript": {".js", ".jsx", ".mjs", ".cjs"},
    }
    for lang in languages:
        extensions.update(lang_to_exts.get(lang, set()))

    # 默认排除目录
    exclude_dirs = {
        ".git", "node_modules", "__pycache__", ".venv", "venv",
        ".mypy_cache", ".pytest_cache", ".ruff_cache", ".egg-info",
        ".codeagent", ".claude", ".idea", ".vscode",
    }

    files: list[Path] = []
    try:
        for entry in root.rglob("*"):
            if entry.is_dir() and entry.name in exclude_dirs:
                continue
            if entry.is_file() and entry.suffix.lower() in extensions:
                files.append(entry)
    except (PermissionError, OSError):
        pass

    return files
