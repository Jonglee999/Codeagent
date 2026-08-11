"""DependencyGraph — 分析模块间的 import/require 依赖关系，构建有向图。"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import networkx as nx

from codeagent.context_engine.dependency_parser import (
    DependencyRecord,
    TreeSitterDependencyAdapter,
)

# ── 导入语句匹配模式 ──────────────────────────────────────────────────────────

# Python import 模式
_PY_IMPORT_RE = re.compile(
    r"^\s*(?:from\s+(\S+)\s+)?import\s+(.+)$", re.MULTILINE
)
_PY_IMPORT_NAME_RE = re.compile(r"(\w+(?:\.\w+)*)")

# JS/TS import 模式
_JS_IMPORT_RE = re.compile(
    r"^\s*import\s+(?:(?:\{[^}]*\}|\*\s+as\s+\w+|\w+(?:,\s*(?:\{[^}]*\}|\*\s+as\s+\w+|\w+))?)\s+from\s+)?['\"]([^'\"]+)['\"]",
    re.MULTILINE,
)
_JS_REQUIRE_RE = re.compile(
    r"(?:const|let|var)\s+.+=\s+require\s*\(\s*['\"]([^'\"]+)['\"]\s*\)",
    re.MULTILINE,
)
_JS_DYNAMIC_IMPORT_RE = re.compile(
    r"import\s*\(\s*['\"]([^'\"]+)['\"]\s*\)", re.MULTILINE
)

# 需要排除的标准库/第三方包前缀（简化版）
_STDLIB_PYTHON: set[str] = {
    "os", "sys", "re", "json", "math", "datetime", "typing", "abc",
    "collections", "pathlib", "functools", "itertools", "logging",
    "asyncio", "fractions", "decimal", "hashlib", "base64",
    "dataclasses", "enum", "uuid", "copy", "inspect", "textwrap",
    "io", "tempfile", "shutil", "subprocess", "multiprocessing",
}

# 第三方包前缀（简化版）—— 不以 "." 开头的模块视为外部
_EXTERNAL_PREFIXES: set[str] = {
    "pytest", "flask", "fastapi", "django", "numpy", "pandas",
    "requests", "click", "rich", "pydantic", "networkx", "tiktoken",
    "tree_sitter", "langgraph", "litellm", "lancedb",
    "sentence_transformers", "structlog",
}


def _is_external(module_name: str) -> bool:
    """判断模块是否为外部依赖（非本地模块）。"""
    if module_name.startswith("."):
        return False  # 相对导入视为本地
    top_level = module_name.split(".")[0]
    if top_level in _STDLIB_PYTHON:
        return True
    if top_level in _EXTERNAL_PREFIXES:
        return True
    # 简单的启发式：不含下划线的单字母顶层名视为外部
    return False


def _normalize_module_to_path(
    module_name: str, base_file: str, project_root: str | None = None
) -> str | None:
    """将模块名转换为相对文件路径。

    正确处理相对导入和包结构。
    如果提供了 project_root，会检测文件系统中实际存在的路径。
    """
    if _is_external(module_name):
        return None

    root_path = Path(project_root) if project_root else None
    base_path = Path(base_file)

    # 处理相对导入（如 from .helper import foo, from ..models import user）
    if module_name.startswith("."):
        dot_count = len(module_name) - len(module_name.lstrip("."))
        rest = module_name.lstrip(".").split(".")

        # 从 base_file 的父目录开始上移
        parent = base_path.parent
        for _ in range(dot_count - 1):
            parent = parent.parent

        if not rest or not rest[0]:
            # from . import something → 父目录的 __init__.py
            if root_path:
                try:
                    rel = parent.relative_to(root_path)
                except ValueError:
                    return None
                rel_str = str(rel).replace("\\", "/")
                return f"{rel_str}/__init__.py" if rel_str else "__init__.py"
            return None

        # from .helper import foo → parent/helper.py
        rel_module = "/".join(rest)
        candidates = [
            f"{rel_module}.py",
            f"{rel_module}/__init__.py",
        ]

        if root_path:
            for candidate in candidates:
                full_path = parent / candidate
                if full_path.exists():
                    try:
                        rel = full_path.relative_to(root_path)
                        return str(rel).replace("\\", "/")
                    except ValueError:
                        continue

        # 找不到实际文件时，基于 parent 的相对路径
        if root_path:
            try:
                parent_rel = parent.relative_to(root_path)
                parent_str = str(parent_rel).replace("\\", "/")
                return f"{parent_str}/{candidates[0]}"
            except ValueError:
                return candidates[0]
        return None

    # 绝对导入
    parts = module_name.split(".")
    if not parts or not parts[0]:
        return None

    candidates: list[str] = [
        "/".join(parts) + ".py",
        "/".join(parts) + "/__init__.py",
    ]

    if root_path:
        for candidate in candidates:
            if (root_path / candidate).exists():
                return candidate

    return candidates[0]


class DependencyGraph:
    """分析模块间的 import/require 依赖关系，构建有向图。"""

    def __init__(self) -> None:
        self._graph: nx.DiGraph = nx.DiGraph()
        self._ext_to_lang: dict[str, str] = {
            ".py": "python",
            ".js": "javascript",
            ".jsx": "javascript",
            ".ts": "typescript",
            ".tsx": "tsx",
            ".mjs": "javascript",
            ".cjs": "javascript",
        }
        self._ast_adapter = TreeSitterDependencyAdapter()
        self._last_warnings: list[str] = []
        self._schema_version = 2

    async def build(self, project_root: str) -> None:
        """全量构建依赖图。"""
        self._graph.clear()
        self._last_warnings.clear()
        root = Path(project_root).resolve()
        if not root.is_dir():
            raise NotADirectoryError(f"Not a directory: {root}")
        self._project_root = str(root)

        # 收集所有源文件
        source_files: list[Path] = []
        for ext in self._ext_to_lang:
            source_files.extend(root.rglob(f"*{ext}"))

        # 排除常见非项目目录
        exclude_dirs = {
            ".git", "node_modules", "__pycache__", ".venv", "venv",
            ".mypy_cache", ".pytest_cache", ".ruff_cache",
        }
        source_files = [
            f for f in source_files
            if not any(d in f.parts for d in exclude_dirs)
        ]

        for file_path in source_files:
            rel_path = str(file_path.relative_to(root)).replace("\\", "/")
            deps = self._extract_dependency_records(file_path)
            self._graph.add_node(rel_path)
            for record in deps:
                dep_path = self._resolve_dependency(record.module, file_path)
                if dep_path:
                    self._graph.add_node(dep_path)
                    self._graph.add_edge(
                        rel_path, dep_path, source=record.source, kind=record.kind,
                    )
                else:
                    external = f"<external>{record.module}"
                    self._graph.add_node(external)
                    self._graph.add_edge(
                        rel_path, external, source=record.source, kind=record.kind,
                    )

    async def update_file(self, file_path: str) -> None:
        """增量更新单个文件的依赖。"""
        # 将路径统一为正斜杠格式，并尝试转为相对路径
        rel_path = self._resolve_rel_path(file_path)

        # 移除旧的出边
        if self._graph.has_node(rel_path):
            out_edges = list(self._graph.out_edges(rel_path))
            self._graph.remove_edges_from(out_edges)

        fp = Path(file_path).resolve()
        if not fp.exists():
            # 安全地删除节点（如果存在）
            if self._graph.has_node(rel_path):
                self._graph.remove_node(rel_path)
            return

        # 重新解析
        deps = self._extract_dependency_records(fp)
        self._graph.add_node(rel_path)
        for record in deps:
            dep_path = self._resolve_dependency(record.module, fp)
            if dep_path:
                self._graph.add_node(dep_path)
                self._graph.add_edge(
                    rel_path, dep_path, source=record.source, kind=record.kind,
                )
            else:
                external = f"<external>{record.module}"
                self._graph.add_node(external)
                self._graph.add_edge(
                    rel_path, external, source=record.source, kind=record.kind,
                )

    def _resolve_rel_path(self, file_path: str) -> str:
        """将文件路径转为相对于项目根目录的路径。"""
        normalized = file_path.replace("\\", "/")
        project_root = getattr(self, '_project_root', None)
        if project_root:
            root = project_root.replace("\\", "/").rstrip("/")
            if normalized.startswith(root + "/"):
                return normalized[len(root) + 1:]
        # 已经是相对路径
        return normalized

    def _resolve_dependency(self, module: str, importer: Path) -> str | None:
        """Resolve Python modules and JS/TS relative modules to project files."""
        if importer.suffix.lower() == ".py":
            return _normalize_module_to_path(
                module, str(importer), getattr(self, "_project_root", None),
            )
        if not module.startswith("."):
            return None
        root_value = getattr(self, "_project_root", None)
        if not root_value:
            return None
        root = Path(root_value)
        base = (importer.parent / module).resolve()
        candidates = [base]
        if not base.suffix:
            candidates.extend(base.with_suffix(ext) for ext in self._ext_to_lang if ext != ".py")
            candidates.extend(base / f"index{ext}" for ext in self._ext_to_lang if ext != ".py")
        for candidate in candidates:
            if candidate.is_file():
                try:
                    return candidate.relative_to(root).as_posix()
                except ValueError:
                    return None
        return None

    def get_dependents(self, file_path: str) -> list[str]:
        """谁依赖这个文件（反向依赖）。"""
        normalized = file_path.replace("\\", "/")
        if not self._graph.has_node(normalized):
            return []
        return list(self._graph.predecessors(normalized))

    def get_dependencies(self, file_path: str) -> list[str]:
        """这个文件依赖谁（正向依赖）。"""
        normalized = file_path.replace("\\", "/")
        if not self._graph.has_node(normalized):
            return []
        return list(self._graph.successors(normalized))

    def get_impact_scope(self, file_path: str) -> list[str]:
        """修改此文件影响哪些文件（传递闭包——反向 DFS）。

        返回按影响距离排序的列表（直接影响在前）。
        """
        normalized = file_path.replace("\\", "/")
        if not self._graph.has_node(normalized):
            return []

        # BFS 遍历反向图
        visited: set[str] = set()
        result: list[str] = []
        queue = [normalized]
        visited.add(normalized)

        while queue:
            current = queue.pop(0)
            for pred in self._graph.predecessors(current):
                if pred not in visited:
                    visited.add(pred)
                    result.append(pred)
                    queue.append(pred)

        return result

    def find_circular_dependencies(self) -> list[list[str]]:
        """检测循环依赖。"""
        cycles: list[list[str]] = []
        try:
            simple_cycles = list(nx.simple_cycles(self._graph))
            for cycle in simple_cycles:
                # 过滤掉外部依赖节点
                filtered = [n for n in cycle if not n.startswith("<external>")]
                if len(filtered) >= 2:
                    cycles.append(filtered)
        except (nx.NetworkXNoCycle, nx.NetworkXError):
            pass
        return cycles

    def to_dict(self) -> dict[str, Any]:
        """将依赖图序列化为字典（供 LLM 消费）。"""
        deps: dict[str, list[str]] = {}
        for node in self._graph.nodes():
            if node.startswith("<external>"):
                continue
            successors = list(self._graph.successors(node))
            internal = [
                s for s in successors if not s.startswith("<external>")
            ]
            if internal:
                deps[node] = internal
        return deps

    def edge_evidence(self, source: str, target: str) -> dict[str, Any]:
        """Return how one dependency edge was discovered."""
        return dict(self._graph.get_edge_data(source, target, default={}))

    def get_index_metadata(self) -> dict[str, Any]:
        return {
            "schema_version": self._schema_version,
            "warnings": list(self._last_warnings),
            "ast_edges": sum(
                data.get("source") == "ast"
                for _, _, data in self._graph.edges(data=True)
            ),
            "fallback_edges": sum(
                data.get("source") == "regex_fallback"
                for _, _, data in self._graph.edges(data=True)
            ),
        }

    @property
    def graph(self) -> nx.DiGraph:
        """获取底层 networkx 有向图。"""
        return self._graph

    # ── 内部方法 ──────────────────────────────────────────────────────────────

    def _extract_dependencies(self, file_path: Path) -> list[str]:
        """解析文件中的导入语句。"""
        ext = file_path.suffix.lower()
        try:
            code = file_path.read_text(encoding="utf-8", errors="ignore")
        except (OSError, PermissionError):
            return []

        if ext == ".py":
            return self._extract_python_imports(code)
        elif ext in (".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx"):
            return self._extract_js_imports(code)
        return []

    def _extract_dependency_records(self, file_path: Path) -> list[DependencyRecord]:
        ext = file_path.suffix.lower()
        language = self._ext_to_lang.get(ext)
        if language is None:
            return []
        try:
            code = file_path.read_text(encoding="utf-8", errors="ignore")
        except (OSError, PermissionError):
            return []
        try:
            return self._ast_adapter.extract(code, language)
        except Exception as exc:
            self._last_warnings.append(
                f"{file_path}: AST dependency parsing unavailable ({type(exc).__name__})"
            )
            modules = (
                self._extract_python_imports(code)
                if ext == ".py" else self._extract_js_imports(code)
            )
            return [
                DependencyRecord(module, "import", "regex_fallback")
                for module in modules
            ]

    def _extract_python_imports(self, code: str) -> list[str]:
        """提取 Python 导入语句。"""
        modules: list[str] = []

        for match in _PY_IMPORT_RE.finditer(code):
            from_part = match.group(1)
            import_part = match.group(2)

            if from_part:
                from_part = from_part.strip()
                if from_part == ".":
                    # from . import X, Y → 每个 X 是当前包的子模块
                    for name_match in _PY_IMPORT_NAME_RE.finditer(import_part):
                        module = name_match.group(1)
                        modules.append(f".{module}")
                elif from_part.startswith("."):
                    # from .foo import bar → 模块是 .foo
                    modules.append(from_part)
                else:
                    # from foo import bar → 模块是 foo
                    modules.append(from_part)
            else:
                # import X, Y
                for name_match in _PY_IMPORT_NAME_RE.finditer(import_part):
                    module = name_match.group(1)
                    modules.append(module.split(".")[0])

        return modules

    def _extract_js_imports(self, code: str) -> list[str]:
        """提取 JS/TS 导入语句。"""
        modules: list[str] = []

        for match in _JS_IMPORT_RE.finditer(code):
            module = match.group(1)
            if module and not module.startswith(".") and "/" in module:
                # 提取包名（@scope/name 或 name）
                if module.startswith("@"):
                    parts = module.split("/")
                    if len(parts) >= 2:
                        module = f"{parts[0]}/{parts[1]}"
                else:
                    module = module.split("/")[0]
            if module:
                modules.append(module)

        for match in _JS_REQUIRE_RE.finditer(code):
            module = match.group(1)
            if module:
                modules.append(module)

        for match in _JS_DYNAMIC_IMPORT_RE.finditer(code):
            module = match.group(1)
            if module:
                modules.append(module)

        return modules
