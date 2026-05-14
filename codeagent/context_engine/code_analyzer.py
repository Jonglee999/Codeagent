"""CodeAnalyzer — 联合 SymbolTable + DependencyGraph 的统一接口。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from codeagent.context_engine.dependency_graph import DependencyGraph
from codeagent.context_engine.symbol_table import Symbol, SymbolTable


class CodeAnalyzer:
    """联合 SymbolTable + DependencyGraph 的统一接口。"""

    def __init__(self, languages: list[str] | None = None) -> None:
        self.symbol_table = SymbolTable(languages)
        self.dependency_graph = DependencyGraph()
        self._project_root: str | None = None

    async def build(self, project_root: str) -> None:
        """全量构建符号表和依赖图。"""
        self._project_root = str(Path(project_root).resolve())
        await self.symbol_table.build(project_root)
        await self.dependency_graph.build(project_root)

    async def update_file(self, file_path: str) -> None:
        """增量更新符号表和依赖图。"""
        await self.symbol_table.update_file(file_path)
        await self.dependency_graph.update_file(file_path)

    def get_symbol_context(self, file_path: str) -> list[Symbol]:
        """获取文件中的符号列表。"""
        return self.symbol_table.get_symbols_in_file(file_path)

    def get_dependency_context(self, file_path: str) -> dict[str, Any]:
        """获取依赖信息摘要（供 ContextAssembler 使用）。"""
        rel_path = self._to_rel_path(file_path)
        dependencies = self.dependency_graph.get_dependencies(rel_path)
        dependents = self.dependency_graph.get_dependents(rel_path)
        impact_scope = self.dependency_graph.get_impact_scope(rel_path)

        deps = [d for d in dependencies if not d.startswith("<external>")]
        deps_ = [d for d in dependents if not d.startswith("<external>")]

        result: dict[str, Any] = {
            "file": rel_path,
            "dependencies": deps,
            "dependents": deps_,
            "impact_scope": impact_scope,
        }

        circles = self.dependency_graph.find_circular_dependencies()
        if circles:
            result["circular_dependencies"] = circles

        return result

    def search_symbol(self, name: str) -> list[Symbol]:
        """精确查询符号。"""
        return self.symbol_table.query(name)

    def fuzzy_search_symbol(self, name: str) -> list[Symbol]:
        """模糊搜索符号。"""
        return self.symbol_table.fuzzy_search(name)

    def get_impact_scope(self, file_path: str) -> list[str]:
        """获取修改文件的影晌范围。"""
        rel_path = self._to_rel_path(file_path)
        return self.dependency_graph.get_impact_scope(rel_path)

    def _to_rel_path(self, file_path: str) -> str:
        """将绝对路径转换为相对路径。"""
        normalized = file_path.replace("\\", "/")
        if self._project_root and normalized.startswith(self._project_root.replace("\\", "/")):
            root = self._project_root.replace("\\", "/")
            if normalized.startswith(root + "/"):
                return normalized[len(root) + 1:]
            elif normalized == root:
                return "."
        return normalized
