"""CodeAnalyzer 单元测试。"""

from __future__ import annotations

import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from codeagent.context_engine.code_analyzer import CodeAnalyzer
from codeagent.context_engine.dependency_graph import DependencyGraph
from codeagent.context_engine.symbol_table import Symbol, SymbolTable


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def temp_dir() -> str:
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    import shutil
    shutil.rmtree(tmpdir)


@pytest.fixture
def sample_project(temp_dir: str) -> str:
    """创建简单的测试项目。"""
    with open(os.path.join(temp_dir, "main.py"), "w") as f:
        f.write("""
from services.auth import AuthService

def main():
    pass
""")

    svc_dir = os.path.join(temp_dir, "services")
    os.makedirs(svc_dir)
    with open(os.path.join(svc_dir, "__init__.py"), "w") as f:
        f.write("")
    with open(os.path.join(svc_dir, "auth.py"), "w") as f:
        f.write("""
class AuthService:
    \"\"\"Auth service.\"\"\"
    def login(self):
        pass
""")

    return temp_dir


# ── Tests: CodeAnalyzer — Build ──────────────────────────────────────────


class TestCodeAnalyzerBuild:
    @pytest.mark.asyncio
    async def test_build_creates_analyzer(self, sample_project: str) -> None:
        ca = CodeAnalyzer()
        await ca.build(sample_project)
        assert ca._project_root is not None
        assert ca.symbol_table is not None
        assert ca.dependency_graph is not None

    @pytest.mark.asyncio
    async def test_build_finds_symbols(self, sample_project: str) -> None:
        ca = CodeAnalyzer()
        await ca.build(sample_project)
        auth_path = os.path.join(sample_project, "services", "auth.py")
        symbols = ca.get_symbol_context(auth_path)
        assert len(symbols) >= 2  # AuthService class + login method
        names = {s.name for s in symbols}
        assert "AuthService" in names

    @pytest.mark.asyncio
    async def test_build_creates_dependency_info(
        self, sample_project: str
    ) -> None:
        ca = CodeAnalyzer()
        await ca.build(sample_project)
        auth_path = os.path.join(sample_project, "services", "auth.py")
        ctx = ca.get_dependency_context(auth_path)
        assert "file" in ctx
        assert "dependencies" in ctx
        assert "dependents" in ctx
        assert "impact_scope" in ctx

    @pytest.mark.asyncio
    async def test_build_empty_project(self, temp_dir: str) -> None:
        ca = CodeAnalyzer()
        await ca.build(temp_dir)
        assert len(ca.symbol_table.get_all_symbols()) == 0


# ── Tests: CodeAnalyzer — Symbol operations ──────────────────────────────


class TestCodeAnalyzerSymbols:
    @pytest.mark.asyncio
    async def test_search_symbol(self, sample_project: str) -> None:
        ca = CodeAnalyzer()
        await ca.build(sample_project)
        results = ca.search_symbol("AuthService")
        assert len(results) == 1
        assert results[0].name == "AuthService"

    @pytest.mark.asyncio
    async def test_search_symbol_not_found(self, sample_project: str) -> None:
        ca = CodeAnalyzer()
        await ca.build(sample_project)
        results = ca.search_symbol("NonExistent")
        assert results == []

    @pytest.mark.asyncio
    async def test_fuzzy_search_symbol(self, sample_project: str) -> None:
        ca = CodeAnalyzer()
        await ca.build(sample_project)
        results = ca.fuzzy_search_symbol("auth")
        assert len(results) >= 1
        assert any(s.name == "AuthService" for s in results)

    @pytest.mark.asyncio
    async def test_get_symbol_context(self, sample_project: str) -> None:
        ca = CodeAnalyzer()
        await ca.build(sample_project)
        auth_path = os.path.join(sample_project, "services", "auth.py")
        symbols = ca.get_symbol_context(auth_path)
        assert all(isinstance(s, Symbol) for s in symbols)


# ── Tests: CodeAnalyzer — Dependency operations ──────────────────────────


class TestCodeAnalyzerDependencies:
    @pytest.mark.asyncio
    async def test_get_dependency_context(
        self, sample_project: str
    ) -> None:
        ca = CodeAnalyzer()
        await ca.build(sample_project)
        auth_path = os.path.join(sample_project, "services", "auth.py")
        ctx = ca.get_dependency_context(auth_path)
        assert isinstance(ctx, dict)
        # services/auth.py has no internal deps (imports from models, not present)
        # But this checks the structure is correct
        assert "dependencies" in ctx

    @pytest.mark.asyncio
    async def test_get_impact_scope(self, sample_project: str) -> None:
        ca = CodeAnalyzer()
        await ca.build(sample_project)
        main_path = os.path.join(sample_project, "main.py")
        scope = ca.get_impact_scope(main_path)
        assert isinstance(scope, list)

    @pytest.mark.asyncio
    async def test_update_file(self, sample_project: str) -> None:
        ca = CodeAnalyzer()
        await ca.build(sample_project)

        # Add a new function to auth.py
        auth_path = os.path.join(sample_project, "services", "auth.py")
        with open(auth_path, "a") as f:
            f.write("\ndef new_func():\n    pass\n")

        await ca.update_file(auth_path)
        symbols = ca.get_symbol_context(auth_path)
        assert any(s.name == "new_func" for s in symbols)


# ── Tests: CodeAnalyzer — Edge cases ─────────────────────────────────────


class TestCodeAnalyzerEdgeCases:
    @pytest.mark.asyncio
    async def test_get_symbol_context_nonexistent_file(
        self, sample_project: str
    ) -> None:
        ca = CodeAnalyzer()
        await ca.build(sample_project)
        symbols = ca.get_symbol_context("/nonexistent.py")
        assert symbols == []

    @pytest.mark.asyncio
    async def test_get_dependency_context_nonexistent_file(
        self, sample_project: str
    ) -> None:
        ca = CodeAnalyzer()
        await ca.build(sample_project)
        ctx = ca.get_dependency_context("/nonexistent.py")
        assert ctx["dependencies"] == []

    @pytest.mark.asyncio
    async def test_get_impact_scope_nonexistent_file(
        self, sample_project: str
    ) -> None:
        ca = CodeAnalyzer()
        await ca.build(sample_project)
        scope = ca.get_impact_scope("/nonexistent.py")
        assert scope == []

    @pytest.mark.asyncio
    async def test_build_twice(self, sample_project: str) -> None:
        """Building twice should not error and should update properly."""
        ca = CodeAnalyzer()
        await ca.build(sample_project)
        # Add a file
        with open(os.path.join(sample_project, "new_module.py"), "w") as f:
            f.write("def new_func(): pass\n")
        await ca.build(sample_project)
        assert len(ca.search_symbol("new_func")) == 1
