"""CodeAnalyzer 集成测试 — 使用 sample_python_project 验证真实代码分析。"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from codeagent.context_engine.code_analyzer import CodeAnalyzer

# 找到 fixtures 目录
_FIXTURES_DIR = Path(__file__).resolve().parent.parent.parent / "fixtures"
SAMPLE_PROJECT = str(_FIXTURES_DIR / "sample_python_project")


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def analyzer() -> CodeAnalyzer:
    return CodeAnalyzer()


# ── Integration Tests ────────────────────────────────────────────────────


class TestCodeAnalyzerIntegration:
    """使用 sample_python_project 的集成测试。"""

    @pytest.mark.asyncio
    async def test_build_on_real_project(self, analyzer: CodeAnalyzer) -> None:
        """验证全量构建成功。"""
        await analyzer.build(SAMPLE_PROJECT)
        symbols = analyzer.symbol_table.get_all_symbols()
        assert len(symbols) > 0, "Should find symbols in real project"

    @pytest.mark.asyncio
    async def test_finds_all_symbols(
        self, analyzer: CodeAnalyzer
    ) -> None:
        """验证能找到所有预期的符号。"""
        await analyzer.build(SAMPLE_PROJECT)
        symbol_names = {s.name for s in analyzer.symbol_table.get_all_symbols()}

        # 来自 models/user.py
        assert "User" in symbol_names
        assert "has_permission" in symbol_names
        assert "deactivate" in symbol_names

        # 来自 services/auth.py
        assert "authenticate_user" in symbol_names
        assert "validate_token" in symbol_names

        # 来自 services/user_service.py
        assert "UserService" in symbol_names
        assert "create_user" in symbol_names
        assert "get_user" in symbol_names
        assert "list_users" in symbol_names

        # 来自 api/routes.py
        assert "setup_routes" in symbol_names

        # 来自 utils/helpers.py
        assert "hash_string" in symbol_names
        assert "to_json" in symbol_names

    @pytest.mark.asyncio
    async def test_symbol_kinds(
        self, analyzer: CodeAnalyzer
    ) -> None:
        """验证符号类型正确。"""
        await analyzer.build(SAMPLE_PROJECT)
        symbols = analyzer.symbol_table.get_all_symbols()
        kinds = {s.name: s.kind for s in symbols}

        assert kinds.get("User") == "class_definition"
        assert kinds.get("UserService") == "class_definition"
        assert kinds.get("authenticate_user") == "function_definition"
        assert kinds.get("create_user") == "function_definition"

    @pytest.mark.asyncio
    async def test_docstring_extraction(
        self, analyzer: CodeAnalyzer
    ) -> None:
        """验证 docstring 提取。"""
        await analyzer.build(SAMPLE_PROJECT)
        syms = analyzer.symbol_table.query("User")
        for s in syms:
            if s.kind == "class_definition":
                assert s.docstring is not None
                assert "Represents a user" in s.docstring
                break

    @pytest.mark.asyncio
    async def test_dependency_graph(
        self, analyzer: CodeAnalyzer
    ) -> None:
        """验证依赖图正确。"""
        await analyzer.build(SAMPLE_PROJECT)
        dg = analyzer.dependency_graph

        # main.py 应依赖 services/auth.py, services/user_service.py, models/user.py, config/settings.py
        main_deps = dg.get_dependencies("main.py")
        expected = [
            "services/auth.py",
            "services/user_service.py",
            "models/user.py",
            "config/settings.py",
        ]
        for exp in expected:
            assert exp in main_deps, f"main.py should depend on {exp}"

    @pytest.mark.asyncio
    async def test_dependents(
        self, analyzer: CodeAnalyzer
    ) -> None:
        """验证反向依赖。"""
        await analyzer.build(SAMPLE_PROJECT)
        dg = analyzer.dependency_graph

        # models/user.py 应有多个依赖者
        user_deps = dg.get_dependents("models/user.py")
        assert "main.py" in user_deps
        assert "services/auth.py" in user_deps
        assert "services/user_service.py" in user_deps

    @pytest.mark.asyncio
    async def test_impact_scope(self, analyzer: CodeAnalyzer) -> None:
        """验证影响范围分析。"""
        await analyzer.build(SAMPLE_PROJECT)
        dg = analyzer.dependency_graph

        # 修改 models/user.py 应影响 main.py, services/auth.py, services/user_service.py, api/routes.py 等
        scope = dg.get_impact_scope("models/user.py")
        assert "main.py" in scope
        assert "services/auth.py" in scope
        assert "services/user_service.py" in scope

    @pytest.mark.asyncio
    async def test_query_symbol(
        self, analyzer: CodeAnalyzer
    ) -> None:
        """验证符号查询。"""
        await analyzer.build(SAMPLE_PROJECT)
        results = analyzer.search_symbol("UserService")
        assert len(results) == 1
        assert results[0].name == "UserService"
        assert results[0].kind == "class_definition"

    @pytest.mark.asyncio
    async def test_fuzzy_search(
        self, analyzer: CodeAnalyzer
    ) -> None:
        """验证模糊搜索。"""
        await analyzer.build(SAMPLE_PROJECT)
        results = analyzer.fuzzy_search_symbol("user")
        names = {s.name for s in results}
        # Should find User, UserService, authenticate_user, create_user, get_user, list_users
        assert "User" in names
        assert "UserService" in names
        assert "authenticate_user" in names
        assert "create_user" in names
        assert "get_user" in names

    @pytest.mark.asyncio
    async def test_get_dependency_context(
        self, analyzer: CodeAnalyzer
    ) -> None:
        """验证依赖上下文。"""
        await analyzer.build(SAMPLE_PROJECT)
        auth_path = os.path.join(SAMPLE_PROJECT, "services", "auth.py")
        ctx = analyzer.get_dependency_context(auth_path)
        assert "dependencies" in ctx
        assert "dependents" in ctx
        assert "impact_scope" in ctx

    @pytest.mark.asyncio
    async def test_no_circular_deps(
        self, analyzer: CodeAnalyzer
    ) -> None:
        """验证示例项目没有循环依赖。"""
        await analyzer.build(SAMPLE_PROJECT)
        circles = analyzer.dependency_graph.find_circular_dependencies()
        assert circles == [], f"Sample project should not have circular deps: {circles}"

    @pytest.mark.asyncio
    async def test_update_file_incremental(
        self, analyzer: CodeAnalyzer
    ) -> None:
        """验证增量更新。"""
        await analyzer.build(SAMPLE_PROJECT)

        auth_path = os.path.join(SAMPLE_PROJECT, "services", "auth.py")

        # 读取原始内容
        with open(auth_path, "r") as f:
            original_content = f.read()

        # 如果之前有残留的测试添加，先清理
        if "new_auth_function" in original_content:
            # 移除之前测试添加的内容
            lines = original_content.splitlines()
            clean_lines = [l for l in lines if "new_auth_function" not in l]
            # 移除尾部空行后的多余空行
            with open(auth_path, "w") as f:
                f.write("\n".join(clean_lines))

        # 重新构建以确保干净状态
        await analyzer.build(SAMPLE_PROJECT)
        before = len(analyzer.get_symbol_context(auth_path))

        # 添加新函数
        with open(auth_path, "a") as f:
            f.write("\n\ndef new_auth_function():\n    pass\n")

        await analyzer.update_file(auth_path)
        after = len(analyzer.get_symbol_context(auth_path))
        assert after == before + 1
        assert len(analyzer.search_symbol("new_auth_function")) == 1

        # 恢复原始文件
        with open(auth_path, "w") as f:
            f.write(original_content)

    @pytest.mark.asyncio
    async def test_symbol_signatures(
        self, analyzer: CodeAnalyzer
    ) -> None:
        """验证符号签名提取。"""
        await analyzer.build(SAMPLE_PROJECT)
        syms = analyzer.symbol_table.query("authenticate_user")
        assert len(syms) == 1
        assert syms[0].signature is not None
        assert "authenticate_user" in syms[0].signature
