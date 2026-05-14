"""ContextEngine 集成测试 — 使用 sample_python_project 做端到端上下文构建。"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from codeagent.context_engine.engine import ContextConfig, ContextEngine
from codeagent.gateway.context_gateway import CodeSnippet, ContextPackage

# 找到 fixtures 目录
_FIXTURES_DIR = Path(__file__).resolve().parent.parent.parent / "fixtures"
SAMPLE_PROJECT = str(_FIXTURES_DIR / "sample_python_project")


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def temp_cache_dir() -> str:
    """创建临时缓存目录（每个测试独立，避免跨测试缓存污染）。"""
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    import shutil
    shutil.rmtree(tmpdir)


@pytest.fixture
def engine(temp_cache_dir: str) -> ContextEngine:
    """创建使用 mock 嵌入模型 + 独立临时缓存的 ContextEngine。"""
    cache_dir = os.path.join(temp_cache_dir, ".codeagent", "cache")
    config = ContextConfig(
        total_budget=8000,
        cache_enabled=True,
        cache_dir=cache_dir,
        languages=("python",),
        search_top_k=10,
        use_mock_embeddings=True,
    )
    return ContextEngine(config=config)


@pytest.fixture
def engine_no_cache(temp_cache_dir: str) -> ContextEngine:
    """创建禁用缓存的 ContextEngine。"""
    config = ContextConfig(
        total_budget=8000,
        cache_enabled=False,
        languages=("python",),
        search_top_k=10,
        use_mock_embeddings=True,
    )
    return ContextEngine(config=config)


# ── Integration Tests ────────────────────────────────────────────────────────


class TestContextEngineIntegration:
    """使用 sample_python_project 的集成测试。"""

    @pytest.mark.asyncio
    async def test_build_context_returns_valid_package(
        self, engine_no_cache: ContextEngine,
    ) -> None:
        """验证 build_context 返回完整的 ContextPackage。"""
        package = await engine_no_cache.build_context(SAMPLE_PROJECT, "user authentication")

        assert isinstance(package, ContextPackage)
        # file_tree 应包含项目结构
        assert package.file_tree is not None
        assert package.file_tree.get("name") == "sample_python_project"
        assert package.file_tree.get("type") == "directory"

        # related_code 应为列表
        assert isinstance(package.related_code, list)

        # dependency_info 应包含依赖数据
        assert isinstance(package.dependency_info, dict)
        assert "dependencies" in package.dependency_info

        # symbol_table 应包含符号
        assert isinstance(package.symbol_table, list)

    @pytest.mark.asyncio
    async def test_build_context_has_symbols(
        self, engine_no_cache: ContextEngine,
    ) -> None:
        """验证符号表包含项目中的符号。"""
        package = await engine_no_cache.build_context(SAMPLE_PROJECT, "auth")

        # 应包含 User 类、authenticate_user 函数等符号
        symbol_names = [s.get("name") for s in package.symbol_table]
        assert "User" in symbol_names, "Should find User class"
        assert "authenticate_user" in symbol_names, (
            "Should find authenticate_user function"
        )

    @pytest.mark.asyncio
    async def test_build_context_has_dependency_info(
        self, engine_no_cache: ContextEngine,
    ) -> None:
        """验证依赖信息包含模块间关系。"""
        package = await engine_no_cache.build_context(SAMPLE_PROJECT, "services")

        dep_info = package.dependency_info
        assert "dependencies" in dep_info
        deps = dep_info["dependencies"]

        # main.py 应依赖 services/auth.py 和 services/user_service.py
        main_deps = deps.get("main.py", [])
        assert len(main_deps) > 0, "main.py should have dependencies"
        assert any("auth" in d for d in main_deps), (
            "main.py should depend on auth module"
        )

    @pytest.mark.asyncio
    async def test_build_context_related_code_scores(
        self, engine_no_cache: ContextEngine,
    ) -> None:
        """验证 related_code 中的 score 在 0-1 范围内。"""
        package = await engine_no_cache.build_context(
            SAMPLE_PROJECT, "authentication",
        )

        for snippet in package.related_code:
            assert 0 <= snippet.score <= 1, (
                f"Score {snippet.score} should be in [0, 1]"
            )

    @pytest.mark.asyncio
    async def test_build_context_multiple_queries(
        self, engine_no_cache: ContextEngine,
    ) -> None:
        """验证不同查询返回不同相关代码（不强制有结果，只验证不崩溃）。"""
        auth_pkg = await engine_no_cache.build_context(
            SAMPLE_PROJECT, "user authentication token",
        )
        helper_pkg = await engine_no_cache.build_context(
            SAMPLE_PROJECT, "string hash utility function",
        )

        # 即使使用 mock 模型，两次查询都应该正常返回
        assert isinstance(auth_pkg.related_code, list)
        assert isinstance(helper_pkg.related_code, list)

    @pytest.mark.asyncio
    async def test_stats_after_build(
        self, engine_no_cache: ContextEngine,
    ) -> None:
        """验证构建后统计信息正确。"""
        await engine_no_cache.build_context(SAMPLE_PROJECT, "test")

        stats = engine_no_cache.get_stats()
        assert stats.total_invocations >= 1
        assert stats.last_build_mode == "full"
        assert stats.last_query == "test"

    @pytest.mark.asyncio
    async def test_full_context_package_fields(
        self, engine_no_cache: ContextEngine,
    ) -> None:
        """验证 ContextPackage 所有字段非空。"""
        package = await engine_no_cache.build_context(SAMPLE_PROJECT, "test")

        # file_tree 应包含目录和文件
        children = package.file_tree.get("children", [])
        child_names = [c.get("name") for c in children]
        assert "services" in child_names
        assert "models" in child_names

        # dependency_info 应有 modules
        total_modules = package.dependency_info.get("total_modules", 0)
        assert total_modules > 0, "Should have module dependencies"

        # symbol_table 应有内容
        assert len(package.symbol_table) > 0, "Should have symbols"

    @pytest.mark.asyncio
    async def test_cache_persistence(
        self, engine: ContextEngine,
    ) -> None:
        """验证缓存机制：首次 full_build，后续 quick_build。"""
        # 第一次构建（full_build）
        pkg1 = await engine.build_context(SAMPLE_PROJECT, "first")
        stats1 = engine.get_stats()
        assert stats1.last_build_mode == "full"
        assert len(pkg1.symbol_table) > 0

        # 第二次构建（应该是 quick_build）
        pkg2 = await engine.build_context(SAMPLE_PROJECT, "second")
        stats2 = engine.get_stats()
        assert stats2.last_build_mode == "quick"
        # 数据仍然正确
        assert len(pkg2.symbol_table) > 0
        assert pkg2.file_tree is not None

    @pytest.mark.asyncio
    async def test_cache_report(
        self, engine: ContextEngine,
    ) -> None:
        """验证缓存报告。"""
        await engine.build_context(SAMPLE_PROJECT, "test")

        report = engine.get_cache_report()
        assert report["cache_enabled"] is True
        assert report["cache_exists"] is True
        assert report["stats"]["total_invocations"] >= 1

    @pytest.mark.asyncio
    async def test_search_semantic_integration(
        self, engine_no_cache: ContextEngine,
    ) -> None:
        """验证 search_semantic 接口正常返回 CodeSnippet 列表。"""
        # 先构建索引
        await engine_no_cache.build_context(SAMPLE_PROJECT, "init")

        # 使用 search_semantic 接口
        results = await engine_no_cache.search_semantic(
            "user authentication", top_k=5,
        )

        assert isinstance(results, list)
        if results:
            assert isinstance(results[0], CodeSnippet)

    @pytest.mark.asyncio
    async def test_update_index(
        self, engine_no_cache: ContextEngine,
    ) -> None:
        """验证 update_index 全量重建索引。"""
        await engine_no_cache.update_index(SAMPLE_PROJECT)

        # update_index 后再构建上下文应正常
        package = await engine_no_cache.build_context(SAMPLE_PROJECT, "test")
        assert isinstance(package, ContextPackage)
        assert len(package.symbol_table) > 0

    @pytest.mark.asyncio
    async def test_stats_avg_time_across_builds(
        self, engine: ContextEngine,
    ) -> None:
        """验证多次构建的统计累积。"""
        await engine.build_context(SAMPLE_PROJECT, "q1")
        await engine.build_context(SAMPLE_PROJECT, "q2")

        stats = engine.get_stats()
        assert stats.total_invocations == 2
        assert stats.last_query == "q2"
        assert stats.cache_hits >= 0  # 第二次可能是 cache hit
