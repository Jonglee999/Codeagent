"""ContextEngine 单元测试 — Mock 所有子模块，测试 Facade 调度逻辑。"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, PropertyMock

import pytest

from codeagent.context_engine.engine import (
    ContextConfig,
    ContextEngine,
)
from codeagent.gateway.context_gateway import CodeSnippet, ContextPackage


# ── Helper: 创建 Mock 子模块的 ContextEngine ───────────────────────────────────


@pytest.fixture
def mock_submodules():
    """创建所有子模块都被 Mock 的 ContextEngine。

    同时 mock FileTreeIndexer, CodeAnalyzer, SemanticSearchEngine,
    ContextAssembler 以避免真实依赖。
    """
    engine = ContextEngine(config=ContextConfig(cache_enabled=False))

    # Mock FileTreeIndexer
    engine.file_tree_indexer = MagicMock()
    engine.file_tree_indexer.scan = AsyncMock(return_value={
        "name": "root", "type": "directory", "children": [],
    })

    # Mock CodeAnalyzer
    mock_symbol_table = MagicMock()
    mock_symbol_table.to_json = MagicMock(return_value=[
        {"name": "func1", "kind": "function_definition", "file_path": "main.py", "start_line": 1},
    ])

    mock_dep_graph = MagicMock()
    mock_dep_graph.to_dict = MagicMock(return_value={"main.py": ["utils.py"]})
    mock_dep_graph.find_circular_dependencies = MagicMock(return_value=[])
    mock_dep_graph.graph.number_of_nodes = PropertyMock(return_value=3)

    engine.code_analyzer = MagicMock()
    engine.code_analyzer.symbol_table = mock_symbol_table
    engine.code_analyzer.dependency_graph = mock_dep_graph
    engine.code_analyzer.build = AsyncMock()
    engine.code_analyzer.update_file = AsyncMock()

    # Mock SemanticSearchEngine
    engine.semantic_search = MagicMock()
    engine.semantic_search.index_project = AsyncMock(return_value={
        "total_files": 5, "total_chunks": 20,
    })
    engine.semantic_search.reindex_file = AsyncMock()
    engine.semantic_search.search = AsyncMock(return_value=[
        MagicMock(
            file_path="services/auth.py",
            start_line=1,
            end_line=10,
            code_snippet="def login(): pass",
            score=0.85,
        ),
    ])

    # Mock ContextAssembler
    engine.context_assembler = MagicMock()
    engine.context_assembler.assemble = MagicMock(
        return_value="<project_tree>\nroot/\n</project_tree>"
    )
    engine.context_assembler.get_budget_report = MagicMock(return_value={
        "total_budget": 8000, "total_used": 4000,
    })

    return engine


@pytest.fixture
def engine_with_cache():
    """创建启用缓存的 ContextEngine，临时目录作为项目根目录。"""
    tmpdir = tempfile.mkdtemp()
    project_root = tmpdir

    config = ContextConfig(
        cache_enabled=True,
        cache_dir=".codeagent/cache",
    )
    engine = ContextEngine(config=config)

    # Mock 子模块（同 mock_submodules）
    engine.file_tree_indexer = MagicMock()
    engine.file_tree_indexer.scan = AsyncMock(return_value={
        "name": "root", "type": "directory", "children": [],
    })

    mock_symbol_table = MagicMock()
    mock_symbol_table.to_json = MagicMock(return_value=[])

    mock_dep_graph = MagicMock()
    mock_dep_graph.to_dict = MagicMock(return_value={})
    mock_dep_graph.find_circular_dependencies = MagicMock(return_value=[])
    mock_dep_graph.graph.number_of_nodes = PropertyMock(return_value=0)

    engine.code_analyzer = MagicMock()
    engine.code_analyzer.symbol_table = mock_symbol_table
    engine.code_analyzer.dependency_graph = mock_dep_graph
    engine.code_analyzer.build = AsyncMock()
    engine.code_analyzer.update_file = AsyncMock()

    engine.semantic_search = MagicMock()
    engine.semantic_search.index_project = AsyncMock()
    engine.semantic_search.reindex_file = AsyncMock()
    engine.semantic_search.search = AsyncMock(return_value=[])

    engine.context_assembler = MagicMock()
    engine.context_assembler.assemble = MagicMock(return_value="")
    engine.context_assembler.get_budget_report = MagicMock(return_value=None)

    yield engine, project_root

    import shutil
    shutil.rmtree(tmpdir)


# ── Tests ──────────────────────────────────────────────────────────────────────


class TestContextEngineInit:
    """ContextEngine 初始化测试。"""

    def test_init_defaults(self) -> None:
        """验证默认配置初始化。"""
        engine = ContextEngine()
        assert engine.config.total_budget == 8000
        assert engine.config.cache_enabled is True
        assert engine.config.cache_dir == ".codeagent/cache"
        assert engine.config.languages == ("python", "typescript", "javascript")
        assert engine.config.search_top_k == 10
        assert engine.config.use_mock_embeddings is False

    def test_init_custom_config(self) -> None:
        """验证自定义配置初始化。"""
        config = ContextConfig(
            total_budget=16000,
            cache_enabled=False,
            languages=("python",),
            search_top_k=20,
            use_mock_embeddings=True,
        )
        engine = ContextEngine(config=config)
        assert engine.config.total_budget == 16000
        assert engine.config.cache_enabled is False
        assert engine.config.languages == ("python",)
        assert engine.config.search_top_k == 20
        assert engine.config.use_mock_embeddings is True

    def test_init_submodules_created(self) -> None:
        """验证子模块在初始化时被创建。"""
        engine = ContextEngine()
        assert engine.file_tree_indexer is not None
        assert engine.code_analyzer is not None
        assert engine.semantic_search is not None
        assert engine.context_assembler is not None

    def test_stats_initial_state(self) -> None:
        """验证初始统计状态。"""
        engine = ContextEngine()
        stats = engine.get_stats()
        assert stats.total_invocations == 0
        assert stats.avg_build_time_ms == 0.0
        assert stats.cache_hits == 0
        assert stats.cache_misses == 0
        assert stats.last_query == ""


class TestContextEngineBuildContext:
    """build_context 调度逻辑测试。"""

    @pytest.mark.asyncio
    async def test_build_context_returns_package(self, mock_submodules) -> None:
        """验证 build_context 返回 ContextPackage。"""
        package = await mock_submodules.build_context("/fake/root", "test query")
        assert isinstance(package, ContextPackage)
        assert package.file_tree is not None
        assert isinstance(package.related_code, list)
        assert isinstance(package.dependency_info, dict)
        assert isinstance(package.symbol_table, list)

    @pytest.mark.asyncio
    async def test_full_build_calls_submodules(self, mock_submodules) -> None:
        """验证 full build 调用所有子模块的 build 方法。"""
        await mock_submodules.build_context("/fake/root", "test query")

        # FileTreeIndexer.scan 被调用
        mock_submodules.file_tree_indexer.scan.assert_awaited_once()

        # CodeAnalyzer.build 被调用
        mock_submodules.code_analyzer.build.assert_awaited_once()

        # SemanticSearchEngine.index_project 被调用
        mock_submodules.semantic_search.index_project.assert_awaited_once()

        # SemanticSearchEngine.search 被调用
        mock_submodules.semantic_search.search.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_build_context_populates_related_code(
        self, mock_submodules,
    ) -> None:
        """验证 related_code 从 SearchResult 正确转换。"""
        package = await mock_submodules.build_context("/fake/root", "auth query")
        assert len(package.related_code) == 1
        snippet = package.related_code[0]
        assert isinstance(snippet, CodeSnippet)
        assert snippet.file_path == "services/auth.py"
        assert snippet.score == 0.85

    @pytest.mark.asyncio
    async def test_build_context_populates_symbol_table(
        self, mock_submodules,
    ) -> None:
        """验证 symbol_table 从 CodeAnalyzer 获取。"""
        package = await mock_submodules.build_context("/fake/root", "query")
        assert len(package.symbol_table) == 1
        assert package.symbol_table[0]["name"] == "func1"

    @pytest.mark.asyncio
    async def test_build_context_populates_dep_info(
        self, mock_submodules,
    ) -> None:
        """验证 dependency_info 包含依赖数据。"""
        package = await mock_submodules.build_context("/fake/root", "query")
        assert "dependencies" in package.dependency_info
        assert "total_modules" in package.dependency_info
        assert package.dependency_info["total_modules"] == 3

    @pytest.mark.asyncio
    async def test_build_context_empty_query(self, mock_submodules) -> None:
        """验证空查询也能正常构建。"""
        package = await mock_submodules.build_context("/fake/root", "")
        assert isinstance(package, ContextPackage)
        mock_submodules.semantic_search.search.assert_awaited_once()


class TestContextEngineCache:
    """缓存机制测试。"""

    @pytest.mark.asyncio
    async def test_no_cache_first_build_uses_full(self, engine_with_cache) -> None:
        """首次构建（无缓存）使用 full_build。"""
        engine, root = engine_with_cache
        await engine.build_context(root, "test")

        engine.code_analyzer.build.assert_awaited_once()
        engine.semantic_search.index_project.assert_awaited_once()
        assert engine._stats.cache_misses == 1
        assert engine._stats.last_build_mode == "full"

    @pytest.mark.asyncio
    async def test_cache_exists_uses_quick(self, engine_with_cache) -> None:
        """缓存存在时使用 quick_build。"""
        engine, root = engine_with_cache

        # 创建缓存文件
        cache_dir = Path(root) / ".codeagent" / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / "mtime_cache.json"
        cache_data = {"project_root": root, "files": {}, "last_build": "2024-01-01T00:00:00"}
        with open(cache_file, "w") as f:
            json.dump(cache_data, f)

        # 第二次构建使用 quick_build
        engine.code_analyzer.build.reset_mock()
        engine.semantic_search.index_project.reset_mock()

        await engine.build_context(root, "test")

        # full 方法不被调用
        engine.code_analyzer.build.assert_not_awaited()
        engine.semantic_search.index_project.assert_not_awaited()
        assert engine._stats.last_build_mode == "quick"

    @pytest.mark.asyncio
    async def test_cache_disabled_always_full(self, mock_submodules) -> None:
        """缓存关闭时总是 full_build。"""
        await mock_submodules.build_context("/fake/root", "test")

        mock_submodules.code_analyzer.build.assert_awaited_once()
        assert mock_submodules._stats.last_build_mode == "full"

    @pytest.mark.asyncio
    async def test_cache_save_creates_file(self, engine_with_cache) -> None:
        """build_context 后缓存文件被创建。"""
        engine, root = engine_with_cache
        cache_path = Path(root) / ".codeagent" / "cache" / "mtime_cache.json"

        assert not cache_path.exists()
        await engine.build_context(root, "test")
        assert cache_path.exists()

    @pytest.mark.asyncio
    async def test_cache_report(self, engine_with_cache) -> None:
        """get_cache_report 返回正确信息。"""
        engine, root = engine_with_cache
        await engine.build_context(root, "test")

        report = engine.get_cache_report()
        assert report["cache_enabled"] is True
        assert report["cache_dir"] == ".codeagent/cache"
        assert report["cache_exists"] is True
        assert report["stats"]["total_invocations"] == 1

    @pytest.mark.asyncio
    async def test_deleted_source_forces_full_rebuild(self, engine_with_cache) -> None:
        engine, root = engine_with_cache
        source = Path(root) / "obsolete.py"
        source.write_text("value = 1\n", encoding="utf-8")
        await engine.build_context(root, "first")

        source.unlink()
        engine.code_analyzer.build.reset_mock()
        engine.semantic_search.index_project.reset_mock()
        await engine.build_context(root, "second")

        engine.code_analyzer.build.assert_awaited_once()
        engine.semantic_search.index_project.assert_awaited_once()
        assert engine._stats.last_build_mode == "full"

    @pytest.mark.asyncio
    async def test_rapid_same_size_edit_invalidates_cache(self, engine_with_cache) -> None:
        engine, root = engine_with_cache
        source = Path(root) / "rapid.py"
        source.write_text("value = 1\n", encoding="utf-8")
        await engine.build_context(root, "first")

        source.write_text("value = 2\n", encoding="utf-8")
        changed, deleted = engine._scan_file_changes()

        assert str(source) in changed
        assert deleted == []


class TestContextEngineStats:
    """性能统计测试。"""

    @pytest.mark.asyncio
    async def test_stats_recorded_after_build(self, mock_submodules) -> None:
        """验证 build_context 后统计信息被更新。"""
        await mock_submodules.build_context("/fake/root", "search query")
        stats = mock_submodules.get_stats()

        assert stats.total_invocations == 1
        assert stats.last_build_time_ms >= 0
        assert stats.last_query == "search query"
        assert stats.last_build_mode == "full"

    @pytest.mark.asyncio
    async def test_stats_multiple_invocations(self, mock_submodules) -> None:
        """验证多次调用后统计信息正确累计。"""
        await mock_submodules.build_context("/fake/root", "q1")
        await mock_submodules.build_context("/fake/root", "q2")

        stats = mock_submodules.get_stats()
        assert stats.total_invocations == 2
        assert stats.last_query == "q2"

    @pytest.mark.asyncio
    async def test_search_semantic_delegates(self, mock_submodules) -> None:
        """验证 search_semantic 委托给 SemanticSearchEngine。"""
        mock_submodules.semantic_search.search = AsyncMock(return_value=[
            MagicMock(
                file_path="main.py", start_line=1, end_line=5,
                code_snippet="print('hello')", score=0.9,
            ),
        ])

        results = await mock_submodules.search_semantic("find main", top_k=3)

        mock_submodules.semantic_search.search.assert_awaited_once_with(
            "find main", top_k=3,
        )
        assert len(results) == 1
        assert isinstance(results[0], CodeSnippet)
        assert results[0].file_path == "main.py"


class TestContextEngineUpdate:
    """update_index 测试。"""

    @pytest.mark.asyncio
    async def test_update_index_calls_submodules(self, mock_submodules) -> None:
        """验证 update_index 调用所有子模块的 build 方法。"""
        await mock_submodules.update_index("/fake/root")

        mock_submodules.code_analyzer.build.assert_awaited_once()
        mock_submodules.semantic_search.index_project.assert_awaited_once()


class TestContextEngineAssemble:
    """ContextAssembler 委托测试。"""

    def test_assemble_context_delegates(self, mock_submodules) -> None:
        """验证 assemble_context 委托给 ContextAssembler。"""
        package = ContextPackage(file_tree={"name": "root"})
        result = mock_submodules.assemble_context(package)

        mock_submodules.context_assembler.assemble.assert_called_once_with(package)
        assert result == "<project_tree>\nroot/\n</project_tree>"

    def test_get_budget_report_delegates(self, mock_submodules) -> None:
        """验证 get_budget_report 委托给 ContextAssembler。"""
        report = mock_submodules.get_budget_report()
        assert report["total_budget"] == 8000


class TestContextEngineEdgeCases:
    """边界情况测试。"""

    @pytest.mark.asyncio
    async def test_determine_build_mode_cache_disabled(self) -> None:
        """缓存禁用时构建模式为 full。"""
        engine = ContextEngine(config=ContextConfig(cache_enabled=False))
        assert engine._determine_build_mode() == "full"

    def test_determine_build_mode_no_cache(self) -> None:
        """无缓存文件时构建模式为 full。"""
        engine = ContextEngine(config=ContextConfig(cache_enabled=True))
        # 未设置 _project_root，缓存不存在
        assert engine._determine_build_mode() == "full"

    def test_config_mutable_defaults(self) -> None:
        """验证多次创建不会共享默认配置。"""
        e1 = ContextEngine()
        e2 = ContextEngine()
        assert e1.config is not e2.config


class TestContextEngineScanChangedFiles:
    """_scan_changed_files 测试。"""

    def test_scan_no_cache_file(self) -> None:
        """无缓存文件时返回空列表。"""
        engine = ContextEngine()
        engine._project_root = "/fake"
        changed = engine._scan_changed_files()
        assert changed == []

    def test_scan_changed_files_detects_change(self, engine_with_cache) -> None:
        """检测到缓存 mtime 与当前文件系统不匹配时返回变更。"""
        engine, root = engine_with_cache
        engine._project_root = root

        # 创建缓存文件
        cache_dir = Path(root) / ".codeagent" / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / "mtime_cache.json"

        # 缓存记录一个不存在的 mtime
        cache_data = {
            "project_root": root,
            "files": {
                "main.py": "2020-01-01T00:00:00",
            },
            "last_build": "2020-01-01T00:00:00",
        }
        with open(cache_file, "w") as f:
            json.dump(cache_data, f)

        # 创建真实的 main.py 文件
        src_file = Path(root) / "main.py"
        src_file.write_text("print('hello')")

        changed = engine._scan_changed_files()
        # main.py 在缓存中有记录但 mtime 不同（或者被检测为新增/修改）
        # 由于 main.py 实际存在，但缓存的 mtime 可能是旧的
        # 实际结果是：main.py 被包含在 changed 中
        assert str(src_file) in changed or len(changed) > 0
