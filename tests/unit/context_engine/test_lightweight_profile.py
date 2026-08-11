from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from codeagent.context_engine.engine import ContextConfig, ContextEngine


@pytest.mark.asyncio
async def test_lightweight_profile_skips_expensive_project_indexes(tmp_path):
    (tmp_path / "example.py").write_text("VALUE = 1\n", encoding="utf-8")
    engine = ContextEngine(ContextConfig(
        cache_enabled=False,
        analysis_enabled=False,
        semantic_enabled=False,
    ))
    engine.file_tree_indexer.scan = AsyncMock(return_value={
        "name": "root",
        "type": "directory",
        "path": ".",
        "children": [],
    })
    engine.code_analyzer = MagicMock()
    engine.code_analyzer.build = AsyncMock()
    engine.semantic_search = MagicMock()
    engine.semantic_search.index_project = AsyncMock()
    engine.semantic_search.search = AsyncMock()

    package = await engine.build_context(str(tmp_path), "find the parser")

    assert package.file_tree["name"] == "root"
    assert package.related_code == []
    assert package.dependency_info == {}
    assert package.symbol_table == []
    assembled = engine.assemble_context(package)
    assert "root/" in assembled
    engine.code_analyzer.build.assert_not_awaited()
    engine.semantic_search.index_project.assert_not_awaited()
    engine.semantic_search.search.assert_not_awaited()


@pytest.mark.asyncio
async def test_semantic_search_is_explicitly_unavailable_in_lightweight_profile():
    engine = ContextEngine(ContextConfig(semantic_enabled=False))
    engine.semantic_search = MagicMock()
    engine.semantic_search.search = AsyncMock()

    assert await engine.search_semantic("anything") == []
    engine.semantic_search.search.assert_not_awaited()
