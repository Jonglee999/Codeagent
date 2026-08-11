from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from codeagent.context_engine.capabilities import context_capability_report
from codeagent.context_engine.dependency_graph import DependencyGraph
from codeagent.context_engine.engine import ContextConfig, ContextEngine
from codeagent.context_engine.semantic_search import SemanticSearchEngine


def test_runtime_context_config_uses_canonical_modes(monkeypatch):
    monkeypatch.setenv("CONTEXT_MODE", "auto")
    monkeypatch.setenv("CONTEXT_TARGET_INPUT_TOKENS", "24000")
    monkeypatch.setenv("CONTEXT_AST_MODE", "on")
    monkeypatch.setenv("CONTEXT_SEMANTIC_MODE", "auto")
    monkeypatch.setenv("CONTEXT_INDEX_BACKGROUND", "true")

    config = ContextConfig.from_env()

    assert config.total_budget == 24000
    assert config.analysis_enabled is True
    assert config.semantic_enabled is True
    assert config.semantic_mode == "auto"
    assert config.index_background is True


def test_capability_report_is_secret_free(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "must-not-leak")
    report = context_capability_report(tmp_path)

    assert report["effective_context_budget"] > 0
    assert report["semantic_index_status"] in {"disabled", "deferred"}
    assert "must-not-leak" not in repr(report)


@pytest.mark.asyncio
async def test_ast_dependency_parser_ignores_comments_and_records_evidence(tmp_path):
    (tmp_path / "real.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "main.py").write_text(
        '# import fake.module\nTEXT = "from fake import value"\n'
        "from real import VALUE\n",
        encoding="utf-8",
    )
    graph = DependencyGraph()

    await graph.build(str(tmp_path))

    assert graph.get_dependencies("main.py") == ["real.py"]
    assert graph.edge_evidence("main.py", "real.py")["source"] == "ast"


@pytest.mark.asyncio
async def test_ast_dependency_parser_resolves_typescript_relative_import(tmp_path):
    (tmp_path / "types.ts").write_text("export type User = {name: string};\n", encoding="utf-8")
    (tmp_path / "app.ts").write_text(
        'import type { User } from "./types";\nconst text = "import fake";\n',
        encoding="utf-8",
    )
    graph = DependencyGraph()

    await graph.build(str(tmp_path))

    assert "types.ts" in graph.get_dependencies("app.ts")
    assert graph.edge_evidence("app.ts", "types.ts")["source"] == "ast"


@pytest.mark.asyncio
async def test_auto_large_repository_starts_semantic_index_in_background(tmp_path):
    (tmp_path / "a.py").write_text("VALUE = 1\n", encoding="utf-8")
    engine = ContextEngine(ContextConfig(
        cache_enabled=False,
        analysis_enabled=False,
        semantic_enabled=True,
        semantic_mode="auto",
        index_background=True,
        auto_file_limit=1,
        use_mock_embeddings=True,
    ))
    engine._source_file_count = lambda: 2  # type: ignore[method-assign]
    started = asyncio.Event()
    release = asyncio.Event()

    async def slow_index(_root):
        started.set()
        await release.wait()
        return {"total_files": 1, "total_chunks": 1}

    engine.semantic_search.index_project = AsyncMock(side_effect=slow_index)

    await engine.build_context(str(tmp_path), "find value")
    await started.wait()
    assert engine.get_capability_report()["semantic_index_status"] == "indexing"
    release.set()
    assert engine._semantic_index_task is not None
    await engine._semantic_index_task
    assert engine.get_capability_report()["semantic_index_status"] == "ready"


@pytest.mark.asyncio
async def test_deleted_file_vectors_are_removed(tmp_path):
    source = tmp_path / "obsolete.py"
    source.write_text("def obsolete_marker():\n    return True\n", encoding="utf-8")
    engine = SemanticSearchEngine(
        db_path=str(tmp_path / ".codeagent" / "lancedb"),
        use_mock=True,
    )
    await engine.index_project(str(tmp_path))
    source.unlink()

    await engine.delete_file(str(source))

    assert all("obsolete.py" not in item.file_path for item in await engine.search("obsolete_marker"))


@pytest.mark.asyncio
async def test_incremental_reindex_keeps_vector_ids_searchable(tmp_path):
    source = tmp_path / "service.py"
    source.write_text("def original_marker():\n    return True\n", encoding="utf-8")
    engine = SemanticSearchEngine(
        db_path=str(tmp_path / ".codeagent" / "lancedb"),
        use_mock=True,
    )
    await engine.index_project(str(tmp_path))
    source.write_text("def replacement_marker():\n    return True\n", encoding="utf-8")

    await engine.reindex_file(str(source))

    results = await engine.search("replacement_marker")
    assert results
    assert any("replacement_marker" in item.code_snippet for item in results)
    assert all("original_marker" not in item.code_snippet for item in results)
