from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from codeagent.tools.lsp.navigate_code import NavigateCodeTool


@pytest.mark.asyncio
async def test_definition_prefers_symbol_index_and_normalizes_path(tmp_path) -> None:
    target = tmp_path / "service.py"
    target.write_text("def authenticate():\n    pass\n", encoding="utf-8")
    analyzer = MagicMock()
    analyzer.query_symbol.return_value = [SimpleNamespace(
        file_path=str(target),
        start_line=1,
        end_line=2,
        kind="function_definition",
        signature="def authenticate():\n    pass",
    )]
    engine = SimpleNamespace(code_analyzer=analyzer)
    tool = NavigateCodeTool(tmp_path, context_engine=engine)

    result = await tool.execute("authenticate", action="definition")

    assert result.success
    assert result.data["engine"] == "ast-symbol-index"
    assert result.data["results"][0]["file_path"] == "service.py"
    assert result.data["results"][0]["signature"] == "def authenticate():"


@pytest.mark.asyncio
async def test_references_fall_back_to_bounded_exact_word_search(tmp_path) -> None:
    (tmp_path / "service.py").write_text(
        "def authenticate():\n    return True\n\nauthenticate()\n",
        encoding="utf-8",
    )
    tool = NavigateCodeTool(tmp_path)

    with patch("asyncio.create_subprocess_exec", side_effect=FileNotFoundError):
        result = await tool.execute("authenticate", action="references")

    assert result.success
    assert result.data["engine"] == "ripgrep-word-fallback"
    assert result.data["total_results"] == 2
    assert all(item["file_path"] == "service.py" for item in result.data["results"])


@pytest.mark.asyncio
async def test_rejects_expression_instead_of_symbol(tmp_path) -> None:
    result = await NavigateCodeTool(tmp_path).execute("service.authenticate()")
    assert result.success is False
    assert result.error_code == "INVALID_SYMBOL"
