from __future__ import annotations

import pytest

from codeagent.tools.file.list_files import ListFilesTool


@pytest.mark.asyncio
async def test_lists_a_bounded_recursive_tree(tmp_path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("print('ok')", encoding="utf-8")
    (tmp_path / "src" / "app.txt").write_text("ok", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("secret-ish", encoding="utf-8")

    result = await ListFilesTool(tmp_path).execute(recursive=True, pattern="*.py", max_results=10)

    assert result.success
    assert result.data["count"] == 1
    assert result.data["entries"][0]["path"] == "src/app.py"
    assert all(".git" not in item["path"] for item in result.data["entries"])


@pytest.mark.asyncio
async def test_rejects_path_traversal(tmp_path) -> None:
    result = await ListFilesTool(tmp_path).execute(path="../")

    assert not result.success
    assert result.error_code == "LIST_FILES_FAILED"


@pytest.mark.asyncio
async def test_reports_truncation(tmp_path) -> None:
    for index in range(3):
        (tmp_path / f"{index}.txt").write_text(str(index), encoding="utf-8")

    result = await ListFilesTool(tmp_path).execute(max_results=2)

    assert result.success
    assert result.data["count"] == 2
    assert result.data["truncated"] is True
