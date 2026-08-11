from __future__ import annotations

import pytest

from codeagent.tools.file.delete_file import DeleteFileTool


@pytest.mark.asyncio
async def test_deletes_file_inside_workspace(tmp_path):
    target = tmp_path / "obsolete.py"
    target.write_text("pass\n", encoding="utf-8")

    result = await DeleteFileTool(tmp_path).execute("obsolete.py")

    assert result.success
    assert result.data["deleted"] == "obsolete.py"
    assert not target.exists()


@pytest.mark.asyncio
async def test_rejects_project_root_and_path_escape(tmp_path):
    tool = DeleteFileTool(tmp_path)

    root_result = await tool.execute(str(tmp_path))
    escape_result = await tool.execute("../outside.txt")

    assert not root_result.success
    assert not escape_result.success
