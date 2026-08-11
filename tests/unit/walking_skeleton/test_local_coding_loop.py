from __future__ import annotations

import subprocess

import pytest

from codeagent.tools.gateway import ToolGateway
from codeagent.workspaces import WorkspaceManager


@pytest.mark.asyncio
async def test_agent_toolset_can_inspect_edit_test_and_diff(tmp_path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "calculator.py").write_text(
        "def add(left: int, right: int) -> int:\n    return left - right\n",
        encoding="utf-8",
    )
    (tmp_path / "test_calculator.py").write_text(
        "from calculator import add\n\ndef test_add():\n    assert add(2, 3) == 5\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "calculator.py", "test_calculator.py"], cwd=tmp_path, check=True)

    gateway = ToolGateway(project_root=str(tmp_path))
    try:
        names = {tool.name for tool in gateway.list_tools()}
        assert {"read_file", "write_file", "search_code", "git", "run_terminal"} <= names

        search = await gateway.execute_tool("search_code", {"query": "return left - right"})
        assert search.success

        read = await gateway.execute_tool("read_file", {"file_path": "calculator.py"})
        assert read.success and "left - right" in read.data["content"]

        write = await gateway.execute_tool(
            "write_file",
            {
                "file_path": "calculator.py",
                "content": "def add(left: int, right: int) -> int:\n    return left + right\n",
                "mode": "modify",
            },
        )
        assert write.success and "left + right" in write.data["diff"]

        test_run = await gateway.execute_tool(
            "run_terminal", {"command": "python -m pytest -q", "timeout": 30}
        )
        assert test_run.success
        assert test_run.data["exit_code"] == 0

        diff = await gateway.execute_tool("git", {"action": "diff"})
        assert diff.success and "left + right" in diff.data["stdout"]
    finally:
        await gateway.aclose()


@pytest.mark.asyncio
async def test_file_tools_confine_changes_to_workspace(tmp_path) -> None:
    gateway = ToolGateway(project_root=str(tmp_path))
    result = await gateway.execute_tool(
        "write_file",
        {"file_path": "../escape.py", "content": "unsafe = True\n", "mode": "create"},
    )
    assert not result.success
    assert result.error_code == "PATH_TRAVERSAL"


@pytest.mark.asyncio
async def test_chat_workspace_supports_create_read_execute_and_delete(tmp_path) -> None:
    workspace = WorkspaceManager(tmp_path).create("Create and remove a probe")
    gateway = ToolGateway(project_root=str(workspace))
    try:
        created = await gateway.execute_tool(
            "write_file",
            {"file_path": "probe.txt", "content": "agent-ready\n", "mode": "create"},
        )
        read = await gateway.execute_tool("read_file", {"file_path": "probe.txt"})
        executed = await gateway.execute_tool(
            "run_terminal",
            {"command": "python -c \"from pathlib import Path; print(Path('probe.txt').read_text().strip())\""},
        )
        deleted = await gateway.execute_tool("delete_file", {"file_path": "probe.txt"})

        assert created.success
        assert read.success and read.data["content"].strip() == "agent-ready"
        assert executed.success and executed.data["stdout"].strip() == "agent-ready"
        assert deleted.success and not (workspace / "probe.txt").exists()
    finally:
        await gateway.aclose()
