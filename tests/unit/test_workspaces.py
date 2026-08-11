from __future__ import annotations

import json
from pathlib import Path

import pytest

from codeagent.workspaces import WorkspaceManager


def test_default_workspace_can_be_isolated_by_environment(tmp_path, monkeypatch):
    repository = tmp_path / "isolated-repository"
    monkeypatch.setenv("CODEAGENT_REPOSITORY_ROOT", str(repository))

    manager = WorkspaceManager()
    workspace = manager.create("isolated")

    assert manager.repository_root == repository.resolve()
    workspace.relative_to(repository.resolve())


def test_chat_workspace_is_unique_writable_and_project_local(tmp_path):
    manager = WorkspaceManager(tmp_path)
    first = manager.create("Create a small parser")
    second = manager.create("Create a small parser")

    assert first != second
    first.relative_to(manager.workspace_root)
    probe = first / "probe.txt"
    probe.write_text("ok", encoding="utf-8")
    assert probe.read_text(encoding="utf-8") == "ok"
    probe.unlink()
    assert not probe.exists()
    marker = json.loads((first / ".codeagent-workspace.json").read_text(encoding="utf-8"))
    assert marker["managed_by"] == "AGENT4CODE"


def test_chat_workspace_inherits_repository_skills_without_copying(tmp_path):
    skill = tmp_path / ".codeagent" / "skills" / "testing" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text(
        "---\nname: testing\ntriggers: [test]\n---\nTesting discipline",
        encoding="utf-8",
    )

    workspace = WorkspaceManager(tmp_path).create("Add tests")

    copied = workspace / ".codeagent" / "skills" / "testing" / "SKILL.md"
    assert not copied.exists()

    from codeagent.extensions import resolve_project_instructions

    resolution = resolve_project_instructions(workspace, "run tests", product_root=tmp_path)
    assert len(resolution.skills) == 1
    assert resolution.skills[0].source == "product"
    assert "Testing discipline" in resolution.instructions


def test_session_delete_requires_matching_managed_conversation(tmp_path):
    manager = WorkspaceManager(tmp_path)
    workspace = manager.create("temporary", conversation_id="conversation-1")

    with pytest.raises(ValueError, match="does not own"):
        manager.delete_session(str(workspace), "conversation-2")

    assert workspace.exists()
    assert manager.delete_session(str(workspace), "conversation-1") is True
    assert not workspace.exists()


def test_project_files_are_durable_until_project_delete(tmp_path):
    manager = WorkspaceManager(tmp_path)
    project = manager.create_project("Demo Project")
    target = manager.resolve_project_file(project["project_id"], "result.md", must_exist=False)
    target.write_text("# result", encoding="utf-8")

    assert manager.list_projects()[0]["name"] == "Demo Project"
    assert manager.list_files(project["project_id"])[0]["path"] == "result.md"
    with pytest.raises(ValueError, match="Invalid project file path"):
        manager.resolve_project_file(project["project_id"], ".codeagent-workspace.json")

    assert manager.delete_project(project["project_id"]) is True
    assert not Path(project["workspace_root"]).exists()
