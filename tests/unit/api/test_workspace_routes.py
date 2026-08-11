from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from codeagent.interaction.api import routes
from codeagent.product_state import ProductStateStore
from codeagent.workspaces import WorkspaceManager


def _client(
    tmp_path,
    monkeypatch,
) -> tuple[TestClient, WorkspaceManager, ProductStateStore]:
    manager = WorkspaceManager(tmp_path)
    state_store = ProductStateStore(tmp_path / "product.sqlite3")
    monkeypatch.setattr(routes, "_workspace_manager", manager)
    monkeypatch.setattr(routes, "_product_state_store", state_store)
    app = FastAPI()
    app.include_router(routes.router)
    return TestClient(app), manager, state_store


def test_project_upload_list_download_and_delete(tmp_path, monkeypatch):
    client, _manager, _state_store = _client(tmp_path, monkeypatch)
    created = client.post("/api/v1/projects", json={"name": "Release notes"})
    assert created.status_code == 201
    project = created.json()["data"]

    uploaded = client.post(
        f"/api/v1/projects/{project['project_id']}/files",
        files={"files": ("notes.md", b"# generated", "text/markdown")},
    )
    assert uploaded.status_code == 200
    listing = client.get(f"/api/v1/projects/{project['project_id']}/files").json()["data"]
    assert listing["files"][0]["path"] == "notes.md"

    downloaded = client.get(f"/api/v1/projects/{project['project_id']}/download/notes.md")
    assert downloaded.content == b"# generated"

    deleted_file = client.delete(f"/api/v1/projects/{project['project_id']}/files/notes.md")
    assert deleted_file.json()["data"]["deleted"] is True
    deleted_project = client.delete(f"/api/v1/projects/{project['project_id']}")
    assert deleted_project.json()["data"]["deleted"] is True


def test_conversation_delete_removes_only_session_workspace(tmp_path, monkeypatch):
    client, manager, state_store = _client(tmp_path, monkeypatch)
    session = manager.create("chat", conversation_id="conversation-1")
    project = manager.create_project("Durable")
    state_store.record_run(
        "task-1",
        conversation_id="conversation-1",
        query="chat",
        workspace_root=str(session),
    )

    response = client.request(
        "DELETE",
        "/api/v1/conversations/conversation-1",
        json={"workspace_root": str(session)},
    )
    assert response.json()["data"] == {
        "workspace_deleted": True,
        "history_deleted": True,
        "reason": "deleted",
    }
    assert not session.exists()
    assert state_store.get_conversation("conversation-1") is None

    retained = client.request(
        "DELETE",
        "/api/v1/conversations/conversation-2",
        json={"workspace_root": project["workspace_root"]},
    )
    assert retained.json()["data"]["reason"] == "project_retained"
    assert manager.resolve_project(project["project_id"]).exists()


def test_upload_rejects_reserved_marker_name(tmp_path, monkeypatch):
    client, _manager, _state_store = _client(tmp_path, monkeypatch)
    project = client.post("/api/v1/projects", json={"name": "Safe"}).json()["data"]
    response = client.post(
        f"/api/v1/projects/{project['project_id']}/files",
        files={"files": (".codeagent-workspace.json", b"bad", "application/json")},
    )
    assert response.status_code == 400


def test_task_preview_discovers_and_serves_real_workspace_html(tmp_path, monkeypatch):
    client, manager, state_store = _client(tmp_path, monkeypatch)
    workspace = manager.create("Build a landing page", conversation_id="conversation-1")
    (workspace / "index.html").write_text(
        '<!doctype html><link rel="stylesheet" href="styles.css"><h1>Live preview</h1>',
        encoding="utf-8",
    )
    (workspace / "styles.css").write_text("h1 { color: blue; }", encoding="utf-8")
    (workspace / ".env").write_text("SECRET=hidden", encoding="utf-8")
    state_store.record_run(
        "task-preview",
        conversation_id="conversation-1",
        query="Build a landing page",
        workspace_root=str(workspace),
    )

    manifest = client.get("/api/v1/tasks/task-preview/preview")

    assert manifest.status_code == 200
    assert manifest.json()["data"]["available"] is True
    assert manifest.json()["data"]["entrypoint"] == "index.html"
    rendered = client.get("/api/v1/tasks/task-preview/preview/index.html")
    assert rendered.status_code == 200
    assert rendered.headers["content-type"].startswith("text/html")
    assert "Live preview" in rendered.text
    assert "frame-ancestors 'self'" in rendered.headers["content-security-policy"]
    assert client.get("/api/v1/tasks/task-preview/preview/styles.css").status_code == 200
    assert client.get("/api/v1/tasks/task-preview/preview/.env").status_code == 400


def test_task_workspace_file_endpoint_serves_and_protects_files(tmp_path, monkeypatch):
    client, manager, state_store = _client(tmp_path, monkeypatch)
    workspace = manager.create("Write a script", conversation_id="conversation-1")
    (workspace / "main.py").write_bytes(b"print('ok')\n")
    (workspace / "notes.md").write_text("# notes", encoding="utf-8")
    (workspace / ".env").write_text("SECRET=hidden", encoding="utf-8")
    state_store.record_run(
        "task-files",
        conversation_id="conversation-1",
        query="Write a script",
        workspace_root=str(workspace),
    )

    served = client.get("/api/v1/tasks/task-files/files/main.py")
    assert served.status_code == 200
    assert served.content == b"print('ok')\n"
    assert served.headers["x-content-type-options"] == "nosniff"

    markdown = client.get("/api/v1/tasks/task-files/files/notes.md")
    assert markdown.status_code == 200
    assert b"# notes" in markdown.content

    # Hidden files and directory traversal stay blocked (clients may normalize
    # "../" away before the resolver sees it, so a refusal is 400 or 404 — never 200).
    assert client.get("/api/v1/tasks/task-files/files/.env").status_code == 400
    assert client.get("/api/v1/tasks/task-files/files/../product.sqlite3").status_code in {400, 404}
    assert client.get("/api/v1/tasks/task-files/files/missing.py").status_code == 404


def test_task_preview_returns_unavailable_without_html(tmp_path, monkeypatch):
    client, manager, state_store = _client(tmp_path, monkeypatch)
    workspace = manager.create("Python only", conversation_id="conversation-1")
    (workspace / "main.py").write_text("print('ok')", encoding="utf-8")
    state_store.record_run(
        "task-python",
        conversation_id="conversation-1",
        query="Python only",
        workspace_root=str(workspace),
    )

    manifest = client.get("/api/v1/tasks/task-python/preview")

    assert manifest.status_code == 200
    assert manifest.json()["data"] == {
        "available": False,
        "entrypoint": None,
        "entries": [],
        "revision": "0",
    }
