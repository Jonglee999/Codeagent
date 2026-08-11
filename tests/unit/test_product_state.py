from __future__ import annotations

from codeagent.product_state import ProductStateStore


def test_default_database_can_be_isolated_by_environment(tmp_path, monkeypatch):
    database = tmp_path / "isolated" / "product.sqlite3"
    monkeypatch.setenv("CODEAGENT_STATE_DB", str(database))

    store = ProductStateStore()

    assert store.path == database.resolve()
    assert database.is_file()


def test_conversation_run_report_and_artifacts_are_durable(tmp_path):
    database = tmp_path / "product.sqlite3"
    store = ProductStateStore(database)
    store.record_run(
        "task-1",
        conversation_id="conversation-1",
        query="Create a parser",
        workspace_root=str(tmp_path / "workspace"),
        project_id="project-1",
    )
    store.update_run_state("task-1", "running")
    store.save_report(
        "task-1",
        {
            "assistant_response": "Implemented.",
            "changes": [{"path": "parser.py", "diff": "+pass"}],
            "validation_results": [{"layer": "tests", "passed": True, "stdout": "1 passed"}],
            "transcript_path": ".codeagent/transcripts/task-1.json",
        },
    )
    store.update_run_state("task-1", "completed")

    reloaded = ProductStateStore(database)
    conversation = reloaded.get_conversation("conversation-1")

    assert conversation is not None
    assert conversation["state"] == "completed"
    assert conversation["project_id"] == "project-1"
    assert [message["role"] for message in conversation["messages"]] == ["user", "assistant"]
    assert conversation["messages"][1]["content"] == "Implemented."


def test_import_is_idempotent_and_delete_cascades(tmp_path):
    store = ProductStateStore(tmp_path / "product.sqlite3")
    entry = {
        "conversation_id": "legacy-1",
        "task_id": "task-old",
        "query": "Legacy conversation",
        "timestamp": "2026-08-07T00:00:00+00:00",
        "state": "completed",
        "workspace_root": str(tmp_path / "legacy"),
        "messages": [
            {
                "id": "message-old",
                "role": "user",
                "content": "Legacy conversation",
                "timestamp": "2026-08-07T00:00:00+00:00",
                "task_id": "task-old",
            }
        ],
    }

    assert store.import_conversations([entry]) == 1
    assert store.import_conversations([entry]) == 1
    store.record_run(
        "task-current",
        conversation_id="legacy-1",
        query="Follow-up",
        workspace_root=str(tmp_path / "legacy"),
    )
    store.save_report("task-current", {"artifacts": [{"kind": "file", "path": "index.html"}]})
    assert len(store.get_conversation("legacy-1")["messages"]) == 2
    assert store.delete_conversation("legacy-1") is True
    assert store.get_conversation("legacy-1") is None
    assert store.get_run("task-current") is None


def test_project_conversations_are_deleted_together(tmp_path):
    store = ProductStateStore(tmp_path / "product.sqlite3")
    for index in range(2):
        store.record_run(
            f"task-{index}",
            conversation_id=f"conversation-{index}",
            query=f"Task {index}",
            workspace_root=str(tmp_path / "project"),
            project_id="shared-project",
        )

    assert store.delete_project_conversations("shared-project") == 2
    assert store.list_conversations() == []
    assert store.get_run("task-0") is None
    assert store.get_run("task-1") is None


def test_get_run_returns_durable_report_after_transient_cache_expiry(tmp_path):
    store = ProductStateStore(tmp_path / "product.sqlite3")
    store.record_run(
        "task-1",
        conversation_id="conversation-1",
        query="Fix it",
        workspace_root=str(tmp_path),
    )
    store.save_report(
        "task-1",
        {
            "task_id": "task-1",
            "assistant_response": "Done",
            "validation_results": [],
        },
    )
    store.update_run_state("task-1", "completed")

    run = store.get_run("task-1")

    assert run is not None
    assert run["state"] == "completed"
    assert run["report"]["assistant_response"] == "Done"


def test_run_persists_recovery_lineage(tmp_path):
    store = ProductStateStore(tmp_path / "product.sqlite3")
    store.record_run(
        "task-source",
        conversation_id="conversation-1",
        query="Original",
        workspace_root=str(tmp_path),
    )
    store.record_run(
        "task-recovery",
        conversation_id="conversation-1",
        query="Continue",
        workspace_root=str(tmp_path),
        recovered_from_task_id="task-source",
    )

    assert store.get_run("task-recovery")["recovered_from_task_id"] == "task-source"


def test_artifact_download_resolution_stays_inside_run_workspace(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    artifact = workspace / ".codeagent/artifacts/task-1/git.diff"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("patch", encoding="utf-8")
    store = ProductStateStore(tmp_path / "product.sqlite3")
    store.record_run(
        "task-1",
        conversation_id="conversation-1",
        query="Fix it",
        workspace_root=str(workspace),
    )
    store.save_report(
        "task-1",
        {
            "artifacts": [{"kind": "git_diff", "path": str(artifact), "size": 5}],
        },
    )

    stored = store.list_artifacts("task-1")

    assert len(stored) == 1
    assert store.resolve_artifact("task-1", stored[0]["artifact_id"]) == artifact
