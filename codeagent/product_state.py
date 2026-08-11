"""Durable product state for conversations, runs, messages, and report artifacts."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ProductStateStore:
    """Small SQLite source of truth shared by the API and optional worker."""

    def __init__(self, path: Path | None = None) -> None:
        root = Path.cwd().resolve()
        configured_path = os.environ.get("CODEAGENT_STATE_DB")
        self.path = (
            path
            or (Path(configured_path) if configured_path else None)
            or root / ".codeagent" / "state" / "product.sqlite3"
        ).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        with self._lock, self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    workspace_root TEXT,
                    project_id TEXT,
                    latest_task_id TEXT,
                    state TEXT NOT NULL DEFAULT 'pending',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    message_id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL REFERENCES conversations(conversation_id)
                        ON DELETE CASCADE,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    task_id TEXT,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_messages_conversation
                    ON messages(conversation_id, created_at);
                CREATE TABLE IF NOT EXISTS runs (
                    task_id TEXT PRIMARY KEY,
                    conversation_id TEXT REFERENCES conversations(conversation_id)
                        ON DELETE SET NULL,
                    query TEXT NOT NULL,
                    workspace_root TEXT NOT NULL,
                    state TEXT NOT NULL,
                    report_json TEXT,
                    recovered_from_task_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_runs_conversation
                    ON runs(conversation_id, created_at);
                CREATE TABLE IF NOT EXISTS artifacts (
                    artifact_id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL REFERENCES runs(task_id) ON DELETE CASCADE,
                    kind TEXT NOT NULL,
                    path TEXT,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )
            columns = {row["name"] for row in connection.execute("PRAGMA table_info(runs)")}
            if "benchmark_instance_id" not in columns:
                connection.execute("ALTER TABLE runs ADD COLUMN benchmark_instance_id TEXT")
            if "recovered_from_task_id" not in columns:
                connection.execute("ALTER TABLE runs ADD COLUMN recovered_from_task_id TEXT")

    def upsert_conversation(
        self,
        conversation_id: str,
        *,
        title: str,
        workspace_root: str | None,
        project_id: str | None = None,
        task_id: str | None = None,
        state: str = "pending",
        timestamp: str | None = None,
    ) -> None:
        stamp = timestamp or _now()
        clean_title = title.strip()[:240] or "New conversation"
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO conversations (
                    conversation_id, title, workspace_root, project_id,
                    latest_task_id, state, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(conversation_id) DO UPDATE SET
                    title = CASE WHEN conversations.title = 'New conversation'
                        THEN excluded.title ELSE conversations.title END,
                    workspace_root = COALESCE(excluded.workspace_root, conversations.workspace_root),
                    project_id = COALESCE(excluded.project_id, conversations.project_id),
                    latest_task_id = COALESCE(excluded.latest_task_id, conversations.latest_task_id),
                    state = excluded.state,
                    updated_at = excluded.updated_at
                """,
                (
                    conversation_id,
                    clean_title,
                    workspace_root,
                    project_id,
                    task_id,
                    state,
                    stamp,
                    stamp,
                ),
            )

    def record_run(
        self,
        task_id: str,
        *,
        conversation_id: str,
        query: str,
        workspace_root: str,
        project_id: str | None = None,
        benchmark_instance_id: str | None = None,
        recovered_from_task_id: str | None = None,
    ) -> None:
        stamp = _now()
        self.upsert_conversation(
            conversation_id,
            title=query,
            workspace_root=workspace_root,
            project_id=project_id,
            task_id=task_id,
            state="pending",
            timestamp=stamp,
        )
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO runs (
                    task_id, conversation_id, query, workspace_root,
                    state, created_at, updated_at, benchmark_instance_id,
                    recovered_from_task_id
                ) VALUES (?, ?, ?, ?, 'pending', ?, ?, ?, ?)
                ON CONFLICT(task_id) DO NOTHING
                """,
                (
                    task_id,
                    conversation_id,
                    query,
                    workspace_root,
                    stamp,
                    stamp,
                    benchmark_instance_id,
                    recovered_from_task_id,
                ),
            )
        self.append_message(
            conversation_id,
            role="user",
            content=query,
            task_id=task_id,
            message_id=f"{task_id}:user",
            timestamp=stamp,
        )

    def append_message(
        self,
        conversation_id: str,
        *,
        role: str,
        content: str,
        task_id: str | None = None,
        message_id: str | None = None,
        timestamp: str | None = None,
    ) -> None:
        if role not in {"user", "assistant"}:
            raise ValueError("Unsupported message role")
        stamp = timestamp or _now()
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO messages (
                    message_id, conversation_id, role, content, task_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (message_id or uuid.uuid4().hex, conversation_id, role, content, task_id, stamp),
            )
            connection.execute(
                "UPDATE conversations SET updated_at = ? WHERE conversation_id = ?",
                (stamp, conversation_id),
            )

    def update_run_state(self, task_id: str, state: str) -> None:
        stamp = _now()
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT conversation_id FROM runs WHERE task_id = ?",
                (task_id,),
            ).fetchone()
            connection.execute(
                "UPDATE runs SET state = ?, updated_at = ? WHERE task_id = ?",
                (state, stamp, task_id),
            )
            if row and row["conversation_id"]:
                connection.execute(
                    """
                    UPDATE conversations SET state = ?, latest_task_id = ?, updated_at = ?
                    WHERE conversation_id = ?
                    """,
                    (state, task_id, stamp, row["conversation_id"]),
                )

    def save_report(self, task_id: str, report: dict[str, Any]) -> None:
        stamp = _now()
        payload = json.dumps(report, ensure_ascii=False)
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT conversation_id FROM runs WHERE task_id = ?",
                (task_id,),
            ).fetchone()
            connection.execute(
                "UPDATE runs SET report_json = ?, updated_at = ? WHERE task_id = ?",
                (payload, stamp, task_id),
            )
        if row is None:
            return
        if row and row["conversation_id"] and report.get("assistant_response"):
            self.append_message(
                row["conversation_id"],
                role="assistant",
                content=str(report["assistant_response"]),
                task_id=task_id,
                message_id=f"{task_id}:assistant",
                timestamp=stamp,
            )
        self.replace_report_artifacts(task_id, report)

    def replace_report_artifacts(self, task_id: str, report: dict[str, Any]) -> None:
        artifacts: list[tuple[str, str | None, dict[str, Any]]] = []
        for change in report.get("changes", []) or []:
            if isinstance(change, dict):
                artifacts.append(("change", change.get("path") or change.get("file"), change))
        for result in report.get("validation_results", []) or []:
            if isinstance(result, dict):
                artifacts.append(("validation", result.get("layer"), result))
        transcript = report.get("transcript_path")
        if transcript:
            artifacts.append(("transcript", str(transcript), {"path": str(transcript)}))
        for artifact in report.get("artifacts", []) or []:
            if isinstance(artifact, dict):
                artifacts.append(
                    (
                        str(artifact.get("kind") or "artifact"),
                        artifact.get("path"),
                        artifact,
                    )
                )
        with self._lock, self._connect() as connection:
            connection.execute("DELETE FROM artifacts WHERE task_id = ?", (task_id,))
            connection.executemany(
                """
                INSERT INTO artifacts (
                    artifact_id, task_id, kind, path, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        uuid.uuid4().hex,
                        task_id,
                        kind,
                        path,
                        json.dumps(data, ensure_ascii=False),
                        _now(),
                    )
                    for kind, path, data in artifacts
                ],
            )

    def get_conversation(self, conversation_id: str) -> dict[str, Any] | None:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM conversations WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
            if row is None:
                return None
            return self._conversation_payload(connection, row)

    def list_conversations(self, limit: int = 200) -> list[dict[str, Any]]:
        bounded = max(1, min(limit, 500))
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM conversations ORDER BY updated_at DESC LIMIT ?",
                (bounded,),
            ).fetchall()
            return [self._conversation_payload(connection, row) for row in rows]

    @staticmethod
    def _conversation_payload(
        connection: sqlite3.Connection,
        row: sqlite3.Row,
    ) -> dict[str, Any]:
        messages = connection.execute(
            """
            SELECT message_id, role, content, task_id, created_at
            FROM messages WHERE conversation_id = ? ORDER BY created_at, rowid
            """,
            (row["conversation_id"],),
        ).fetchall()
        return {
            "conversation_id": row["conversation_id"],
            "task_id": row["latest_task_id"] or "",
            "query": row["title"],
            "timestamp": row["updated_at"],
            "state": row["state"],
            "workspace_root": row["workspace_root"],
            "project_id": row["project_id"],
            "messages": [
                {
                    "id": message["message_id"],
                    "role": message["role"],
                    "content": message["content"],
                    "timestamp": message["created_at"],
                    "task_id": message["task_id"],
                }
                for message in messages
            ],
        }

    def delete_conversation(self, conversation_id: str) -> bool:
        with self._lock, self._connect() as connection:
            # A conversation owns its task runs and report artifacts.  The schema's
            # historical ON DELETE SET NULL rule kept orphaned runs after users
            # removed a conversation, so delete them explicitly first.
            connection.execute(
                "DELETE FROM runs WHERE conversation_id = ?",
                (conversation_id,),
            )
            cursor = connection.execute(
                "DELETE FROM conversations WHERE conversation_id = ?",
                (conversation_id,),
            )
            return cursor.rowcount > 0

    def delete_project_conversations(self, project_id: str) -> int:
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                DELETE FROM runs
                WHERE conversation_id IN (
                    SELECT conversation_id FROM conversations WHERE project_id = ?
                )
                """,
                (project_id,),
            )
            cursor = connection.execute(
                "DELETE FROM conversations WHERE project_id = ?",
                (project_id,),
            )
            return cursor.rowcount

    def benchmark_runs(self) -> list[dict[str, Any]]:
        """Return the latest durable run for each explicitly tagged benchmark instance."""
        rows = self.benchmark_run_history()
        latest: dict[str, dict[str, Any]] = {}
        for row in rows:
            instance_id = row["instance_id"]
            if instance_id not in latest:
                latest[instance_id] = row
        return list(latest.values())

    def benchmark_run_history(self) -> list[dict[str, Any]]:
        """Return all explicitly tagged benchmark runs, newest first."""
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM runs
                WHERE benchmark_instance_id IS NOT NULL
                ORDER BY updated_at DESC
                """
            ).fetchall()
        history: list[dict[str, Any]] = []
        for row in rows:
            instance_id = str(row["benchmark_instance_id"])
            report = json.loads(row["report_json"]) if row["report_json"] else None
            history.append({
                "instance_id": instance_id,
                "task_id": row["task_id"],
                "state": row["state"],
                "workspace_root": row["workspace_root"],
                "report": report,
                "updated_at": row["updated_at"],
            })
        return history

    def get_run(self, task_id: str) -> dict[str, Any] | None:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM runs WHERE task_id = ?",
                (task_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "task_id": row["task_id"],
            "conversation_id": row["conversation_id"],
            "query": row["query"],
            "workspace_root": row["workspace_root"],
            "state": row["state"],
            "report": json.loads(row["report_json"]) if row["report_json"] else None,
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "benchmark_instance_id": row["benchmark_instance_id"],
            "recovered_from_task_id": row["recovered_from_task_id"],
        }

    def list_artifacts(self, task_id: str) -> list[dict[str, Any]]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT artifact_id, kind, metadata_json, created_at
                FROM artifacts WHERE task_id = ? ORDER BY created_at, rowid
                """,
                (task_id,),
            ).fetchall()
        return [
            {
                "artifact_id": row["artifact_id"],
                "kind": row["kind"],
                "metadata": json.loads(row["metadata_json"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def resolve_artifact(self, task_id: str, artifact_id: str) -> Path:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                """
                SELECT artifacts.path, runs.workspace_root
                FROM artifacts JOIN runs ON artifacts.task_id = runs.task_id
                WHERE artifacts.task_id = ? AND artifacts.artifact_id = ?
                """,
                (task_id, artifact_id),
            ).fetchone()
        if row is None or not row["path"]:
            raise KeyError(artifact_id)
        workspace = Path(row["workspace_root"]).resolve()
        candidate = Path(row["path"]).resolve()
        try:
            candidate.relative_to(workspace)
        except ValueError as exc:
            raise ValueError("Artifact path escaped its workspace") from exc
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        return candidate

    def import_conversations(self, entries: Iterable[dict[str, Any]]) -> int:
        imported = 0
        for entry in entries:
            conversation_id = str(entry.get("conversation_id") or "")[:100]
            if not conversation_id:
                continue
            self.upsert_conversation(
                conversation_id,
                title=str(entry.get("query") or "New conversation"),
                workspace_root=entry.get("workspace_root"),
                project_id=entry.get("project_id"),
                task_id=entry.get("task_id"),
                state=str(entry.get("state") or "completed"),
                timestamp=entry.get("timestamp"),
            )
            for message in entry.get("messages", []) or []:
                if not isinstance(message, dict) or message.get("role") not in {
                    "user",
                    "assistant",
                }:
                    continue
                self.append_message(
                    conversation_id,
                    role=message["role"],
                    content=str(message.get("content") or ""),
                    task_id=message.get("task_id"),
                    message_id=str(message.get("id") or uuid.uuid4().hex),
                    timestamp=message.get("timestamp"),
                )
            imported += 1
        return imported


product_state_store = ProductStateStore()
