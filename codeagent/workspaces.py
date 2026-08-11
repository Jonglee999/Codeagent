"""Safe project-local workspaces for chat-first tasks."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class WorkspaceManager:
    """Create isolated writable task directories below the CodeAgent repository."""

    def __init__(self, repository_root: Path | None = None) -> None:
        configured_root = os.environ.get("CODEAGENT_REPOSITORY_ROOT")
        self.repository_root = (
            repository_root or (Path(configured_root) if configured_root else None) or Path.cwd()
        ).resolve()
        self.workspace_root = self.repository_root / ".codeagent" / "workspaces" / "sessions"
        self.project_root = self.repository_root / ".codeagent" / "workspaces" / "projects"

    def create(self, query: str, conversation_id: str | None = None) -> Path:
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        workspace = (self.workspace_root / f"chat-{timestamp}-{uuid.uuid4().hex[:10]}").resolve()
        workspace.relative_to(self.workspace_root.resolve())
        workspace.mkdir(mode=0o700)
        marker = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "query_preview": query.strip()[:240],
            "managed_by": "AGENT4CODE",
            "kind": "session",
            "conversation_id": conversation_id,
        }
        (workspace / ".codeagent-workspace.json").write_text(
            json.dumps(marker, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return workspace

    def create_project(self, name: str) -> dict[str, Any]:
        """Create a durable managed project that may own multiple conversations."""
        clean_name = " ".join(name.strip().split())[:80]
        if not clean_name:
            raise ValueError("Project name cannot be empty")
        self.project_root.mkdir(parents=True, exist_ok=True)
        project_id = uuid.uuid4().hex
        workspace = (self.project_root / f"project-{project_id}").resolve()
        workspace.relative_to(self.project_root.resolve())
        workspace.mkdir(mode=0o700)
        marker = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "managed_by": "AGENT4CODE",
            "kind": "project",
            "project_id": project_id,
            "name": clean_name,
        }
        self._write_marker(workspace, marker)
        return self._project_metadata(workspace, marker)

    def list_projects(self) -> list[dict[str, Any]]:
        if not self.project_root.exists():
            return []
        projects: list[dict[str, Any]] = []
        for workspace in self.project_root.glob("project-*"):
            marker = self._read_marker(workspace)
            if marker and marker.get("kind") == "project":
                projects.append(self._project_metadata(workspace, marker))
        return sorted(projects, key=lambda item: item["created_at"], reverse=True)

    def resolve_project(self, project_id: str) -> Path:
        if not project_id or any(char not in "0123456789abcdef" for char in project_id.lower()):
            raise KeyError("Unknown project")
        workspace = (self.project_root / f"project-{project_id}").resolve()
        workspace.relative_to(self.project_root.resolve())
        marker = self._read_marker(workspace)
        if not marker or marker.get("project_id") != project_id or marker.get("kind") != "project":
            raise KeyError("Unknown project")
        return workspace

    def list_files(self, project_id: str) -> list[dict[str, Any]]:
        workspace = self.resolve_project(project_id)
        files: list[dict[str, Any]] = []
        for path in workspace.rglob("*"):
            if not path.is_file() or path.name == ".codeagent-workspace.json":
                continue
            relative = path.relative_to(workspace)
            if relative.parts and relative.parts[0] in {".codeagent", ".git"}:
                continue
            stat = path.stat()
            files.append(
                {
                    "path": relative.as_posix(),
                    "name": path.name,
                    "size": stat.st_size,
                    "modified_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                }
            )
            if len(files) >= 500:
                break
        return sorted(files, key=lambda item: item["path"].lower())

    def resolve_project_file(
        self,
        project_id: str,
        file_path: str,
        *,
        must_exist: bool = True,
    ) -> Path:
        workspace = self.resolve_project(project_id)
        candidate = (workspace / file_path).resolve()
        candidate.relative_to(workspace)
        if candidate == workspace or candidate.name == ".codeagent-workspace.json":
            raise ValueError("Invalid project file path")
        if must_exist and not candidate.is_file():
            raise FileNotFoundError(file_path)
        return candidate

    def delete_session(self, workspace_path: str, conversation_id: str) -> bool:
        """Delete only an AGENT4CODE-owned ephemeral session workspace."""
        workspace = Path(workspace_path).expanduser().resolve()
        workspace.relative_to(self.workspace_root.resolve())
        if workspace == self.workspace_root.resolve() or not workspace.exists():
            return False
        marker = self._read_marker(workspace)
        if not marker or marker.get("managed_by") != "AGENT4CODE":
            raise ValueError("Workspace is not managed by AGENT4CODE")
        if marker.get("kind", "session") != "session":
            raise ValueError("Project workspaces must be deleted from the project UI")
        marker_conversation = marker.get("conversation_id")
        if marker_conversation and marker_conversation != conversation_id:
            raise ValueError("Conversation does not own this workspace")
        shutil.rmtree(workspace)
        return True

    def delete_project(self, project_id: str) -> bool:
        workspace = self.resolve_project(project_id)
        shutil.rmtree(workspace)
        return True

    def metadata_for(self, workspace_path: str | Path) -> dict[str, Any] | None:
        """Return a verified marker only for an exact managed workspace directory."""
        workspace = Path(workspace_path).expanduser().resolve()
        allowed = False
        for root in (self.workspace_root.resolve(), self.project_root.resolve()):
            try:
                workspace.relative_to(root)
                allowed = workspace != root
                if allowed:
                    break
            except ValueError:
                continue
        if not allowed or not workspace.is_dir():
            return None
        return self._read_marker(workspace)

    @staticmethod
    def _read_marker(workspace: Path) -> dict[str, Any] | None:
        marker_path = workspace / ".codeagent-workspace.json"
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return marker if marker.get("managed_by") == "AGENT4CODE" else None

    @staticmethod
    def _write_marker(workspace: Path, marker: dict[str, Any]) -> None:
        (workspace / ".codeagent-workspace.json").write_text(
            json.dumps(marker, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _project_metadata(
        workspace: Path,
        marker: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "project_id": marker["project_id"],
            "name": marker["name"],
            "created_at": marker["created_at"],
            "workspace_root": str(workspace),
        }
