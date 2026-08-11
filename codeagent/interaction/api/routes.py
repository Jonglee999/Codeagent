"""FastAPI REST routes — 任务生命周期管理端点。"""

from __future__ import annotations

import asyncio
import logging
import mimetypes
import os
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse, JSONResponse

from codeagent.gateway.orchestration_gateway import (
    HumanDecision,
    UserRequest,
)
from codeagent.gateway.orchestration_gateway_impl import OrchestrationGatewayImpl
from codeagent.benchmarks import BenchmarkCatalog, PredictionExporter
from codeagent.config import (
    get_env,
    get_evolution_enabled,
    get_memory_enabled,
    get_memory_use_vector,
    get_mcp_enabled,
    get_sandbox_enabled,
    get_skills_enabled,
)
from codeagent.tools.gateway import ToolGateway
from codeagent.model_routing import ModelRouter
from codeagent.extensions import discover_skills, resolve_project_instructions
from codeagent.extensions.mcp import resolve_mcp_config
from codeagent.orchestration.capabilities import select_mcp_server_names
from codeagent.context_engine.capabilities import context_capability_report
from codeagent.workspaces import WorkspaceManager
from codeagent.product_state import product_state_store

from .auth import verify_api_key
from .models import (
    ApiResponse,
    ConversationDeleteRequest,
    ConversationImportRequest,
    DecisionRequest,
    RecoveryRequest,
    SteeringRequest,
    ProjectCreateRequest,
    TaskCreateRequest,
    TaskReportResponse,
    TaskStatusResponse,
)
from .rate_limit import RateLimiter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1")


def _get_gateway(request: Request) -> OrchestrationGatewayImpl:
    """从 app.state 获取 Gateway 单例。"""
    gateway: OrchestrationGatewayImpl | None = getattr(request.app.state, "gateway", None)
    if gateway is None:
        raise HTTPException(status_code=503, detail="Gateway not initialized")
    return gateway


# 速率限制器（单例，使用 Redis URL）
_rate_limiter = RateLimiter()
_benchmark_catalog = BenchmarkCatalog()
_workspace_manager = WorkspaceManager()
_product_state_store = product_state_store
_MAX_UPLOAD_BYTES = 25 * 1024 * 1024
_PREVIEW_EXTENSIONS = {
    ".html",
    ".htm",
    ".css",
    ".js",
    ".mjs",
    ".json",
    ".map",
    ".svg",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".ico",
    ".woff",
    ".woff2",
    ".ttf",
    ".otf",
    ".mp3",
    ".wav",
    ".mp4",
    ".webm",
}
_PREVIEW_IGNORED_DIRECTORIES = {
    ".codeagent",
    ".git",
    ".pytest_cache",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
}


def _conversation_history_for_run(
    conversation_id: str,
    supplied_history: list[dict[str, str]],
    current_query: str,
) -> list[dict[str, str]]:
    """Return bounded, de-duplicated history for the next conversation turn.

    Persisted messages are authoritative once a conversation exists.  Falling
    back to the client payload keeps the first follow-up compatible with older
    frontends, while removing a trailing copy of the current query prevents the
    model from seeing the same user turn twice.
    """
    stored = _product_state_store.get_conversation(conversation_id)
    source = (stored or {}).get("messages") or supplied_history
    normalized: list[dict[str, str]] = []
    for item in source:
        role = item.get("role")
        content = item.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str):
            continue
        content = content.strip()
        if not content:
            continue
        message = {"role": role, "content": content[:12000]}
        if normalized and normalized[-1] == message:
            continue
        normalized.append(message)

    query = current_query.strip()
    while (
        normalized
        and normalized[-1]["role"] == "user"
        and normalized[-1]["content"] == query
    ):
        normalized.pop()
    return normalized[-20:]


def _preview_workspace(task_id: str) -> Path:
    run = _product_state_store.get_run(task_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Task not found")
    workspace = Path(str(run["workspace_root"])).resolve()
    if not workspace.is_dir():
        raise HTTPException(status_code=404, detail="Task workspace is unavailable")
    return workspace


def _preview_files(workspace: Path) -> list[Path]:
    files: list[Path] = []
    for current_root, directories, names in os.walk(workspace):
        directories[:] = [
            name
            for name in directories
            if name not in _PREVIEW_IGNORED_DIRECTORIES and not name.startswith(".")
        ]
        current = Path(current_root)
        for name in names:
            path = current / name
            if name.startswith(".") or path.suffix.lower() not in _PREVIEW_EXTENSIONS:
                continue
            files.append(path)
            if len(files) >= 10_000:
                return files
    return files


def _resolve_workspace_file(workspace: Path, file_path: str, *, purpose: str = "file") -> Path:
    """Resolve a file inside a task workspace with path-traversal protection.

    Hidden paths (dot-directories/files such as .env, .git) are always rejected so
    neither the preview nor the general file endpoint can leak workspace secrets.
    """
    relative = Path(file_path.replace("\\", "/"))
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} or part.startswith(".") for part in relative.parts)
    ):
        raise HTTPException(status_code=400, detail=f"Invalid {purpose} path")
    candidate = (workspace / relative).resolve()
    try:
        candidate.relative_to(workspace)
    except ValueError as exc:
        raise HTTPException(
            status_code=400, detail=f"{purpose.capitalize()} path escaped workspace"
        ) from exc
    if not candidate.is_file():
        raise HTTPException(status_code=404, detail=f"{purpose.capitalize()} file not found")
    return candidate


def _resolve_preview_file(workspace: Path, file_path: str) -> Path:
    candidate = _resolve_workspace_file(workspace, file_path, purpose="preview")
    if candidate.suffix.lower() not in _PREVIEW_EXTENSIONS:
        raise HTTPException(status_code=404, detail="Preview file not found")
    return candidate


def _report_artifacts(task_id: str) -> list[dict[str, Any]]:
    return [
        {
            "artifact_id": item["artifact_id"],
            "kind": item["kind"],
            "size": item["metadata"].get("size"),
            "download_url": (f"/api/v1/tasks/{task_id}/artifacts/{item['artifact_id']}"),
        }
        for item in _product_state_store.list_artifacts(task_id)
        if item["kind"] in {"git_diff", "validation_output", "report", "transcript"}
    ]


# 二进制文件扩展名（快速判断，避免逐个采样）；其余未知扩展名用 null-byte 采样兜底。
_BINARY_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".bmp", ".avif",
    ".woff", ".woff2", ".ttf", ".otf", ".eot",
    ".mp3", ".wav", ".mp4", ".webm", ".mov", ".avi",
    ".zip", ".gz", ".tar", ".bz2", ".xz", ".7z", ".whl", ".so", ".dll", ".exe",
    ".pdf", ".bin", ".pyc", ".class", ".jar", ".dat", ".db", ".sqlite", ".sqlite3",
}


def _is_binary_file(path: Path) -> bool:
    """Return True for known binary extensions or content containing null bytes."""
    if path.suffix.lower() in _BINARY_SUFFIXES:
        return True
    try:
        with path.open("rb") as fh:
            return b"\x00" in fh.read(4096)
    except OSError:
        return True


def _list_task_files(workspace: Path, *, max_files: int = 10_000) -> list[dict[str, Any]]:
    """List non-hidden workspace files with text/binary markers for the frontend.

    Mirrors `_preview_files` traversal but without the extension allowlist, so any
    file the agent produced (Python, markdown, configs, images, ...) is surfaced.
    """
    files: list[dict[str, Any]] = []
    for current_root, directories, names in os.walk(workspace):
        directories[:] = [
            name
            for name in directories
            if name not in _PREVIEW_IGNORED_DIRECTORIES and not name.startswith(".")
        ]
        current = Path(current_root)
        for name in names:
            if name.startswith("."):
                continue
            path = current / name
            if not path.is_file():
                continue
            files.append(
                {
                    "path": path.relative_to(workspace).as_posix(),
                    "name": name,
                    "size": path.stat().st_size,
                    "modified": int(path.stat().st_mtime),
                    "binary": _is_binary_file(path),
                }
            )
            if len(files) >= max_files:
                return files
    return files


@router.get("/catalog/tasks", response_model=ApiResponse)
async def list_benchmark_tasks(
    _auth: str = Depends(verify_api_key),
) -> ApiResponse:
    tasks = []
    for task in _benchmark_catalog.list_tasks():
        item = asdict(task)
        item["workspace_path"] = str(_benchmark_catalog.workspace_for(task.instance_id))
        item["workspace_ready"] = (
            _benchmark_catalog.repository_cache_for(task.instance_id) / ".git"
        ).exists()
        tasks.append(item)
    return ApiResponse(success=True, data={"tasks": tasks, "count": len(tasks)})


@router.post("/catalog/tasks/{instance_id}/prepare", response_model=ApiResponse)
async def prepare_benchmark_workspace(
    instance_id: str,
    _auth: str = Depends(verify_api_key),
) -> ApiResponse:
    try:
        workspace = await _benchmark_catalog.prepare_workspace(instance_id)
        return ApiResponse(
            success=True,
            data={"instance_id": instance_id, "project_root": str(workspace)},
        )
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown benchmark task: {instance_id}")
    except Exception as exc:
        logger.exception("Failed to prepare benchmark workspace")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/catalog/runs", response_model=ApiResponse)
async def benchmark_run_status(
    _auth: str = Depends(verify_api_key),
) -> ApiResponse:
    exporter = PredictionExporter(_benchmark_catalog, _product_state_store)
    return ApiResponse(success=True, data=exporter.statuses())


@router.post("/catalog/predictions/export", response_model=ApiResponse)
async def export_benchmark_predictions(
    _auth: str = Depends(verify_api_key),
) -> ApiResponse:
    destination = (
        _benchmark_catalog.repository_root / ".codeagent" / "benchmarks" / "predictions.jsonl"
    )
    exporter = PredictionExporter(_benchmark_catalog, _product_state_store)
    count = exporter.export(
        destination,
        get_env("LLM_MODEL", "deepseek/deepseek-v4-flash"),
    )
    return ApiResponse(
        success=True,
        data={"count": count, "path": str(destination.resolve())},
    )


@router.get("/projects", response_model=ApiResponse)
async def list_projects(_auth: str = Depends(verify_api_key)) -> ApiResponse:
    return ApiResponse(success=True, data={"projects": _workspace_manager.list_projects()})


@router.post("/projects", response_model=ApiResponse, status_code=status.HTTP_201_CREATED)
async def create_project(
    body: ProjectCreateRequest,
    _auth: str = Depends(verify_api_key),
) -> ApiResponse:
    try:
        return ApiResponse(success=True, data=_workspace_manager.create_project(body.name))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete("/projects/{project_id}", response_model=ApiResponse)
async def delete_project(
    project_id: str,
    _auth: str = Depends(verify_api_key),
) -> ApiResponse:
    try:
        deleted = _workspace_manager.delete_project(project_id)
        conversations_deleted = _product_state_store.delete_project_conversations(project_id)
        return ApiResponse(
            success=True,
            data={
                "deleted": deleted,
                "conversations_deleted": conversations_deleted,
            },
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Project not found") from exc


@router.get("/projects/{project_id}/files", response_model=ApiResponse)
async def list_project_files(
    project_id: str,
    _auth: str = Depends(verify_api_key),
) -> ApiResponse:
    try:
        return ApiResponse(success=True, data={"files": _workspace_manager.list_files(project_id)})
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Project not found") from exc


@router.post("/projects/{project_id}/files", response_model=ApiResponse)
async def upload_project_files(
    project_id: str,
    files: list[UploadFile] = File(...),
    _auth: str = Depends(verify_api_key),
) -> ApiResponse:
    uploaded_files: list[dict[str, Any]] = []
    try:
        for uploaded in files:
            filename = Path(uploaded.filename or "").name
            if not filename or filename == ".codeagent-workspace.json":
                raise HTTPException(status_code=400, detail="Invalid upload filename")
            content = await uploaded.read(_MAX_UPLOAD_BYTES + 1)
            if len(content) > _MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail=f"File exceeds 25 MB: {filename}")
            target = _workspace_manager.resolve_project_file(
                project_id,
                filename,
                must_exist=False,
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            uploaded_files.append({"name": filename, "path": filename, "size": len(content)})
        return ApiResponse(success=True, data={"files": uploaded_files})
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Project not found") from exc
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        for uploaded in files:
            await uploaded.close()


@router.get("/projects/{project_id}/download/{file_path:path}")
async def download_project_file(
    project_id: str,
    file_path: str,
    _auth: str = Depends(verify_api_key),
) -> FileResponse:
    try:
        target = _workspace_manager.resolve_project_file(project_id, file_path)
        return FileResponse(target, filename=target.name)
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail="File not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete("/projects/{project_id}/files/{file_path:path}", response_model=ApiResponse)
async def delete_project_file(
    project_id: str,
    file_path: str,
    _auth: str = Depends(verify_api_key),
) -> ApiResponse:
    try:
        target = _workspace_manager.resolve_project_file(project_id, file_path)
        target.unlink()
        return ApiResponse(success=True, data={"deleted": True, "path": file_path})
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail="File not found") from exc
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/conversations", response_model=ApiResponse)
async def list_conversations(
    limit: int = 200,
    _auth: str = Depends(verify_api_key),
) -> ApiResponse:
    return ApiResponse(
        success=True,
        data={"conversations": _product_state_store.list_conversations(limit)},
    )


@router.post("/conversations/import", response_model=ApiResponse)
async def import_conversations(
    body: ConversationImportRequest,
    _auth: str = Depends(verify_api_key),
) -> ApiResponse:
    imported = _product_state_store.import_conversations(body.conversations)
    return ApiResponse(success=True, data={"imported": imported})


@router.delete("/conversations/{conversation_id}", response_model=ApiResponse)
async def delete_conversation(
    conversation_id: str,
    body: ConversationDeleteRequest,
    _auth: str = Depends(verify_api_key),
) -> ApiResponse:
    """Delete an ephemeral conversation workspace; project-owned files are retained."""
    stored = _product_state_store.get_conversation(conversation_id)
    workspace_root = stored.get("workspace_root") if stored else body.workspace_root
    if not workspace_root:
        history_deleted = _product_state_store.delete_conversation(conversation_id)
        return ApiResponse(
            success=True,
            data={
                "workspace_deleted": False,
                "history_deleted": history_deleted,
                "reason": "no_workspace",
            },
        )
    resolved = Path(workspace_root).expanduser().resolve()
    try:
        resolved.relative_to(_workspace_manager.workspace_root.resolve())
    except ValueError:
        try:
            resolved.relative_to(_workspace_manager.project_root.resolve())
            reason = "project_retained"
        except ValueError:
            reason = "external_workspace_retained"
        history_deleted = _product_state_store.delete_conversation(conversation_id)
        return ApiResponse(
            success=True,
            data={
                "workspace_deleted": False,
                "history_deleted": history_deleted,
                "reason": reason,
            },
        )
    try:
        deleted = _workspace_manager.delete_session(str(resolved), conversation_id)
        history_deleted = _product_state_store.delete_conversation(conversation_id)
        return ApiResponse(
            success=True,
            data={
                "workspace_deleted": deleted,
                "history_deleted": history_deleted,
                "reason": "deleted" if deleted else "already_absent",
            },
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/system/capabilities", response_model=ApiResponse)
async def get_system_capabilities(
    request: Request,
    project_root: str | None = None,
    query: str = "",
    _auth: str = Depends(verify_api_key),
) -> ApiResponse:
    product_root = Path.cwd().resolve()
    workspace = Path(project_root).expanduser().resolve() if project_root else product_root
    product_skills = discover_skills(product_root, source="product")
    resolution = resolve_project_instructions(
        workspace,
        query,
        product_root=product_root,
        skills_enabled=get_skills_enabled(),
    )
    memory_root = workspace / ".codeagent" / "memory"
    transcript_root = workspace / ".codeagent" / "transcripts"
    tool_gateway = ToolGateway(project_root=str(workspace))
    try:
        tools = sorted(tool.name for tool in tool_gateway.list_tools())
    finally:
        await tool_gateway.aclose()
    mcp_resolution = resolve_mcp_config(workspace)
    mcp_enabled = get_mcp_enabled()
    selected_mcp = (
        select_mcp_server_names(query, set(mcp_resolution.servers))
        if query and mcp_enabled
        else set()
    )
    mcp_servers = [
        {
            **server.public_metadata(),
            "available": False,
            "deferred": server.name not in selected_mcp,
            "matched": server.name in selected_mcp,
            "tool_count": 0,
        }
        for server in mcp_resolution.servers.values()
    ]
    mcp_status = {
        "enabled": mcp_enabled,
        "configured": bool(mcp_resolution.servers or mcp_resolution.config_paths),
        "available": False,
        "servers": mcp_servers,
        "tools": [],
        "blocked_tools": [],
        "warnings": list(mcp_resolution.warnings),
        "detail": (
            "disabled by MCP_ENABLED=false"
            if not mcp_enabled
            else f"{len(selected_mcp)} server(s) match the current query; connection deferred until task execution"
            if selected_mcp
            else f"{len(mcp_servers)} configured server(s); connection deferred until needed"
            if mcp_servers
            else "No MCP configuration resolved for this workspace"
        ),
    }
    model_resilience = ModelRouter.from_env().snapshot()
    return ApiResponse(
        success=True,
        data={
            "inline_runner": bool(getattr(request.app.state, "inline_mode", False)),
            "llm_configured": bool(os.environ.get("LLM_API_KEY")),
            "model": os.environ.get("LLM_MODEL", "deepseek/deepseek-v4-flash"),
            "model_resilience": model_resilience,
            "context": context_capability_report(workspace),
            "sandbox_enabled": get_sandbox_enabled(),
            "memory_enabled": get_memory_enabled(),
            "vector_memory_enabled": get_memory_use_vector(),
            "memory": {
                "enabled": get_memory_enabled(),
                "stored_count": len(list(memory_root.glob("**/*.md")))
                if memory_root.exists()
                else 0,
                "transcript_count": len(list(transcript_root.glob("*.json")))
                if transcript_root.exists()
                else 0,
                "detail": (
                    "disabled by MEMORY_ENABLED=false"
                    if not get_memory_enabled()
                    else "lexical recall active"
                    if not get_memory_use_vector()
                    else "vector and lexical recall active"
                ),
            },
            "reflection_enabled": get_evolution_enabled(),
            "tools": tools,
            "skills": {
                "configured": bool(product_skills) and get_skills_enabled(),
                "count": len(product_skills),
                "resolved_count": len(resolution.skills),
                "active": [skill.public_metadata() for skill in resolution.skills],
                "allowed_tools": resolution.allowed_tools,
                "warnings": resolution.warnings,
                "detail": (
                    "disabled by SKILLS_ENABLED=false"
                    if not get_skills_enabled()
                    else f"{len(resolution.skills)} matched for current query"
                    if query
                    else "Skills are matched per request"
                ),
            },
            "mcp": mcp_status,
            "benchmark_tasks": len(_benchmark_catalog.list_tasks()),
        },
    )


@router.post("/tasks", response_model=ApiResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_task(
    req: TaskCreateRequest,
    request: Request,
    gateway: OrchestrationGatewayImpl = Depends(_get_gateway),
    _auth: str = Depends(verify_api_key),
    _rl: None = Depends(_rate_limiter.check_rate_limit),
) -> ApiResponse | JSONResponse:
    """提交新 Agent 任务。

    支持两种模式：
    - Celery 模式：通过 gateway.start_task 提交到 Celery 队列
    - Inline 模式：直接在当前 asyncio 事件循环中启动后台任务
    """
    try:
        inline_runner = getattr(request.app.state, "inline_runner", None)
        if inline_runner is not None:
            if not os.environ.get("LLM_API_KEY"):
                raise HTTPException(
                    status_code=503,
                    detail="LLM_API_KEY is not configured. Update .env and restart the Harness.",
                )
        conversation_id = req.conversation_id or uuid.uuid4().hex
        conversation_history = _conversation_history_for_run(
            conversation_id,
            req.conversation_history,
            req.query,
        )
        project_path = (
            Path(req.project_root).expanduser().resolve()
            if req.project_root
            else _workspace_manager.create(req.query, conversation_id)
        )
        benchmark_task = None
        if req.benchmark_instance_id:
            try:
                benchmark_task = _benchmark_catalog.get_task(req.benchmark_instance_id)
                valid_workspace, reason = _benchmark_catalog.validate_workspace(
                    req.benchmark_instance_id,
                    project_path,
                    require_clean=not bool(req.recovered_from_task_id),
                )
            except KeyError as exc:
                raise HTTPException(status_code=400, detail="Unknown benchmark instance") from exc
            if not valid_workspace:
                raise HTTPException(
                    status_code=400,
                    detail=f"Benchmark workspace is not ready: {reason}",
                )
        if req.recovered_from_task_id:
            source_run = _product_state_store.get_run(req.recovered_from_task_id)
            if source_run is None:
                raise HTTPException(status_code=400, detail="Recovery source task was not found")
            if source_run["state"] not in {"failed", "cancelled"}:
                raise HTTPException(
                    status_code=409,
                    detail="Recovery source must be failed or cancelled",
                )
            if Path(str(source_run["workspace_root"])).resolve() != project_path:
                raise HTTPException(
                    status_code=400,
                    detail="Recovery must reuse the source task workspace",
                )
        if inline_runner is not None:
            if not project_path.is_dir():
                raise HTTPException(
                    status_code=400,
                    detail=f"Project root does not exist or is not a directory: {project_path}",
                )
        user_req = UserRequest(
            query=req.query,
            project_root=str(project_path),
            auto_mode=req.auto_mode,
            max_retries=req.max_retries,
            conversation_history=conversation_history,
            response_mode=req.response_mode,
            direct_execution=req.direct_execution,
            conversation_id=conversation_id,
            benchmark_instance_id=req.benchmark_instance_id,
            benchmark_fail_to_pass=(benchmark_task.fail_to_pass if benchmark_task else []),
            benchmark_pass_to_pass=(benchmark_task.pass_to_pass if benchmark_task else []),
            recovered_from_task_id=req.recovered_from_task_id,
        )
        task_id = await gateway.start_task(user_req, skip_celery=inline_runner is not None)
        metadata = _workspace_manager.metadata_for(project_path) or {}
        _product_state_store.record_run(
            task_id,
            conversation_id=conversation_id,
            query=req.query,
            workspace_root=str(project_path),
            project_id=metadata.get("project_id"),
            benchmark_instance_id=req.benchmark_instance_id,
            recovered_from_task_id=req.recovered_from_task_id,
        )

        # Inline 模式：直接在后台协程中运行 Agent
        if inline_runner is not None:
            redis_url = getattr(request.app.state, "redis_url", None)
            inline_task = asyncio.create_task(
                inline_runner(
                    task_id,
                    {
                        "query": req.query,
                        "project_root": str(project_path),
                        "auto_mode": req.auto_mode,
                        "max_retries": req.max_retries,
                        "conversation_history": conversation_history,
                        "response_mode": req.response_mode,
                        "direct_execution": req.direct_execution,
                        "conversation_id": conversation_id,
                        "benchmark_instance_id": req.benchmark_instance_id,
                        "benchmark_fail_to_pass": (
                            benchmark_task.fail_to_pass if benchmark_task else []
                        ),
                        "benchmark_pass_to_pass": (
                            benchmark_task.pass_to_pass if benchmark_task else []
                        ),
                        "recovered_from_task_id": req.recovered_from_task_id,
                    },
                    redis_url,
                )
            )
            inline_tasks = cast(
                dict[str, asyncio.Task[Any]],
                getattr(request.app.state, "inline_tasks", None),
            )
            if inline_tasks is None:
                inline_tasks = {}
                request.app.state.inline_tasks = inline_tasks
            inline_tasks[task_id] = inline_task

            def forget_inline_task(_completed: asyncio.Task[Any]) -> None:
                inline_tasks.pop(task_id, None)

            inline_task.add_done_callback(forget_inline_task)

        return ApiResponse(
            success=True,
            data={
                "task_id": task_id,
                "status": "pending",
                "project_root": str(project_path),
                "conversation_id": conversation_id,
            },
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to create task")
        return JSONResponse(
            status_code=500,
            content=ApiResponse(success=False, error=str(exc)).model_dump(),
        )


@router.get("/tasks/{task_id}", response_model=ApiResponse)
async def get_task_status(
    task_id: str,
    gateway: OrchestrationGatewayImpl = Depends(_get_gateway),
    _auth: str = Depends(verify_api_key),
    _rl: None = Depends(_rate_limiter.check_rate_limit),
) -> ApiResponse | JSONResponse:
    """查询任务状态。"""
    try:
        status = await gateway.get_task_status(task_id)
        resp = TaskStatusResponse(
            task_id=status.task_id,
            state=status.state.value,
            progress=status.progress,
            current_step=status.current_step,
            errors=status.errors,
        )
        return ApiResponse(success=True, data=resp.model_dump())
    except KeyError:
        durable = _product_state_store.get_run(task_id)
        if durable is not None:
            durable_state = str(durable["state"])
            durable_report = durable.get("report") or {}
            resp = TaskStatusResponse(
                task_id=task_id,
                state=durable_state,
                progress=(1.0 if durable_state in {"completed", "failed", "cancelled"} else 0.0),
                current_step=None,
                errors=([str(durable_report["error"])] if durable_report.get("error") else []),
            )
            return ApiResponse(success=True, data=resp.model_dump())
        return JSONResponse(
            status_code=404,
            content=ApiResponse(success=False, error=f"Task not found: {task_id}").model_dump(),
        )
    except Exception as exc:
        logger.exception("Failed to get task status")
        return JSONResponse(
            status_code=500,
            content=ApiResponse(success=False, error=str(exc)).model_dump(),
        )


@router.post("/tasks/{task_id}/decision", response_model=ApiResponse)
async def submit_decision(
    task_id: str,
    body: DecisionRequest,
    request: Request,
    gateway: OrchestrationGatewayImpl = Depends(_get_gateway),
    _auth: str = Depends(verify_api_key),
    _rl: None = Depends(_rate_limiter.check_rate_limit),
) -> ApiResponse | JSONResponse:
    """提交 Human Review 决策。"""
    try:
        decision = HumanDecision(
            task_id=task_id,
            decision=body.decision,
            modifications=body.modifications,
        )
        await gateway.submit_decision(task_id, decision)
        inline_tasks = getattr(request.app.state, "inline_tasks", {})
        active = inline_tasks.get(task_id)
        resume_runner = getattr(request.app.state, "resume_inline_checkpoint", None)
        run = _product_state_store.get_run(task_id)
        resume_started = False
        if (
            resume_runner is not None
            and (active is None or active.done())
            and run is not None
            and run.get("state") in {"pending", "running"}
        ):
            resumed = asyncio.create_task(
                resume_runner(
                    task_id,
                    body.decision,
                    getattr(request.app.state, "redis_url", None),
                )
            )
            inline_tasks[task_id] = resumed
            resumed.add_done_callback(lambda _done: inline_tasks.pop(task_id, None))
            resume_started = True
        return ApiResponse(
            success=True,
            data={
                "task_id": task_id,
                "decision": body.decision,
                "checkpoint_resume_started": resume_started,
            },
        )
    except Exception as exc:
        logger.exception("Failed to submit decision")
        return JSONResponse(
            status_code=500,
            content=ApiResponse(success=False, error=str(exc)).model_dump(),
        )


@router.delete("/tasks/{task_id}", response_model=ApiResponse)
async def cancel_task(
    task_id: str,
    request: Request,
    gateway: OrchestrationGatewayImpl = Depends(_get_gateway),
    _auth: str = Depends(verify_api_key),
    _rl: None = Depends(_rate_limiter.check_rate_limit),
) -> ApiResponse | JSONResponse:
    """取消任务。"""
    try:
        result = await gateway.cancel_task(task_id)
        inline_tasks = getattr(request.app.state, "inline_tasks", {})
        inline_task = inline_tasks.get(task_id)
        if inline_task is not None and not inline_task.done():
            inline_task.cancel()
        return ApiResponse(success=True, data={"cancelled": result})
    except Exception as exc:
        logger.exception("Failed to cancel task")
        return JSONResponse(
            status_code=500,
            content=ApiResponse(success=False, error=str(exc)).model_dump(),
        )


@router.post("/tasks/{task_id}/steer", response_model=ApiResponse)
async def steer_task(
    task_id: str,
    body: SteeringRequest,
    gateway: OrchestrationGatewayImpl = Depends(_get_gateway),
    _auth: str = Depends(verify_api_key),
    _rl: None = Depends(_rate_limiter.check_rate_limit),
) -> ApiResponse | JSONResponse:
    """Queue an instruction for the next safe Agent execution boundary."""
    try:
        queued = await gateway.steer_task(task_id, body.instruction)
        return ApiResponse(success=True, data={"queued": queued, "task_id": task_id})
    except ValueError as exc:
        return JSONResponse(
            status_code=409,
            content=ApiResponse(success=False, error=str(exc)).model_dump(),
        )
    except Exception as exc:
        logger.exception("Failed to steer task")
        return JSONResponse(
            status_code=500,
            content=ApiResponse(success=False, error=str(exc)).model_dump(),
        )


@router.post(
    "/tasks/{task_id}/recover",
    response_model=ApiResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def recover_task(
    task_id: str,
    body: RecoveryRequest,
    request: Request,
    gateway: OrchestrationGatewayImpl = Depends(_get_gateway),
    _auth: str = Depends(verify_api_key),
    _rl: None = Depends(_rate_limiter.check_rate_limit),
) -> ApiResponse | JSONResponse:
    """Start a new auditable run from a failed/cancelled run's workspace."""
    source = _product_state_store.get_run(task_id)
    if source is None:
        return JSONResponse(
            status_code=404,
            content=ApiResponse(success=False, error=f"Task not found: {task_id}").model_dump(),
        )
    if source["state"] not in {"failed", "cancelled"}:
        return JSONResponse(
            status_code=409,
            content=ApiResponse(
                success=False,
                error="Only failed or cancelled tasks can be recovered",
            ).model_dump(),
        )
    workspace = Path(str(source["workspace_root"])).resolve()
    if not workspace.is_dir():
        return JSONResponse(
            status_code=409,
            content=ApiResponse(
                success=False,
                error="The source workspace is no longer available",
            ).model_dump(),
        )

    conversation_id = str(source.get("conversation_id") or uuid.uuid4().hex)
    conversation = _product_state_store.get_conversation(conversation_id)
    history = [
        {"role": str(item["role"]), "content": str(item["content"])}
        for item in (conversation or {}).get("messages", [])[-20:]
        if item.get("role") in {"user", "assistant"}
    ]
    recovery_query = (
        f"Recover task {task_id} from its existing workspace.\n"
        f"Original goal: {source['query']}\n"
        f"Recovery instruction: {body.instruction.strip()}\n"
        "Treat the current workspace as the source of truth: inspect partial changes before editing, "
        "avoid repeating completed work, then run relevant validation."
    )
    result = await create_task(
        TaskCreateRequest(
            query=recovery_query,
            project_root=str(workspace),
            auto_mode=body.auto_mode,
            max_retries=body.max_retries,
            conversation_history=history,
            response_mode="execute",
            conversation_id=conversation_id,
            benchmark_instance_id=source.get("benchmark_instance_id"),
            recovered_from_task_id=task_id,
        ),
        request,
        gateway,
        _auth,
        _rl,
    )
    if isinstance(result, ApiResponse) and isinstance(result.data, dict):
        result.data["recovered_from_task_id"] = task_id
    return result


@router.get("/tasks/{task_id}/report", response_model=ApiResponse)
async def get_report(
    task_id: str,
    gateway: OrchestrationGatewayImpl = Depends(_get_gateway),
    _auth: str = Depends(verify_api_key),
    _rl: None = Depends(_rate_limiter.check_rate_limit),
) -> ApiResponse | JSONResponse:
    """获取任务完整报告。"""
    try:
        report = await gateway.get_report(task_id)
        resp = TaskReportResponse(
            task_id=report.task_id,
            status=report.status,
            plan=report.plan,
            changes=report.changes,
            validation_results=report.validation_results,
            duration=report.duration,
            token_usage=report.token_usage,
            assistant_response=report.assistant_response,
            response_mode=report.response_mode,
            run_profile=report.run_profile,
            tool_manifest=report.tool_manifest,
            context_manifest=report.context_manifest,
            steering_instructions=report.steering_instructions,
            memory_hits=report.memory_hits,
            resolved_skills=report.resolved_skills,
            warnings=report.warnings,
            reflection=report.reflection,
            transcript_path=report.transcript_path,
            mcp_servers=report.mcp_servers,
            model_runtime=report.model_runtime,
            infrastructure_runtime=report.infrastructure_runtime,
            benchmark_metrics=report.benchmark_metrics,
            benchmark_instance_id=report.benchmark_instance_id,
            artifacts=_report_artifacts(task_id),
            error=report.error,
            recovered_from_task_id=report.recovered_from_task_id,
        )
        return ApiResponse(success=True, data=resp.model_dump())
    except KeyError:
        durable = _product_state_store.get_run(task_id)
        if durable is not None and durable.get("report") is not None:
            durable_report = dict(durable["report"])
            durable_report["artifacts"] = _report_artifacts(task_id)
            resp = TaskReportResponse.model_validate(durable_report)
            return ApiResponse(success=True, data=resp.model_dump())
        return JSONResponse(
            status_code=404,
            content=ApiResponse(
                success=False, error=f"Report not found for task: {task_id}"
            ).model_dump(),
        )
    except Exception as exc:
        logger.exception("Failed to get report")
        return JSONResponse(
            status_code=500,
            content=ApiResponse(success=False, error=str(exc)).model_dump(),
        )


@router.get("/tasks/{task_id}/artifacts/{artifact_id}")
async def download_task_artifact(
    task_id: str,
    artifact_id: str,
    _auth: str = Depends(verify_api_key),
) -> FileResponse:
    try:
        artifact = _product_state_store.resolve_artifact(task_id, artifact_id)
        return FileResponse(artifact, filename=artifact.name)
    except (KeyError, FileNotFoundError):
        raise HTTPException(status_code=404, detail="Artifact not found")
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/tasks/{task_id}/files", response_model=ApiResponse)
async def list_task_workspace_files(
    task_id: str,
    _auth: str = Depends(verify_api_key),
) -> ApiResponse:
    """List files the agent produced inside the task's authorized workspace."""
    workspace = _preview_workspace(task_id)
    files = _list_task_files(workspace)
    return ApiResponse(success=True, data={"files": files, "count": len(files)})


@router.get("/tasks/{task_id}/preview", response_model=ApiResponse)
async def get_task_preview(
    task_id: str,
    _auth: str = Depends(verify_api_key),
) -> ApiResponse:
    """Discover real HTML entrypoints inside the task's authorized workspace."""
    workspace = _preview_workspace(task_id)
    preview_files = _preview_files(workspace)
    html_entries = sorted(
        path.relative_to(workspace).as_posix()
        for path in preview_files
        if path.suffix.lower() in {".html", ".htm"}
    )
    html_entries.sort(
        key=lambda value: (
            value.lower() != "index.html",
            Path(value).name.lower() != "index.html",
            value,
        )
    )
    revision = max((path.stat().st_mtime_ns for path in preview_files), default=0)
    return ApiResponse(
        success=True,
        data={
            "available": bool(html_entries),
            "entrypoint": html_entries[0] if html_entries else None,
            "entries": html_entries[:100],
            "revision": str(revision),
        },
    )


@router.get("/tasks/{task_id}/files/{file_path:path}")
async def get_task_workspace_file(
    task_id: str,
    file_path: str,
    _auth: str = Depends(verify_api_key),
) -> FileResponse:
    """Serve a workspace file produced during the task for review or download.

    Unlike the sandboxed preview route, this endpoint has no extension allowlist so
    any non-hidden file the agent wrote (Python, markdown, images, ...) can be opened
    in the browser or downloaded from the frontend report.
    """
    workspace = _preview_workspace(task_id)
    target = _resolve_workspace_file(workspace, file_path)
    media_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
    headers = {
        "Cache-Control": "no-store",
        "X-Content-Type-Options": "nosniff",
    }
    return FileResponse(target, media_type=media_type, headers=headers)


@router.get("/tasks/{task_id}/preview/{file_path:path}")
async def serve_task_preview_file(
    task_id: str,
    file_path: str,
    _auth: str = Depends(verify_api_key),
) -> FileResponse:
    """Serve a sandboxed-preview asset without exposing arbitrary workspace files."""
    workspace = _preview_workspace(task_id)
    target = _resolve_preview_file(workspace, file_path)
    media_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
    headers = {
        "Cache-Control": "no-store",
        "X-Content-Type-Options": "nosniff",
    }
    if target.suffix.lower() in {".html", ".htm"}:
        headers["Content-Security-Policy"] = (
            "default-src 'self' data: blob: http: https:; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval' blob: http: https:; "
            "style-src 'self' 'unsafe-inline' http: https:; "
            "connect-src 'self' data: blob: http: https: ws: wss:; "
            "img-src 'self' data: blob: http: https:; frame-ancestors 'self'"
        )
    elif target.suffix.lower() == ".svg":
        headers["Content-Security-Policy"] = (
            "default-src 'none'; style-src 'unsafe-inline'; img-src data:; "
            "script-src 'none'; sandbox"
        )
    return FileResponse(target, media_type=media_type, headers=headers)
