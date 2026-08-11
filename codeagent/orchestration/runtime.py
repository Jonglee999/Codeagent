"""Runtime assembly for memory and reflection services."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from codeagent.config import (
    get_benchmark_discovery_warning_ratio,
    get_benchmark_invalid_tool_warning_ratio,
    get_benchmark_token_warning_ratio,
    get_evolution_enabled,
    get_max_tokens_per_task,
    get_memory_enabled,
)
from codeagent.context_engine.evolution import (
    SelfEvolutionManager,
    StrategyApplier,
    StrategyExtractor,
    StrategyStore,
    TrajectoryRecorder,
)
from codeagent.memory.manager import MemoryManager


def to_serializable(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return {
            field: to_serializable(getattr(value, field)) for field in value.__dataclass_fields__
        }
    if isinstance(value, (list, tuple)):
        return [to_serializable(item) for item in value]
    if isinstance(value, dict):
        return {key: to_serializable(item) for key, item in value.items()}
    return value


def execution_response(final_state: Any, success: bool, errors: list[str]) -> str:
    for entry in reversed(getattr(final_state, "execution_log", []) or []):
        if entry.get("type") == "llm_response" and entry.get("content"):
            content = str(entry["content"]).strip()
            if content:
                return content[:6000]
    changes = getattr(final_state, "accumulated_changes", []) or []
    validations = getattr(final_state, "validation_results", []) or []
    if success:
        parts = ["任务已完成。"]
        if changes:
            paths = sorted(
                {
                    str(change.get("file_path") or change.get("path") or "")
                    for change in changes
                    if isinstance(change, dict)
                }
                - {""}
            )
            if paths:
                parts.append(f"变更文件：{', '.join(paths)}。")
        if validations:
            layers = []
            for item in validations:
                label = str(getattr(item, "layer", "") or getattr(item, "type", "") or "验证")
                passed = bool(getattr(item, "passed", False))
                layers.append(f"{label}{'通过' if passed else '未通过'}")
            parts.append(f"验证：{', '.join(layers)}。")
        return "\n".join(parts)
    detail = "; ".join(str(error) for error in errors[:3]) if errors else "执行或验证未通过"
    return f"任务未能完成：{detail}"


def build_execution_report(
    task_id: str,
    final_state: Any,
    *,
    duration: float,
    success: bool,
    errors: list[str],
    model_runtime: dict[str, Any],
    infrastructure_runtime: dict[str, Any],
    extension_warnings: list[str] | None = None,
    mcp_servers: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Canonical report shape shared by inline and distributed execution."""
    provider_diagnostics = [
        to_serializable(entry)
        for entry in (getattr(final_state, "execution_log", []) or [])
        if isinstance(entry, dict) and entry.get("type") == "provider_response_diagnostic"
    ][-5:]
    benchmark_metrics, benchmark_warnings = _build_benchmark_metrics(
        final_state, model_runtime, errors
    )
    return {
        "task_id": task_id,
        "status": "completed" if success else "failed",
        "benchmark_instance_id": getattr(final_state, "benchmark_instance_id", None),
        "recovered_from_task_id": getattr(final_state, "recovered_from_task_id", None),
        "plan": to_serializable(getattr(final_state, "plan", []) or []),
        "changes": to_serializable(getattr(final_state, "accumulated_changes", []) or []),
        "validation_results": to_serializable(getattr(final_state, "validation_results", []) or []),
        "duration": duration,
        "token_usage": int(getattr(final_state, "estimated_tokens", 0) or 0),
        "assistant_response": execution_response(final_state, success, errors),
        "response_mode": "execute",
        "run_profile": to_serializable(getattr(final_state, "run_profile", {}) or {}),
        "tool_manifest": to_serializable(getattr(final_state, "tool_manifest", {}) or {}),
        "context_manifest": to_serializable(getattr(final_state, "context_manifest", {}) or {}),
        "steering_instructions": to_serializable(
            getattr(final_state, "steering_instructions", []) or []
        ),
        "memory_hits": to_serializable(getattr(final_state, "memory_hits", []) or []),
        "resolved_skills": to_serializable(getattr(final_state, "resolved_skills", []) or []),
        "warnings": [
            *(getattr(final_state, "warnings", []) or []),
            *(extension_warnings or []),
            *benchmark_warnings,
        ],
        "reflection": to_serializable(getattr(final_state, "reflection", None)),
        "transcript_path": getattr(final_state, "transcript_path", None),
        "mcp_servers": mcp_servers or [],
        "model_runtime": model_runtime,
        "infrastructure_runtime": infrastructure_runtime,
        "provider_response_diagnostics": provider_diagnostics,
        "benchmark_metrics": benchmark_metrics,
        "error": errors[0] if errors else None,
    }


def _build_benchmark_metrics(
    final_state: Any,
    model_runtime: dict[str, Any],
    errors: list[str],
) -> tuple[dict[str, Any], list[str]]:
    """Derive no-cost, private-reasoning-free benchmark efficiency metrics."""

    if not getattr(final_state, "benchmark_instance_id", None):
        return {}, []
    log = [
        item
        for item in (getattr(final_state, "execution_log", []) or [])
        if isinstance(item, dict)
    ]
    tool_calls = [item for item in log if item.get("type") == "tool_call"]
    by_name: dict[str, int] = {}
    for item in tool_calls:
        name = str(item.get("tool_name") or "unknown")
        by_name[name] = by_name.get(name, 0) + 1
    invalid_calls = sum(
        str(item.get("error_code") or "").upper()
        in {"INVALID_PARAMS", "VALIDATION_ERROR", "CAPABILITY_NOT_EXPOSED"}
        for item in tool_calls
    )
    capability_violations = sum(
        str(item.get("error_code") or "").upper() == "CAPABILITY_NOT_EXPOSED"
        for item in tool_calls
    )
    executed_tool_calls = sum(
        str(item.get("error_code") or "").upper() != "CAPABILITY_NOT_EXPOSED"
        for item in tool_calls
    )
    duplicate_failures = sum(
        item.get("type") == "duplicate_tool_failure_breaker" for item in log
    )
    discovery_calls = sum(
        item.get("tool_name")
        in {"read_file", "list_files", "search_code", "get_diagnostics"}
        for item in tool_calls
    )
    mutations = [
        item
        for item in tool_calls
        if item.get("tool_name") in {"write_file", "apply_patch", "delete_file"}
    ]
    token_usage = int(getattr(final_state, "estimated_tokens", 0) or 0)
    total_tool_calls = len(tool_calls)
    discovery_ratio = discovery_calls / total_tool_calls if total_tool_calls else 0.0
    invalid_ratio = invalid_calls / total_tool_calls if total_tool_calls else 0.0
    token_ratio = token_usage / max(1, get_max_tokens_per_task())
    warnings: list[str] = []
    if discovery_ratio > get_benchmark_discovery_warning_ratio():
        warnings.append(
            f"Benchmark discovery ratio is high ({discovery_ratio:.0%})."
        )
    if invalid_ratio > get_benchmark_invalid_tool_warning_ratio():
        warnings.append(
            f"Benchmark invalid-tool ratio is high ({invalid_ratio:.0%})."
        )
    if token_ratio > get_benchmark_token_warning_ratio():
        warnings.append(f"Benchmark token budget usage is high ({token_ratio:.0%}).")
    stop_reason = (
        "token_budget_exhausted"
        if getattr(final_state, "review_type", None) == "token_budget_exhausted"
        else "duplicate_tool_failure"
        if duplicate_failures
        else "validation_degraded"
        if getattr(final_state, "validation_state", "") == "validation_degraded"
        else "error"
        if errors
        else "completed"
    )
    return {
        "provider_calls": int(getattr(final_state, "llm_call_count", 0) or 0),
        "provider_attempts": int(model_runtime.get("attempt_count", 0) or 0),
        "prompt_tokens": int(model_runtime.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(model_runtime.get("completion_tokens", 0) or 0),
        "total_tokens": token_usage,
        "tool_calls": total_tool_calls,
        "executed_tool_calls": executed_tool_calls,
        "tool_calls_by_name": by_name,
        "invalid_tool_calls": invalid_calls,
        "capability_violations": capability_violations,
        "duplicate_failures": duplicate_failures,
        "context_reductions": int(
            model_runtime.get("context_reduction_count", 0) or 0
        ),
        "discovery_ratio": round(discovery_ratio, 4),
        "invalid_tool_ratio": round(invalid_ratio, 4),
        "mutation_latency_ms": round(
            sum(float(item.get("duration_ms", 0) or 0) for item in mutations), 2
        ),
        "validation_state": getattr(final_state, "validation_state", "not_run"),
        "patch_produced": bool(getattr(final_state, "accumulated_changes", []) or []),
        "stop_reason": stop_reason,
        "memory_mode": getattr(final_state, "memory_mode", "off"),
        "learning_mode": getattr(final_state, "learning_mode", "off"),
    }, warnings


def _run_git(root: Path, arguments: list[str]) -> str:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return f"# git artifact unavailable: {type(exc).__name__}: {exc}\n"
    if result.returncode not in {0, 1}:
        return f"# git artifact unavailable (exit {result.returncode})\n{result.stderr}"
    return result.stdout


def _git_repository_root(root: Path) -> Path | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    return Path(result.stdout.strip()).resolve()


def _standalone_workspace_files(root: Path) -> list[str]:
    excluded_parts = {".codeagent", ".git", ".pytest_cache", "__pycache__"}
    files: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if relative.name == ".codeagent-workspace.json":
            continue
        if any(part in excluded_parts for part in relative.parts):
            continue
        files.append(relative.as_posix())
    return sorted(files)


def collect_git_patch(project_root: str) -> str:
    """Collect the complete tracked and untracked working-tree patch."""
    root = Path(project_root).resolve()
    repository_root = _git_repository_root(root)
    if repository_root == root:
        tracked = _run_git(root, ["diff", "--binary", "--no-ext-diff", "HEAD", "--", "."])
        untracked_raw = _run_git(
            root,
            ["ls-files", "--others", "--exclude-standard", "-z", "--", "."],
        )
        if untracked_raw.startswith("# git artifact unavailable"):
            return tracked
        relative_files = [item for item in untracked_raw.split("\0") if item]
        patches = [tracked]
    else:
        # Session workspaces live below the CodeAgent repository. Treat them as
        # standalone roots so Git cannot include unrelated parent-repository files.
        relative_files = _standalone_workspace_files(root)
        patches = []
    for relative in relative_files:
        if relative.startswith(".codeagent/") or relative == ".codeagent-workspace.json":
            continue
        patches.append(
            _run_git(root, ["diff", "--no-index", "--binary", "--", "/dev/null", relative])
        )
    return "".join(patches)


def persist_run_artifacts(
    project_root: str,
    task_id: str,
    report: dict[str, Any],
) -> list[dict[str, Any]]:
    """Persist replayable report, validation output, and the complete Git patch."""
    safe_task_id = "".join(char for char in task_id if char.isalnum() or char in "-_")
    if not safe_task_id:
        raise ValueError("Invalid task id")
    root = Path(project_root).resolve() / ".codeagent" / "artifacts" / safe_task_id
    root.mkdir(parents=True, exist_ok=True)
    patch_path = root / "git.diff"
    validation_path = root / "validation.json"
    report_path = root / "report.json"
    patch_path.write_text(collect_git_patch(project_root), encoding="utf-8")
    validation_path.write_text(
        json.dumps(report.get("validation_results", []), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    artifacts: list[dict[str, Any]] = [
        {"kind": "git_diff", "path": str(patch_path), "size": patch_path.stat().st_size},
        {
            "kind": "validation_output",
            "path": str(validation_path),
            "size": validation_path.stat().st_size,
        },
        {"kind": "report", "path": str(report_path)},
    ]
    report["artifacts"] = artifacts
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    artifacts[-1]["size"] = report_path.stat().st_size
    return artifacts


def assess_task_completion(
    final_state: Any,
    *,
    require_patch: bool = False,
    require_test_evidence: bool = False,
) -> tuple[bool, list[str]]:
    """Require planned work and proportional validation evidence before success."""
    errors = list(getattr(final_state, "errors", []) or [])
    plan = list(getattr(final_state, "plan", []) or [])
    current_step = int(getattr(final_state, "current_step_index", 0) or 0)
    validations = list(getattr(final_state, "validation_results", []) or [])

    if plan and current_step < len(plan):
        errors.append(f"Task incomplete: executed {current_step} of {len(plan)} planned steps")
    if not validations:
        errors.append("Task incomplete: final validation did not run")
    elif not all(result.passed for result in validations):
        errors.append("Task validation failed")

    if require_patch:
        root = Path(str(getattr(final_state, "project_root", ""))).resolve()
        repository_root = _git_repository_root(root)
        if repository_root == root:
            patch = collect_git_patch(str(root)).strip()
            has_patch = bool(patch) and not patch.startswith("# git artifact unavailable")
        else:
            # Non-Git workspaces cannot distinguish baseline files from changes;
            # rely on mutation records instead of treating every file as new.
            has_patch = bool(getattr(final_state, "accumulated_changes", []) or [])
        if not has_patch:
            errors.append("Task incomplete: benchmark run produced no code patch")

    if require_test_evidence:
        runtime_result = validations[-1] if validations else None
        output = str(getattr(runtime_result, "output", "") or "").strip()
        if not output:
            errors.append(
                "Task incomplete: benchmark run has no executed test output"
            )

    return len(errors) == 0, errors


def build_memory_service(
    project_root: str,
    llm_client: Any,
    auto_mode: bool,
) -> MemoryManager | None:
    """Build only memory, for lightweight chat writeback paths."""
    if not get_memory_enabled():
        return None
    root = Path(project_root).resolve()
    return MemoryManager.create(
        project_path=str(root),
        llm_client=llm_client,
        auto_mode=auto_mode,
    )


def build_learning_services(
    project_root: str,
    llm_client: Any,
    auto_mode: bool,
) -> tuple[MemoryManager | None, SelfEvolutionManager | None, StrategyApplier | None]:
    """Build project-scoped memory and trajectory learning services.

    Vector retrieval remains configuration-controlled and every service degrades
    non-fatally inside its own gateway methods.
    """
    root = Path(project_root).resolve()
    memory = build_memory_service(project_root, llm_client, auto_mode)
    if not get_evolution_enabled():
        return memory, None, None

    trajectory_root = root / ".codeagent" / "trajectories"
    strategy_root = root / ".codeagent" / "strategies"
    recorder = TrajectoryRecorder(base_path=str(trajectory_root))
    store = StrategyStore(base_path=str(strategy_root))
    applier = StrategyApplier(store=store)
    extractor = StrategyExtractor(
        llm_client=llm_client,
        recorder=recorder,
        store=store,
        use_semantic_dedup=False,
    )
    evolution = SelfEvolutionManager(
        recorder=recorder,
        extractor=extractor,
        store=store,
        applier=applier,
    )
    return memory, evolution, applier
