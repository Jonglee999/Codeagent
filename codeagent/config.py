"""Configuration management — centralized env/flag access.

Centralizes environment variable reading (from .env and os.environ)
so modules don't need to call os.environ directly.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal


FeatureMode = Literal["off", "on", "auto"]
_CONTEXT_DEFAULT_BUDGET = 16000


def load_env_file(env_path: Path) -> None:
    """Load .env file into environment variables.

    Supports KEY=VALUE lines, comments (#), and quoted values.
    Does not override already-set environment variables.
    """
    if not env_path.exists():
        return
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("\"'")
            if key and key not in os.environ:
                os.environ[key] = value


def get_env(key: str, default: str = "") -> str:
    """Get an environment variable value with an optional default."""
    return os.environ.get(key, default)


def get_api_key() -> str:
    """Get the LLM API key from environment."""
    return get_env("LLM_API_KEY")


def get_model() -> str:
    """Get the LLM model name from environment."""
    return get_env("LLM_MODEL", "deepseek/deepseek-v4-flash")


def get_timeout() -> int:
    """Get the LLM timeout in seconds."""
    return int(get_env("LLM_TIMEOUT", "300"))


def get_max_retries() -> int:
    """Get the default max retry count."""
    return int(get_env("MAX_RETRIES", "3"))


def get_context_budget() -> int:
    """Get the context budget in tokens."""
    # CONTEXT_TARGET_INPUT_TOKENS is the canonical setting.  Keep the old
    # variable as a compatibility alias for existing deployments.
    value = get_env(
        "CONTEXT_TARGET_INPUT_TOKENS",
        get_env("CONTEXT_BUDGET_TOKENS", str(_CONTEXT_DEFAULT_BUDGET)),
    )
    return max(1, int(value))


def get_context_mode() -> FeatureMode:
    """Return the overall context mode (off/on/auto)."""
    return _feature_mode("CONTEXT_MODE", "auto")


def get_context_ast_mode() -> FeatureMode:
    """Return AST analysis mode, honoring the legacy boolean setting."""
    legacy = get_env("CONTEXT_ANALYSIS_ENABLED", "").strip()
    default: FeatureMode = (
        "on" if legacy.lower() == "true" else "off" if legacy.lower() == "false" else "auto"
    )
    return _feature_mode("CONTEXT_AST_MODE", default)


def get_context_semantic_mode() -> FeatureMode:
    """Return semantic indexing mode, honoring the legacy boolean setting."""
    legacy = get_env("CONTEXT_SEMANTIC_ENABLED", "").strip()
    default: FeatureMode = (
        "on" if legacy.lower() == "true" else "off" if legacy.lower() == "false" else "auto"
    )
    return _feature_mode("CONTEXT_SEMANTIC_MODE", default)


def get_context_index_background() -> bool:
    """Whether large semantic indexes may be built without blocking a run."""
    return get_env("CONTEXT_INDEX_BACKGROUND", "true").strip().lower() == "true"


def get_context_auto_file_limit() -> int:
    """Maximum source-file count that auto mode indexes synchronously."""
    return max(1, int(get_env("CONTEXT_AUTO_FILE_LIMIT", "300")))


def get_context_preinject_max_files() -> int:
    """Maximum related snippets pre-injected before model reasoning starts."""
    return max(1, min(20, int(get_env("CONTEXT_PREINJECT_MAX_FILES", "5"))))


def get_model_context_window() -> int:
    """Configured provider context window used for capability reporting."""
    return max(1, int(get_env("LLM_CONTEXT_WINDOW", "128000")))


def get_context_min_input_tokens() -> int:
    """Smallest adaptive repository-context profile."""
    return max(1, int(get_env("CONTEXT_MIN_INPUT_TOKENS", "16000")))


def get_context_max_input_tokens() -> int:
    """Hard upper bound for repository context before request overhead."""
    return max(
        get_context_min_input_tokens(),
        int(get_env("CONTEXT_MAX_INPUT_TOKENS", "80000")),
    )


def get_context_safety_margin_tokens() -> int:
    return max(0, int(get_env("CONTEXT_SAFETY_MARGIN_TOKENS", "8000")))


def get_context_tool_overhead_tokens() -> int:
    return max(0, int(get_env("CONTEXT_TOOL_OVERHEAD_TOKENS", "8000")))


def get_model_routing_enabled() -> bool:
    return get_env("MODEL_ROUTING_ENABLED", "true").strip().lower() == "true"


def get_model_routing_mode() -> Literal["off", "shadow", "on"]:
    """Gray-release mode; the legacy boolean remains the master kill switch."""
    if not get_model_routing_enabled():
        return "off"
    value = get_env("MODEL_ROUTING_MODE", "on").strip().lower()
    if value not in {"off", "shadow", "on"}:
        raise ValueError("MODEL_ROUTING_MODE must be one of: off, shadow, on")
    return value  # type: ignore[return-value]


def get_a2a_server_enabled() -> bool:
    """Whether the inbound A2A protocol surface is registered."""
    return get_env("A2A_SERVER_ENABLED", "false").strip().lower() == "true"


def get_a2a_client_enabled() -> bool:
    """Whether outbound A2A discovery and delegation are allowed."""
    return get_env("A2A_CLIENT_ENABLED", "false").strip().lower() == "true"


def get_a2a_base_url() -> str:
    return get_env("A2A_BASE_URL", "http://127.0.0.1:8000").rstrip("/")


def get_a2a_delegate_url() -> str:
    return get_env("A2A_DELEGATE_URL", "").rstrip("/")


def get_a2a_project_root() -> str:
    """Workspace exposed to inbound agents; remote callers cannot override it."""
    return get_env("A2A_PROJECT_ROOT", str(Path.cwd()))


def get_a2a_api_keys() -> set[str]:
    return {value.strip() for value in get_env("A2A_API_KEYS", "").split(",") if value.strip()}


def get_a2a_remote_api_key() -> str:
    return get_env("A2A_REMOTE_API_KEY", "")


def get_a2a_allowlist() -> set[str]:
    return {
        value.strip().lower()
        for value in get_env("A2A_ALLOWLIST", "127.0.0.1,localhost").split(",")
        if value.strip()
    }


def get_a2a_timeout() -> int:
    return max(1, int(get_env("A2A_TIMEOUT_SECONDS", "900")))


def get_a2a_max_message_bytes() -> int:
    return max(1024, int(get_env("A2A_MAX_MESSAGE_BYTES", "1048576")))


def get_a2a_max_artifact_bytes() -> int:
    return max(1024, int(get_env("A2A_MAX_ARTIFACT_BYTES", "4194304")))


def get_a2a_max_delegation_depth() -> int:
    return max(0, int(get_env("A2A_MAX_DELEGATION_DEPTH", "1")))


def get_a2a_task_db_path() -> Path:
    return Path(get_env("A2A_TASK_DB_PATH", ".codeagent/state/a2a_tasks.db"))


def get_checkpoint_enabled() -> bool:
    return get_env("CHECKPOINT_ENABLED", "true").strip().lower() == "true"


def get_checkpoint_ttl_days() -> int:
    return max(1, int(get_env("CHECKPOINT_TTL_DAYS", "30")))


def _feature_mode(key: str, default: FeatureMode) -> FeatureMode:
    value = get_env(key, default).strip().lower()
    if value not in {"off", "on", "auto"}:
        raise ValueError(f"{key} must be one of: off, on, auto")
    return value  # type: ignore[return-value]


# ── LLM 成本控制配置 ──────────────────────────────────────


def get_max_llm_calls_per_task() -> int:
    """单次任务最大 LLM 调用次数，默认 50。"""
    return int(get_env("MAX_LLM_CALLS_PER_TASK", "20"))


def get_max_tokens_per_task() -> int:
    """单次任务最大 token 消耗估算，默认 100000。"""
    return int(get_env("MAX_TOKENS_PER_TASK", "500000"))


def get_max_completion_tokens_per_call() -> int:
    """Maximum model output tokens for one planning/execution call."""
    return int(get_env("MAX_COMPLETION_TOKENS_PER_CALL", "32768"))


def _bounded_ratio(key: str, default: str) -> float:
    return max(0.0, min(1.0, float(get_env(key, default))))


def get_benchmark_discovery_warning_ratio() -> float:
    return _bounded_ratio("BENCHMARK_DISCOVERY_WARNING_RATIO", "0.60")


def get_benchmark_invalid_tool_warning_ratio() -> float:
    return _bounded_ratio("BENCHMARK_INVALID_TOOL_WARNING_RATIO", "0.20")


def get_benchmark_token_warning_ratio() -> float:
    return _bounded_ratio("BENCHMARK_TOKEN_WARNING_RATIO", "0.90")


# ── Sandbox / Docker 配置 ──────────────────────────────────────


def get_sandbox_enabled() -> bool:
    """获取沙箱模式是否启用。

    从 SANDBOX_ENABLED 环境变量读取，默认 false。
    启用后代码执行将在 Docker 容器中隔离运行。
    """
    return get_env("SANDBOX_ENABLED", "false").lower() == "true"


def get_sandbox_timeout() -> int:
    """获取沙箱命令执行超时秒数。

    从 SANDBOX_TIMEOUT 环境变量读取，默认 60 秒。
    """
    return int(get_env("SANDBOX_TIMEOUT", "60"))


# ── Memory 系统配置 ──────────────────────────────────────


def get_memory_global_root() -> Path:
    """获取全局记忆存储根目录。

    默认 ~/.codeagent/memory/，可通过 MEMORY_GLOBAL_ROOT 环境变量覆盖。
    """
    return Path(get_env("MEMORY_GLOBAL_ROOT", str(Path.home() / ".codeagent" / "memory")))


def get_memory_project_root(project_path: str) -> Path:
    """获取项目记忆存储根目录。

    默认 {project_path}/.codeagent/memory/。
    """
    return Path(project_path) / ".codeagent" / "memory"


def get_memory_index_max_lines() -> int:
    """获取 MEMORY.md 索引最大行数，默认 200。"""
    return int(get_env("MEMORY_INDEX_MAX_LINES", "200"))


def get_memory_session_ttl_days() -> int:
    """获取 session 类型记忆过期天数，默认 7。"""
    return int(get_env("MEMORY_SESSION_TTL_DAYS", "7"))


def get_memory_token_budget() -> int:
    """获取记忆注入 System Prompt 的 token 预算，默认 800。"""
    return int(get_env("MEMORY_TOKEN_BUDGET", "800"))


def get_memory_enabled() -> bool:
    """Whether persistent memory recall and extraction are enabled."""
    return get_env("MEMORY_ENABLED", "true").lower() == "true"


def get_skills_enabled() -> bool:
    """Whether manifest-matched Skills are injected into tasks."""
    return get_env("SKILLS_ENABLED", "true").lower() == "true"


def get_mcp_enabled() -> bool:
    """Whether MCP server discovery and dynamic tools are enabled."""
    return get_env("MCP_ENABLED", "true").lower() == "true"


def get_sandbox_memory_mb() -> int:
    """获取沙箱容器内存限制（MB）。

    从 SANDBOX_MEMORY_MB 环境变量读取，默认 512 MB。
    """
    return int(get_env("SANDBOX_MEMORY_MB", "512"))


# ── 自进化系统配置（Phase 7.1） ────────────────────────────────


def get_evolution_enabled() -> bool:
    """是否启用自进化机制，默认 True。"""
    return get_env("EVOLUTION_ENABLED", "true").lower() == "true"


def get_trajectory_base_path() -> str:
    """轨迹存储路径，默认 .codeagent/trajectories/。"""
    return get_env("TRAJECTORY_BASE_PATH", ".codeagent/trajectories")


def get_strategy_base_path() -> str:
    """策略存储路径，默认 .codeagent/strategies/。"""
    return get_env("STRATEGY_BASE_PATH", ".codeagent/strategies")


def get_memory_use_vector() -> bool:
    """记忆检索是否启用向量语义检索，默认 True（需要 sentence-transformers）。"""
    return get_env("MEMORY_USE_VECTOR", "false").lower() == "true"


def get_checkpoint_db_path() -> str:
    """Checkpoint SQLite 数据库路径，默认 ~/.codeagent/checkpoints.db。"""
    return get_env(
        "CHECKPOINT_DB_PATH",
        str(Path.home() / ".codeagent" / "checkpoints.db"),
    )


# ── Phase 8: Celery / Redis 配置 ──────────────────────────────────


def get_redis_url() -> str:
    """获取 Redis 连接 URL，默认 redis://localhost:6379。"""
    return get_env("REDIS_URL", "redis://127.0.0.1:6379")


def get_celery_task_timeout() -> int:
    """Agent 任务超时秒数，默认 600（10 分钟）。"""
    return int(get_env("CELERY_TASK_TIMEOUT", "600"))
