"""Configuration management — centralized env/flag access.

Centralizes environment variable reading (from .env and os.environ)
so modules don't need to call os.environ directly.
"""

from __future__ import annotations

import os
from pathlib import Path


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
    return int(get_env("LLM_TIMEOUT", "60"))


def get_max_retries() -> int:
    """Get the default max retry count."""
    return int(get_env("MAX_RETRIES", "3"))


def get_context_budget() -> int:
    """Get the context budget in tokens."""
    return int(get_env("CONTEXT_BUDGET_TOKENS", "8000"))


# ── LLM 成本控制配置 ──────────────────────────────────────


def get_max_llm_calls_per_task() -> int:
    """单次任务最大 LLM 调用次数，默认 50。"""
    return int(get_env("MAX_LLM_CALLS_PER_TASK", "50"))


def get_max_tokens_per_task() -> int:
    """单次任务最大 token 消耗估算，默认 100000。"""
    return int(get_env("MAX_TOKENS_PER_TASK", "100000"))


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
    return get_env("MEMORY_USE_VECTOR", "true").lower() == "true"


def get_checkpoint_db_path() -> str:
    """Checkpoint SQLite 数据库路径，默认 ~/.codeagent/checkpoints.db。"""
    return get_env(
        "CHECKPOINT_DB_PATH",
        str(Path.home() / ".codeagent" / "checkpoints.db"),
    )
