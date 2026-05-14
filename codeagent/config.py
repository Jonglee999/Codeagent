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
