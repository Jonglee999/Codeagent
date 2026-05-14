"""Application settings."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Settings:
    """Application configuration."""

    app_name: str = "sample_app"
    debug: bool = False
    secret_key: str = "default-secret"
    max_login_attempts: int = 3

    @classmethod
    def from_env(cls) -> Settings:
        """Load settings from environment."""
        return cls()
