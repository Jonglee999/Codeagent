"""User model."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class User:
    """User model representing a system user."""

    username: str
    email: str = ""
    role: str = "user"
    is_active: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert user to dictionary."""
        return {
            "username": self.username,
            "email": self.email,
            "role": self.role,
            "is_active": self.is_active,
        }
