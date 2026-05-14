"""User model."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class User:
    """Represents a user in the system."""

    name: str
    email: str
    roles: list[str] = field(default_factory=list)
    is_active: bool = True

    def has_permission(self, permission: str) -> bool:
        """Check if user has a specific permission."""
        return permission in self.roles

    def deactivate(self) -> None:
        """Deactivate the user."""
        self.is_active = False
