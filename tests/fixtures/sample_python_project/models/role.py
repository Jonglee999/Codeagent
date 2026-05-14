"""Role model."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Role:
    """Represents a role with permissions."""

    name: str
    permissions: list[str] = field(default_factory=list)

    def add_permission(self, permission: str) -> None:
        """Add a permission to the role."""
        if permission not in self.permissions:
            self.permissions.append(permission)
