"""User service implementation."""

from __future__ import annotations


from config.settings import Settings
from models.user import User


class UserService:
    """Service for managing users."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._users: dict[str, User] = {}

    def create_user(self, name: str, email: str) -> User:
        """Create a new user."""
        user = User(name=name, email=email)
        self._users[user.name] = user
        return user

    def get_user(self, name: str) -> User | None:
        """Get a user by name."""
        return self._users.get(name)

    def list_users(self) -> list[User]:
        """List all users."""
        return list(self._users.values())
