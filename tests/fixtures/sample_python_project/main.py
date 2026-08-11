"""Main entry point."""

from services.auth import authenticate_user
from services.user_service import UserService
from models.user import User
from config.settings import Settings


def main() -> None:
    """Run the application."""
    settings: Settings = Settings()
    user: User = User(name="test_user", email="test@example.com")
    svc: UserService = UserService(settings)
    token: str = authenticate_user(svc, user)
    print(f"Authenticated: {token}")
