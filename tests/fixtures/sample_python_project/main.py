"""Main entry point."""

from services.auth import authenticate_user
from services.user_service import UserService
from models.user import User
from config.settings import Settings


def main() -> None:
    """Run the application."""
    settings = Settings()
    user = User(name="test_user", email="test@example.com")
    svc = UserService(settings)
    token = authenticate_user(svc, user)
    print(f"Authenticated: {token}")
