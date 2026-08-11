"""API routes."""

from __future__ import annotations

from services.auth import authenticate_user
from services.user_service import UserService


def setup_routes(svc: UserService) -> dict[str, object]:
    """Set up API routes."""
    routes: dict[str, object] = {}

    def login(name: str, email: str) -> dict[str, str]:
        user = svc.create_user(name, email)
        token = authenticate_user(svc, user)
        return {"token": token}

    def health() -> dict[str, str]:
        return {"status": "ok"}

    routes["login"] = login
    routes["health"] = health
    return routes
