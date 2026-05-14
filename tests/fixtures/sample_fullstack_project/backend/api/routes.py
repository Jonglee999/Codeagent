"""API Routes — uses Basic Auth middleware for protected endpoints."""

from typing import Any

from fastapi import APIRouter, Depends, Request

from backend.auth.auth_middleware import authenticate_request

router = APIRouter()


@router.get("/health")
async def health_check() -> dict[str, str]:
    """Public health check endpoint."""
    return {"status": "ok"}


@router.get("/users/me")
async def get_current_user(
    request: Request,
    user: dict[str, Any] = Depends(authenticate_request),  # type: ignore[arg-type]
) -> dict[str, Any]:
    """Get current authenticated user's profile.

    Uses Basic Auth via the authenticate_request dependency.
    """
    return {
        "username": user["username"],
        "role": user["role"],
    }


@router.get("/users/{username}")
async def get_user(
    username: str,
    request: Request,
    user: dict[str, Any] = Depends(authenticate_request),  # type: ignore[arg-type]
) -> dict[str, Any]:
    """Get a specific user's public profile (authenticated)."""
    return {"username": username, "message": "User profile endpoint"}
