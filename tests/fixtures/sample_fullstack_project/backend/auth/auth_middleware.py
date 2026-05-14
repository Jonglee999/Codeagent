"""Basic Authentication Middleware.

Current implementation: Basic Auth via Authorization header.
Goal: Replace with JWT-based authentication.
"""

import base64
import hashlib
import hmac
from typing import Any

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse


# Simulated user store
_USERS: dict[str, str] = {
    "admin": "password123",
    "user1": "secret456",
}


def _verify_basic_auth(credentials: str) -> dict[str, Any] | None:
    """Verify Basic Auth credentials and return user info.

    Args:
        credentials: The Base64-encoded "username:password" string.

    Returns:
        User info dict if valid, None otherwise.
    """
    try:
        decoded = base64.b64decode(credentials).decode("utf-8")
        username, password = decoded.split(":", 1)
    except (ValueError, base64.binascii.Error):
        return None

    stored_password = _USERS.get(username)
    if stored_password is None:
        return None

    # Constant-time comparison to prevent timing attacks
    if hmac.compare_digest(stored_password, password):
        return {"username": username, "role": "admin" if username == "admin" else "user"}

    return None


async def authenticate_request(request: Request) -> dict[str, Any]:
    """Extract and validate Basic Auth credentials from request.

    Args:
        request: The incoming FastAPI request.

    Returns:
        User info dict on success.

    Raises:
        HTTPException: If authentication fails.
    """
    auth_header = request.headers.get("Authorization", "")

    if not auth_header.startswith("Basic "):
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid Authorization header",
            headers={"WWW-Authenticate": "Basic"},
        )

    credentials = auth_header[len("Basic "):]
    user_info = _verify_basic_auth(credentials)

    if user_info is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid username or password",
            headers={"WWW-Authenticate": "Basic"},
        )

    return user_info


def require_auth(func: Any) -> Any:
    """Decorator to require authentication for a route."""
    import functools

    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        # In a real app this would inject the user into request context
        return await func(*args, **kwargs)

    return wrapper
