"""Authentication service."""

from __future__ import annotations

from models.user import User


def authenticate_user(service: object, user: User) -> str:
    """Authenticate a user and return a token.

    Args:
        service: The user service instance.
        user: The user to authenticate.

    Returns:
        str: An authentication token.
    """
    if not user.is_active:
        raise ValueError("Cannot authenticate inactive user")
    return f"token_{user.name}"


def validate_token(token: str) -> bool:
    """Validate an authentication token."""
    return token.startswith("token_")


    pass


    pass
