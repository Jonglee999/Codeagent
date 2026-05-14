"""Authentication tests — verify Basic Auth middleware behavior."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest


class TestBasicAuth:
    """Tests for the Basic Auth middleware."""

    def test_verify_valid_credentials(self) -> None:
        """Valid Basic Auth credentials should return user info."""
        import base64

        from backend.auth.auth_middleware import _verify_basic_auth

        credentials = base64.b64encode(b"admin:password123").decode()
        result = _verify_basic_auth(credentials)
        assert result is not None
        assert result["username"] == "admin"
        assert result["role"] == "admin"

    def test_verify_invalid_password(self) -> None:
        """Invalid password should return None."""
        import base64

        from backend.auth.auth_middleware import _verify_basic_auth

        credentials = base64.b64encode(b"admin:wrongpassword").decode()
        result = _verify_basic_auth(credentials)
        assert result is None

    def test_verify_nonexistent_user(self) -> None:
        """Non-existent user should return None."""
        import base64

        from backend.auth.auth_middleware import _verify_basic_auth

        credentials = base64.b64encode(b"nonexistent:password").decode()
        result = _verify_basic_auth(credentials)
        assert result is None

    def test_missing_auth_header_raises(self) -> None:
        """Missing Authorization header should raise 401."""
        import httpx

        from backend.auth.auth_middleware import authenticate_request

        request = MagicMock(spec=httpx.Request)
        request.headers = {}

        import pytest

        with pytest.raises(Exception):
            import asyncio
            asyncio.run(authenticate_request(request))
