"""API Client — uses Basic Auth for all API requests.

Current implementation: Sends Basic Auth header with every request.
When auth migrates to JWT: change to send Bearer token instead.
"""

import base64
from typing import Any


class ApiClient:
    """HTTP API client using Basic Authentication."""

    def __init__(self, base_url: str, username: str, password: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.username = username
        self.password = password
        self._auth_token: str | None = None

    def _make_basic_auth_header(self) -> dict[str, str]:
        """Create Basic Auth Authorization header."""
        credentials = f"{self.username}:{self.password}"
        encoded = base64.b64encode(credentials.encode()).decode()
        return {"Authorization": f"Basic {encoded}"}

    def _get_headers(self) -> dict[str, str]:
        """Get request headers including auth."""
        headers: dict[str, str] = {
            "Content-Type": "application/json",
        }
        headers.update(self._make_basic_auth_header())
        return headers

    async def get(self, path: str) -> dict[str, Any]:
        """Send a GET request with Basic Auth."""
        import httpx

        url = f"{self.base_url}{path}"
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, headers=self._get_headers())
            resp.raise_for_status()
            return resp.json()

    async def post(self, path: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
        """Send a POST request with Basic Auth."""
        import httpx

        url = f"{self.base_url}{path}"
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, json=data, headers=self._get_headers())
            resp.raise_for_status()
            return resp.json()
