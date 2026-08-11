"""A2A trust-boundary validation."""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse

from codeagent.config import get_a2a_allowlist


class A2ASecurityError(ValueError):
    pass


def validate_remote_url(url: str, allowlist: set[str] | None = None) -> str:
    """Reject credentials, non-HTTP schemes and non-allowlisted/SSRF targets."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise A2ASecurityError("A2A URL must use http or https")
    if parsed.username or parsed.password or parsed.fragment:
        raise A2ASecurityError("A2A URL may not contain credentials or fragments")
    host = parsed.hostname.lower().rstrip(".")
    if parsed.scheme == "http" and host not in {"localhost", "127.0.0.1", "::1"}:
        raise A2ASecurityError("Remote A2A endpoints must use HTTPS")
    allowed = allowlist if allowlist is not None else get_a2a_allowlist()
    if host not in allowed:
        raise A2ASecurityError(f"A2A host is not allowlisted: {host}")
    try:
        default_port = 443 if parsed.scheme == "https" else 80
        addresses = {item[4][0] for item in socket.getaddrinfo(host, parsed.port or default_port)}
    except socket.gaierror as exc:
        raise A2ASecurityError(f"A2A host cannot be resolved: {host}") from exc
    for address in addresses:
        ip = ipaddress.ip_address(address)
        if (ip.is_private or ip.is_loopback or ip.is_link_local) and host not in {"localhost", "127.0.0.1", "::1"}:
            raise A2ASecurityError("A2A host resolves to a private or link-local address")
    return url.rstrip("/")


def validate_payload_size(value: str | bytes, maximum: int, label: str) -> None:
    size = len(value if isinstance(value, bytes) else value.encode("utf-8"))
    if size > maximum:
        raise A2ASecurityError(f"{label} exceeds {maximum} bytes")
