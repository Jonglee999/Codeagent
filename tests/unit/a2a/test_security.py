import socket

import pytest

from codeagent.a2a.security import A2ASecurityError, validate_payload_size, validate_remote_url


def test_url_rejects_scheme_credentials_and_non_allowlisted_host():
    with pytest.raises(A2ASecurityError):
        validate_remote_url("file:///etc/passwd", {"localhost"})
    with pytest.raises(A2ASecurityError):
        validate_remote_url("http://user:pass@localhost", {"localhost"})
    with pytest.raises(A2ASecurityError):
        validate_remote_url("https://example.com", {"localhost"})
    with pytest.raises(A2ASecurityError, match="HTTPS"):
        validate_remote_url("http://agent.example", {"agent.example"})


def test_url_rejects_dns_rebinding_to_private_address(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", lambda *_a, **_k: [(2, 1, 6, "", ("10.0.0.4", 443))])
    with pytest.raises(A2ASecurityError, match="private"):
        validate_remote_url("https://agent.example", {"agent.example"})


def test_localhost_and_payload_limit():
    assert validate_remote_url("http://localhost:8000", {"localhost"}) == "http://localhost:8000"
    validate_payload_size("1234", 4, "message")
    with pytest.raises(A2ASecurityError, match="exceeds"):
        validate_payload_size("12345", 4, "message")
