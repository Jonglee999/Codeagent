"""Scoped MCP configuration with secret-safe policy resolution."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

MCPTransport = Literal["stdio", "streamable-http"]
RiskLevel = Literal["low", "high"]


@dataclass(frozen=True)
class MCPServerConfig:
    name: str
    source: str
    source_root: Path
    transport: MCPTransport = "stdio"
    command: str | None = None
    args: tuple[str, ...] = ()
    url: str | None = None
    enabled: bool = True
    allowed_tools: tuple[str, ...] = ()
    env_allowlist: tuple[str, ...] = ()
    required_env: tuple[str, ...] = ()
    header_allowlist: tuple[str, ...] = ()
    timeout_seconds: int = 30
    max_concurrency: int = 2
    max_output_chars: int = 32_000
    risk_level: RiskLevel = "low"
    failure_threshold: int = 3
    cooldown_seconds: int = 30
    environment: dict[str, str] = field(default_factory=dict, repr=False)
    headers: dict[str, str] = field(default_factory=dict, repr=False)

    def public_metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source": self.source,
            "transport": self.transport,
            "enabled": self.enabled,
            "allowed_tools": list(self.allowed_tools),
            "required_env": list(self.required_env),
            "timeout_seconds": self.timeout_seconds,
            "max_concurrency": self.max_concurrency,
            "max_output_chars": self.max_output_chars,
            "risk_level": self.risk_level,
        }

    def missing_required_env(self) -> tuple[str, ...]:
        """Return required variable names whose values were not resolved."""
        return tuple(name for name in self.required_env if not self.environment.get(name))


@dataclass
class MCPConfigResolution:
    servers: dict[str, MCPServerConfig] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    config_paths: list[str] = field(default_factory=list)


def _bounded_int(raw: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        return max(minimum, min(maximum, int(raw)))
    except (TypeError, ValueError):
        return default


def _string_tuple(raw: Any, field_name: str) -> tuple[str, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
        raise ValueError(f"{field_name} must be a list of strings")
    return tuple(dict.fromkeys(item.strip() for item in raw if item.strip()))


def _read_json(path: Path, max_bytes: int = 64 * 1024) -> dict[str, Any]:
    if path.stat().st_size > max_bytes:
        raise ValueError(f"configuration exceeds {max_bytes} bytes")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("configuration root must be an object")
    return payload


def _secret_values(root: Path, server_name: str) -> tuple[dict[str, str], dict[str, str]]:
    path = root / ".codeagent" / "mcp.secrets.json"
    if not path.is_file():
        return {}, {}
    payload = _read_json(path, max_bytes=32 * 1024)
    raw = payload.get("servers", {}).get(server_name, {})
    if not isinstance(raw, dict):
        return {}, {}
    env = raw.get("env", {})
    headers = raw.get("headers", {})
    safe_env = env if isinstance(env, dict) else {}
    safe_headers = headers if isinstance(headers, dict) else {}
    return (
        {str(k): str(v) for k, v in safe_env.items()},
        {str(k): str(v) for k, v in safe_headers.items()},
    )


def _parse_server(
    name: str,
    raw: Any,
    *,
    source: str,
    root: Path,
) -> MCPServerConfig:
    if not isinstance(raw, dict):
        raise ValueError("server definition must be an object")
    if "env" in raw or "headers" in raw:
        raise ValueError(
            "inline env/headers are forbidden; use env_allowlist/header_allowlist and "
            ".codeagent/mcp.secrets.json or process environment"
        )

    transport = str(raw.get("transport", "stdio"))
    if transport not in {"stdio", "streamable-http"}:
        raise ValueError("transport must be stdio or streamable-http")
    command = raw.get("command")
    args = _string_tuple(raw.get("args", []), "args")
    url = raw.get("url")
    if transport == "stdio":
        if not isinstance(command, str) or not command.strip():
            raise ValueError("stdio command must be a non-empty string")
        url = None
    else:
        if not isinstance(url, str) or urlparse(url).scheme not in {"http", "https"}:
            raise ValueError("streamable-http url must use http or https")
        command = None
        args = ()

    env_allowlist = _string_tuple(raw.get("env_allowlist"), "env_allowlist")
    required_env = _string_tuple(raw.get("required_env"), "required_env")
    if not set(required_env).issubset(env_allowlist):
        raise ValueError("required_env must be a subset of env_allowlist")
    header_allowlist = _string_tuple(raw.get("header_allowlist"), "header_allowlist")
    secret_env, secret_headers = _secret_values(root, name)
    environment = {
        key: secret_env.get(key, os.environ.get(key, ""))
        for key in env_allowlist
        if secret_env.get(key, os.environ.get(key)) is not None
    }
    headers = {
        key: secret_headers.get(key, os.environ.get(key, ""))
        for key in header_allowlist
        if secret_headers.get(key, os.environ.get(key)) is not None
    }
    risk = str(raw.get("risk_level", "low"))
    if risk not in {"low", "high"}:
        raise ValueError("risk_level must be low or high")

    return MCPServerConfig(
        name=name,
        source=source,
        source_root=root,
        transport=transport,  # type: ignore[arg-type]
        command=command,
        args=args,
        url=url,
        enabled=bool(raw.get("enabled", True)),
        allowed_tools=_string_tuple(raw.get("allowed_tools"), "allowed_tools"),
        env_allowlist=env_allowlist,
        required_env=required_env,
        header_allowlist=header_allowlist,
        timeout_seconds=_bounded_int(raw.get("timeout_seconds"), 30, 1, 120),
        max_concurrency=_bounded_int(raw.get("max_concurrency"), 2, 1, 8),
        max_output_chars=_bounded_int(raw.get("max_output_chars"), 32_000, 512, 1_000_000),
        risk_level=risk,  # type: ignore[arg-type]
        failure_threshold=_bounded_int(raw.get("failure_threshold"), 3, 1, 10),
        cooldown_seconds=_bounded_int(raw.get("cooldown_seconds"), 30, 1, 600),
        environment=environment,
        headers=headers,
    )


def resolve_mcp_config(
    workspace_root: str | Path,
    *,
    product_root: str | Path | None = None,
    user_root: str | Path | None = None,
) -> MCPConfigResolution:
    """Resolve product -> user -> workspace MCP servers; later names override."""
    workspace = Path(workspace_root).resolve()
    product = Path(product_root or Path.cwd()).resolve()
    user = Path(user_root).expanduser().resolve() if user_root else Path.home()
    scopes: list[tuple[str, Path]] = [("product", product)]
    if user != product:
        scopes.append(("user", user))
    if workspace not in {product, user}:
        scopes.append(("workspace", workspace))
    result = MCPConfigResolution()

    for source, root in scopes:
        path = root / ".codeagent" / "mcp.json"
        if not path.is_file():
            continue
        result.config_paths.append(str(path))
        try:
            payload = _read_json(path)
            raw_servers = payload.get("servers", {})
            if not isinstance(raw_servers, dict):
                raise ValueError("servers must be an object")
            for name, raw in raw_servers.items():
                try:
                    server = _parse_server(str(name), raw, source=source, root=root)
                    if server.enabled:
                        result.servers[server.name] = server
                    else:
                        result.servers.pop(server.name, None)
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    result.warnings.append(f"{source} MCP server {name}: {exc}")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            result.warnings.append(f"{source} MCP config: {exc}")

    # A workspace override executes relative stdio commands from the active workspace,
    # while inherited product/user definitions retain their own source directory.
    result.servers = {name: replace(server) for name, server in result.servers.items()}
    return result
