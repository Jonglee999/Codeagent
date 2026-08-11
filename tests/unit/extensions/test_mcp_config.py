from __future__ import annotations

import json
from pathlib import Path

from codeagent.extensions.mcp import resolve_mcp_config


def _config(root: Path, servers: dict) -> None:
    directory = root / ".codeagent"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "mcp.json").write_text(
        json.dumps({"servers": servers}), encoding="utf-8"
    )


def test_workspace_server_overrides_product_server(tmp_path: Path) -> None:
    product = tmp_path / "product"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _config(product, {"tools": {"command": "product-python", "allowed_tools": ["read"]}})
    _config(workspace, {"tools": {"command": "workspace-python", "allowed_tools": ["write"]}})

    resolution = resolve_mcp_config(
        workspace, product_root=product, user_root=tmp_path / "user"
    )

    server = resolution.servers["tools"]
    assert server.source == "workspace"
    assert server.command == "workspace-python"
    assert server.allowed_tools == ("write",)


def test_disabled_workspace_server_removes_inherited_server(tmp_path: Path) -> None:
    product = tmp_path / "product"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _config(product, {"tools": {"command": "python"}})
    _config(workspace, {"tools": {"command": "python", "enabled": False}})

    resolution = resolve_mcp_config(workspace, product_root=product)

    assert "tools" not in resolution.servers


def test_secrets_are_allowlisted_and_never_public(tmp_path: Path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _config(workspace, {
        "remote": {
            "transport": "streamable-http",
            "url": "https://example.invalid/mcp",
            "header_allowlist": ["Authorization"],
        }
    })
    secret_path = workspace / ".codeagent" / "mcp.secrets.json"
    secret_path.write_text(
        json.dumps({"servers": {"remote": {"headers": {"Authorization": "Bearer secret"}}}}),
        encoding="utf-8",
    )

    server = resolve_mcp_config(workspace, product_root=tmp_path).servers["remote"]

    assert server.headers == {"Authorization": "Bearer secret"}
    assert "secret" not in json.dumps(server.public_metadata())


def test_inline_secret_values_are_rejected(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _config(workspace, {"unsafe": {"command": "python", "env": {"TOKEN": "secret"}}})

    resolution = resolve_mcp_config(workspace, product_root=tmp_path)

    assert "unsafe" not in resolution.servers
    assert any("inline env/headers are forbidden" in warning for warning in resolution.warnings)


def test_required_env_is_named_publicly_without_exposing_value(
    tmp_path: Path, monkeypatch
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setenv("SAFE_TOKEN", "private-value")
    _config(workspace, {
        "secured": {
            "command": "server",
            "env_allowlist": ["SAFE_TOKEN"],
            "required_env": ["SAFE_TOKEN"],
        }
    })

    server = resolve_mcp_config(workspace, product_root=tmp_path).servers["secured"]
    metadata = server.public_metadata()

    assert server.missing_required_env() == ()
    assert metadata["required_env"] == ["SAFE_TOKEN"]
    assert "private-value" not in json.dumps(metadata)


def test_required_env_must_be_allowlisted(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _config(workspace, {
        "secured": {
            "command": "server",
            "required_env": ["TOKEN"],
        }
    })

    resolution = resolve_mcp_config(workspace, product_root=tmp_path)

    assert "secured" not in resolution.servers
    assert any("required_env must be a subset" in item for item in resolution.warnings)
