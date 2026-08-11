"""Deterministic MCP server for real stdio/Streamable HTTP contract runs."""

from __future__ import annotations

import os
import sys

from mcp.server.fastmcp import FastMCP

transport = sys.argv[1] if len(sys.argv) > 1 else "stdio"
port = int(os.environ.get("CODEAGENT_MCP_CONTRACT_PORT", "8765"))
mcp = FastMCP(
    "codeagent-contract",
    host="127.0.0.1",
    port=port,
    stateless_http=True,
)


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


@mcp.tool()
def restricted_value() -> str:
    """A tool intentionally excluded by acceptance-test allowlists."""
    return "should-not-be-visible"


@mcp.tool()
def failure_probe() -> str:
    """Always fail so the client can verify isolation and circuit behavior."""
    raise RuntimeError("intentional MCP contract failure")


if __name__ == "__main__":
    mcp.run(transport=transport)  # type: ignore[arg-type]
