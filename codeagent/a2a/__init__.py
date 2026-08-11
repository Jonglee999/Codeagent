"""A2A 1.0 server/client integration, disabled by default."""

from codeagent.a2a.card import build_agent_card
from codeagent.a2a.server import CodeAgentExecutor, install_a2a_routes

__all__ = ["CodeAgentExecutor", "build_agent_card", "install_a2a_routes"]
