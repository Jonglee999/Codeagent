"""Public CodeAgent Agent Card."""

from a2a.types import AgentCapabilities, AgentCard, AgentInterface, AgentSkill
from a2a.utils.constants import PROTOCOL_VERSION_1_0, TransportProtocol

from codeagent.config import get_a2a_base_url


_SKILLS = {
    "inspect_repository": "Inspect repository structure and relevant source code",
    "plan_change": "Create an implementation and validation plan",
    "implement_change": "Implement a scoped code change",
    "validate_change": "Run targeted checks and summarize their results",
    "review_patch": "Review a patch for defects, risk, and regressions",
}


def build_agent_card(base_url: str | None = None) -> AgentCard:
    base = (base_url or get_a2a_base_url()).rstrip("/")
    return AgentCard(
        name="CodeAgent",
        description="Repository-aware coding agent with local validation",
        version="0.1.0",
        supported_interfaces=[AgentInterface(
            url=f"{base}/a2a",
            protocol_binding=TransportProtocol.JSONRPC,
            protocol_version=PROTOCOL_VERSION_1_0,
        )],
        capabilities=AgentCapabilities(streaming=True),
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain", "application/json", "text/x-diff"],
        skills=[AgentSkill(id=key, name=key.replace("_", " ").title(), description=value, tags=["coding", "repository"]) for key, value in _SKILLS.items()],
    )
