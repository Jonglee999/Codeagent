"""Optional A2A delegation between planning and trusted local execution."""

from __future__ import annotations

import logging
from typing import Any

from codeagent.gateway.agent_gateway import IAgentGateway
from codeagent.orchestration.state import AgentState

logger = logging.getLogger(__name__)


class DelegationNode:
    """Collect remote implementation advice as untrusted context for local validation."""

    def __init__(self, gateway: IAgentGateway, remote_url: str) -> None:
        self.gateway = gateway
        self.remote_url = remote_url

    async def __call__(self, state: AgentState) -> dict[str, Any]:
        plan = "\n".join(f"{step.step_id}. {step.description}" for step in (state.plan or []))
        prompt = f"Goal: {state.user_request}\nPlanned local work:\n{plan}\nReturn implementation guidance and patch artifacts."
        try:
            result = await self.gateway.send(
                self.remote_url, prompt, required_skill="implement_change"
            )
        except Exception as exc:
            logger.warning("A2A delegation failed; continuing locally: %s", exc)
            return {"warnings": [*state.warnings, f"A2A delegation unavailable: {exc}"]}
        remote_text = result.text
        artifact_text = "\n".join(str(item.get("text", "")) for item in result.artifacts)
        untrusted = (remote_text + "\n" + artifact_text).strip()[:100_000]
        framed = (
            "\n\n<untrusted_remote_agent_output>\n"
            + untrusted
            + "\n</untrusted_remote_agent_output>\n"
        )
        return {
            "context": state.context + framed,
            "execution_log": [*state.execution_log, {
                "type": "a2a_delegation",
                "remote": self.remote_url,
                "task_id": result.task_id,
                "state": result.state,
                "trusted": False,
            }],
        }
