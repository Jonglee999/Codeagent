from codeagent.gateway.agent_gateway import AgentResult
from codeagent.orchestration.nodes.delegation_node import DelegationNode
from codeagent.orchestration.state import AgentState, PlanStep


class StubAgentGateway:
    async def send(self, url, message, required_skill=None):
        assert required_skill == "implement_change"
        return AgentResult("remote-1", "completed", "Use a narrow patch", [{"text": "diff"}])


async def test_delegation_marks_remote_content_untrusted():
    state = AgentState("fix it", ".", plan=[PlanStep(1, "modify file", "modify")])
    result = await DelegationNode(StubAgentGateway(), "https://agent.example")(state)  # type: ignore[arg-type]
    assert "<untrusted_remote_agent_output>" in result["context"]
    assert result["execution_log"][-1]["trusted"] is False
