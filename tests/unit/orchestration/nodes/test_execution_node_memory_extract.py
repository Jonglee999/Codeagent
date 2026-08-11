"""P2 memory extraction happens after evidence, not inside ExecutionNode."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from codeagent.gateway.memory_gateway import IMemoryGateway
from codeagent.gateway.tool_gateway import IToolGateway
from codeagent.gateway.validation_gateway import IValidationGateway
from codeagent.orchestration.nodes.execution_node import ExecutionNode
from codeagent.orchestration.state import AgentState


@pytest.mark.asyncio
async def test_execution_does_not_extract_unvalidated_memory() -> None:
    memory = AsyncMock(spec=IMemoryGateway)
    tools = AsyncMock(spec=IToolGateway)
    tools.list_tools.return_value = []
    llm = AsyncMock()
    llm.return_value.choices = [
        type("Choice", (), {"message": type("Message", (), {"content": "done", "tool_calls": None})()})(),
    ]
    node = ExecutionNode(
        llm=llm,
        tool_gateway=tools,
        validation_gateway=AsyncMock(spec=IValidationGateway),
        memory_gateway=memory,
    )

    result = await node(AgentState(user_request="inspect", project_root="."))

    memory.auto_extract.assert_not_awaited()
    assert result.get("memory_extracted") is not True


@pytest.mark.asyncio
async def test_memory_recall_is_audited_without_exposing_body() -> None:
    memory = AsyncMock(spec=IMemoryGateway)
    memory.recall.return_value = (
        '<relevant_memories><memory type="project" name="known-fix" '
        'score="0.91" scope="project">secret body</memory></relevant_memories>'
    )
    events: list[dict] = []
    tools = AsyncMock(spec=IToolGateway)
    tools.list_tools.return_value = []
    llm = AsyncMock()
    llm.return_value.choices = [
        type("Choice", (), {"message": type("Message", (), {"content": "done", "tool_calls": None})()})(),
    ]
    state = AgentState(user_request="inspect", project_root=".")
    node = ExecutionNode(
        llm=llm,
        tool_gateway=tools,
        validation_gateway=AsyncMock(spec=IValidationGateway),
        memory_gateway=memory,
        progress_callback=events.append,
    )

    result = await node(state)

    assert result["memory_hits"][0]["name"] == "known-fix"
    assert "secret body" not in str(events)
    assert any(event["type"] == "memory_recalled" for event in events)


def test_duplicate_workspace_prefix_is_normalized(tmp_path) -> None:
    workspace = tmp_path / "session-123"
    workspace.mkdir()
    state = AgentState(user_request="fix", project_root=str(workspace))

    normalized = ExecutionNode._normalize_workspace_args(
        state, {"file_path": "session-123/src/app.py", "cwd": "session-123"},
    )

    assert normalized == {"file_path": "src/app.py", "cwd": "."}
