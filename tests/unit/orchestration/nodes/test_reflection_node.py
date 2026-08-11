from __future__ import annotations

import pytest

from codeagent.gateway.validation_gateway import ValidationError, ValidationResult
from codeagent.orchestration.nodes.reflection_node import ReflectionNode
from codeagent.orchestration.routing import route_after_reflection
from codeagent.orchestration.state import AgentState


@pytest.mark.asyncio
async def test_reflection_finishes_when_validation_passes() -> None:
    state = AgentState(
        user_request="fix it",
        project_root=".",
        validation_results=[ValidationResult(passed=True)],
    )
    update = await ReflectionNode()(state)
    state.reflection = update["reflection"]
    assert update["reflection"]["next_action"] == "finish"
    assert route_after_reflection(state) == "end"


@pytest.mark.asyncio
async def test_reflection_requests_repair_with_failed_evidence() -> None:
    state = AgentState(
        user_request="fix it",
        project_root=".",
        retry_count=1,
        validation_results=[
            ValidationResult(
                passed=False,
                errors=[ValidationError(file_path="test.py", message="one failure")],
            ),
        ],
    )
    update = await ReflectionNode()(state)
    state.reflection = update["reflection"]
    assert update["reflection"]["failed_validations"][0]["errors"] == ["one failure"]
    assert route_after_reflection(state) == "execution"
