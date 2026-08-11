from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from codeagent.benchmarks.official import OfficialEvaluationStore
from codeagent.gateway.tool_gateway import IToolGateway, ToolDefinition, ToolResult
from codeagent.orchestration.nodes.execution_node import ExecutionNode
from codeagent.orchestration.state import AgentState, PlanStep
from codeagent.tools.base import ConcreteTool
from codeagent.tools.gateway import ToolGateway


FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "benchmark_hardening"
    / "smoke15_failures.json"
)


def _cases() -> dict[str, Any]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))["cases"]


def _official_report_path(
    store: OfficialEvaluationStore,
    run_id: str,
    model_slug: str,
    instance_id: str,
) -> Path:
    return (
        store.paths.result_root
        / run_id
        / "logs"
        / "run_evaluation"
        / run_id
        / model_slug
        / instance_id
        / "report.json"
    )


def test_gap_corrupt_container_is_not_counted_as_model_unresolved(tmp_path) -> None:
    case = _cases()["corrupt_official_container"]
    store = OfficialEvaluationStore(tmp_path)
    run_id = "corrupt-run"
    instance_id = "owner__repo-1"
    report = _official_report_path(store, run_id, "test__model", instance_id)
    report.parent.mkdir(parents=True)
    report.write_text(
        json.dumps({instance_id: {"resolved": case["report_resolved"]}}),
        encoding="utf-8",
    )
    (report.parent / "run_instance.log").write_text(case["log"], encoding="utf-8")
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(
        json.dumps({
            "instance_id": instance_id,
            "model_name_or_path": "test/model",
            "model_patch": "diff --git a/a.py b/a.py\n",
        }) + "\n",
        encoding="utf-8",
    )

    result = store.import_run(
        run_id,
        model_name="test/model",
        instance_ids=[instance_id],
        predictions_path=predictions,
    )
    status = result["instances"][instance_id]

    assert status["evaluation_state"] == "infrastructure_error"
    assert status["completed"] is False
    assert status["resolved"] is None
    assert result["run"]["infrastructure_errors"] == 1


def test_gap_empty_paid_prediction_has_an_explicit_official_state(tmp_path) -> None:
    store = OfficialEvaluationStore(tmp_path)
    run_id = "empty-run"
    instance_id = "owner__repo-1"
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(
        json.dumps({
            "instance_id": instance_id,
            "model_name_or_path": "test/model",
            "model_patch": "",
        }) + "\n",
        encoding="utf-8",
    )

    result = store.import_run(
        run_id,
        model_name="test/model",
        instance_ids=[instance_id],
        predictions_path=predictions,
    )
    status = result["instances"][instance_id]

    assert status["evaluation_state"] == "empty_patch"
    assert status["resolved"] is False
    assert status["error"] is None
    assert result["run"]["empty_patches"] == 1


def test_gap_tool_schema_validation_returns_actionable_details() -> None:
    result = ConcreteTool().validate_params_detailed(msg=123)

    assert result.valid is False
    assert result.errors[0].path == "$.msg"
    assert "string" in result.errors[0].message


@pytest.mark.asyncio
async def test_gap_apply_patch_can_infer_one_declared_relative_target(tmp_path) -> None:
    case = _cases()["patch_without_explicit_file_path"]
    target = tmp_path / "src" / "example.py"
    target.parent.mkdir()
    target.write_text("old = 1\n", encoding="utf-8")
    gateway = ToolGateway(project_root=str(tmp_path))

    result = await gateway.execute_tool(case["tool"], case["arguments"])

    assert result.success is True
    assert target.read_text(encoding="utf-8") == "old = 2\n"


@pytest.mark.asyncio
async def test_gap_budget_preflight_rejects_call_that_cannot_fit(monkeypatch) -> None:
    monkeypatch.setattr(
        "codeagent.orchestration.nodes.execution_node.codeagent_config.get_max_tokens_per_task",
        lambda: 100,
    )
    monkeypatch.setattr(
        "codeagent.orchestration.nodes.execution_node.codeagent_config.get_max_completion_tokens_per_call",
        lambda: 50,
    )
    llm = AsyncMock(return_value=SimpleNamespace(usage=None))
    node = ExecutionNode(
        llm=llm,
        tool_gateway=AsyncMock(),
        validation_gateway=AsyncMock(),
    )
    state = AgentState(
        user_request="test",
        project_root=".",
        estimated_tokens=90,
    )

    response = await node._call_llm_with_limit(
        state,
        messages=[{"role": "user", "content": "x" * 1000}],
        tools=None,
        tool_choice=None,
    )

    assert response is None
    llm.assert_not_awaited()
    assert state.review_type == "token_budget_exhausted"


@dataclass
class _Function:
    name: str
    arguments: str


@dataclass
class _ToolCall:
    id: str = "same-call"
    type: str = "function"
    function: _Function = field(default_factory=lambda: _Function(
        name="apply_patch",
        arguments=json.dumps({
            "file_path": "src/example.py",
            "patch": "@@\n-old = 1\n+old = 2\n",
        }),
    ))


@pytest.mark.asyncio
async def test_gap_duplicate_invalid_mutation_is_bounded(tmp_path) -> None:
    gateway = AsyncMock(spec=IToolGateway)
    gateway.list_tools.return_value = [ToolDefinition(
        name="apply_patch",
        description="Apply a patch",
        parameters_schema={"type": "object"},
        category="mutation",
    )]
    gateway.execute_tool.return_value = ToolResult(
        success=False,
        error_message="Invalid parameters",
        error_code="INVALID_PARAMS",
    )
    message = SimpleNamespace(content=None, tool_calls=[_ToolCall()])
    llm = AsyncMock(return_value=SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=SimpleNamespace(total_tokens=1),
    ))
    state = AgentState(
        user_request="fix it",
        project_root=str(tmp_path),
        benchmark_instance_id="owner__repo-1",
        plan=[PlanStep(
            step_id=1,
            description="Modify the source",
            action="modify",
            target_file="src/example.py",
        )],
    )
    node = ExecutionNode(
        llm=llm,
        tool_gateway=gateway,
        validation_gateway=AsyncMock(),
        max_tool_calls=6,
    )

    await node(state)

    assert gateway.execute_tool.await_count <= 2


@pytest.mark.asyncio
async def test_gap_missing_dependency_is_validation_degraded(tmp_path) -> None:
    case = _cases()["missing_dependency_validation"]
    gateway = AsyncMock(spec=IToolGateway)
    gateway.list_tools.return_value = [ToolDefinition(
        name="run_terminal",
        description="Run tests",
        parameters_schema={"type": "object"},
        category="execution",
    )]
    gateway.execute_tool.return_value = ToolResult(
        success=False,
        data={"exit_code": case["exit_code"], "stderr": case["stderr"]},
        error_message=case["stderr"],
        error_code="NON_ZERO_EXIT",
    )
    tool_call = _ToolCall(
        function=_Function(
            name="run_terminal",
            arguments=json.dumps({"command": case["command"]}),
        )
    )
    llm = AsyncMock(return_value=SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(
            content=None,
            tool_calls=[tool_call],
        ))],
        usage=SimpleNamespace(total_tokens=1),
    ))
    state = AgentState(
        user_request="run tests",
        project_root=str(tmp_path),
        benchmark_instance_id="owner__repo-1",
        plan=[PlanStep(
            step_id=1,
            description="Run the targeted tests",
            action="command",
        )],
    )
    node = ExecutionNode(
        llm=llm,
        tool_gateway=gateway,
        validation_gateway=AsyncMock(),
        max_tool_calls=1,
    )

    result = await node(state)

    assert result["validation_state"] == "validation_degraded"
    assert result["current_step_index"] == 1
