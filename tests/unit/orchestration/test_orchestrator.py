"""Orchestrator 单元测试。

Mock 所有 Gateway 和 LLM，测试 run/resume 的调度逻辑。
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from codeagent.orchestration.orchestrator import Orchestrator
from codeagent.orchestration.state import AgentState
from codeagent.gateway.memory_gateway import IMemoryGateway
from codeagent.gateway.validation_gateway import ValidationResult


@pytest.fixture
def mock_gateways():
    """创建所有 mock gateway 和 LLM。"""
    return {
        "context_gateway": AsyncMock(),
        "tool_gateway": MagicMock(),
        "validation_gateway": AsyncMock(),
        "llm": AsyncMock(),
    }


@pytest.fixture
def orchestrator(mock_gateways: dict) -> Orchestrator:
    orch = Orchestrator(
        context_gateway=mock_gateways["context_gateway"],
        tool_gateway=mock_gateways["tool_gateway"],
        validation_gateway=mock_gateways["validation_gateway"],
        llm=mock_gateways["llm"],
        model_name="test-model",
    )
    # Mock 编译后的 graph，避免 MemorySaver 产生 unawaited coroutine
    mock_graph = AsyncMock()
    mock_graph.ainvoke.side_effect = (
        lambda state, config=None: state
        if state is not None
        else AgentState(user_request="", project_root="")
    )
    mock_graph.aupdate_state = AsyncMock()
    mock_graph.aget_state = AsyncMock()
    orch._graph = mock_graph
    return orch


class TestOrchestratorRun:
    """Orchestrator.run() 测试。"""

    async def test_run_returns_agent_state(self, orchestrator: Orchestrator) -> None:
        result = await orchestrator.run(
            user_request="test request",
            project_root="/test/project",
        )
        assert result is not None
        assert result.user_request == "test request"
        assert result.project_root == "/test/project"

    async def test_run_stores_config(self, orchestrator: Orchestrator) -> None:
        await orchestrator.run(
            user_request="test",
            project_root="/root",
        )
        checkpoints = orchestrator.get_checkpoints()
        assert len(checkpoints) == 1
        assert "thread_id" in checkpoints[0]

    async def test_run_auto_mode(self, orchestrator: Orchestrator) -> None:
        result = await orchestrator.run(
            user_request="test",
            project_root="/root",
            auto_mode=True,
        )
        assert result.auto_mode is True

    async def test_run_preserves_external_task_id(self, orchestrator: Orchestrator) -> None:
        result = await orchestrator.run(
            user_request="test",
            project_root="/root",
            task_id="api-task-123",
        )
        assert result.task_id == "api-task-123"
        assert result.conversation_history == [
            {"role": "user", "content": "test"},
        ]

    async def test_run_preserves_benchmark_instance_id(
        self, orchestrator: Orchestrator, tmp_path
    ) -> None:
        result = await orchestrator.run(
            user_request="fix benchmark",
            project_root=str(tmp_path),
            benchmark_instance_id="owner__repo-1",
        )

        assert result.benchmark_instance_id == "owner__repo-1"
        assert result.direct_execution is True
        assert result.run_profile["workflow"] == "direct"
        assert result.run_profile["benchmark"] is True

    async def test_run_multiple_times(self, orchestrator: Orchestrator) -> None:
        await orchestrator.run(user_request="req1", project_root="/root")
        await orchestrator.run(user_request="req2", project_root="/root")
        assert len(orchestrator.get_checkpoints()) == 2

    async def test_run_error_sets_errors_in_state(self, orchestrator: Orchestrator, mock_gateways: dict) -> None:
        orchestrator._graph.ainvoke.side_effect = RuntimeError("fail")
        result = await orchestrator.run(
            user_request="test",
            project_root="/root",
        )
        assert any("Orchestrator run failed" in e for e in result.errors)

    async def test_evidence_gated_memory_writeback_runs_only_after_completion(
        self, tmp_path, mock_gateways: dict,
    ) -> None:
        memory = AsyncMock(spec=IMemoryGateway)
        memory.auto_extract.return_value = []
        events: list[dict] = []
        orch = Orchestrator(
            **mock_gateways,
            memory_gateway=memory,
            progress_callback=events.append,
        )
        graph = AsyncMock()

        async def completed(state, config=None):
            state.validation_results = [ValidationResult(passed=True)]
            state.reflection = {"next_action": "finish"}
            return state

        graph.ainvoke.side_effect = completed
        orch._graph = graph

        result = await orch.run(
            "Remember this project preference: always use the parser fixture",
            str(tmp_path),
            task_id="task-memory",
        )

        memory.auto_extract.assert_not_awaited()
        assert result.memory_extracted is False
        assert result.transcript_path is not None
        assert not any(event["type"] == "memory_extracted" for event in events)
        assert next(event for event in events if event["type"] == "transcript_saved")["visibility"] == "internal"

        saved = await orch.persist_memory(
            result,
            user_request="Remember this project preference: always use the parser fixture",
            task_id="task-memory",
            success=True,
            completion_errors=[],
        )

        assert saved == 1
        memory.auto_extract.assert_awaited_once()
        assert result.memory_extracted is True

    async def test_routine_task_skips_foreground_learning(
        self, tmp_path, mock_gateways: dict,
    ) -> None:
        memory = AsyncMock(spec=IMemoryGateway)
        events: list[dict] = []
        orch = Orchestrator(
            **mock_gateways,
            memory_gateway=memory,
            progress_callback=events.append,
        )
        graph = AsyncMock()

        async def completed(state, config=None):
            state.validation_results = [ValidationResult(passed=True)]
            return state

        graph.ainvoke.side_effect = completed
        orch._graph = graph

        result = await orch.run("fix parser", str(tmp_path), task_id="task-routine")

        memory.auto_extract.assert_not_awaited()
        assert result.learning_mode == "off"
        profile_events = [event for event in events if event["type"] == "run_profile_selected"]
        assert profile_events[0]["data"]["workflow"] == "direct"

    async def test_recovery_lineage_is_emitted_and_persisted_in_state(
        self, tmp_path, mock_gateways: dict,
    ) -> None:
        events: list[dict] = []
        orch = Orchestrator(**mock_gateways, progress_callback=events.append)
        graph = AsyncMock()

        async def completed(state, config=None):
            state.validation_results = [ValidationResult(passed=True)]
            return state

        graph.ainvoke.side_effect = completed
        orch._graph = graph

        result = await orch.run(
            "continue repair",
            str(tmp_path),
            task_id="task-recovery",
            recovered_from_task_id="task-source",
        )

        assert result.recovered_from_task_id == "task-source"
        assert any(
            event["type"] == "recovery_started"
            and event["data"]["recovered_from_task_id"] == "task-source"
            for event in events
        )


class TestOrchestratorResume:
    """Orchestrator.resume() 测试。"""

    async def test_resume_unknown_thread_id_raises(self, orchestrator: Orchestrator) -> None:
        with pytest.raises(ValueError, match="Unknown thread_id"):
            await orchestrator.resume(
                thread_id="nonexistent",
                human_decision="approve",
            )

    async def test_resume_after_run_accepts_approve(self, orchestrator: Orchestrator) -> None:
        await orchestrator.run(user_request="test", project_root="/root")
        checkpoints = orchestrator.get_checkpoints()
        thread_id = checkpoints[0]["thread_id"]

        result = await orchestrator.resume(
            thread_id=thread_id,
            human_decision="approve",
        )
        assert result is not None

    async def test_resume_after_run_accepts_abort(self, orchestrator: Orchestrator) -> None:
        await orchestrator.run(user_request="test", project_root="/root")
        thread_id = orchestrator.get_checkpoints()[0]["thread_id"]

        result = await orchestrator.resume(
            thread_id=thread_id,
            human_decision="abort",
        )
        assert result is not None

    async def test_resume_after_run_accepts_modify(self, orchestrator: Orchestrator) -> None:
        await orchestrator.run(user_request="test", project_root="/root")
        thread_id = orchestrator.get_checkpoints()[0]["thread_id"]

        result = await orchestrator.resume(
            thread_id=thread_id,
            human_decision="modify",
        )
        assert result is not None


class TestOrchestratorCheckpoints:
    """Checkpoint 管理测试。"""

    def test_get_checkpoints_empty_initially(self, orchestrator: Orchestrator) -> None:
        assert orchestrator.get_checkpoints() == []

    async def test_get_checkpoints_after_run(self, orchestrator: Orchestrator) -> None:
        await orchestrator.run(user_request="test", project_root="/root")
        checkpoints = orchestrator.get_checkpoints()
        assert len(checkpoints) == 1
        assert "thread_id" in checkpoints[0]
        assert "config" in checkpoints[0]

    async def test_get_checkpoints_contains_config(self, orchestrator: Orchestrator) -> None:
        await orchestrator.run(user_request="test", project_root="/root")
        cp = orchestrator.get_checkpoints()[0]
        assert "configurable" in cp["config"]
        assert "thread_id" in cp["config"]["configurable"]
