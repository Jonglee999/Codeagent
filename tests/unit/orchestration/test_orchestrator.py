"""Orchestrator 单元测试。

Mock 所有 Gateway 和 LLM，测试 run/resume 的调度逻辑。
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from codeagent.orchestration.orchestrator import Orchestrator


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
    return Orchestrator(
        context_gateway=mock_gateways["context_gateway"],
        tool_gateway=mock_gateways["tool_gateway"],
        validation_gateway=mock_gateways["validation_gateway"],
        llm=mock_gateways["llm"],
        model_name="test-model",
    )


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

    async def test_run_multiple_times(self, orchestrator: Orchestrator) -> None:
        await orchestrator.run(user_request="req1", project_root="/root")
        await orchestrator.run(user_request="req2", project_root="/root")
        assert len(orchestrator.get_checkpoints()) == 2

    async def test_run_error_sets_errors_in_state(self, orchestrator: Orchestrator, mock_gateways: dict) -> None:
        mock_gateways["context_gateway"].build_context.side_effect = RuntimeError("fail")
        result = await orchestrator.run(
            user_request="test",
            project_root="/root",
        )
        assert any("Orchestrator run failed" in e for e in result.errors)


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
