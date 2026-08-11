from __future__ import annotations

from unittest.mock import AsyncMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from codeagent.interaction.api.routes import _get_gateway, router


def _client_with_inline_runner() -> tuple[TestClient, AsyncMock]:
    gateway = AsyncMock()
    app = FastAPI()
    app.include_router(router)
    app.state.inline_runner = AsyncMock()
    app.dependency_overrides[_get_gateway] = lambda: gateway
    return TestClient(app), gateway


def test_inline_task_fails_fast_when_llm_is_not_configured(monkeypatch, tmp_path):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    client, gateway = _client_with_inline_runner()

    response = client.post(
        "/api/v1/tasks",
        json={"query": "fix it", "project_root": str(tmp_path)},
    )

    assert response.status_code == 503
    assert "LLM_API_KEY" in response.json()["detail"]
    gateway.start_task.assert_not_awaited()


def test_inline_task_fails_fast_for_missing_project(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_API_KEY", "test-only-key")
    client, gateway = _client_with_inline_runner()

    response = client.post(
        "/api/v1/tasks",
        json={"query": "fix it", "project_root": str(tmp_path / "missing")},
    )

    assert response.status_code == 400
    assert "does not exist" in response.json()["detail"]
    gateway.start_task.assert_not_awaited()
