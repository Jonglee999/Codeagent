"""API 路由单元测试。

使用 FastAPI TestClient + Mock Gateway 测试 5 个端点。
覆盖：正常路径、异常路径、统一响应格式、404 处理、CORS 头。
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

from codeagent.gateway.orchestration_gateway import (
    TaskReport,
    TaskState,
    TaskStatus,
)
from codeagent.interaction.api.routes import _get_gateway, router


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_gateway():
    """Create an AsyncMock gateway for testing."""
    gw = AsyncMock()

    # start_task
    gw.start_task.return_value = "test-task-id-123"

    # get_task_status
    gw.get_task_status.return_value = TaskStatus(
        task_id="test-task-id-123",
        state=TaskState.COMPLETED,
        progress=1.0,
        current_step="Done",
        errors=[],
    )

    # cancel_task
    gw.cancel_task.return_value = True

    # get_report
    gw.get_report.return_value = TaskReport(
        task_id="test-task-id-123",
        plan=[{"step": "analyze", "action": "read files"}],
        changes=[{"file": "main.py", "action": "modify"}],
        validation_results=[{"layer": "syntax", "passed": True}],
        duration=12.5,
        token_usage=1500,
    )

    # submit_decision
    gw.submit_decision.return_value = None

    return gw


@pytest.fixture
def app(mock_gateway):
    """Create a test FastAPI app with mocked gateway."""
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[_get_gateway] = lambda: mock_gateway
    return app


@pytest.fixture
def client(app):
    """Create a TestClient for the test app."""
    return TestClient(app)


# =============================================================================
# POST /api/v1/tasks
# =============================================================================


class TestCreateTask:
    def test_success(self, client, mock_gateway):
        """POST /tasks 成功返回 202 + task_id。"""
        mock_gateway.start_task.return_value = "task-42"
        resp = client.post(
            "/api/v1/tasks",
            json={
                "query": "Add logging to main.py",
                "project_root": "/tmp/test",
            },
        )
        assert resp.status_code == 202
        data = resp.json()
        assert data["success"] is True
        assert data["data"]["task_id"] == "task-42"
        assert data["data"]["status"] == "pending"
        assert data["error"] is None

    def test_success_auto_mode(self, client, mock_gateway):
        """POST /tasks 支持 auto_mode 和 max_retries。"""
        mock_gateway.start_task.return_value = "task-99"
        resp = client.post(
            "/api/v1/tasks",
            json={
                "query": "Refactor utils",
                "project_root": "/tmp/p",
                "auto_mode": True,
                "max_retries": 5,
            },
        )
        assert resp.status_code == 202
        data = resp.json()
        assert data["success"] is True
        assert data["data"]["task_id"] == "task-99"
        # Verify the gateway received correct parameters
        call_args = mock_gateway.start_task.await_args
        assert call_args is not None
        user_req = call_args.args[0]
        assert user_req.auto_mode is True
        assert user_req.max_retries == 5

    def test_missing_query(self, client):
        """POST /tasks 缺少 query 时返回 422。"""
        resp = client.post(
            "/api/v1/tasks",
            json={"project_root": "/tmp/test"},
        )
        assert resp.status_code == 422

    def test_empty_query(self, client):
        """POST /tasks query 为空字符串时返回 422。"""
        resp = client.post(
            "/api/v1/tasks",
            json={"query": "", "project_root": "/tmp/test"},
        )
        assert resp.status_code == 422

    def test_missing_project_root(self, client):
        """POST /tasks 缺少 project_root 时返回 422。"""
        resp = client.post(
            "/api/v1/tasks",
            json={"query": "Add logging"},
        )
        assert resp.status_code == 422

    def test_gateway_error(self, client, mock_gateway):
        """POST /tasks gateway 异常时返回 500。"""
        mock_gateway.start_task.side_effect = RuntimeError("Redis connection failed")
        resp = client.post(
            "/api/v1/tasks",
            json={
                "query": "Add logging",
                "project_root": "/tmp/test",
            },
        )
        assert resp.status_code == 500
        data = resp.json()
        assert data["success"] is False
        assert "Redis connection failed" in data["error"]


# =============================================================================
# GET /api/v1/tasks/{task_id}
# =============================================================================


class TestGetTaskStatus:
    def test_success_running(self, client, mock_gateway):
        """GET /tasks/{id} 返回 running 状态。"""
        mock_gateway.get_task_status.return_value = TaskStatus(
            task_id="task-1",
            state=TaskState.RUNNING,
            progress=0.5,
            current_step="Executing plan",
            errors=[],
        )
        resp = client.get("/api/v1/tasks/task-1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["data"]["state"] == "running"
        assert data["data"]["progress"] == 0.5
        assert data["data"]["current_step"] == "Executing plan"

    def test_success_pending(self, client, mock_gateway):
        """GET /tasks/{id} 返回 pending 状态。"""
        mock_gateway.get_task_status.return_value = TaskStatus(
            task_id="task-1",
            state=TaskState.PENDING,
            progress=0.0,
            current_step=None,
            errors=[],
        )
        resp = client.get("/api/v1/tasks/task-1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["data"]["state"] == "pending"
        assert data["data"]["progress"] == 0.0

    def test_not_found(self, client, mock_gateway):
        """GET /tasks/{id} 不存在的任务返回 404。"""
        mock_gateway.get_task_status.side_effect = KeyError("nonexistent")
        resp = client.get("/api/v1/tasks/nonexistent")
        assert resp.status_code == 404
        data = resp.json()
        assert data["success"] is False
        assert "not found" in data["error"].lower()

    def test_gateway_error(self, client, mock_gateway):
        """GET /tasks/{id} gateway 异常时返回 500。"""
        mock_gateway.get_task_status.side_effect = RuntimeError("Redis error")
        resp = client.get("/api/v1/tasks/task-1")
        assert resp.status_code == 500
        data = resp.json()
        assert data["success"] is False
        assert "Redis error" in data["error"]


# =============================================================================
# POST /api/v1/tasks/{task_id}/decision
# =============================================================================


class TestSubmitDecision:
    def test_approve(self, client, mock_gateway):
        """POST /tasks/{id}/decision 提交 approve 决策。"""
        resp = client.post(
            "/api/v1/tasks/task-1/decision",
            json={"decision": "approve"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["data"]["decision"] == "approve"
        mock_gateway.submit_decision.assert_awaited_once()

    def test_reject(self, client, mock_gateway):
        """POST /tasks/{id}/decision 提交 reject 决策。"""
        resp = client.post(
            "/api/v1/tasks/task-1/decision",
            json={"decision": "reject"},
        )
        assert resp.status_code == 200
        assert resp.json()["success"] is True
        mock_gateway.submit_decision.assert_awaited_once()

    def test_modify(self, client, mock_gateway):
        """POST /tasks/{id}/decision 提交 modify 决策（含修改建议）。"""
        resp = client.post(
            "/api/v1/tasks/task-1/decision",
            json={
                "decision": "modify",
                "modifications": {"feedback": "use safer approach"},
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        # Verify HumanDecision 传递了 modifications
        call_args = mock_gateway.submit_decision.await_args
        assert call_args is not None
        decision = call_args.args[1]
        assert decision.modifications == {"feedback": "use safer approach"}

    def test_invalid_decision_value(self, client):
        """POST /tasks/{id}/decision 无效 decision 值返回 422。"""
        resp = client.post(
            "/api/v1/tasks/task-1/decision",
            json={"decision": "invalid_option"},
        )
        assert resp.status_code == 422

    def test_gateway_error(self, client, mock_gateway):
        """POST /tasks/{id}/decision gateway 异常时返回 500。"""
        mock_gateway.submit_decision.side_effect = RuntimeError("Redis error")
        resp = client.post(
            "/api/v1/tasks/task-1/decision",
            json={"decision": "approve"},
        )
        assert resp.status_code == 500
        data = resp.json()
        assert data["success"] is False


# =============================================================================
# DELETE /api/v1/tasks/{task_id}
# =============================================================================


class TestCancelTask:
    def test_success(self, client, mock_gateway):
        """DELETE /tasks/{id} 成功取消任务。"""
        mock_gateway.cancel_task.return_value = True
        resp = client.delete("/api/v1/tasks/task-1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["data"]["cancelled"] is True
        mock_gateway.cancel_task.assert_awaited_once_with("task-1")

    def test_cancel_not_found(self, client, mock_gateway):
        """DELETE /tasks/{id} 取消不存在的任务返回 cancelled=False。"""
        mock_gateway.cancel_task.return_value = False
        resp = client.delete("/api/v1/tasks/task-1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["data"]["cancelled"] is False

    def test_gateway_error(self, client, mock_gateway):
        """DELETE /tasks/{id} gateway 异常时返回 500。"""
        mock_gateway.cancel_task.side_effect = RuntimeError("Revoke failed")
        resp = client.delete("/api/v1/tasks/task-1")
        assert resp.status_code == 500
        data = resp.json()
        assert data["success"] is False


# =============================================================================
# GET /api/v1/tasks/{task_id}/report
# =============================================================================


class TestGetReport:
    def test_success(self, client, mock_gateway):
        """GET /tasks/{id}/report 返回完整报告。"""
        resp = client.get("/api/v1/tasks/task-1/report")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["data"]["task_id"] == "test-task-id-123"
        assert data["data"]["duration"] == 12.5
        assert data["data"]["token_usage"] == 1500
        assert len(data["data"]["plan"]) == 1
        assert len(data["data"]["changes"]) == 1
        assert len(data["data"]["validation_results"]) == 1

    def test_not_found(self, client, mock_gateway):
        """GET /tasks/{id}/report 不存在的报告返回 404。"""
        mock_gateway.get_report.side_effect = KeyError("nonexistent")
        resp = client.get("/api/v1/tasks/nonexistent/report")
        assert resp.status_code == 404
        data = resp.json()
        assert data["success"] is False
        assert "not found" in data["error"].lower()

    def test_gateway_error(self, client, mock_gateway):
        """GET /tasks/{id}/report gateway 异常时返回 500。"""
        mock_gateway.get_report.side_effect = RuntimeError("Redis error")
        resp = client.get("/api/v1/tasks/task-1/report")
        assert resp.status_code == 500
        data = resp.json()
        assert data["success"] is False


# =============================================================================
# 统一响应格式
# =============================================================================


class TestApiResponseFormat:
    def test_success_response_structure(self, client):
        """成功响应包含 success/data/error 三个字段。"""
        resp = client.get("/api/v1/tasks/test-id/report")
        data = resp.json()
        assert "success" in data
        assert "data" in data
        assert "error" in data
        assert data["success"] is True
        assert data["error"] is None

    def test_error_response_structure(self, client, mock_gateway):
        """错误响应也包含 success/data/error 三个字段。"""
        mock_gateway.get_task_status.side_effect = KeyError("nonexistent")
        resp = client.get("/api/v1/tasks/nonexistent")
        data = resp.json()
        assert "success" in data
        assert "error" in data
        assert data["success"] is False


# =============================================================================
# CORS 头
# =============================================================================


class TestCORS:
    def test_cors_headers_allowed_origin(self):
        """预检请求返回正确的 CORS 头。"""
        test_app = FastAPI()
        test_app.add_middleware(
            CORSMiddleware,
            allow_origins=["http://localhost:5173"],
            allow_methods=["*"],
            allow_headers=["*"],
        )
        test_app.include_router(router)
        test_app.dependency_overrides[_get_gateway] = lambda: AsyncMock()
        client = TestClient(test_app)

        resp = client.options(
            "/api/v1/tasks",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert resp.headers.get("access-control-allow-origin") == "http://localhost:5173"

    def test_cors_headers_disallowed_origin(self):
        """未允许的来源不返回 CORS 头。"""
        test_app = FastAPI()
        test_app.add_middleware(
            CORSMiddleware,
            allow_origins=["http://localhost:5173"],
            allow_methods=["*"],
            allow_headers=["*"],
        )
        test_app.include_router(router)
        test_app.dependency_overrides[_get_gateway] = lambda: AsyncMock()
        client = TestClient(test_app)

        resp = client.options(
            "/api/v1/tasks",
            headers={
                "Origin": "http://evil.com",
                "Access-Control-Request-Method": "POST",
            },
        )
        cors_header = resp.headers.get("access-control-allow-origin")
        assert cors_header != "http://evil.com"


# =============================================================================
# 健康检查端点（来自 main.py）
# =============================================================================


class TestHealth:
    def test_health_endpoint(self):
        """GET /health 返回 ok。"""
        from codeagent.interaction.api.main import app as real_app

        real_app.dependency_overrides[_get_gateway] = lambda: AsyncMock()
        client = TestClient(real_app)
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert "inline_mode" in body
        assert "redis" in body
