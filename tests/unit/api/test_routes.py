"""API 路由单元测试。

使用 FastAPI TestClient + Mock Gateway 测试 5 个端点。
覆盖：正常路径、异常路径、统一响应格式、404 处理、CORS 头。
"""

from __future__ import annotations

from pathlib import Path
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
from codeagent.interaction.api import routes
from codeagent.product_state import ProductStateStore


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
    gw.steer_task.return_value = True

    # get_report
    gw.get_report.return_value = TaskReport(
        task_id="test-task-id-123",
        plan=[{"step": "analyze", "action": "read files"}],
        changes=[{"file": "main.py", "action": "modify"}],
        validation_results=[{"layer": "syntax", "passed": True}],
        duration=12.5,
        token_usage=1500,
        benchmark_instance_id="owner__repo-1",
        model_runtime={"active_model": "test/model", "retry_count": 1},
        infrastructure_runtime={
            "redis": {"recovery_count": 1},
            "tools": {"timeout_count": 1},
        },
    )

    # submit_decision
    gw.submit_decision.return_value = None

    return gw


@pytest.fixture
def state_store(tmp_path):
    return ProductStateStore(tmp_path / "product.sqlite3")


@pytest.fixture
def app(mock_gateway, state_store, monkeypatch):
    """Create a test FastAPI app with mocked gateway."""
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[_get_gateway] = lambda: mock_gateway
    monkeypatch.setattr(routes, "_product_state_store", state_store)
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

    def test_accepts_long_official_benchmark_statement(self, client, mock_gateway):
        mock_gateway.start_task.return_value = "benchmark-task"

        resp = client.post(
            "/api/v1/tasks",
            json={"query": "x" * 15_000, "project_root": "/tmp/test"},
        )

        assert resp.status_code == 202

    def test_missing_project_root_creates_managed_workspace(
        self, client, mock_gateway, monkeypatch, tmp_path,
    ):
        from codeagent.workspaces import WorkspaceManager

        monkeypatch.setattr(
            "codeagent.interaction.api.routes._workspace_manager",
            WorkspaceManager(tmp_path),
        )
        resp = client.post(
            "/api/v1/tasks",
            json={"query": "Add logging"},
        )
        assert resp.status_code == 202
        workspace = Path(resp.json()["data"]["project_root"])
        assert workspace.is_dir()
        workspace.relative_to(tmp_path / ".codeagent" / "workspaces" / "sessions")
        request = mock_gateway.start_task.await_args.args[0]
        assert request.project_root == str(workspace)

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

    def test_follow_up_uses_durable_history_and_removes_current_turn_duplicate(
        self, client, mock_gateway, state_store, tmp_path,
    ):
        state_store.record_run(
            "task-first",
            conversation_id="conversation-1",
            query="Create random_generator.py",
            workspace_root=str(tmp_path),
        )
        state_store.save_report("task-first", {"assistant_response": "Created it. Output: 50"})

        response = client.post(
            "/api/v1/tasks",
            json={
                "query": "Run it again",
                "project_root": str(tmp_path),
                "conversation_id": "conversation-1",
                "conversation_history": [
                    {"role": "user", "content": "stale client copy"},
                    {"role": "user", "content": "Run it again"},
                ],
            },
        )

        assert response.status_code == 202
        request = mock_gateway.start_task.await_args.args[0]
        assert request.conversation_history == [
            {"role": "user", "content": "Create random_generator.py"},
            {"role": "assistant", "content": "Created it. Output: 50"},
        ]


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

    def test_falls_back_to_durable_status(
        self, client, mock_gateway, state_store, tmp_path,
    ):
        state_store.record_run(
            "durable-task",
            conversation_id="conversation-1",
            query="Fix it",
            workspace_root=str(tmp_path),
        )
        state_store.update_run_state("durable-task", "completed")
        mock_gateway.get_task_status.side_effect = KeyError("expired")

        response = client.get("/api/v1/tasks/durable-task")

        assert response.status_code == 200
        assert response.json()["data"]["state"] == "completed"
        assert response.json()["data"]["progress"] == 1.0

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

    def test_restarts_persisted_inline_checkpoint(
        self, client, app, state_store, tmp_path,
    ):
        resume = AsyncMock()
        app.state.inline_tasks = {}
        app.state.resume_inline_checkpoint = resume
        app.state.redis_url = "redis://test"
        state_store.record_run(
            "task-resume",
            conversation_id="conversation-resume",
            query="Continue after review",
            workspace_root=str(tmp_path),
        )
        state_store.update_run_state("task-resume", "running")

        response = client.post(
            "/api/v1/tasks/task-resume/decision",
            json={"decision": "approve"},
        )

        assert response.status_code == 200
        assert response.json()["data"]["checkpoint_resume_started"] is True


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


class TestSteerTask:
    def test_queues_runtime_instruction(self, client, mock_gateway):
        response = client.post(
            "/api/v1/tasks/task-1/steer",
            json={"instruction": "Keep the public API compatible"},
        )
        assert response.status_code == 200
        assert response.json()["data"]["queued"] is True
        mock_gateway.steer_task.assert_awaited_once_with(
            "task-1", "Keep the public API compatible"
        )

    def test_rejects_empty_runtime_instruction(self, client):
        response = client.post(
            "/api/v1/tasks/task-1/steer", json={"instruction": ""}
        )
        assert response.status_code == 422


class TestRecoverTask:
    def test_starts_new_run_in_existing_workspace(
        self, client, mock_gateway, state_store, tmp_path,
    ):
        state_store.record_run(
            "task-source",
            conversation_id="conversation-1",
            query="Finish the parser",
            workspace_root=str(tmp_path),
        )
        state_store.update_run_state("task-source", "cancelled")
        mock_gateway.start_task.return_value = "task-recovery"

        response = client.post(
            "/api/v1/tasks/task-source/recover",
            json={"instruction": "Continue with the tests", "auto_mode": True},
        )

        assert response.status_code == 202
        assert response.json()["data"]["task_id"] == "task-recovery"
        assert response.json()["data"]["recovered_from_task_id"] == "task-source"
        recovered = mock_gateway.start_task.await_args.args[0]
        assert recovered.project_root == str(tmp_path.resolve())
        assert recovered.conversation_id == "conversation-1"
        assert recovered.response_mode == "execute"
        assert "Original goal: Finish the parser" in recovered.query
        assert "Recovery instruction: Continue with the tests" in recovered.query
        assert recovered.recovered_from_task_id == "task-source"
        assert state_store.get_run("task-recovery")["recovered_from_task_id"] == "task-source"

    def test_rejects_recovery_of_active_or_successful_run(
        self, client, state_store, tmp_path,
    ):
        state_store.record_run(
            "task-source",
            conversation_id="conversation-1",
            query="Finish the parser",
            workspace_root=str(tmp_path),
        )
        state_store.update_run_state("task-source", "completed")

        response = client.post("/api/v1/tasks/task-source/recover", json={})

        assert response.status_code == 409

    def test_rejects_recovery_when_workspace_is_missing(
        self, client, state_store, tmp_path,
    ):
        missing = tmp_path / "removed-workspace"
        state_store.record_run(
            "task-source",
            conversation_id="conversation-1",
            query="Finish the parser",
            workspace_root=str(missing),
        )
        state_store.update_run_state("task-source", "failed")

        response = client.post("/api/v1/tasks/task-source/recover", json={})

        assert response.status_code == 409


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
        assert data["data"]["model_runtime"]["active_model"] == "test/model"
        assert data["data"]["infrastructure_runtime"]["redis"]["recovery_count"] == 1
        assert data["data"]["benchmark_instance_id"] == "owner__repo-1"

    def test_not_found(self, client, mock_gateway):
        """GET /tasks/{id}/report 不存在的报告返回 404。"""
        mock_gateway.get_report.side_effect = KeyError("nonexistent")
        resp = client.get("/api/v1/tasks/nonexistent/report")
        assert resp.status_code == 404
        data = resp.json()
        assert data["success"] is False
        assert "not found" in data["error"].lower()

    def test_falls_back_to_durable_report(
        self, client, mock_gateway, state_store, tmp_path,
    ):
        state_store.record_run(
            "durable-task",
            conversation_id="conversation-1",
            query="Fix it",
            workspace_root=str(tmp_path),
        )
        report = TaskReport(
            task_id="durable-task",
            plan=[],
            changes=[],
            validation_results=[],
            duration=1.0,
            token_usage=10,
            assistant_response="Durable answer",
        )
        state_store.save_report("durable-task", report.__dict__)
        mock_gateway.get_report.side_effect = KeyError("expired")

        response = client.get("/api/v1/tasks/durable-task/report")

        assert response.status_code == 200
        assert response.json()["data"]["assistant_response"] == "Durable answer"

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


class TestSystemCapabilities:
    def test_does_not_connect_to_mcp_during_page_load(
        self, client, monkeypatch, tmp_path,
    ) -> None:
        probe = AsyncMock(side_effect=AssertionError("MCP must stay cold"))
        monkeypatch.setattr(routes.ToolGateway, "initialize_extensions", probe)

        response = client.get(
            "/api/v1/system/capabilities",
            params={"project_root": str(tmp_path)},
        )

        assert response.status_code == 200
        assert response.json()["success"] is True
        probe.assert_not_awaited()


# =============================================================================
# 任务工作区文件浏览
# =============================================================================


class TestTaskFiles:
    def _seed_workspace(self, tmp_path: Path) -> None:
        (tmp_path / "main.py").write_text("def hello():\n    return 1\n", encoding="utf-8")
        (tmp_path / "notes.md").write_text("# notes\n", encoding="utf-8")
        (tmp_path / "logo.png").write_bytes(b"\x89PNG\x00\x00\x00\x00binary")
        (tmp_path / ".env").write_text("SECRET=1\n", encoding="utf-8")
        node_modules = tmp_path / "node_modules" / "pkg"
        node_modules.mkdir(parents=True)
        (node_modules / "index.js").write_text("module.exports = 1\n", encoding="utf-8")
        hidden_dir = tmp_path / ".codeagent"
        hidden_dir.mkdir()
        (hidden_dir / "report.json").write_text("{}", encoding="utf-8")

    def test_lists_files_with_binary_markers(
        self, client, state_store, tmp_path,
    ):
        state_store.record_run(
            "file-task",
            conversation_id="conversation-1",
            query="Create a generator",
            workspace_root=str(tmp_path),
        )
        self._seed_workspace(tmp_path)

        resp = client.get("/api/v1/tasks/file-task/files")

        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        files = {item["path"]: item for item in data["data"]["files"]}
        assert "main.py" in files
        assert files["main.py"]["binary"] is False
        assert "logo.png" in files
        assert files["logo.png"]["binary"] is True
        # 隐藏文件与依赖目录被跳过
        assert ".env" not in files
        assert "node_modules/pkg/index.js" not in files
        assert ".codeagent/report.json" not in files

    def test_file_list_unknown_task_returns_404(self, client):
        resp = client.get("/api/v1/tasks/missing-task/files")
        assert resp.status_code == 404

    def test_serves_text_file_content(
        self, client, state_store, tmp_path,
    ):
        state_store.record_run(
            "file-task",
            conversation_id="conversation-1",
            query="Create a generator",
            workspace_root=str(tmp_path),
        )
        (tmp_path / "main.py").write_text("def hello():\n    return 1\n", encoding="utf-8")

        resp = client.get("/api/v1/tasks/file-task/files/main.py")

        assert resp.status_code == 200
        assert "def hello():" in resp.text

    def test_rejects_path_escape(
        self, client, state_store, tmp_path,
    ):
        # Literal "../" is normalized away by the HTTP client before routing, so use the
        # URL-encoded form to exercise the server-side traversal guard.
        state_store.record_run(
            "file-task",
            conversation_id="conversation-1",
            query="Create a generator",
            workspace_root=str(tmp_path),
        )
        resp = client.get("/api/v1/tasks/file-task/files/%2e%2e%2foutside.txt")
        assert resp.status_code == 400

    def test_rejects_hidden_file(
        self, client, state_store, tmp_path,
    ):
        state_store.record_run(
            "file-task",
            conversation_id="conversation-1",
            query="Create a generator",
            workspace_root=str(tmp_path),
        )
        (tmp_path / ".env").write_text("SECRET=1\n", encoding="utf-8")
        resp = client.get("/api/v1/tasks/file-task/files/.env")
        assert resp.status_code == 400
