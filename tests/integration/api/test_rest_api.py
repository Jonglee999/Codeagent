"""REST API 集成测试。

使用 fakeredis 模拟 Redis 后端，测试完整的 HTTP 请求-响应流程，
包括请求序列化、状态转换、错误处理和响应格式。

测试场景（≥15 tests）：
  Scene A — 完整任务生命周期（POST → GET polling → report）
  Scene B — Human Review 决策流程
  Scene C — 取消任务
  Scene D — 错误处理（404 / 422）
  Scene E — 健康检查与响应格式
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from codeagent.gateway.orchestration_gateway_impl import OrchestrationGatewayImpl


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def fake_redis():
    """Shared fakeredis instance — data persists across gateway calls."""
    import fakeredis.aioredis

    return fakeredis.aioredis.FakeRedis()


@pytest.fixture(autouse=True)
def _patch_redis(mocker, fake_redis):
    """All aioredis.from_url calls return the same fakeredis instance."""
    mocker.patch(
        "codeagent.gateway.orchestration_gateway_impl.aioredis.from_url",
        return_value=fake_redis,
    )
    mocker.patch(
        "codeagent.interaction.api.websocket.aioredis.from_url",
        return_value=fake_redis,
    )


@pytest.fixture(autouse=True)
def _patch_celery(mocker):
    """Replace Celery with MagicMock so no real broker connection."""
    mocker.patch(
        "codeagent.gateway.orchestration_gateway_impl.Celery",
        return_value=MagicMock(),
    )


@pytest.fixture
async def async_client(fake_redis):
    """httpx AsyncClient with gateway injection via dependency_overrides.

    Uses dependency_overrides to bypass ASGI scope[app] limitation in httpx.
    """
    from codeagent.interaction.api.main import app
    from codeagent.interaction.api.routes import _get_gateway

    gateway = OrchestrationGatewayImpl(redis_url="redis://localhost:6379")

    # Override the gateway dependency — avoids needing request.app.state
    app.dependency_overrides[_get_gateway] = lambda: gateway

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as c:
        yield c

    app.dependency_overrides.clear()


# =============================================================================
# Scene A: 完整任务生命周期
# =============================================================================


class TestFullLifecycle:
    """Scene A: POST → GET polling → report 的完整生命周期。"""

    @pytest.mark.asyncio
    async def test_create_task_returns_202_and_task_id(self, async_client):
        """POST /api/v1/tasks 返回 202 + task_id。"""
        resp = await async_client.post(
            "/api/v1/tasks",
            json={
                "query": "Add logging to main.py",
                "project_root": "/tmp/test",
            },
        )
        assert resp.status_code == 202
        data = resp.json()
        assert data["success"] is True
        assert isinstance(data["data"]["task_id"], str)
        assert len(data["data"]["task_id"]) > 0
        assert data["data"]["status"] == "pending"
        assert data["error"] is None

    @pytest.mark.asyncio
    async def test_create_task_with_auto_mode(self, async_client):
        """POST 支持 auto_mode 和 max_retries 参数。"""
        resp = await async_client.post(
            "/api/v1/tasks",
            json={
                "query": "Refactor utils",
                "project_root": "/tmp/p",
                "auto_mode": True,
                "max_retries": 5,
            },
        )
        assert resp.status_code == 202
        assert resp.json()["success"] is True

    @pytest.mark.asyncio
    async def test_task_initial_status_in_redis(self, async_client, fake_redis):
        """任务创建后 Redis 中初始状态为 pending。"""
        resp = await async_client.post(
            "/api/v1/tasks",
            json={"query": "fix bug", "project_root": "/tmp/test"},
        )
        task_id = resp.json()["data"]["task_id"]

        raw = await fake_redis.get(f"task:{task_id}:status")
        assert raw is not None
        status = json.loads(raw)
        assert status["state"] == "pending"
        assert status["progress"] == 0.0
        assert status["current_step"] is None

    @pytest.mark.asyncio
    async def test_get_task_status_pending(self, async_client, fake_redis):
        """GET /tasks/{id} 读取 Redis 中的 pending 状态。"""
        await fake_redis.setex(
            "task:status-pending:status",
            3600,
            json.dumps({
                "task_id": "status-pending",
                "state": "pending",
                "progress": 0.0,
                "current_step": None,
                "errors": [],
            }),
        )
        resp = await async_client.get("/api/v1/tasks/status-pending")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["data"]["state"] == "pending"
        assert data["data"]["progress"] == 0.0

    @pytest.mark.asyncio
    async def test_get_task_status_running(self, async_client, fake_redis):
        """GET /tasks/{id} 读取 running 状态。"""
        await fake_redis.setex(
            "task:status-running:status",
            3600,
            json.dumps({
                "task_id": "status-running",
                "state": "running",
                "progress": 0.5,
                "current_step": "Executing planning",
                "errors": [],
            }),
        )
        resp = await async_client.get("/api/v1/tasks/status-running")
        assert resp.status_code == 200
        data = resp.json()
        assert data["data"]["state"] == "running"
        assert data["data"]["progress"] == 0.5
        assert data["data"]["current_step"] == "Executing planning"

    @pytest.mark.asyncio
    async def test_get_task_status_completed(self, async_client, fake_redis):
        """GET /tasks/{id} 读取 completed 状态。"""
        await fake_redis.setex(
            "task:status-completed:status",
            3600,
            json.dumps({
                "task_id": "status-completed",
                "state": "completed",
                "progress": 1.0,
                "current_step": "Done",
                "errors": [],
            }),
        )
        resp = await async_client.get("/api/v1/tasks/status-completed")
        assert resp.status_code == 200
        data = resp.json()
        assert data["data"]["state"] == "completed"
        assert data["data"]["progress"] == 1.0
        assert data["data"]["current_step"] == "Done"

    @pytest.mark.asyncio
    async def test_get_report_after_completion(self, async_client, fake_redis):
        """任务完成后 GET /report 返回完整报告。"""
        await fake_redis.setex(
            "task:report-test:report",
            3600,
            json.dumps({
                "task_id": "report-test",
                "plan": [{"step": "analyze", "action": "read files"}],
                "changes": [{"file": "main.py", "action": "modify"}],
                "validation_results": [{"layer": "syntax", "passed": True}],
                "duration": 12.5,
                "token_usage": 1500,
            }),
        )
        resp = await async_client.get("/api/v1/tasks/report-test/report")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["data"]["duration"] == 12.5
        assert data["data"]["token_usage"] == 1500
        assert len(data["data"]["plan"]) == 1
        assert len(data["data"]["changes"]) == 1
        assert len(data["data"]["validation_results"]) == 1


# =============================================================================
# Scene B: Human Review 流程
# =============================================================================


class TestHumanReviewFlow:
    """Scene B: 通过 API 提交 Human Review 决策。"""

    @pytest.mark.asyncio
    async def test_submit_approve(self, async_client):
        """POST /tasks/{id}/decision 提交 approve。"""
        resp = await async_client.post(
            "/api/v1/tasks/t-1/decision",
            json={"decision": "approve"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["data"]["decision"] == "approve"

    @pytest.mark.asyncio
    async def test_submit_reject(self, async_client):
        """POST /tasks/{id}/decision 提交 reject。"""
        resp = await async_client.post(
            "/api/v1/tasks/t-1/decision",
            json={"decision": "reject"},
        )
        assert resp.status_code == 200
        assert resp.json()["success"] is True

    @pytest.mark.asyncio
    async def test_submit_modify_with_modifications(self, async_client, fake_redis):
        """POST /tasks/{id}/decision 提交 modify + modifications，验证 Redis 存储。"""
        resp = await async_client.post(
            "/api/v1/tasks/t-modify/decision",
            json={
                "decision": "modify",
                "modifications": {"feedback": "use safer approach"},
            },
        )
        assert resp.status_code == 200

        # 验证 Redis 中存储的 decision 内容
        raw = await fake_redis.get("task:t-modify:decision")
        assert raw is not None
        stored = json.loads(raw)
        assert stored["decision"] == "modify"
        assert stored["modifications"] == {"feedback": "use safer approach"}

    @pytest.mark.asyncio
    async def test_submit_decision_persists_to_redis(self, async_client, fake_redis):
        """decision 提交后持久化到 Redis，含 task_id。"""
        resp = await async_client.post(
            "/api/v1/tasks/t-persist/decision",
            json={"decision": "approve"},
        )
        assert resp.status_code == 200

        raw = await fake_redis.get("task:t-persist:decision")
        assert raw is not None
        stored = json.loads(raw)
        assert stored["task_id"] == "t-persist"
        assert stored["decision"] == "approve"


# =============================================================================
# Scene C: 取消任务
# =============================================================================


class TestCancelTask:
    """Scene C: 取消任务流程。"""

    @pytest.mark.asyncio
    async def test_cancel_existing_task(self, async_client, fake_redis):
        """DELETE /tasks/{id} 取消存在的任务返回 cancelled=True。"""
        await fake_redis.setex(
            "task:cancel-me:status",
            3600,
            json.dumps({
                "task_id": "cancel-me",
                "state": "running",
                "progress": 0.3,
                "current_step": "executing",
                "errors": [],
            }),
        )
        resp = await async_client.delete("/api/v1/tasks/cancel-me")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["data"]["cancelled"] is True

    @pytest.mark.asyncio
    async def test_cancel_nonexistent_returns_false(self, async_client):
        """DELETE /tasks/{id} 取消不存在的任务返回 cancelled=False。"""
        resp = await async_client.delete("/api/v1/tasks/nonexistent")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["data"]["cancelled"] is False

    @pytest.mark.asyncio
    async def test_cancel_updates_redis_to_cancelled(self, async_client, fake_redis):
        """取消任务后 Redis 中状态为 CANCELLED。"""
        await fake_redis.setex(
            "task:status-check:status",
            3600,
            json.dumps({
                "task_id": "status-check",
                "state": "running",
                "progress": 0.3,
                "errors": [],
            }),
        )
        resp = await async_client.delete("/api/v1/tasks/status-check")
        assert resp.status_code == 200

        raw = await fake_redis.get("task:status-check:status")
        assert raw is not None
        updated = json.loads(raw)
        assert updated["state"] == "cancelled"
        assert updated["progress"] == 0.3  # original progress preserved


# =============================================================================
# Scene D: 错误处理
# =============================================================================


class TestErrorHandling:
    """Scene D: 404 / 422 等错误处理。"""

    @pytest.mark.asyncio
    async def test_get_nonexistent_task_404(self, async_client):
        """不存在的 task_id 返回 404。"""
        resp = await async_client.get("/api/v1/tasks/__missing__")
        assert resp.status_code == 404
        data = resp.json()
        assert data["success"] is False
        assert "not found" in data["error"].lower()

    @pytest.mark.asyncio
    async def test_get_nonexistent_report_404(self, async_client):
        """不存在的报告返回 404。"""
        resp = await async_client.get("/api/v1/tasks/__missing__/report")
        assert resp.status_code == 404
        data = resp.json()
        assert data["success"] is False
        assert "not found" in data["error"].lower()

    @pytest.mark.asyncio
    async def test_create_missing_query_422(self, async_client):
        """缺少 query 字段返回 422。"""
        resp = await async_client.post(
            "/api/v1/tasks",
            json={"project_root": "/tmp/test"},
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_create_empty_query_422(self, async_client):
        """query 为空字符串返回 422。"""
        resp = await async_client.post(
            "/api/v1/tasks",
            json={"query": "", "project_root": "/tmp/test"},
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_create_missing_project_root_422(self, async_client):
        """缺少 project_root 返回 422。"""
        resp = await async_client.post(
            "/api/v1/tasks",
            json={"query": "add logging"},
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_submit_invalid_decision_422(self, async_client):
        """无效 decision 值返回 422。"""
        resp = await async_client.post(
            "/api/v1/tasks/t-1/decision",
            json={"decision": "invalid_option"},
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_submit_decision_missing_field_422(self, async_client):
        """缺少 decision 字段返回 422。"""
        resp = await async_client.post(
            "/api/v1/tasks/t-1/decision",
            json={},
        )
        assert resp.status_code == 422


# =============================================================================
# Scene E: 健康检查与响应格式
# =============================================================================


class TestHealthEndpoint:
    """健康检查端点验证。"""

    @pytest.mark.asyncio
    async def test_health_returns_ok(self, async_client):
        """GET /health 返回 ok 状态。"""
        resp = await async_client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"

    @pytest.mark.asyncio
    async def test_health_contains_redis_info(self, async_client):
        """GET /health 包含 Redis 连接信息。"""
        resp = await async_client.get("/health")
        body = resp.json()
        assert "redis" in body


class TestResponseFormat:
    """统一响应格式验证。"""

    @pytest.mark.asyncio
    async def test_success_has_three_fields(self, async_client):
        """成功响应含 success/data/error 字段，error 为 null。"""
        resp = await async_client.get("/api/v1/tasks/t-1/report")
        data = resp.json()
        assert "success" in data
        assert "data" in data
        assert "error" in data

    @pytest.mark.asyncio
    async def test_error_response_structure(self, async_client):
        """错误响应含 success/data/error，success 为 false。"""
        resp = await async_client.get("/api/v1/tasks/__missing__")
        data = resp.json()
        assert "success" in data
        assert "error" in data
        assert data["success"] is False
