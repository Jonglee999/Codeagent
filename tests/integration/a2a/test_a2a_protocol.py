from __future__ import annotations

from dataclasses import dataclass
import asyncio
import time

import httpx
import pytest
from fastapi import FastAPI
from a2a.server.tasks import InMemoryTaskStore

from codeagent.a2a.client import A2AClientGateway
from codeagent.a2a.server import CodeAgentExecutor, install_a2a_routes
from codeagent.gateway.orchestration_gateway import TaskReport, TaskState, TaskStatus


@dataclass
class FakeGateway:
    starts: int = 0
    cancelled: bool = False
    pending: bool = False

    async def start_task(self, request, skip_celery=False):
        self.starts += 1
        self.request = request
        return "internal-1"

    async def get_task_status(self, task_id):
        state = TaskState.CANCELLED if self.cancelled else (TaskState.RUNNING if self.pending else TaskState.COMPLETED)
        return TaskStatus(task_id, state, 1.0, "done")

    async def get_report(self, task_id):
        return TaskReport(task_id, [], [], ["pytest passed"], 0.1, 12, assistant_response="finished")

    async def cancel_task(self, task_id):
        self.cancelled = True
        return True


@pytest.fixture
async def local_agent(monkeypatch):
    monkeypatch.setenv("A2A_CLIENT_ENABLED", "true")
    monkeypatch.setenv("A2A_ALLOWLIST", "localhost")
    app = FastAPI()
    executor = CodeAgentExecutor()
    gateway = FakeGateway()
    executor.configure(gateway)  # type: ignore[arg-type]
    install_a2a_routes(app, executor, "http://localhost", InMemoryTaskStore())
    http = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://localhost")
    client = A2AClientGateway(http_client=http, cache_ttl=30)
    try:
        yield client, gateway
    finally:
        await http.aclose()


async def test_card_discovery_capability_matching_and_stream(local_agent):
    client, gateway = local_agent
    remote = await client.discover("http://localhost")
    assert remote.version == "0.1.0"
    assert "implement_change" in remote.skills

    updates = [item async for item in client.stream(
        "http://localhost", "inspect this repository", required_skill="inspect_repository"
    )]
    assert updates[-1].state == "completed"
    assert updates[-1].text == "finished"
    assert any(item.artifacts for item in updates)
    assert gateway.starts == 1
    persisted = await client.get("http://localhost", updates[-1].task_id)
    assert persisted.state == "completed"


async def test_discovery_is_cached_and_skill_mismatch_fails(local_agent):
    client, _gateway = local_agent
    first = await client.discover("http://localhost")
    second = await client.discover("http://localhost")
    assert first is second
    with pytest.raises(ValueError, match="does not advertise"):
        await client.send("http://localhost", "do it", required_skill="deploy_production")


async def test_cancel_active_stream_and_persist_cancelled_state(local_agent):
    _streaming_client, gateway = local_agent
    gateway.pending = True
    app = FastAPI()
    executor = CodeAgentExecutor()
    executor.configure(gateway)  # type: ignore[arg-type]
    install_a2a_routes(app, executor, "http://localhost", InMemoryTaskStore())
    http = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://localhost")
    client = A2AClientGateway(http_client=http, streaming=False)
    first = await client.send("http://localhost", "long task")
    assert first.task_id
    cancelled = await client.cancel("http://localhost", first.task_id)
    assert cancelled.state == "canceled"
    persisted = await client.get("http://localhost", first.task_id)
    assert persisted.state == "canceled"
    await http.aclose()


@pytest.mark.benchmark
async def test_twenty_concurrent_a2a_streams_complete(local_agent):
    client, gateway = local_agent

    async def run(index: int):
        return await client.send("http://localhost", f"task {index}")

    started = time.perf_counter()
    results = await asyncio.gather(*(run(index) for index in range(20)))
    elapsed = time.perf_counter() - started
    assert all(result.state == "completed" for result in results)
    assert gateway.starts == 20
    assert elapsed < 5


async def test_task_survives_server_recreation_for_sse_reconnect(monkeypatch, tmp_path):
    monkeypatch.setenv("A2A_CLIENT_ENABLED", "true")
    monkeypatch.setenv("A2A_ALLOWLIST", "localhost")
    monkeypatch.setenv("A2A_TASK_DB_PATH", str(tmp_path / "a2a.db"))

    first_app = FastAPI()
    first_executor = CodeAgentExecutor()
    first_executor.configure(FakeGateway())  # type: ignore[arg-type]
    install_a2a_routes(first_app, first_executor, "http://localhost")
    first_http = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=first_app), base_url="http://localhost"
    )
    first_client = A2AClientGateway(http_client=first_http)
    completed = await first_client.send("http://localhost", "persist this task")
    await first_http.aclose()

    second_app = FastAPI()
    install_a2a_routes(second_app, CodeAgentExecutor(), "http://localhost")
    second_http = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=second_app), base_url="http://localhost"
    )
    second_client = A2AClientGateway(http_client=second_http)
    restored = await second_client.get("http://localhost", completed.task_id)
    assert restored.state == "completed"
    assert restored.artifacts
    await second_http.aclose()
