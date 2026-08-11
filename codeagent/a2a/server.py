"""Official A2A SDK adapter for the local orchestration gateway."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import asdict
from typing import Any, Awaitable, Callable

from a2a.helpers.proto_helpers import new_task_from_user_message, new_text_part
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import DatabaseTaskStore, TaskStore, TaskUpdater
from a2a.types import TaskState as A2ATaskState
from fastapi import FastAPI
from sqlalchemy.ext.asyncio import create_async_engine

from codeagent.a2a.card import build_agent_card
from codeagent.a2a.security import validate_payload_size
from codeagent.config import (
    get_a2a_max_artifact_bytes,
    get_a2a_max_delegation_depth,
    get_a2a_max_message_bytes,
    get_a2a_project_root,
    get_a2a_timeout,
    get_a2a_task_db_path,
)
from codeagent.gateway.orchestration_gateway import IOrchestrationGateway, TaskState, UserRequest

logger = logging.getLogger(__name__)
InlineStarter = Callable[[str, dict[str, Any]], Awaitable[None]]


class CodeAgentExecutor(AgentExecutor):
    """Maps A2A tasks to CodeAgent task IDs and mirrors terminal artifacts."""

    def __init__(self) -> None:
        self.gateway: IOrchestrationGateway | None = None
        self.inline_starter: InlineStarter | None = None
        self.task_map: dict[str, str] = {}
        self.idempotency_map: dict[str, str] = {}
        self._lock = asyncio.Lock()

    def configure(self, gateway: IOrchestrationGateway, inline_starter: InlineStarter | None = None) -> None:
        self.gateway = gateway
        self.inline_starter = inline_starter

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        if self.gateway is None:
            raise RuntimeError("A2A server is not initialized")
        external_id = context.task_id or ""
        context_id = context.context_id or external_id
        updater = TaskUpdater(event_queue, external_id, context_id)
        message = context.get_user_input()
        validate_payload_size(message, get_a2a_max_message_bytes(), "A2A message")
        depth = int(context.metadata.get("delegation_depth", 0) or 0)
        if depth > get_a2a_max_delegation_depth():
            raise ValueError("A2A delegation depth limit exceeded")
        logger.info("A2A execute trace=%s task=%s depth=%s", context_id, external_id, depth)
        idempotency_key = str(context.metadata.get("idempotency_key", "") or external_id)
        if context.current_task is None:
            if context.message is None:
                raise ValueError("A2A request must contain a message")
            await event_queue.enqueue_event(new_task_from_user_message(context.message))

        async with self._lock:
            internal_id = self.idempotency_map.get(idempotency_key)
            if internal_id is None:
                request = UserRequest(
                    query=message,
                    project_root=get_a2a_project_root(),
                    auto_mode=True,
                    response_mode="auto",
                    conversation_id=context_id,
                )
                try:
                    internal_id = await self.gateway.start_task(
                        request, skip_celery=self.inline_starter is not None  # type: ignore[call-arg]
                    )
                except TypeError:
                    internal_id = await self.gateway.start_task(request)
                self.idempotency_map[idempotency_key] = internal_id
                if self.inline_starter is not None:
                    await self.inline_starter(internal_id, asdict(request))
            self.task_map[external_id] = internal_id

        await updater.update_status(
            A2ATaskState.TASK_STATE_WORKING,
            metadata={"codeagent_task_id": internal_id},
        )
        try:
            await asyncio.wait_for(self._wait_for_result(internal_id, updater), timeout=get_a2a_timeout())
        except TimeoutError:
            await self.gateway.cancel_task(internal_id)
            await updater.failed(updater.new_agent_message([new_text_part("CodeAgent task deadline exceeded")]))

    async def _wait_for_result(self, internal_id: str, updater: TaskUpdater) -> None:
        assert self.gateway is not None
        previous: tuple[str, str | None] | None = None
        while True:
            status = await self.gateway.get_task_status(internal_id)
            marker = (status.state.value, status.current_step)
            if marker != previous and status.current_step:
                await updater.update_status(
                    A2ATaskState.TASK_STATE_WORKING,
                    updater.new_agent_message([new_text_part(status.current_step)]),
                    metadata={"codeagent_task_id": internal_id, "progress": status.progress},
                )
            previous = marker
            if status.state == TaskState.COMPLETED:
                report = await self.gateway.get_report(internal_id)
                artifact = json.dumps(asdict(report), ensure_ascii=False, default=str)
                validate_payload_size(artifact, get_a2a_max_artifact_bytes(), "A2A artifact")
                await updater.add_artifact(
                    [new_text_part(artifact, "application/json")],
                    name="codeagent-report.json",
                    metadata={"codeagent_task_id": internal_id, "untrusted": False},
                    last_chunk=True,
                )
                text = report.assistant_response or f"CodeAgent task {internal_id} completed"
                await updater.complete(updater.new_agent_message([new_text_part(text)]))
                return
            if status.state == TaskState.CANCELLED:
                await updater.cancel(updater.new_agent_message([new_text_part("CodeAgent task cancelled")]))
                return
            if status.state == TaskState.FAILED:
                await updater.failed(updater.new_agent_message([new_text_part("; ".join(status.errors) or "CodeAgent task failed")]))
                return
            await asyncio.sleep(0.1)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        external_id = context.task_id or ""
        updater = TaskUpdater(event_queue, external_id, context.context_id or external_id)
        internal_id = self.task_map.get(external_id)
        if self.gateway is not None and internal_id:
            await self.gateway.cancel_task(internal_id)
        await updater.cancel()


def install_a2a_routes(
    app: FastAPI,
    executor: CodeAgentExecutor,
    base_url: str | None = None,
    task_store: TaskStore | None = None,
) -> None:
    """Register the v1 JSON-RPC/SSE endpoint and Agent Card."""
    card = build_agent_card(base_url)
    if task_store is None:
        path = get_a2a_task_db_path().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        engine = create_async_engine(f"sqlite+aiosqlite:///{path.as_posix()}")
        task_store = DatabaseTaskStore(engine)
    handler = DefaultRequestHandler(agent_executor=executor, task_store=task_store, agent_card=card)
    for route in [
        *create_agent_card_routes(card),
        *create_jsonrpc_routes(handler, "/a2a"),
    ]:
        app.router.routes.append(route)
