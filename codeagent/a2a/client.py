"""Secure, cached outbound A2A client based on the official SDK."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator

import httpx
from google.protobuf.json_format import MessageToJson
from a2a.client import Client, ClientConfig, ClientFactory
from a2a.helpers.proto_helpers import get_artifact_text, get_message_text, new_text_message
from a2a.types import (
    CancelTaskRequest,
    Artifact,
    GetTaskRequest,
    Role,
    SendMessageConfiguration,
    SendMessageRequest,
    Task,
    TaskState,
)
from a2a.utils.constants import PROTOCOL_VERSION_1_0

from codeagent.a2a.security import validate_payload_size, validate_remote_url
from codeagent.config import (
    get_a2a_client_enabled,
    get_a2a_max_artifact_bytes,
    get_a2a_max_message_bytes,
    get_a2a_remote_api_key,
    get_a2a_timeout,
)
from codeagent.gateway.agent_gateway import AgentResult, IAgentGateway, RemoteAgent


class A2AClientGateway(IAgentGateway):
    def __init__(
        self,
        *,
        http_client: httpx.AsyncClient | None = None,
        cache_ttl: float = 300,
        streaming: bool = True,
    ) -> None:
        remote_key = get_a2a_remote_api_key()
        headers = {"Authorization": f"Bearer {remote_key}"} if remote_key else None
        self._http = http_client or httpx.AsyncClient(
            timeout=get_a2a_timeout(), follow_redirects=False, headers=headers
        )
        self._streaming = streaming
        self._factory = ClientFactory(ClientConfig(streaming=streaming, polling=True, httpx_client=self._http))
        self._cache_ttl = cache_ttl
        self._cache: dict[str, tuple[float, Client, RemoteAgent]] = {}

    def _assert_enabled(self) -> None:
        if not get_a2a_client_enabled():
            raise RuntimeError("A2A client is disabled")

    async def _resolve(self, url: str) -> tuple[Client, RemoteAgent]:
        self._assert_enabled()
        normalized = validate_remote_url(url)
        cached = self._cache.get(normalized)
        if cached and cached[0] > time.monotonic():
            return cached[1], cached[2]
        client = await self._factory.create_from_url(normalized)
        card = client._card  # type: ignore[attr-defined]  # Official client discovery card.
        if not card.version or not card.supported_interfaces:
            raise ValueError("Remote Agent Card is incomplete")
        if not any(item.protocol_version == PROTOCOL_VERSION_1_0 for item in card.supported_interfaces):
            raise ValueError("Remote agent does not support A2A protocol 1.0")
        remote = RemoteAgent(normalized, card.name, card.version, frozenset(skill.id for skill in card.skills))
        self._cache[normalized] = (time.monotonic() + self._cache_ttl, client, remote)
        return client, remote

    async def discover(self, url: str) -> RemoteAgent:
        _client, remote = await self._resolve(url)
        return remote

    async def stream(self, url: str, message: str, *, required_skill: str | None = None) -> AsyncIterator[AgentResult]:
        validate_payload_size(message, get_a2a_max_message_bytes(), "A2A message")
        client, remote = await self._resolve(url)
        if required_skill and required_skill not in remote.skills:
            raise ValueError(f"Remote agent does not advertise skill: {required_skill}")
        request = SendMessageRequest(
            message=new_text_message(message, role=Role.ROLE_USER),
            configuration=SendMessageConfiguration(return_immediately=not self._streaming),
        )
        async with asyncio.timeout(get_a2a_timeout()):
            async for response in client.send_message(request):
                kind = response.WhichOneof("payload")
                if kind == "task":
                    task = response.task
                    yield self._result(task)
                elif kind == "message":
                    yield AgentResult("", "message", get_message_text(response.message))
                elif kind == "status_update":
                    update = response.status_update
                    state = TaskState.Name(update.status.state).removeprefix("TASK_STATE_").lower()
                    text = get_message_text(update.status.message) if update.status.HasField("message") else ""
                    yield AgentResult(update.task_id, state, text)
                elif kind == "artifact_update":
                    update = response.artifact_update
                    self._validate_artifact(update.artifact)
                    text = get_artifact_text(update.artifact)
                    validate_payload_size(text, get_a2a_max_artifact_bytes(), "remote A2A artifact")
                    yield AgentResult(update.task_id, "working", artifacts=[{
                        "id": update.artifact.artifact_id,
                        "name": update.artifact.name,
                        "text": text,
                        "untrusted": True,
                    }])

    async def send(self, url: str, message: str, *, required_skill: str | None = None) -> AgentResult:
        result = AgentResult("", "unknown")
        async for result in self.stream(url, message, required_skill=required_skill):
            pass
        return result

    async def get(self, url: str, task_id: str) -> AgentResult:
        client, _remote = await self._resolve(url)
        return self._result(await client.get_task(GetTaskRequest(id=task_id)))

    async def cancel(self, url: str, task_id: str) -> AgentResult:
        client, _remote = await self._resolve(url)
        return self._result(await client.cancel_task(CancelTaskRequest(id=task_id)))

    @staticmethod
    def _result(task: Task) -> AgentResult:
        artifacts = []
        for artifact in task.artifacts:
            A2AClientGateway._validate_artifact(artifact)
            text = get_artifact_text(artifact)
            validate_payload_size(text, get_a2a_max_artifact_bytes(), "remote A2A artifact")
            artifacts.append({"id": artifact.artifact_id, "name": artifact.name, "text": text, "untrusted": True})
        text = get_message_text(task.status.message) if task.status.HasField("message") else ""
        state = TaskState.Name(task.status.state).removeprefix("TASK_STATE_").lower()
        return AgentResult(task.id, state, text, artifacts)

    @staticmethod
    def _validate_artifact(artifact: Artifact) -> None:
        maximum = get_a2a_max_artifact_bytes()
        for part in artifact.parts:
            kind = part.WhichOneof("content")
            if kind == "url":
                validate_remote_url(part.url)
            elif kind == "raw":
                validate_payload_size(part.raw, maximum, "remote A2A artifact")
            elif kind == "data":
                validate_payload_size(MessageToJson(part.data), maximum, "remote A2A artifact")
