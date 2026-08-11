"""Outbound agent-to-agent gateway contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator


@dataclass(frozen=True)
class RemoteAgent:
    url: str
    name: str
    version: str
    skills: frozenset[str]


@dataclass
class AgentResult:
    task_id: str
    state: str
    text: str = ""
    artifacts: list[dict[str, object]] = field(default_factory=list)


class IAgentGateway(ABC):
    @abstractmethod
    async def discover(self, url: str) -> RemoteAgent: ...

    @abstractmethod
    async def send(self, url: str, message: str, *, required_skill: str | None = None) -> AgentResult: ...

    @abstractmethod
    def stream(self, url: str, message: str, *, required_skill: str | None = None) -> AsyncIterator[AgentResult]: ...

    @abstractmethod
    async def get(self, url: str, task_id: str) -> AgentResult: ...

    @abstractmethod
    async def cancel(self, url: str, task_id: str) -> AgentResult: ...
