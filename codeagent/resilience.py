"""Shared asynchronous circuit-breaker primitives."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Callable, Literal

CircuitState = Literal["closed", "open", "half_open"]


class CircuitOpenError(RuntimeError):
    pass


@dataclass(frozen=True)
class CircuitTransition:
    previous: CircuitState
    current: CircuitState
    reason: str


class CircuitBreaker:
    """Small async closed/open/half-open state machine with one probe."""

    def __init__(
        self,
        failure_threshold: int = 3,
        cooldown_seconds: float = 30,
        *,
        label: str = "circuit",
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.failure_threshold = max(1, failure_threshold)
        self.cooldown_seconds = max(0.0, cooldown_seconds)
        self.label = label
        self.state: CircuitState = "closed"
        self.failure_count = 0
        self.opened_at: float | None = None
        self._probe_in_flight = False
        self._lock = asyncio.Lock()
        self._clock = clock

    async def before_call(self) -> CircuitTransition | None:
        async with self._lock:
            if self.state == "open":
                elapsed = self._clock() - (self.opened_at or 0)
                if elapsed < self.cooldown_seconds:
                    raise CircuitOpenError(
                        f"{self.label} circuit open; retry after "
                        f"{self.cooldown_seconds - elapsed:.1f}s"
                    )
                if self._probe_in_flight:
                    raise CircuitOpenError(f"{self.label} half-open probe already in flight")
                previous = self.state
                self.state = "half_open"
                self._probe_in_flight = True
                return CircuitTransition(previous, self.state, "cooldown elapsed")
            if self.state == "half_open" and self._probe_in_flight:
                raise CircuitOpenError(f"{self.label} half-open probe already in flight")
            return None

    async def record_success(self) -> CircuitTransition | None:
        async with self._lock:
            previous = self.state
            self.failure_count = 0
            self.opened_at = None
            self._probe_in_flight = False
            self.state = "closed"
            if previous != self.state:
                return CircuitTransition(previous, self.state, "probe succeeded")
            return None

    async def record_failure(self, reason: str) -> CircuitTransition | None:
        async with self._lock:
            previous = self.state
            self.failure_count += 1
            self._probe_in_flight = False
            if self.state == "half_open" or self.failure_count >= self.failure_threshold:
                self.state = "open"
                self.opened_at = self._clock()
            if previous != self.state:
                return CircuitTransition(previous, self.state, reason)
            return None

    def snapshot(self) -> dict[str, object]:
        return {
            "state": self.state,
            "failure_count": self.failure_count,
            "failure_threshold": self.failure_threshold,
            "cooldown_seconds": self.cooldown_seconds,
        }
