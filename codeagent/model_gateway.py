"""Unified model calls with classification, retries, circuits, and fallback."""

from __future__ import annotations

import asyncio
import inspect
import logging
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Awaitable, Callable, Literal

from codeagent.config import get_env
from codeagent.interaction.api.metrics import observe_llm_call
from codeagent.resilience import CircuitBreaker, CircuitOpenError, CircuitTransition

logger = logging.getLogger(__name__)

ModelErrorKind = Literal[
    "authentication",
    "rate_limit",
    "timeout",
    "connection",
    "provider_5xx",
    "context_limit",
    "content_policy",
    "invalid_request",
    "circuit_open",
    "unknown",
]

EventCallback = Callable[[dict[str, Any]], Any]
ProviderCall = Callable[..., Awaitable[Any]]
SleepCall = Callable[[float], Awaitable[None]]


@dataclass(frozen=True)
class ClassifiedModelError:
    kind: ModelErrorKind
    retryable: bool
    fallback_allowed: bool
    status_code: int | None = None
    retry_after_seconds: float | None = None


@dataclass(frozen=True)
class ModelEndpoint:
    role: Literal["primary", "fallback"]
    model: str
    api_key: str = field(default="", repr=False)
    api_base: str = ""
    timeout_seconds: float = 60.0

    def public_metadata(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "model": self.model,
            "configured": bool(self.model and self.api_key),
            "custom_api_base": bool(self.api_base),
            "timeout_seconds": self.timeout_seconds,
        }


@dataclass(frozen=True)
class ResiliencePolicy:
    max_retries: int = 2
    retry_base_seconds: float = 0.75
    retry_max_seconds: float = 8.0
    retry_jitter_seconds: float = 0.25
    retry_after_cap_seconds: float = 300.0
    failure_threshold: int = 3
    cooldown_seconds: float = 30.0


class ModelGatewayError(RuntimeError):
    def __init__(self, endpoint: ModelEndpoint, classified: ClassifiedModelError) -> None:
        self.endpoint = endpoint
        self.classified = classified
        status = f" (HTTP {classified.status_code})" if classified.status_code else ""
        super().__init__(f"Model {endpoint.role} failed: {classified.kind}{status}")


def _status_code(exc: BaseException) -> int | None:
    raw = getattr(exc, "status_code", None)
    if raw is None:
        response = getattr(exc, "response", None)
        raw = getattr(response, "status_code", None)
    try:
        return int(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def _headers(exc: BaseException) -> dict[str, str]:
    raw = getattr(exc, "headers", None)
    if raw is None:
        raw = getattr(getattr(exc, "response", None), "headers", None)
    if raw is None:
        return {}
    try:
        return {str(key).lower(): str(value) for key, value in raw.items()}
    except (AttributeError, TypeError, ValueError):
        return {}


def _retry_after(exc: BaseException) -> float | None:
    value = _headers(exc).get("retry-after")
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            target = parsedate_to_datetime(value)
            if target.tzinfo is None:
                target = target.replace(tzinfo=timezone.utc)
            return max(0.0, (target - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return None


def classify_model_error(exc: BaseException) -> ClassifiedModelError:
    """Classify provider errors without depending on one SDK's exception classes."""
    status = _status_code(exc)
    text = f"{type(exc).__name__}: {exc}".lower()
    retry_after = _retry_after(exc)

    if status in {401, 403} or any(token in text for token in ("authentication", "api key", "unauthorized", "forbidden")):
        return ClassifiedModelError("authentication", False, True, status)
    if status == 429 or "rate limit" in text or "ratelimit" in text:
        return ClassifiedModelError("rate_limit", True, True, status, retry_after)
    if status in {408, 504} or isinstance(exc, (asyncio.TimeoutError, TimeoutError)) or "timeout" in text or "timed out" in text:
        return ClassifiedModelError("timeout", True, True, status)
    if status is not None and 500 <= status <= 599:
        return ClassifiedModelError("provider_5xx", True, True, status, retry_after)
    if any(token in text for token in ("connection", "connecterror", "connection refused", "dns", "unavailable")):
        return ClassifiedModelError("connection", True, True, status)
    if any(token in text for token in ("context length", "context window", "maximum context", "too many tokens")):
        return ClassifiedModelError("context_limit", False, False, status)
    if any(token in text for token in ("content policy", "content filter", "safety policy", "moderation")):
        return ClassifiedModelError("content_policy", False, False, status)
    if status is not None and 400 <= status <= 499:
        return ClassifiedModelError("invalid_request", False, False, status)
    return ClassifiedModelError("unknown", False, False, status)


class ModelGateway:
    """Task-scoped gateway shared by chat, planning, execution, memory, and reflection."""

    def __init__(
        self,
        primary: ModelEndpoint,
        fallback: ModelEndpoint | None = None,
        *,
        policy: ResiliencePolicy | None = None,
        provider_call: ProviderCall | None = None,
        event_callback: EventCallback | None = None,
        sleep: SleepCall = asyncio.sleep,
        random_uniform: Callable[[float, float], float] = random.uniform,
    ) -> None:
        self.primary = primary
        self.fallback = fallback if fallback and fallback.model else None
        self.policy = policy or ResiliencePolicy()
        self._provider_call = provider_call or self._default_provider_call
        self._event_callback = event_callback
        self._sleep = sleep
        self._random_uniform = random_uniform
        self._circuits = {
            primary.role: self._new_circuit(primary),
        }
        if self.fallback:
            self._circuits[self.fallback.role] = self._new_circuit(self.fallback)
        self._attempt_count = 0
        self._retry_count = 0
        self._fallback_count = 0
        self._successful_calls = 0
        self._context_reduction_count = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._active_model = primary.model
        self._last_failure: dict[str, Any] | None = None
        self._fallback_reason: str | None = None

    @classmethod
    def from_env(
        cls,
        *,
        event_callback: EventCallback | None = None,
        provider_call: ProviderCall | None = None,
    ) -> "ModelGateway":
        timeout = float(get_env("LLM_TIMEOUT", "300"))
        primary_key = get_env("LLM_API_KEY", "")
        primary_base = get_env("LLM_API_BASE", "")
        primary = ModelEndpoint(
            role="primary",
            model=get_env("LLM_MODEL", "deepseek/deepseek-v4-flash"),
            api_key=primary_key,
            api_base=primary_base,
            timeout_seconds=timeout,
        )
        fallback_model = get_env("LLM_FALLBACK_MODEL", "").strip()
        fallback = None
        if fallback_model:
            fallback = ModelEndpoint(
                role="fallback",
                model=fallback_model,
                api_key=get_env("LLM_FALLBACK_API_KEY", "") or primary_key,
                api_base=get_env("LLM_FALLBACK_API_BASE", "") or primary_base,
                timeout_seconds=float(get_env("LLM_FALLBACK_TIMEOUT", str(timeout))),
            )
        policy = ResiliencePolicy(
            max_retries=max(0, int(get_env("LLM_MAX_RETRIES", "1"))),
            retry_base_seconds=max(0.0, float(get_env("LLM_RETRY_BASE_SECONDS", "0.75"))),
            retry_max_seconds=max(0.0, float(get_env("LLM_RETRY_MAX_SECONDS", "8"))),
            retry_jitter_seconds=max(0.0, float(get_env("LLM_RETRY_JITTER_SECONDS", "0.25"))),
            retry_after_cap_seconds=max(0.0, float(get_env("LLM_RETRY_AFTER_CAP_SECONDS", "300"))),
            failure_threshold=max(1, int(get_env("LLM_CIRCUIT_FAILURE_THRESHOLD", "3"))),
            cooldown_seconds=max(0.0, float(get_env("LLM_CIRCUIT_COOLDOWN_SECONDS", "30"))),
        )
        return cls(
            primary,
            fallback,
            policy=policy,
            provider_call=provider_call,
            event_callback=event_callback,
        )

    def _new_circuit(self, endpoint: ModelEndpoint) -> CircuitBreaker:
        return CircuitBreaker(
            failure_threshold=self.policy.failure_threshold,
            cooldown_seconds=self.policy.cooldown_seconds,
            label=f"model {endpoint.model}",
        )

    @staticmethod
    async def _default_provider_call(**kwargs: Any) -> Any:
        import litellm

        litellm.set_verbose = False
        return await litellm.acompletion(**kwargs)

    async def _emit(self, event: dict[str, Any]) -> None:
        if self._event_callback is None:
            return
        try:
            result = self._event_callback(event)
            if inspect.isawaitable(result):
                await result
        except Exception:
            logger.warning("Model resilience event delivery failed", exc_info=True)

    async def _emit_transition(
        self,
        endpoint: ModelEndpoint,
        transition: CircuitTransition | None,
    ) -> None:
        if transition is None:
            return
        await self._emit({
            "type": "circuit_state_changed",
            "summary": (
                f"Model {endpoint.model}: {transition.previous} -> {transition.current}"
            ),
            "data": {
                "capability": "model",
                "endpoint": endpoint.role,
                "model": endpoint.model,
                "previous": transition.previous,
                "current": transition.current,
                "reason": transition.reason,
            },
        })

    async def __call__(self, **kwargs: Any) -> Any:
        try:
            return await self._call_endpoint(self.primary, kwargs)
        except ModelGatewayError as primary_error:
            if self.fallback is None or not primary_error.classified.fallback_allowed:
                raise
            self._fallback_count += 1
            self._fallback_reason = primary_error.classified.kind
            await self._emit({
                "type": "fallback_activated",
                "summary": (
                    f"Switched model from {self.primary.model} to {self.fallback.model} "
                    f"after {primary_error.classified.kind}"
                ),
                "data": {
                    "capability": "model",
                    "from_model": self.primary.model,
                    "to_model": self.fallback.model,
                    "reason": primary_error.classified.kind,
                },
            })
            return await self._call_endpoint(self.fallback, kwargs)

    async def _call_endpoint(self, endpoint: ModelEndpoint, kwargs: dict[str, Any]) -> Any:
        circuit = self._circuits[endpoint.role]
        try:
            await self._emit_transition(endpoint, await circuit.before_call())
        except CircuitOpenError:
            classified = ClassifiedModelError("circuit_open", False, True)
            self._record_failure(endpoint, classified)
            raise ModelGatewayError(endpoint, classified) from None

        last_classified = ClassifiedModelError("unknown", False, False)
        attempt = 0
        context_reduced = False
        call_input = dict(kwargs)
        while True:
            self._attempt_count += 1
            started = time.monotonic()
            try:
                call_kwargs = {
                    **{key: value for key, value in call_input.items() if value is not None},
                    "model": endpoint.model,
                    "timeout": endpoint.timeout_seconds,
                }
                if endpoint.api_key:
                    call_kwargs["api_key"] = endpoint.api_key
                if endpoint.api_base:
                    call_kwargs["api_base"] = endpoint.api_base
                async with asyncio.timeout(endpoint.timeout_seconds):
                    response = await self._provider_call(**call_kwargs)
                elapsed = time.monotonic() - started
                prompt_tokens, completion_tokens = self._usage(response)
                observe_llm_call(
                    endpoint.model,
                    elapsed,
                    tokens_prompt=prompt_tokens,
                    tokens_completion=completion_tokens,
                    status="success",
                )
                await self._emit_transition(endpoint, await circuit.record_success())
                self._successful_calls += 1
                self._prompt_tokens += prompt_tokens
                self._completion_tokens += completion_tokens
                self._active_model = endpoint.model
                return response
            except Exception as exc:
                elapsed = time.monotonic() - started
                last_classified = classify_model_error(exc)
                observe_llm_call(
                    endpoint.model,
                    elapsed,
                    status="timeout" if last_classified.kind == "timeout" else "error",
                )
                self._record_failure(endpoint, last_classified)
                if last_classified.kind == "context_limit" and not context_reduced:
                    compacted, details = self._compact_context(call_input)
                    if compacted is not None:
                        call_input = compacted
                        context_reduced = True
                        self._context_reduction_count += 1
                        await self._emit({
                            "type": "capability_degraded",
                            "summary": "Model context was compacted after a provider context-limit response",
                            "data": {
                                "capability": "context",
                                "reason": "provider_context_limit",
                                "fallback": "compacted_messages",
                                **details,
                            },
                        })
                        continue
                if not last_classified.retryable or attempt >= self.policy.max_retries:
                    break
                delay = self._retry_delay(attempt, last_classified.retry_after_seconds)
                self._retry_count += 1
                await self._emit({
                    "type": "model_retry_scheduled",
                    "summary": (
                        f"Retrying {endpoint.model} after {last_classified.kind} "
                        f"in {delay:.2f}s"
                    ),
                    "data": {
                        "capability": "model",
                        "endpoint": endpoint.role,
                        "model": endpoint.model,
                        "reason": last_classified.kind,
                        "attempt": attempt + 1,
                        "delay_seconds": delay,
                        "retry_after_seconds": last_classified.retry_after_seconds,
                    },
                })
                await self._sleep(delay)
                attempt += 1

        await self._emit_transition(
            endpoint,
            await circuit.record_failure(last_classified.kind),
        )
        await self._emit({
            "type": "model_call_failed",
            "summary": f"Model {endpoint.model} failed: {last_classified.kind}",
            "data": {
                "capability": "model",
                "endpoint": endpoint.role,
                "model": endpoint.model,
                "reason": last_classified.kind,
                "status_code": last_classified.status_code,
            },
        })
        raise ModelGatewayError(endpoint, last_classified)

    @staticmethod
    def _compact_context(
        kwargs: dict[str, Any],
    ) -> tuple[dict[str, Any] | None, dict[str, int]]:
        """Create one bounded, auditable message reduction for context overflow."""
        messages = kwargs.get("messages")
        if not isinstance(messages, list) or not messages:
            return None, {}
        original_chars = sum(
            len(str(item.get("content", "")))
            for item in messages
            if isinstance(item, dict)
        )
        if original_chars <= 24_000:
            return None, {}

        selected: list[dict[str, Any]] = []
        first = messages[0] if isinstance(messages[0], dict) else None
        has_system = bool(first and first.get("role") == "system")
        if has_system and first is not None:
            selected.append({**first, "content": str(first.get("content", ""))[:4_000]})

        tail_start = 1 if has_system else 0
        tail = [item for item in messages[tail_start:] if isinstance(item, dict)][-6:]
        for index, item in enumerate(tail):
            limit = 8_000 if index == len(tail) - 1 else 2_000
            content = str(item.get("content", ""))
            selected.append({**item, "content": content[-limit:]})

        compacted_chars = sum(len(str(item.get("content", ""))) for item in selected)
        if compacted_chars >= original_chars:
            return None, {}
        return (
            {**kwargs, "messages": selected},
            {
                "original_message_count": len(messages),
                "compacted_message_count": len(selected),
                "original_chars": original_chars,
                "compacted_chars": compacted_chars,
            },
        )

    def _retry_delay(self, attempt: int, retry_after: float | None) -> float:
        exponential = min(
            self.policy.retry_max_seconds,
            self.policy.retry_base_seconds * (2 ** attempt),
        )
        jitter = self._random_uniform(0.0, self.policy.retry_jitter_seconds)
        delay = exponential + jitter
        if retry_after is not None:
            delay = max(delay, min(retry_after, self.policy.retry_after_cap_seconds))
        return delay

    def _record_failure(
        self,
        endpoint: ModelEndpoint,
        classified: ClassifiedModelError,
    ) -> None:
        self._last_failure = {
            "endpoint": endpoint.role,
            "model": endpoint.model,
            "reason": classified.kind,
            "status_code": classified.status_code,
        }

    @staticmethod
    def _usage(response: Any) -> tuple[int, int]:
        usage = getattr(response, "usage", None)
        if usage is None:
            return 0, 0
        return (
            int(getattr(usage, "prompt_tokens", 0) or 0),
            int(getattr(usage, "completion_tokens", 0) or 0),
        )

    def snapshot(self) -> dict[str, Any]:
        endpoints = []
        for endpoint in (self.primary, self.fallback):
            if endpoint is None:
                continue
            endpoints.append({
                **endpoint.public_metadata(),
                "circuit": self._circuits[endpoint.role].snapshot(),
            })
        return {
            "active_model": self._active_model,
            "fallback_configured": self.fallback is not None,
            "fallback_activated": self._fallback_count > 0,
            "fallback_reason": self._fallback_reason,
            "attempt_count": self._attempt_count,
            "retry_count": self._retry_count,
            "fallback_count": self._fallback_count,
            "successful_calls": self._successful_calls,
            "context_reduction_count": self._context_reduction_count,
            "prompt_tokens": self._prompt_tokens,
            "completion_tokens": self._completion_tokens,
            "total_tokens": self._prompt_tokens + self._completion_tokens,
            "last_failure": self._last_failure,
            "policy": {
                "max_retries": self.policy.max_retries,
                "failure_threshold": self.policy.failure_threshold,
                "cooldown_seconds": self.policy.cooldown_seconds,
                "retry_after_cap_seconds": self.policy.retry_after_cap_seconds,
            },
            "endpoints": endpoints,
        }
