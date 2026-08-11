from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from codeagent.model_gateway import (
    ModelEndpoint,
    ModelGateway,
    ModelGatewayError,
    ResiliencePolicy,
    classify_model_error,
)


class ProviderError(RuntimeError):
    def __init__(self, message: str, status_code: int, headers: dict | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.headers = headers or {}


def response(text: str = "ok") -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2),
    )


@pytest.mark.asyncio
async def test_snapshot_accumulates_provider_reported_tokens() -> None:
    async def provider(**_):
        return response()

    gateway = ModelGateway(endpoint(), provider_call=provider)

    await gateway(messages=[{"role": "user", "content": "hello"}])

    snapshot = gateway.snapshot()
    assert snapshot["prompt_tokens"] == 3
    assert snapshot["completion_tokens"] == 2
    assert snapshot["total_tokens"] == 5


def endpoint(role: str = "primary", model: str = "provider/primary") -> ModelEndpoint:
    return ModelEndpoint(role=role, model=model, api_key="secret-key", timeout_seconds=2)  # type: ignore[arg-type]


def policy(**overrides) -> ResiliencePolicy:
    values = {
        "max_retries": 2,
        "retry_base_seconds": 0.1,
        "retry_max_seconds": 1,
        "retry_jitter_seconds": 0,
        "failure_threshold": 1,
        "cooldown_seconds": 30,
    }
    values.update(overrides)
    return ResiliencePolicy(**values)


@pytest.mark.parametrize(
    ("error", "kind", "retryable", "fallback_allowed"),
    [
        (ProviderError("bad key", 401), "authentication", False, True),
        (ProviderError("limited", 429), "rate_limit", True, True),
        (ProviderError("down", 503), "provider_5xx", True, True),
        (asyncio.TimeoutError(), "timeout", True, True),
        (RuntimeError("connection refused"), "connection", True, True),
        (RuntimeError("maximum context length exceeded"), "context_limit", False, False),
        (RuntimeError("content policy rejection"), "content_policy", False, False),
        (ProviderError("bad request", 400), "invalid_request", False, False),
    ],
)
def test_classifies_model_failures(error, kind, retryable, fallback_allowed):
    classified = classify_model_error(error)
    assert classified.kind == kind
    assert classified.retryable is retryable
    assert classified.fallback_allowed is fallback_allowed


@pytest.mark.asyncio
async def test_rate_limit_honors_retry_after_then_succeeds():
    calls = 0
    sleeps: list[float] = []
    events: list[dict] = []

    async def provider(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ProviderError("rate limited", 429, {"Retry-After": "2.5"})
        return response()

    async def sleep(delay: float):
        sleeps.append(delay)

    gateway = ModelGateway(
        endpoint(),
        policy=policy(),
        provider_call=provider,
        sleep=sleep,
        event_callback=events.append,
    )

    result = await gateway(messages=[])

    assert result.choices[0].message.content == "ok"
    assert calls == 2
    assert sleeps == [2.5]
    retry_event = next(event for event in events if event["type"] == "model_retry_scheduled")
    assert retry_event["data"]["retry_after_seconds"] == 2.5
    assert gateway.snapshot()["retry_count"] == 1


@pytest.mark.asyncio
async def test_server_error_retries_then_uses_explicit_fallback():
    seen_models: list[str] = []
    events: list[dict] = []

    async def provider(**kwargs):
        seen_models.append(kwargs["model"])
        if kwargs["model"] == "provider/primary":
            raise ProviderError("upstream down", 503)
        return response("fallback-ok")

    gateway = ModelGateway(
        endpoint(),
        endpoint("fallback", "provider/fallback"),
        policy=policy(max_retries=1),
        provider_call=provider,
        sleep=lambda _delay: asyncio.sleep(0),
        random_uniform=lambda _a, _b: 0,
        event_callback=events.append,
    )

    result = await gateway(messages=[])

    assert result.choices[0].message.content == "fallback-ok"
    assert seen_models == ["provider/primary", "provider/primary", "provider/fallback"]
    assert any(event["type"] == "fallback_activated" for event in events)
    assert any(
        event["type"] == "circuit_state_changed"
        and event["data"]["capability"] == "model"
        and event["data"]["current"] == "open"
        for event in events
    )
    snapshot = gateway.snapshot()
    assert snapshot["active_model"] == "provider/fallback"
    assert snapshot["fallback_activated"] is True
    assert snapshot["fallback_reason"] == "provider_5xx"


@pytest.mark.asyncio
async def test_auth_failure_skips_primary_retry_and_falls_back():
    seen_models: list[str] = []

    async def provider(**kwargs):
        seen_models.append(kwargs["model"])
        if kwargs["model"] == "provider/primary":
            raise ProviderError("invalid API key", 401)
        return response()

    gateway = ModelGateway(
        endpoint(),
        endpoint("fallback", "provider/fallback"),
        policy=policy(),
        provider_call=provider,
    )

    await gateway(messages=[])
    assert seen_models == ["provider/primary", "provider/fallback"]
    assert gateway.snapshot()["retry_count"] == 0


@pytest.mark.asyncio
async def test_context_limit_does_not_retry_or_fallback():
    calls = 0

    async def provider(**kwargs):
        nonlocal calls
        calls += 1
        raise ProviderError("maximum context length exceeded", 400)

    gateway = ModelGateway(
        endpoint(),
        endpoint("fallback", "provider/fallback"),
        policy=policy(),
        provider_call=provider,
    )

    with pytest.raises(ModelGatewayError) as caught:
        await gateway(messages=[])

    assert caught.value.classified.kind == "context_limit"
    assert calls == 1
    assert gateway.snapshot()["fallback_count"] == 0


@pytest.mark.asyncio
async def test_context_limit_compacts_large_history_once_and_emits_degradation():
    calls: list[int] = []
    events: list[dict] = []

    async def provider(**kwargs):
        chars = sum(len(item["content"]) for item in kwargs["messages"])
        calls.append(chars)
        if len(calls) == 1:
            raise ProviderError("maximum context length exceeded", 400)
        return response()

    gateway = ModelGateway(
        endpoint(),
        policy=policy(max_retries=0),
        provider_call=provider,
        event_callback=events.append,
    )

    await gateway(messages=[
        {"role": "system", "content": "s" * 10_000},
        *({"role": "user", "content": "u" * 8_000} for _ in range(5)),
        {"role": "user", "content": "latest" * 2_000},
    ])

    assert calls[1] < calls[0]
    assert gateway.snapshot()["context_reduction_count"] == 1
    assert any(
        event["type"] == "capability_degraded"
        and event["data"]["fallback"] == "compacted_messages"
        for event in events
    )


@pytest.mark.asyncio
async def test_open_primary_circuit_routes_next_call_to_fallback_without_primary_io():
    primary_calls = 0
    fallback_calls = 0

    async def provider(**kwargs):
        nonlocal primary_calls, fallback_calls
        if kwargs["model"] == "provider/primary":
            primary_calls += 1
            raise ProviderError("down", 503)
        fallback_calls += 1
        return response()

    gateway = ModelGateway(
        endpoint(),
        endpoint("fallback", "provider/fallback"),
        policy=policy(max_retries=0, failure_threshold=1),
        provider_call=provider,
    )

    await gateway(messages=[])
    await gateway(messages=[])

    assert primary_calls == 1
    assert fallback_calls == 2
    primary_status = gateway.snapshot()["endpoints"][0]
    assert primary_status["circuit"]["state"] == "open"


@pytest.mark.asyncio
async def test_snapshot_and_events_never_include_api_keys():
    events: list[dict] = []

    async def provider(**kwargs):
        raise ProviderError("bad key secret-key", 401)

    gateway = ModelGateway(
        endpoint(),
        policy=policy(),
        provider_call=provider,
        event_callback=events.append,
    )

    with pytest.raises(ModelGatewayError):
        await gateway(messages=[])

    assert "secret-key" not in repr(gateway.snapshot())
    assert "secret-key" not in repr(events)
