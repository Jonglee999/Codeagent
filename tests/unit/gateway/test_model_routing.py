from __future__ import annotations

from types import SimpleNamespace

import pytest

from codeagent.model_routing import ModelRegistry, ModelRouter


def test_default_registry_uses_flash_for_l1_l2_and_pro_for_l3(monkeypatch):
    monkeypatch.delenv("MODEL_L1_NAME", raising=False)
    monkeypatch.delenv("MODEL_L2_NAME", raising=False)
    monkeypatch.delenv("MODEL_L3_NAME", raising=False)
    monkeypatch.setenv("LLM_MODEL", "deepseek/deepseek-v4-flash")
    registry = ModelRegistry.from_env()

    assert registry.get("L1").model == "deepseek/deepseek-v4-flash"
    assert registry.get("L2").model == "deepseek/deepseek-v4-flash"
    assert registry.get("L3").model == "deepseek/deepseek-v4-pro"


def test_router_selects_roles_and_escalates_complex_work():
    router = ModelRouter(ModelRegistry.from_env(), provider_call=lambda **_: None)

    assert router.select("summary", {}) == "L1"
    assert router.select("planning", {"request": "修复一个默认值"}) == "L2"
    assert router.select("execution", {"request": "跨模块认证重构"}) == "L3"
    assert router.select("execution", {"validation_failures": 2}) == "L3"


@pytest.mark.asyncio
async def test_router_calls_selected_model_and_hides_routing_kwargs(monkeypatch):
    calls = []

    async def provider(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2)
        )

    monkeypatch.setenv("LLM_API_KEY", "test-key")
    router = ModelRouter(ModelRegistry.from_env(), provider_call=provider)
    await router(
        messages=[{"role": "user", "content": "跨模块安全重构"}],
        model_role="execution",
        routing_context={"request": "跨模块安全重构"},
        max_tokens=100_000,
    )

    assert calls[0]["model"] == "deepseek/deepseek-v4-pro"
    assert calls[0]["max_tokens"] == 32_768
    assert "routing_context" not in calls[0]
    assert router.snapshot()["selection_count"]["L3"] == 1


@pytest.mark.asyncio
async def test_shadow_mode_records_l3_proposal_but_calls_l2(monkeypatch):
    calls = []

    async def provider(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1))

    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("MODEL_ROUTING_ENABLED", "true")
    monkeypatch.setenv("MODEL_ROUTING_MODE", "shadow")
    router = ModelRouter(ModelRegistry.from_env(), provider_call=provider)
    await router(
        messages=[{"role": "user", "content": "security migration"}],
        model_role="execution",
        routing_context={"request": "cross-module security migration"},
    )

    assert calls[0]["model"] == "deepseek/deepseek-v4-flash"
    snapshot = router.snapshot()
    assert snapshot["last_selection"]["tier"] == "L2"
    assert snapshot["last_selection"]["proposed_tier"] == "L3"
    assert snapshot["shadow_difference_count"] == 1
