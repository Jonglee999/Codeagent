"""Quota-consuming smoke test for real three-tier provider routing."""

from __future__ import annotations

import os

import pytest

from codeagent.model_routing import ModelRouter


pytestmark = [pytest.mark.e2e, pytest.mark.requires_network]


@pytest.mark.asyncio
async def test_flash_and_pro_tiers_are_reachable() -> None:
    if os.environ.get("CONFIRM_LLM_API_COST", "").lower() != "true":
        pytest.skip("Set CONFIRM_LLM_API_COST=true to run quota-consuming model routing E2E")

    router = ModelRouter.from_env()
    messages = [{"role": "user", "content": "Reply with the single word OK."}]

    flash = await router(
        messages=messages,
        max_tokens=16,
        model_role="summary",
    )
    pro = await router(
        messages=messages,
        max_tokens=16,
        model_role="execution",
        routing_context={"request": "cross-module security architecture refactor"},
    )

    assert (
        flash.choices[0].message.content
        or getattr(flash.choices[0].message, "reasoning_content", "")
    )
    assert (
        pro.choices[0].message.content
        or getattr(pro.choices[0].message, "reasoning_content", "")
    )
    snapshot = router.snapshot()
    assert snapshot["selection_count"]["L1"] == 1
    assert snapshot["selection_count"]["L3"] == 1
    assert snapshot["gateways"]["L1"]["active_model"].endswith("flash")
    assert snapshot["gateways"]["L3"]["active_model"].endswith("pro")
