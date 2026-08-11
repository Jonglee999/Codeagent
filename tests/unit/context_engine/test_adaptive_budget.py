from codeagent.context_engine.budget import select_context_budget


def test_simple_request_uses_compact_profile():
    result = select_context_budget(
        "解释这个函数",
        target_tokens=80_000,
        min_tokens=16_000,
        max_tokens=80_000,
        context_window=128_000,
        max_output_tokens=32_000,
        safety_margin_tokens=8_000,
        tool_overhead_tokens=8_000,
    )
    assert result.profile == "compact"
    assert result.effective_tokens == 16_000


def test_complex_request_reaches_80k_when_window_allows_it():
    result = select_context_budget(
        "跨模块架构重构",
        target_tokens=80_000,
        min_tokens=16_000,
        max_tokens=80_000,
        context_window=128_000,
        max_output_tokens=32_000,
        safety_margin_tokens=8_000,
        tool_overhead_tokens=8_000,
    )
    assert result.profile == "extended"
    assert result.effective_tokens == 80_000


def test_budget_is_clamped_to_complete_request_window():
    result = select_context_budget(
        "跨模块架构重构",
        target_tokens=80_000,
        min_tokens=16_000,
        max_tokens=80_000,
        context_window=64_000,
        max_output_tokens=24_000,
        safety_margin_tokens=8_000,
        tool_overhead_tokens=8_000,
    )
    assert result.effective_tokens == 24_000
    assert "clamped_to_model_window" in result.reason


def test_explicit_target_below_minimum_remains_a_hard_ceiling():
    result = select_context_budget(
        "修复函数",
        target_tokens=8_000,
        min_tokens=16_000,
        max_tokens=80_000,
        context_window=128_000,
        max_output_tokens=32_000,
        safety_margin_tokens=8_000,
        tool_overhead_tokens=8_000,
    )
    assert result.effective_tokens == 8_000
