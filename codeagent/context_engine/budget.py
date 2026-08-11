"""Adaptive request budget calculation for repository context."""

from __future__ import annotations

from dataclasses import dataclass
import re


_COMPLEX = re.compile(
    r"跨模块|跨服务|架构|重构|迁移|并发|安全|认证|调用链|"
    r"\b(architecture|refactor|migration|concurren|security|auth|cross[- ]module)\b",
    re.I,
)
_MEDIUM = re.compile(
    r"实现|修复|新增|测试|接口|数据库|依赖|"
    r"\b(implement|fix|add|test|api|database|dependency)\b",
    re.I,
)


@dataclass(frozen=True)
class ContextBudgetDecision:
    profile: str
    requested_tokens: int
    effective_tokens: int
    model_available_tokens: int
    reason: str


def select_context_budget(
    query: str,
    *,
    target_tokens: int,
    min_tokens: int,
    max_tokens: int,
    context_window: int,
    max_output_tokens: int,
    safety_margin_tokens: int,
    tool_overhead_tokens: int,
) -> ContextBudgetDecision:
    """Select 16K/32K/48K/80K-style input profiles and enforce the model window."""
    text = query or ""
    if _COMPLEX.search(text) or len(text) > 2_000:
        profile, requested, reason = "extended", 80_000, "complex_or_cross_module_request"
    elif len(_MEDIUM.findall(text)) >= 2 or len(text) > 600:
        profile, requested, reason = "large", 48_000, "multi_constraint_change_request"
    elif _MEDIUM.search(text):
        profile, requested, reason = "standard", 32_000, "code_change_request"
    else:
        profile, requested, reason = "compact", 16_000, "simple_or_read_only_request"

    configured_ceiling = max(1, min(target_tokens, max_tokens))
    requested = min(configured_ceiling, max(min_tokens, requested))
    available = max(
        1,
        context_window
        - max_output_tokens
        - safety_margin_tokens
        - tool_overhead_tokens,
    )
    effective = max(1, min(requested, available))
    if effective < requested:
        reason += ";clamped_to_model_window"
    return ContextBudgetDecision(profile, requested, effective, available, reason)
