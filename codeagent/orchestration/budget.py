"""Conservative, provider-independent model-call budget preflight."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any


_MIN_COMPLETION_TOKENS = 32


@dataclass(frozen=True)
class ModelCallBudget:
    """The token envelope available for one model request."""

    allowed: bool
    remaining_tokens: int
    estimated_input_tokens: int
    max_completion_tokens: int
    minimum_completion_tokens: int
    reason: str | None = None


def estimate_request_tokens(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
) -> int:
    """Estimate request tokens conservatively without a provider tokenizer.

    UTF-8 byte length divided by two intentionally overestimates ordinary English
    and JSON prompts while remaining deterministic for every configured provider.
    A small per-message allowance covers chat envelope fields that are not present
    in the serialized payload.
    """

    payload: dict[str, Any] = {"messages": messages}
    if tools:
        payload["tools"] = tools
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    structural_overhead = 8 + (len(messages) * 4) + (len(tools or []) * 8)
    return max(1, math.ceil(len(serialized) / 2) + structural_overhead)


def preflight_model_call(
    *,
    consumed_tokens: int,
    max_task_tokens: int,
    max_completion_tokens: int,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    minimum_completion_tokens: int = _MIN_COMPLETION_TOKENS,
) -> ModelCallBudget:
    """Return a hard pre-call token envelope for a model request."""

    remaining = max(0, int(max_task_tokens) - max(0, int(consumed_tokens)))
    estimated_input = estimate_request_tokens(messages, tools)
    available_completion = max(0, remaining - estimated_input)
    completion = min(max(0, int(max_completion_tokens)), available_completion)
    minimum = max(1, int(minimum_completion_tokens))
    allowed = completion >= minimum
    reason = None if allowed else "insufficient_input_and_completion_budget"
    return ModelCallBudget(
        allowed=allowed,
        remaining_tokens=remaining,
        estimated_input_tokens=estimated_input,
        max_completion_tokens=completion,
        minimum_completion_tokens=minimum,
        reason=reason,
    )
