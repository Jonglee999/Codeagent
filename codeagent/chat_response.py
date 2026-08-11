"""Helpers for producing a user-visible chat response from an LLM call."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable


EMPTY_VISIBLE_RESPONSE = "模型没有返回可显示的内容。"


def response_text(response: object) -> str:
    """Return only provider-designated visible content, never hidden reasoning."""
    try:
        return str(response.choices[0].message.content or "").strip()  # type: ignore[attr-defined]
    except (AttributeError, IndexError, TypeError):
        return ""


def response_tokens(response: object) -> int:
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0
    return int(
        getattr(usage, "total_tokens", 0)
        or getattr(usage, "prompt_tokens", 0)
        + getattr(usage, "completion_tokens", 0)
    )


@dataclass(frozen=True)
class VisibleChatResponse:
    response: object
    text: str
    token_usage: int
    attempts: int


async def request_visible_chat(
    model_call: Callable[..., Awaitable[Any]],
    *,
    max_attempts: int = 2,
    **kwargs: Any,
) -> VisibleChatResponse:
    """Retry once when a provider returns reasoning but no visible answer.

    Thinking-capable models may occasionally finish successfully with an empty
    ``content`` field. Hidden ``reasoning_content`` must not be shown to the
    user, so a bounded second call is safer than treating it as the answer.
    Token usage includes every attempt, including discarded empty responses.
    """
    attempts = max(1, int(max_attempts))
    token_usage = 0
    last_response: object | None = None

    for attempt in range(1, attempts + 1):
        last_response = await model_call(**kwargs)
        token_usage += response_tokens(last_response)
        text = response_text(last_response)
        if text:
            return VisibleChatResponse(last_response, text, token_usage, attempt)

    assert last_response is not None
    return VisibleChatResponse(
        last_response,
        EMPTY_VISIBLE_RESPONSE,
        token_usage,
        attempts,
    )
