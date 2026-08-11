from types import SimpleNamespace

import pytest

from codeagent.chat_response import EMPTY_VISIBLE_RESPONSE, request_visible_chat


def _response(content: str | None, *, reasoning: str = "", tokens: int = 0):
    message = SimpleNamespace(content=content, reasoning_content=reasoning)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=SimpleNamespace(total_tokens=tokens),
    )


@pytest.mark.asyncio
async def test_retries_reasoning_only_response_without_exposing_reasoning() -> None:
    responses = iter([
        _response(None, reasoning="private reasoning", tokens=13),
        _response(" visible answer ", tokens=17),
    ])
    calls = 0

    async def model_call(**_kwargs):
        nonlocal calls
        calls += 1
        return next(responses)

    result = await request_visible_chat(model_call, messages=[])

    assert calls == 2
    assert result.text == "visible answer"
    assert result.token_usage == 30
    assert result.attempts == 2


@pytest.mark.asyncio
async def test_uses_fallback_after_bounded_empty_responses() -> None:
    async def model_call(**_kwargs):
        return _response(None, reasoning="private reasoning", tokens=5)

    result = await request_visible_chat(model_call, messages=[])

    assert result.text == EMPTY_VISIBLE_RESPONSE
    assert result.token_usage == 10
    assert result.attempts == 2
