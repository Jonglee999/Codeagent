from codeagent.orchestration.budget import (
    estimate_request_tokens,
    preflight_model_call,
)


def test_estimate_request_tokens_grows_with_prompt_and_tools() -> None:
    short = estimate_request_tokens([{"role": "user", "content": "x"}])
    long = estimate_request_tokens(
        [{"role": "user", "content": "x" * 1000}],
        [{"type": "function", "function": {"name": "read_file"}}],
    )

    assert short > 0
    assert long > short


def test_preflight_reserves_input_and_minimum_completion() -> None:
    budget = preflight_model_call(
        consumed_tokens=90,
        max_task_tokens=100,
        max_completion_tokens=50,
        messages=[{"role": "user", "content": "x" * 1000}],
    )

    assert budget.allowed is False
    assert budget.max_completion_tokens == 0
    assert budget.reason == "insufficient_input_and_completion_budget"


def test_preflight_caps_completion_to_remaining_envelope() -> None:
    budget = preflight_model_call(
        consumed_tokens=100,
        max_task_tokens=500,
        max_completion_tokens=800,
        messages=[{"role": "user", "content": "hello"}],
    )

    assert budget.allowed is True
    assert budget.max_completion_tokens < 400
    assert (
        budget.estimated_input_tokens + budget.max_completion_tokens
        <= budget.remaining_tokens
    )
