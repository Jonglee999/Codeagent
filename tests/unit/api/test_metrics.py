"""Unit tests for codeagent/interaction/api/metrics.py — Prometheus metrics.
Coverage targets: ≥ 10 tests covering:
- /metrics endpoint returns valid Prometheus text
- Counter/Histogram/Gauge increment and observe
- Multi-label correctness
- Metrics helper functions (observe_llm_call, observe_tool_call, etc.)
"""

from __future__ import annotations

import pytest
from prometheus_client import REGISTRY

from codeagent.interaction.api import metrics as m


def _clear_registry() -> None:
    """Remove codeagent_ metrics from REGISTRY between tests."""
    collectors = list(REGISTRY._collector_to_names.keys())
    for collector in collectors:
        for name in list(REGISTRY._collector_to_names.get(collector, [])):
            if name.startswith("codeagent_"):
                REGISTRY.unregister(collector)
                break


@pytest.fixture(autouse=True)
def reset_metrics():
    """Reset all codeagent_ metrics before each test."""
    collectors_to_unreg = []
    for collector, names in list(REGISTRY._collector_to_names.items()):
        if any(n.startswith("codeagent_") for n in names):
            collectors_to_unreg.append(collector)
    for c in collectors_to_unreg:
        try:
            REGISTRY.unregister(c)
        except KeyError:
            pass
    # Re-import metrics module to re-register
    import importlib
    importlib.reload(m)
    yield


def test_get_metrics_endpoint_returns_text() -> None:
    """get_metrics_endpoint returns non-empty Prometheus text."""
    text = m.get_metrics_endpoint()
    assert isinstance(text, str)
    assert len(text) > 0
    # Prometheus text format includes TYPE lines
    assert "# TYPE" in text or "# HELP" in text


def test_metrics_contains_codeagent_prefix() -> None:
    """Metrics output contains codeagent_ prefixed metrics."""
    # Trigger a metric recording so it appears in output
    m.observe_llm_call("test-model", 0.5, tokens_prompt=10, tokens_completion=20, status="success")
    m.observe_tool_call("read_file", 0.05, success=True)

    text = m.get_metrics_endpoint()
    assert "codeagent_llm_calls_total" in text
    assert "codeagent_tool_calls_total" in text


def test_llm_call_counter_increment() -> None:
    """llm_calls_total counter increments with labels."""
    m.observe_llm_call("gpt-4", 1.0, status="success")
    m.observe_llm_call("gpt-4", 0.5, status="success")

    value = m.llm_calls_total.labels(model="gpt-4", status="success")._value.get()
    assert value == 2


def test_llm_call_error_counter() -> None:
    """LLM error calls are counted separately."""
    m.observe_llm_call("gpt-4", 1.0, status="error")
    value = m.llm_calls_total.labels(model="gpt-4", status="error")._value.get()
    assert value == 1


def test_tool_call_counter_with_labels() -> None:
    """tool_calls_total increments for different tool names and statuses."""
    m.observe_tool_call("read_file", 0.1, success=True)
    m.observe_tool_call("write_file", 0.2, success=True)
    m.observe_tool_call("read_file", 0.3, success=False)

    read_ok = m.tool_calls_total.labels(tool_name="read_file", status="success")._value.get()
    read_err = m.tool_calls_total.labels(tool_name="read_file", status="error")._value.get()
    write_ok = m.tool_calls_total.labels(tool_name="write_file", status="success")._value.get()

    assert read_ok == 1
    assert read_err == 1
    assert write_ok == 1


def test_active_tasks_gauge() -> None:
    """active_tasks gauge can be set and read."""
    m.active_tasks.set(3)
    assert m.active_tasks._value.get() == 3
    m.active_tasks.set(1)
    assert m.active_tasks._value.get() == 1


def test_validation_results_counter() -> None:
    """validation_results_total counter works with layer/passed labels."""
    m.observe_validation("syntax", passed=True)
    m.observe_validation("syntax", passed=False)
    m.observe_validation("static", passed=True)

    syntax_pass = m.validation_results_total.labels(layer="syntax", passed="true")._value.get()
    syntax_fail = m.validation_results_total.labels(layer="syntax", passed="false")._value.get()
    static_pass = m.validation_results_total.labels(layer="static", passed="true")._value.get()

    assert syntax_pass == 1
    assert syntax_fail == 1
    assert static_pass == 1


def test_llm_tokens_counter() -> None:
    """llm_tokens_total records prompt and completion tokens."""
    m.observe_llm_call("gpt-4", 1.0, tokens_prompt=100, tokens_completion=200, status="success")

    prompt_val = m.llm_tokens_total.labels(model="gpt-4", type="prompt")._value.get()
    completion_val = m.llm_tokens_total.labels(model="gpt-4", type="completion")._value.get()

    assert prompt_val == 100
    assert completion_val == 200
