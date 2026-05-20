"""Prometheus metrics definitions for CodeAgent.

All metrics use the ``codeagent_`` prefix. Every ``Histogram`` specifies
``buckets``. Metric recording functions are wrapped in try/except so they
never raise.
"""

from __future__ import annotations

import functools
import logging
import time
from typing import Any, Callable

from prometheus_client import Counter, Gauge, Histogram, REGISTRY, generate_latest

logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════
# LLM metrics
# ════════════════════════════════════════════════════════════════

llm_calls_total = Counter(
    "codeagent_llm_calls_total",
    "Total number of LLM calls",
    labelnames=["model", "status"],  # status: success / error / timeout
)

llm_call_duration_seconds = Histogram(
    "codeagent_llm_call_duration_seconds",
    "LLM call duration in seconds",
    labelnames=["model"],
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0),
)

llm_tokens_total = Counter(
    "codeagent_llm_tokens_total",
    "Total LLM tokens consumed",
    labelnames=["model", "type"],  # type: prompt / completion
)

# ════════════════════════════════════════════════════════════════
# Tool call metrics
# ════════════════════════════════════════════════════════════════

tool_calls_total = Counter(
    "codeagent_tool_calls_total",
    "Total number of tool calls",
    labelnames=["tool_name", "status"],  # status: success / error
)

tool_call_duration_seconds = Histogram(
    "codeagent_tool_call_duration_seconds",
    "Tool call duration in seconds",
    labelnames=["tool_name"],
    buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0),
)

# ════════════════════════════════════════════════════════════════
# Task metrics
# ════════════════════════════════════════════════════════════════

task_duration_seconds = Histogram(
    "codeagent_task_duration_seconds",
    "End‑to‑end task execution duration in seconds",
    labelnames=["status"],  # status: completed / failed / cancelled
    buckets=(10, 30, 60, 120, 300, 600, 1800),
)

active_tasks = Gauge(
    "codeagent_active_tasks",
    "Number of currently active (in‑progress) tasks",
)

task_steps_total = Counter(
    "codeagent_task_steps_total",
    "Total number of task execution steps",
    labelnames=["status"],
)

# ════════════════════════════════════════════════════════════════
# Validation metrics
# ════════════════════════════════════════════════════════════════

validation_results_total = Counter(
    "codeagent_validation_results_total",
    "Total number of validation results",
    labelnames=["layer", "passed"],  # layer: syntax / static / runtime
)

# ════════════════════════════════════════════════════════════════
# Memory metrics
# ════════════════════════════════════════════════════════════════

memory_retrieval_duration_seconds = Histogram(
    "codeagent_memory_retrieval_duration_seconds",
    "Memory retrieval duration in seconds",
    buckets=(0.01, 0.05, 0.1, 0.5, 1.0),
)


# ════════════════════════════════════════════════════════════════
# Public helpers
# ════════════════════════════════════════════════════════════════


def get_metrics_endpoint() -> str:
    """Return the ``/metrics`` endpoint content in Prometheus text format."""
    return generate_latest(REGISTRY).decode("utf-8")


def observe_llm_call(
    model: str,
    duration: float,
    tokens_prompt: int = 0,
    tokens_completion: int = 0,
    status: str = "success",
) -> None:
    """Record LLM call metrics.

    Args:
        model: Model name (e.g. ``"deepseek/deepseek-v4-flash"``).
        duration: Call duration in seconds.
        tokens_prompt: Number of prompt tokens consumed.
        tokens_completion: Number of completion tokens consumed.
        status: ``"success"``, ``"error"``, or ``"timeout"``.
    """
    try:
        llm_calls_total.labels(model=model, status=status).inc()
        llm_call_duration_seconds.labels(model=model).observe(duration)
        if tokens_prompt:
            llm_tokens_total.labels(model=model, type="prompt").inc(tokens_prompt)
        if tokens_completion:
            llm_tokens_total.labels(model=model, type="completion").inc(tokens_completion)
    except Exception as exc:
        logger.warning("Failed to record LLM metrics: %s", exc)


def observe_tool_call(tool_name: str, duration: float, success: bool) -> None:
    """Record tool call metrics.

    Args:
        tool_name: Name of the tool (e.g. ``"read_file"``).
        duration: Call duration in seconds.
        success: Whether the call succeeded.
    """
    try:
        status = "success" if success else "error"
        tool_calls_total.labels(tool_name=tool_name, status=status).inc()
        tool_call_duration_seconds.labels(tool_name=tool_name).observe(duration)
    except Exception as exc:
        logger.warning("Failed to record tool call metrics: %s", exc)


def observe_task_duration(duration: float, status: str) -> None:
    """Record task execution metrics.

    Args:
        duration: End‑to‑end duration in seconds.
        status: ``"completed"``, ``"failed"``, or ``"cancelled"``.
    """
    try:
        task_duration_seconds.labels(status=status).observe(duration)
        active_tasks.set(0)
    except Exception as exc:
        logger.warning("Failed to record task metrics: %s", exc)


def observe_validation(layer: str, passed: bool) -> None:
    """Record a validation result.

    Args:
        layer: ``"syntax"``, ``"static"``, or ``"runtime"``.
        passed: Whether the layer passed.
    """
    try:
        validation_results_total.labels(layer=layer, passed=str(passed).lower()).inc()
    except Exception as exc:
        logger.warning("Failed to record validation metrics: %s", exc)


def observe_memory_retrieval(duration: float) -> None:
    """Record memory retrieval duration.

    Args:
        duration: Retrieval duration in seconds.
    """
    try:
        memory_retrieval_duration_seconds.observe(duration)
    except Exception as exc:
        logger.warning("Failed to record memory retrieval metrics: %s", exc)


def wrap_llm_call(model: str, fn: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap an LLM call function with metrics recording.

    Usage::

        llm_with_metrics = wrap_llm_call("my-model", original_llm_call)
        response = await llm_with_metrics(model=..., messages=..., ...)

    The wrapper records ``llm_calls_total``, ``llm_call_duration_seconds``,
    and ``llm_tokens_total`` (when ``usage`` information is available on the
    response).
    """

    @functools.wraps(fn)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        t0 = time.monotonic()
        try:
            response = await fn(*args, **kwargs)
            elapsed = time.monotonic() - t0

            tokens_prompt = 0
            tokens_completion = 0
            if hasattr(response, "usage") and response.usage:
                tokens_prompt = getattr(response.usage, "prompt_tokens", 0) or 0
                tokens_completion = getattr(response.usage, "completion_tokens", 0) or 0

            observe_llm_call(
                model=model,
                duration=elapsed,
                tokens_prompt=tokens_prompt,
                tokens_completion=tokens_completion,
                status="success",
            )
            return response
        except Exception as exc:
            elapsed = time.monotonic() - t0
            status = "timeout" if "timeout" in str(exc).lower() else "error"
            observe_llm_call(model=model, duration=elapsed, status=status)
            raise

    return wrapper
