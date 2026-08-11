"""Compatibility exports for the shared circuit breaker."""

from codeagent.resilience import (
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
    CircuitTransition,
)

__all__ = [
    "CircuitBreaker",
    "CircuitOpenError",
    "CircuitState",
    "CircuitTransition",
]
