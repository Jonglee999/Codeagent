"""Structured logging configuration for CodeAgent.

Provides:
- configure_logging(env): Dev → rich console, Production → JSON stdout
- get_logger(name, task_id=None): Factory with automatic field binding
"""

from __future__ import annotations

import structlog
from structlog.processors import JSONRenderer, TimeStamper, add_log_level


def configure_logging(env: str = "development") -> None:
    """Configure global structured logging.

    Args:
        env: "development" → rich console output (colorful, human-readable)
             "production"  → JSON lines output (for log aggregators)
    """
    shared_processors: list[structlog.typing.Processor] = [
        TimeStamper(fmt="iso", utc=True),
        add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    if env == "production":
        shared_processors.append(JSONRenderer())
    else:
        shared_processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=shared_processors,
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str, task_id: str | None = None) -> structlog.stdlib.BoundLogger:
    """Get a structured logger with module and optional task_id bound.

    Args:
        name: Usually ``__name__`` of the calling module.
        task_id: Optional task identifier to bind for traceability.

    Returns:
        A ``structlog.stdlib.BoundLogger`` with ``module`` and optional ``task_id`` fields.
    """
    return structlog.get_logger(module=name, task_id=task_id)  # type: ignore[return-value]
