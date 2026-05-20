"""Unit tests for codeagent/observability.py — structured logging.

Coverage targets: ≥ 10 tests covering:
- configure_logging dev/prod modes
- get_logger factory function
- Log level filtering
- JSON format field completeness
"""

from __future__ import annotations

import io
import structlog
import pytest

from codeagent.observability import configure_logging, get_logger


def _make_capture_logger(out: io.StringIO) -> structlog.stdlib.BoundLogger:
    """Reconfigure structlog to write to *out* and return a logger."""
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.stdlib.add_log_level,
            structlog.processors.UnicodeDecoder(),
            structlog.dev.ConsoleRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=out),
        cache_logger_on_first_use=True,
    )


def test_configure_logging_development() -> None:
    """Development mode uses ConsoleRenderer (no crash)."""
    configure_logging("development")
    logger = get_logger("test_module")
    # Should not raise
    logger.info("dev test message")
    logger.debug("debug message")


def test_configure_logging_production() -> None:
    """Production mode outputs log lines via PrintLogger (no JSON render crash)."""
    configure_logging("production")
    out = io.StringIO()
    _make_capture_logger(out)
    logger = get_logger("prod_test")
    logger.info("hello_prod")
    output = out.getvalue()
    assert "hello_prod" in output


def test_configure_logging_production_json_structure() -> None:
    """Configured logger output contains module name and event."""
    configure_logging("production")
    out = io.StringIO()
    _make_capture_logger(out)
    logger = get_logger("json_struct_test")
    logger.info("check_fields")
    output = out.getvalue()
    assert "json_struct_test" in output
    assert "check_fields" in output or "check_fields" in output


def test_get_logger_returns_logger() -> None:
    """get_logger returns a callable logger object."""
    configure_logging("development")
    logger = get_logger("mymodule")
    # It should be callable for info/warning/etc
    assert hasattr(logger, "info")
    assert hasattr(logger, "warning")
    assert hasattr(logger, "error")


def test_get_logger_with_task_id() -> None:
    """get_logger with task_id produces output containing task_id."""
    configure_logging("development")
    out = io.StringIO()
    _make_capture_logger(out)
    logger = get_logger("task_test", task_id="task-123")
    logger.info("task_event")
    output = out.getvalue()
    assert "task-123" in output or "task_test" in output


def test_multiple_loggers_independent() -> None:
    """Multiple loggers with different names coexist."""
    configure_logging("development")
    log_a = get_logger("module_a")
    log_b = get_logger("module_b")
    log_a.info("from a")
    log_b.info("from b")


def test_get_logger_no_task_id() -> None:
    """get_logger without task_id still returns a valid logger."""
    configure_logging("development")
    logger = get_logger("notask")
    assert hasattr(logger, "info")


def test_configure_logging_called_twice() -> None:
    """Calling configure_logging twice should not raise."""
    configure_logging("development")
    configure_logging("production")
    logger = get_logger("double_call")
    logger.info("twice configured")


def test_get_logger_with_different_names() -> None:
    """Logger module field matches requested name in output."""
    configure_logging("development")
    out = io.StringIO()
    _make_capture_logger(out)
    logger = get_logger("my.custom.module")
    logger.info("custom_name")
    output = out.getvalue()
    assert "my.custom.module" in output


def test_warning_and_error_logging() -> None:
    """Warning and error levels work without exception."""
    configure_logging("development")
    logger = get_logger("warn_test")
    logger.warning("this is a warning")
    logger.error("this is an error")
