"""Structured logging helpers for Atropos runtime and API services."""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any, Final

ENV_LOG_FORMAT: Final[str] = "ATROPOS_LOG_FORMAT"
DEFAULT_LOG_FORMAT: Final[str] = "json"

STABLE_LOG_FIELDS: Final[tuple[str, ...]] = (
    "env_id",
    "request_id",
    "batch_id",
    "worker_id",
    "endpoint",
    "current_step",
)


class StructuredJsonFormatter(logging.Formatter):
    """Formatter that emits one JSON payload per log record."""

    _reserved = {
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
        "message",
        "asctime",
    }

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        for field in STABLE_LOG_FIELDS:
            payload[field] = getattr(record, field, None)

        extra = {
            key: value
            for key, value in record.__dict__.items()
            if key not in self._reserved
            and key not in STABLE_LOG_FIELDS
            and not key.startswith("_")
        }
        if extra:
            payload["extra"] = extra
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class PrettyLogFormatter(logging.Formatter):
    """Human-friendly formatter for local development."""

    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.fromtimestamp(record.created, tz=timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        context_bits = [
            f"{field}={getattr(record, field)}"
            for field in STABLE_LOG_FIELDS
            if getattr(record, field, None) is not None
        ]
        context = f" [{' '.join(context_bits)}]" if context_bits else ""
        return f"{timestamp} {record.levelname} {record.name} - {record.getMessage()}{context}"


def resolve_log_format(raw_format: str | None = None) -> str:
    """Resolve requested output format with safe fallback."""

    requested_value = raw_format
    if requested_value is None:
        requested_value = os.getenv(ENV_LOG_FORMAT)
    if requested_value is None:
        requested_value = DEFAULT_LOG_FORMAT
    requested = requested_value.lower()
    aliases = {"text": "pretty", "human": "pretty"}
    resolved = aliases.get(requested, requested)
    if resolved not in {"json", "pretty"}:
        logging.getLogger("atroposlib.logging").warning(
            "Invalid log format requested; falling back to JSON",
            extra={"requested_log_format": requested},
        )
        return DEFAULT_LOG_FORMAT
    return resolved


def configure_logging(
    logger_name: str = "atroposlib",
    *,
    level: int = logging.INFO,
    log_format: str | None = None,
) -> logging.Logger:
    """Configure a logger for structured output and return it."""

    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.propagate = False
    logger.handlers.clear()

    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(level)

    resolved_format = resolve_log_format(log_format)
    if resolved_format == "pretty":
        handler.setFormatter(PrettyLogFormatter())
    else:
        handler.setFormatter(StructuredJsonFormatter())

    logger.addHandler(handler)
    return logger


def build_log_context(**context: Any) -> dict[str, Any]:
    """Return stable context keys with optional values for `logging.extra`."""

    payload = {field: context.get(field) for field in STABLE_LOG_FIELDS}
    for key, value in context.items():
        if key not in payload:
            payload[key] = value
    return payload
