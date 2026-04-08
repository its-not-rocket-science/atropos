"""Error categorization helpers for resilient batch processing."""

from __future__ import annotations

from typing import Literal

ErrorCategory = Literal[
    "timeout",
    "network",
    "resource",
    "config",
    "recoverable",
    "fatal",
    "unknown",
]


def categorize_error(error: Exception) -> ErrorCategory:
    """Map exceptions to operational error categories."""
    msg = str(error).lower()

    if isinstance(error, TimeoutError) or "timeout" in msg:
        return "timeout"
    if "connection" in msg or "network" in msg or "dns" in msg:
        return "network"
    if "out of memory" in msg or "cuda out of memory" in msg or "resource" in msg:
        return "resource"
    if isinstance(error, (KeyError, ValueError)) or "missing required" in msg or "invalid" in msg:
        return "config"
    if isinstance(error, (OSError, RuntimeError)):
        return "recoverable"
    if isinstance(error, (TypeError, AssertionError)):
        return "fatal"
    return "unknown"


def is_recoverable(error: Exception) -> bool:
    """Whether an error should be retried."""
    category = categorize_error(error)
    return category in {"timeout", "network", "resource", "recoverable"}
