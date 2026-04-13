"""Backward-compatible LoggingManager shim."""

from __future__ import annotations

from dataclasses import dataclass

from .metrics_logger import MetricsLogger


@dataclass
class LoggingManager(MetricsLogger):
    """Compatibility alias for MetricsLogger."""
