"""Structured logging and metric tracking for environment execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MetricsLogger:
    """Capture events and metrics without hidden global side effects.

    Production deployments may wrap this manager with W&B or other telemetry
    sinks while preserving the same method-level contract.
    """

    events: list[dict[str, Any]] = field(default_factory=list)
    metrics: list[dict[str, float]] = field(default_factory=list)

    def log_event(self, event: str, level: str = "info", **metadata: Any) -> None:
        self.events.append({"level": level, "event": event, "metadata": metadata})

    def record_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        payload = dict(metrics)
        if step is not None:
            payload["step"] = float(step)
        self.metrics.append(payload)

    def reset(self) -> None:
        self.events.clear()
        self.metrics.clear()
