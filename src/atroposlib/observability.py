"""Production observability primitives for runtime API + RL environments."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

try:  # pragma: no cover - optional dependency in minimal installs.
    from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram
    from prometheus_client import generate_latest as prometheus_generate_latest
except Exception:  # pragma: no cover - fallback when prometheus_client is unavailable.
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"
    CollectorRegistry = None
    Counter = Gauge = Histogram = None
    prometheus_generate_latest = None

try:  # pragma: no cover - optional dependency in minimal installs.
    from opentelemetry import trace
except Exception:  # pragma: no cover - fallback when opentelemetry is unavailable.
    trace = None


@dataclass(slots=True)
class NoOpMetric:
    """Null-object metric used when Prometheus client is unavailable."""

    def labels(self, **_: str) -> NoOpMetric:
        return self

    def inc(self, amount: float = 1.0) -> None:
        _ = amount

    def set(self, value: float) -> None:
        _ = value

    def observe(self, value: float) -> None:
        _ = value


class RuntimeMetrics:
    """Prometheus-compatible metrics for rollout/runtime operations."""

    def __init__(self) -> None:
        self._registry = CollectorRegistry() if CollectorRegistry else None

        if self._registry and Counter and Gauge and Histogram:
            self.api_requests_total = Counter(
                "atropos_api_requests_total",
                "Count of API requests by endpoint and status.",
                labelnames=("method", "path", "status_code"),
                registry=self._registry,
            )
            self.api_errors_total = Counter(
                "atropos_api_errors_total",
                "Count of API errors (HTTP status >= 400).",
                labelnames=("method", "path", "status_code"),
                registry=self._registry,
            )
            self.rollout_latency_seconds = Histogram(
                "atropos_rollout_latency_seconds",
                "Latency distribution for rollout endpoints.",
                labelnames=("method", "path", "env"),
                registry=self._registry,
                buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10),
            )
            self.queue_size = Gauge(
                "atropos_queue_size",
                "Queue size per environment.",
                labelnames=("env",),
                registry=self._registry,
            )
            self.worker_utilization = Gauge(
                "atropos_worker_utilization",
                "Worker utilization ratio in [0, 1].",
                labelnames=("env",),
                registry=self._registry,
            )
        else:
            self.api_requests_total = NoOpMetric()
            self.api_errors_total = NoOpMetric()
            self.rollout_latency_seconds = NoOpMetric()
            self.queue_size = NoOpMetric()
            self.worker_utilization = NoOpMetric()

    @property
    def content_type(self) -> str:
        return CONTENT_TYPE_LATEST

    def generate_latest(self) -> bytes:
        if self._registry and prometheus_generate_latest:
            return prometheus_generate_latest(self._registry)
        return b""


def build_structured_log_payload(
    *,
    event: str,
    level: str = "info",
    **fields: Any,
) -> str:
    """Render a structured JSON log payload.

    Producing JSON directly keeps logs structured even when no JSON formatter is configured.
    """

    payload: dict[str, Any] = {
        "ts": time.time(),
        "event": event,
        "level": level,
        **fields,
    }
    return json.dumps(payload, separators=(",", ":"), default=str)


class BaseEnvTracingHooks:
    """OpenTelemetry hooks around BaseEnv step execution."""

    def __init__(self, env_name: str = "base_env") -> None:
        self.env_name = env_name
        self._tracer = trace.get_tracer("atropos.env") if trace else None

    @contextmanager
    def step_span(
        self,
        *,
        worker_count: int,
        payload: dict[str, Any],
    ) -> Generator[object | None, None, None]:
        if self._tracer is None:
            yield None
            return

        with self._tracer.start_as_current_span("base_env.step") as span:
            span.set_attribute("atropos.env.name", self.env_name)
            span.set_attribute("atropos.worker.requested", worker_count)
            span.set_attribute("atropos.payload.keys", ",".join(sorted(payload.keys())))
            yield span


def log_json_event(
    logger: logging.Logger,
    *,
    event: str,
    level: str = "info",
    **fields: Any,
) -> None:
    """Emit structured JSON log event via the requested logger."""

    message = build_structured_log_payload(event=event, level=level, **fields)
    log_method = getattr(logger, level.lower(), logger.info)
    log_method(message)
