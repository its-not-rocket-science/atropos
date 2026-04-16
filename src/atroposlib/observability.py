"""Observability primitives for runtime APIs and environment orchestration."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from importlib.util import find_spec
from time import perf_counter
from typing import Any

from .logging_utils import configure_logging

if find_spec("prometheus_client") is not None:
    from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
else:  # pragma: no cover - exercised when optional dependency is missing
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"

    _FALLBACK_METRIC_NAMES: set[str] = set()

    class _NoOpMetric:
        def __init__(self, name: str, metric_type: str) -> None:
            self._name = name
            self._metric_type = metric_type
            _FALLBACK_METRIC_NAMES.add(name)

        def labels(self, **_: str) -> _NoOpMetric:
            return self

        def inc(self, amount: float = 1.0) -> None:
            _ = amount

        def set(self, value: float) -> None:
            _ = value

        def observe(self, value: float) -> None:
            _ = value

    def _counter(name: str, *_: Any, **__: Any) -> _NoOpMetric:
        return _NoOpMetric(name=name, metric_type="counter")

    def _gauge(name: str, *_: Any, **__: Any) -> _NoOpMetric:
        return _NoOpMetric(name=name, metric_type="gauge")

    def _histogram(name: str, *_: Any, **__: Any) -> _NoOpMetric:
        return _NoOpMetric(name=name, metric_type="histogram")

    def generate_latest() -> bytes:  # type: ignore[misc]
        lines = [f"{name} 0" for name in sorted(_FALLBACK_METRIC_NAMES)]
        return ("\n".join(lines) + "\n").encode("utf-8")

    Counter = _counter
    Gauge = _gauge
    Histogram = _histogram

if find_spec("opentelemetry") is not None:
    from opentelemetry import trace as otel_trace
else:  # pragma: no cover - exercised when optional dependency is missing
    otel_trace = None


def setup_json_logging(logger_name: str = "atropos", *, level: int = logging.INFO) -> None:
    """Enable structured JSON logs for the provided logger hierarchy."""

    configure_logging(logger_name=logger_name, level=level, log_format="json")


@dataclass(slots=True)
class Observability:
    """Metrics and tracing helpers shared by runtime and env layers."""

    api_requests_total: Any
    api_request_latency_seconds: Any
    api_errors_total: Any
    runtime_queue_depth: Any
    runtime_queue_oldest_age_seconds: Any
    buffered_groups: Any
    ingestion_records_total: Any
    duplicate_ingestion_rejections_total: Any
    batch_formation_latency_seconds: Any
    rollout_latency_seconds: Any
    worker_count: Any
    worker_utilization_ratio: Any
    trainer_queue_depth: Any
    env_rate_limit_ratio: Any
    failed_sends_total: Any
    eval_duration_seconds: Any

    @classmethod
    def create(cls) -> Observability:
        return cls(
            api_requests_total=Counter(
                "atropos_api_requests_total",
                "Total API requests",
                ["method", "path", "status"],
            ),
            api_request_latency_seconds=Histogram(
                "atropos_api_request_latency_seconds",
                "API request latency in seconds",
                ["method", "path"],
            ),
            api_errors_total=Counter(
                "atropos_api_errors_total",
                "Total API error responses",
                ["method", "path", "status"],
            ),
            runtime_queue_depth=Gauge(
                "atropos_runtime_queue_depth",
                "Queue depth by environment",
                ["env"],
            ),
            runtime_queue_oldest_age_seconds=Gauge(
                "atropos_runtime_queue_oldest_age_seconds",
                "Oldest queued group age in seconds by environment",
                ["env"],
            ),
            buffered_groups=Gauge(
                "atropos_buffered_groups",
                "Buffered scored-data groups by environment",
                ["env"],
            ),
            ingestion_records_total=Counter(
                "atropos_ingestion_records_total",
                "Total ingested scored-data records",
                ["env"],
            ),
            duplicate_ingestion_rejections_total=Counter(
                "atropos_duplicate_ingestion_rejections_total",
                "Total duplicate ingestion requests rejected",
                ["env", "endpoint"],
            ),
            batch_formation_latency_seconds=Histogram(
                "atropos_batch_formation_latency_seconds",
                "Latency between buffered and batched group states",
                ["env"],
            ),
            rollout_latency_seconds=Histogram(
                "atropos_rollout_latency_seconds",
                "Rollout latency in seconds",
                ["env"],
            ),
            worker_count=Gauge(
                "atropos_worker_count",
                "Selected worker count by environment",
                ["env"],
            ),
            worker_utilization_ratio=Gauge(
                "atropos_worker_utilization_ratio",
                "Worker utilization ratio by environment",
                ["env"],
            ),
            trainer_queue_depth=Gauge(
                "atropos_trainer_queue_depth",
                "Observed trainer/API queue depth by environment",
                ["env"],
            ),
            env_rate_limit_ratio=Gauge(
                "atropos_env_rate_limit_ratio",
                "Adaptive environment rate limit ratio",
                ["env"],
            ),
            failed_sends_total=Counter(
                "atropos_failed_sends_total",
                "Total failed sends from runtime to API/model transport",
                ["env"],
            ),
            eval_duration_seconds=Histogram(
                "atropos_eval_duration_seconds",
                "End-to-end evaluation duration in seconds",
                ["env"],
            ),
        )

    def observe_api_request(
        self,
        *,
        method: str,
        path: str,
        status: int,
        duration_seconds: float,
    ) -> None:
        labels = {"method": method, "path": path}
        status_labels = {**labels, "status": str(status)}
        self.api_requests_total.labels(**status_labels).inc()
        self.api_request_latency_seconds.labels(**labels).observe(duration_seconds)
        if status >= 400:
            self.api_errors_total.labels(**status_labels).inc()

    def set_queue_depth(self, *, env: str, queue_depth: int) -> None:
        self.runtime_queue_depth.labels(env=env).set(queue_depth)

    def set_queue_oldest_age(self, *, env: str, oldest_age_seconds: float) -> None:
        self.runtime_queue_oldest_age_seconds.labels(env=env).set(max(0.0, oldest_age_seconds))

    def set_buffered_groups(self, *, env: str, group_count: int) -> None:
        self.buffered_groups.labels(env=env).set(max(0, group_count))

    def observe_ingestion(self, *, env: str, accepted_count: int) -> None:
        if accepted_count > 0:
            self.ingestion_records_total.labels(env=env).inc(accepted_count)

    def observe_duplicate_rejection(self, *, env: str, endpoint: str) -> None:
        self.duplicate_ingestion_rejections_total.labels(env=env, endpoint=endpoint).inc()

    def observe_batch_formation_latency(self, *, env: str, latency_seconds: float) -> None:
        self.batch_formation_latency_seconds.labels(env=env).observe(max(0.0, latency_seconds))

    def observe_rollout(self, *, env: str, latency_seconds: float) -> None:
        self.rollout_latency_seconds.labels(env=env).observe(latency_seconds)

    def set_worker_count(self, *, env: str, worker_count: int) -> None:
        self.worker_count.labels(env=env).set(max(0, worker_count))

    def set_worker_utilization(self, *, env: str, utilization_ratio: float) -> None:
        clipped = max(0.0, min(utilization_ratio, 1.0))
        self.worker_utilization_ratio.labels(env=env).set(clipped)

    def set_trainer_queue_depth(self, *, env: str, queue_depth: int) -> None:
        self.trainer_queue_depth.labels(env=env).set(max(0, queue_depth))

    def set_env_rate_limit(self, *, env: str, rate_limit: float) -> None:
        clipped = max(0.0, min(rate_limit, 1.0))
        self.env_rate_limit_ratio.labels(env=env).set(clipped)

    def observe_failed_send(self, *, env: str) -> None:
        self.failed_sends_total.labels(env=env).inc()

    def observe_eval_duration(self, *, env: str, duration_seconds: float) -> None:
        self.eval_duration_seconds.labels(env=env).observe(max(0.0, duration_seconds))


OBSERVABILITY = Observability.create()


@contextmanager
def tracing_span(name: str, *, attributes: dict[str, Any] | None = None) -> Any:
    """OpenTelemetry span wrapper with no-op fallback when SDK is unavailable."""

    if otel_trace is None:
        yield None
        return

    tracer = otel_trace.get_tracer("atroposlib.observability")
    with tracer.start_as_current_span(name) as span:
        for key, value in (attributes or {}).items():
            span.set_attribute(key, value)
        yield span


@contextmanager
def timed_rollout(env: str) -> Any:
    """Measure rollout latency and publish it as a histogram observation."""

    started = perf_counter()
    try:
        yield
    finally:
        OBSERVABILITY.observe_rollout(env=env, latency_seconds=perf_counter() - started)


def render_metrics() -> tuple[bytes, str]:
    """Render Prometheus-compatible exposition payload and content type."""

    return generate_latest(), CONTENT_TYPE_LATEST
