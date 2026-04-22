"""Observability primitives for runtime APIs and environment orchestration."""

from __future__ import annotations

import logging
import os
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

_TRACING_CONFIGURED = False
_TRACING_CONFIG_ERROR: str | None = None


@dataclass(frozen=True, slots=True)
class TracingConfig:
    """Toggleable OpenTelemetry tracing configuration."""

    enabled: bool = False
    service_name: str = "atropos-runtime"
    exporter: str = "otlp"
    endpoint: str | None = None
    insecure: bool = True
    sample_ratio: float = 1.0


def tracing_config_from_env() -> TracingConfig:
    """Build tracing config from environment variables."""

    enabled = os.getenv("ATROPOS_TRACING_ENABLED", "false").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    exporter = os.getenv("ATROPOS_TRACING_EXPORTER", "otlp").strip().lower()
    endpoint = os.getenv("ATROPOS_TRACING_ENDPOINT")
    service_name = os.getenv("ATROPOS_TRACING_SERVICE_NAME", "atropos-runtime")
    insecure = os.getenv("ATROPOS_TRACING_INSECURE", "true").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    try:
        sample_ratio = float(os.getenv("ATROPOS_TRACING_SAMPLE_RATIO", "1.0"))
    except ValueError:
        sample_ratio = 1.0
    clipped_sample_ratio = max(0.0, min(sample_ratio, 1.0))
    return TracingConfig(
        enabled=enabled,
        service_name=service_name,
        exporter=exporter,
        endpoint=endpoint,
        insecure=insecure,
        sample_ratio=clipped_sample_ratio,
    )


def configure_tracing(config: TracingConfig | None = None) -> bool:
    """Configure global OpenTelemetry tracer provider.

    Returns True when tracing is enabled and initialized, otherwise False.
    """

    global _TRACING_CONFIGURED, _TRACING_CONFIG_ERROR
    if _TRACING_CONFIGURED:
        return True
    resolved = config or tracing_config_from_env()
    if not resolved.enabled:
        return False
    if find_spec("opentelemetry.sdk") is None:
        _TRACING_CONFIG_ERROR = (
            "Tracing enabled but OpenTelemetry SDK is not installed "
            "(install opentelemetry-sdk + exporter package)."
        )
        return False

    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.sdk.trace.sampling import TraceIdRatioBased

    exporter: Any
    if resolved.exporter == "console":
        exporter = ConsoleSpanExporter()
    elif resolved.exporter == "jaeger":
        if find_spec("opentelemetry.exporter.jaeger.thrift") is None:
            _TRACING_CONFIG_ERROR = (
                "Jaeger exporter selected but package is missing "
                "(install opentelemetry-exporter-jaeger-thrift)."
            )
            return False
        from opentelemetry.exporter.jaeger.thrift import JaegerExporter

        exporter = JaegerExporter(
            collector_endpoint=resolved.endpoint or "http://localhost:14268/api/traces",
        )
    else:
        if find_spec("opentelemetry.exporter.otlp.proto.http.trace_exporter") is None:
            _TRACING_CONFIG_ERROR = (
                "OTLP exporter selected but package is missing "
                "(install opentelemetry-exporter-otlp-proto-http)."
            )
            return False
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        exporter = OTLPSpanExporter(
            endpoint=resolved.endpoint or "http://localhost:4318/v1/traces",
        )

    provider = TracerProvider(
        resource=Resource.create({"service.name": resolved.service_name}),
        sampler=TraceIdRatioBased(resolved.sample_ratio),
    )
    provider.add_span_processor(BatchSpanProcessor(exporter))
    if otel_trace is None:
        return False
    otel_trace.set_tracer_provider(provider)
    _TRACING_CONFIGURED = True
    _TRACING_CONFIG_ERROR = None
    return True


def setup_json_logging(logger_name: str = "atropos", *, level: int = logging.INFO) -> None:
    """Enable structured JSON logs for the provided logger hierarchy."""

    configure_logging(logger_name=logger_name, level=level, log_format="json")


@dataclass(slots=True)
class Observability:
    """Metrics and tracing helpers shared by runtime and env layers."""

    api_requests_total: Any
    api_request_latency_seconds: Any
    api_errors_total: Any
    api_requests_by_env_total: Any
    runtime_queue_depth: Any
    runtime_queue_oldest_age_seconds: Any
    buffered_groups: Any
    ingestion_records_total: Any
    duplicate_ingestion_rejections_total: Any
    duplicate_ingestion_groups_total: Any
    batch_formation_latency_seconds: Any
    rollout_latency_seconds: Any
    worker_count: Any
    worker_utilization_ratio: Any
    trainer_queue_depth: Any
    env_rate_limit_ratio: Any
    failed_sends_total: Any
    eval_duration_seconds: Any
    worker_dependency_checks_total: Any
    worker_dependency_failures_total: Any
    worker_dependency_ready: Any
    runtime_groups_by_state: Any

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
            api_requests_by_env_total=Counter(
                "atropos_api_requests_by_env_total",
                "Total API requests partitioned by environment",
                ["env", "method", "path", "status"],
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
            duplicate_ingestion_groups_total=Counter(
                "atropos_duplicate_ingestion_groups_total",
                "Total duplicate scored-data groups rejected",
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
            worker_dependency_checks_total=Counter(
                "atropos_worker_dependency_checks_total",
                "Total worker dependency readiness checks",
                ["dependency"],
            ),
            worker_dependency_failures_total=Counter(
                "atropos_worker_dependency_failures_total",
                "Total failed worker dependency checks",
                ["dependency"],
            ),
            worker_dependency_ready=Gauge(
                "atropos_worker_dependency_ready",
                "Latest worker dependency readiness state (1=ready, 0=not ready)",
                ["dependency"],
            ),
            runtime_groups_by_state=Gauge(
                "atropos_runtime_groups_by_state",
                "Count of scored-data groups by lifecycle state and environment",
                ["env", "state"],
            ),
        )

    def observe_api_request(
        self,
        *,
        method: str,
        path: str,
        status: int,
        duration_seconds: float,
        env: str | None = None,
    ) -> None:
        labels = {"method": method, "path": path}
        status_labels = {**labels, "status": str(status)}
        self.api_requests_total.labels(**status_labels).inc()
        self.api_request_latency_seconds.labels(**labels).observe(duration_seconds)
        if env:
            env_labels = {"env": env, **status_labels}
            self.api_requests_by_env_total.labels(**env_labels).inc()
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

    def observe_duplicate_groups(self, *, env: str, endpoint: str, duplicate_groups: int) -> None:
        if duplicate_groups > 0:
            self.duplicate_ingestion_groups_total.labels(env=env, endpoint=endpoint).inc(
                duplicate_groups
            )

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

    def observe_worker_dependency_check(self, *, dependency: str, ready: bool) -> None:
        self.worker_dependency_checks_total.labels(dependency=dependency).inc()
        if not ready:
            self.worker_dependency_failures_total.labels(dependency=dependency).inc()
        self.worker_dependency_ready.labels(dependency=dependency).set(1.0 if ready else 0.0)

    def set_runtime_groups_by_state(self, *, env: str, state: str, count: int) -> None:
        self.runtime_groups_by_state.labels(env=env, state=state).set(max(0, count))


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
        try:
            yield span
        except Exception as exc:
            span.record_exception(exc)
            raise


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
