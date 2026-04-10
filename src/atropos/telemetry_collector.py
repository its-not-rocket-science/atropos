"""Active telemetry collection from running inference servers.

This module provides collectors that can query metrics from running
inference servers like vLLM, TGI, Triton, etc.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from .logging_config import get_logger
from .telemetry import TelemetryData

logger = get_logger("telemetry")


@dataclass
class CollectionConfig:
    """Configuration for telemetry collection.

    Attributes:
        collection_duration_sec: How long to collect metrics for.
        sampling_interval_sec: How often to sample metrics.
        warmup_requests: Number of warmup requests before collecting.
        benchmark_prompt: Prompt to use for benchmarking.
        max_tokens: Maximum tokens to generate.
    """

    collection_duration_sec: float = 60.0
    sampling_interval_sec: float = 5.0
    warmup_requests: int = 10
    benchmark_prompt: str = "Explain the concept of machine learning in simple terms."
    max_tokens: int = 256


@dataclass
class CollectionResult:
    """Result of a telemetry collection session.

    Attributes:
        success: Whether collection was successful.
        samples: List of telemetry samples collected.
        aggregated: Aggregated telemetry data (averages).
        error_message: Error message if collection failed.
        metadata: Additional collection metadata.
    """

    success: bool
    samples: list[TelemetryData] = field(default_factory=list)
    aggregated: TelemetryData | None = None
    error_message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class TelemetryRequestError(ConnectionError):
    """Raised when telemetry endpoint requests fail."""


class TelemetryParseError(ValueError):
    """Raised when telemetry payloads cannot be parsed."""


class TelemetryCollector(ABC):
    """Abstract base class for telemetry collectors."""

    RETRYABLE_HTTP_CODES = {408, 425, 429, 500, 502, 503, 504}

    def __init__(self, base_url: str, config: CollectionConfig | None = None):
        """Initialize the collector.

        Args:
            base_url: Base URL of the inference server.
            config: Collection configuration.
        """
        self.base_url = base_url.rstrip("/")
        self.config = config or CollectionConfig()

    @abstractmethod
    def check_health(self) -> bool:
        """Check if the inference server is healthy and reachable.

        Returns:
            True if server is healthy.
        """
        raise NotImplementedError

    @abstractmethod
    def collect(self) -> CollectionResult:
        """Collect telemetry from the inference server.

        Returns:
            CollectionResult with samples and aggregated data.
        """
        raise NotImplementedError

    def _request_text(
        self,
        endpoint: str,
        *,
        method: str = "GET",
        data: dict[str, Any] | None = None,
        timeout: float = 30,
        retries: int = 3,
        backoff_base_sec: float = 0.25,
        retry_on_status: set[int] | None = None,
    ) -> str:
        """Request endpoint text with retry/backoff for transient failures."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        req = urllib.request.Request(url, method=method)
        req.add_header("Content-Type", "application/json")

        if data is not None:
            req.data = json.dumps(data).encode("utf-8")

        retryable_status = retry_on_status or self.RETRYABLE_HTTP_CODES
        last_error: Exception | None = None

        for attempt in range(1, retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=timeout) as response:
                    payload = cast(bytes, response.read())
                    return payload.decode("utf-8")
            except urllib.error.HTTPError as exc:
                last_error = exc
                if exc.code not in retryable_status or attempt == retries:
                    break
                sleep_sec = backoff_base_sec * (2 ** (attempt - 1))
                logger.warning(
                    "Transient HTTP %s for %s (%s/%s); retrying in %.2fs",
                    exc.code,
                    url,
                    attempt,
                    retries,
                    sleep_sec,
                )
                time.sleep(sleep_sec)
            except urllib.error.URLError as exc:
                last_error = exc
                if attempt == retries:
                    break
                sleep_sec = backoff_base_sec * (2 ** (attempt - 1))
                logger.warning(
                    "Network error for %s (%s/%s); retrying in %.2fs: %s",
                    url,
                    attempt,
                    retries,
                    sleep_sec,
                    exc,
                )
                time.sleep(sleep_sec)

        raise TelemetryRequestError(f"Failed to request {url}: {last_error}") from last_error

    def _make_request(
        self,
        endpoint: str,
        method: str = "GET",
        data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Make JSON HTTP request to the inference server.

        Raises:
            TelemetryRequestError: If request fails.
            TelemetryParseError: If response is not JSON.
        """
        text = self._request_text(endpoint, method=method, data=data)
        try:
            result: dict[str, Any] = json.loads(text)
            return result
        except json.JSONDecodeError as exc:
            url = f"{self.base_url}/{endpoint.lstrip('/')}"
            raise TelemetryParseError(f"Invalid JSON from {url}") from exc


class VLLMCollector(TelemetryCollector):
    """Collector for vLLM inference server metrics."""

    FIELD_QUALITY: dict[str, str] = {
        "memory_gb": "estimated_from_metric",  # gpu_cache_usage_perc * assumed 80GB
        "throughput_toks_per_sec": "estimated_from_metric",  # 1 / time_per_token
        "latency_ms_per_request": "estimated_from_metric",  # time_per_token * configured tokens
        "tokens_per_request": "estimated_configured",
    }

    def check_health(self) -> bool:
        """Check vLLM health endpoint."""
        try:
            response = self._make_request("/health")
        except (TelemetryRequestError, TelemetryParseError) as exc:
            logger.warning("vLLM health check failed: %s", exc)
            return False
        return response.get("status") == "healthy"

    def collect(self) -> CollectionResult:
        """Collect telemetry from vLLM server.

        Uses vLLM's /metrics endpoint for Prometheus metrics
        and optionally runs a benchmark workload.
        """
        try:
            if not self.check_health():
                return CollectionResult(success=False, error_message="vLLM server is not healthy")

            self._run_warmup()

            samples: list[TelemetryData] = []
            start_time = time.time()

            while time.time() - start_time < self.config.collection_duration_sec:
                sample = self._collect_sample()
                if sample:
                    samples.append(sample)
                time.sleep(self.config.sampling_interval_sec)

            if not samples:
                return CollectionResult(success=False, error_message="No samples collected")

            aggregated = self._aggregate_samples(samples)

            return CollectionResult(
                success=True,
                samples=samples,
                aggregated=aggregated,
                metadata={
                    "collection_duration_sec": self.config.collection_duration_sec,
                    "sample_count": len(samples),
                    "server_type": "vllm",
                    "field_quality": self.FIELD_QUALITY,
                },
            )

        except (TelemetryRequestError, TelemetryParseError, ValueError) as exc:
            return CollectionResult(success=False, error_message=str(exc))

    def _run_warmup(self) -> None:
        """Send warmup requests to stabilize the server."""
        for _ in range(self.config.warmup_requests):
            try:
                self._make_request(
                    "/v1/completions",
                    method="POST",
                    data={
                        "model": "default",
                        "prompt": self.config.benchmark_prompt,
                        "max_tokens": 10,
                    },
                )
            except (TelemetryRequestError, TelemetryParseError) as exc:
                logger.debug("vLLM warmup request failed (continuing): %s", exc)

    def _collect_sample(self) -> TelemetryData | None:
        """Collect a single telemetry sample."""
        try:
            metrics_text = self._request_text("/metrics", timeout=10, retries=2)
            return self._parse_prometheus_metrics(metrics_text)
        except (TelemetryRequestError, TelemetryParseError) as metrics_error:
            logger.info(
                "vLLM metrics endpoint unavailable, using request estimate: %s",
                metrics_error,
            )
            return self._estimate_from_request(metrics_error)

    def _parse_prometheus_metrics(self, metrics_text: str) -> TelemetryData:
        """Parse Prometheus-style metrics from vLLM."""
        metrics: dict[str, float] = {}

        for line in metrics_text.split("\n"):
            if "vllm:gpu_cache_usage_perc" in line and not line.startswith("#"):
                try:
                    metrics["gpu_cache_usage"] = float(line.split()[-1])
                except (ValueError, IndexError) as exc:
                    raise TelemetryParseError("Malformed vLLM gpu_cache_usage metric") from exc
            elif "vllm:num_requests_running" in line and not line.startswith("#"):
                try:
                    metrics["requests_running"] = float(line.split()[-1])
                except (ValueError, IndexError) as exc:
                    raise TelemetryParseError("Malformed vLLM requests_running metric") from exc
            elif "vllm:time_per_output_token_seconds" in line and not line.startswith("#"):
                try:
                    metrics["time_per_token_ms"] = float(line.split()[-1]) * 1000
                except (ValueError, IndexError) as exc:
                    raise TelemetryParseError(
                        "Malformed vLLM time_per_output_token metric"
                    ) from exc

        if not metrics:
            raise TelemetryParseError(
                "vLLM metrics endpoint returned no recognized telemetry fields"
            )

        time_per_token_ms = metrics.get("time_per_token_ms", 0.0)
        throughput = 1000 / time_per_token_ms if time_per_token_ms > 0 else 0.0

        return TelemetryData(
            source="vllm",
            memory_gb=metrics.get("gpu_cache_usage", 0.0) * 80,
            throughput_toks_per_sec=throughput,
            latency_ms_per_request=time_per_token_ms * self.config.max_tokens,
            tokens_per_request=float(self.config.max_tokens),
            timestamp=str(time.time()),
            raw_metrics={
                "source_type": "prometheus",
                "metrics": metrics,
                "metric_quality": self.FIELD_QUALITY,
                "heuristics": {
                    "memory_gb": "gpu_cache_usage_perc multiplied by assumed 80GB GPU memory",
                    "throughput_toks_per_sec": "inverse of time_per_output_token_seconds",
                    "latency_ms_per_request": (
                        "time_per_output_token_seconds scaled by configured max_tokens"
                    ),
                },
            },
        )

    def _estimate_from_request(self, metrics_error: Exception) -> TelemetryData | None:
        """Estimate metrics from a single completion request."""
        start_time = time.time()

        try:
            response = self._make_request(
                "/v1/completions",
                method="POST",
                data={
                    "model": "default",
                    "prompt": self.config.benchmark_prompt,
                    "max_tokens": self.config.max_tokens,
                },
            )
        except (TelemetryRequestError, TelemetryParseError) as exc:
            logger.warning("vLLM fallback request failed: %s", exc)
            return None

        elapsed_ms = (time.time() - start_time) * 1000
        tokens_generated = response.get("usage", {}).get("completion_tokens", 0)
        throughput = (tokens_generated / elapsed_ms) * 1000 if elapsed_ms > 0 else 0.0

        return TelemetryData(
            source="vllm",
            memory_gb=0.0,
            throughput_toks_per_sec=throughput,
            latency_ms_per_request=elapsed_ms,
            tokens_per_request=float(tokens_generated or self.config.max_tokens),
            timestamp=str(time.time()),
            raw_metrics={
                "source_type": "request_fallback",
                "fallback_reason": str(metrics_error),
                "response": response,
                "metric_quality": {
                    "memory_gb": "unavailable",
                    "throughput_toks_per_sec": "measured_from_request",
                    "latency_ms_per_request": "measured_from_request",
                    "tokens_per_request": (
                        "measured_from_request" if tokens_generated else "estimated_configured"
                    ),
                },
                "heuristics": {
                    "tokens_per_request": "configured max_tokens when completion_tokens missing"
                },
            },
        )

    def _aggregate_samples(self, samples: list[TelemetryData]) -> TelemetryData:
        """Aggregate multiple samples into a single TelemetryData."""
        if not samples:
            raise ValueError("No samples to aggregate")

        n = len(samples)
        return TelemetryData(
            source="vllm",
            memory_gb=sum(s.memory_gb for s in samples) / n,
            throughput_toks_per_sec=sum(s.throughput_toks_per_sec for s in samples) / n,
            latency_ms_per_request=sum(s.latency_ms_per_request for s in samples) / n,
            tokens_per_request=sum(s.tokens_per_request for s in samples) / n,
            timestamp=str(time.time()),
            raw_metrics={"sample_count": n, "metric_quality": "aggregated_mean"},
        )


class TGICollector(TelemetryCollector):
    """Collector for Text Generation Inference (TGI) server metrics."""

    FIELD_QUALITY: dict[str, str] = {
        "memory_gb": "unavailable",
        "throughput_toks_per_sec": "unavailable",
        "latency_ms_per_request": "unavailable",
        "tokens_per_request": "estimated_configured",
    }

    def check_health(self) -> bool:
        """Check TGI health endpoint."""
        try:
            self._make_request("/health")
        except (TelemetryRequestError, TelemetryParseError) as exc:
            logger.warning("TGI health check failed: %s", exc)
            return False
        return True

    def collect(self) -> CollectionResult:
        """Collect telemetry from TGI server."""
        try:
            if not self.check_health():
                return CollectionResult(success=False, error_message="TGI server is not healthy")

            samples: list[TelemetryData] = []
            start_time = time.time()

            while time.time() - start_time < self.config.collection_duration_sec:
                sample = self._collect_sample()
                if sample:
                    samples.append(sample)
                time.sleep(self.config.sampling_interval_sec)

            if not samples:
                return CollectionResult(success=False, error_message="No samples collected")

            aggregated = self._aggregate_samples(samples)

            return CollectionResult(
                success=True,
                samples=samples,
                aggregated=aggregated,
                metadata={
                    "collection_duration_sec": self.config.collection_duration_sec,
                    "sample_count": len(samples),
                    "server_type": "tgi",
                    "field_quality": self.FIELD_QUALITY,
                },
            )

        except (TelemetryRequestError, TelemetryParseError, ValueError) as exc:
            return CollectionResult(success=False, error_message=str(exc))

    def _collect_sample(self) -> TelemetryData | None:
        """Collect a single telemetry sample from TGI."""
        try:
            metrics_text = self._request_text("/metrics", timeout=10, retries=2)
            return self._parse_prometheus_metrics(metrics_text)
        except (TelemetryRequestError, TelemetryParseError) as exc:
            logger.warning("TGI metrics collection failed: %s", exc)
            return None

    def _parse_prometheus_metrics(self, metrics_text: str) -> TelemetryData:
        """Parse Prometheus-style metrics from TGI."""
        metrics: dict[str, float] = {}

        for line in metrics_text.split("\n"):
            if "tgi_batch_current_size" in line and not line.startswith("#"):
                try:
                    metrics["batch_size"] = float(line.split()[-1])
                except (ValueError, IndexError) as exc:
                    raise TelemetryParseError("Malformed TGI batch size metric") from exc
            elif "tgi_queue_size" in line and not line.startswith("#"):
                try:
                    metrics["queue_size"] = float(line.split()[-1])
                except (ValueError, IndexError) as exc:
                    raise TelemetryParseError("Malformed TGI queue size metric") from exc

        if not metrics:
            raise TelemetryParseError(
                "TGI metrics endpoint returned no recognized telemetry fields"
            )

        return TelemetryData(
            source="tgi",
            memory_gb=0.0,
            throughput_toks_per_sec=0.0,
            latency_ms_per_request=0.0,
            tokens_per_request=float(self.config.max_tokens),
            timestamp=str(time.time()),
            raw_metrics={
                "source_type": "prometheus",
                "metrics": metrics,
                "metric_quality": self.FIELD_QUALITY,
            },
        )

    def _aggregate_samples(self, samples: list[TelemetryData]) -> TelemetryData:
        """Aggregate multiple samples."""
        if not samples:
            raise ValueError("No samples to aggregate")
        n = len(samples)
        return TelemetryData(
            source="tgi",
            memory_gb=sum(s.memory_gb for s in samples) / n,
            throughput_toks_per_sec=sum(s.throughput_toks_per_sec for s in samples) / n,
            latency_ms_per_request=sum(s.latency_ms_per_request for s in samples) / n,
            tokens_per_request=sum(s.tokens_per_request for s in samples) / n,
            timestamp=str(time.time()),
            raw_metrics={"sample_count": n, "metric_quality": "aggregated_mean"},
        )


class TritonCollector(TelemetryCollector):
    """Collector for NVIDIA Triton Inference Server metrics."""

    FIELD_QUALITY: dict[str, str] = {
        "memory_gb": "unavailable",
        "throughput_toks_per_sec": "estimated_from_counter",
        "latency_ms_per_request": "measured_from_stats",
        "tokens_per_request": "estimated_configured",
    }

    def check_health(self) -> bool:
        """Check Triton health endpoint."""
        try:
            self._make_request("/v2/health/ready")
        except (TelemetryRequestError, TelemetryParseError) as exc:
            logger.warning("Triton health check failed: %s", exc)
            return False
        return True

    def collect(self) -> CollectionResult:
        """Collect telemetry from Triton server."""
        try:
            if not self.check_health():
                return CollectionResult(success=False, error_message="Triton server is not healthy")

            stats = self._make_request("/v2/models/stats")

            samples: list[TelemetryData] = []
            start_time = time.time()

            while time.time() - start_time < self.config.collection_duration_sec:
                sample = self._collect_sample(stats)
                if sample:
                    samples.append(sample)
                time.sleep(self.config.sampling_interval_sec)

            if not samples:
                return CollectionResult(success=False, error_message="No samples collected")

            aggregated = self._aggregate_samples(samples)

            return CollectionResult(
                success=True,
                samples=samples,
                aggregated=aggregated,
                metadata={
                    "collection_duration_sec": self.config.collection_duration_sec,
                    "sample_count": len(samples),
                    "server_type": "triton",
                    "field_quality": self.FIELD_QUALITY,
                },
            )

        except (TelemetryRequestError, TelemetryParseError, ValueError) as exc:
            return CollectionResult(success=False, error_message=str(exc))

    def _collect_sample(self, model_stats: dict[str, Any]) -> TelemetryData | None:
        """Collect a single telemetry sample from Triton."""
        model_stats_data = model_stats.get("model_stats")
        if not isinstance(model_stats_data, dict) or not model_stats_data:
            logger.warning("Triton model stats missing or malformed")
            return None

        model_name = next(iter(model_stats_data.keys()))
        stats = model_stats_data.get(model_name)
        if not isinstance(stats, dict):
            logger.warning("Triton stats for model '%s' are malformed", model_name)
            return None

        inference_stats = stats.get("inference_stats", {})
        success_count = inference_stats.get("success", {}).get("count", 0)
        compute_time_ns = inference_stats.get("compute_infer", {}).get("ns", 0)

        if not isinstance(success_count, (int, float)) or not isinstance(
            compute_time_ns, (int, float)
        ):
            logger.warning("Triton counters are not numeric")
            return None

        compute_time_ms = compute_time_ns / 1e6

        return TelemetryData(
            source="triton",
            memory_gb=0.0,
            throughput_toks_per_sec=float(success_count),
            latency_ms_per_request=compute_time_ms,
            tokens_per_request=float(self.config.max_tokens),
            timestamp=str(time.time()),
            raw_metrics={
                "source_type": "model_stats",
                "model_name": model_name,
                "stats": stats,
                "metric_quality": self.FIELD_QUALITY,
                "heuristics": {
                    "throughput_toks_per_sec": (
                        "derived from cumulative success counter; not interval-normalized"
                    )
                },
            },
        )

    def _aggregate_samples(self, samples: list[TelemetryData]) -> TelemetryData:
        """Aggregate multiple samples."""
        if not samples:
            raise ValueError("No samples to aggregate")
        n = len(samples)
        return TelemetryData(
            source="triton",
            memory_gb=sum(s.memory_gb for s in samples) / n,
            throughput_toks_per_sec=sum(s.throughput_toks_per_sec for s in samples) / n,
            latency_ms_per_request=sum(s.latency_ms_per_request for s in samples) / n,
            tokens_per_request=sum(s.tokens_per_request for s in samples) / n,
            timestamp=str(time.time()),
            raw_metrics={"sample_count": n, "metric_quality": "aggregated_mean"},
        )


COLLECTORS: dict[str, type[TelemetryCollector]] = {
    "vllm": VLLMCollector,
    "tgi": TGICollector,
    "triton": TritonCollector,
}


def get_collector(
    server_type: str, base_url: str, config: CollectionConfig | None = None
) -> TelemetryCollector:
    """Get a telemetry collector by server type.

    Args:
        server_type: Type of server (vllm, tgi, triton).
        base_url: Base URL of the server.
        config: Optional collection configuration.

    Returns:
        TelemetryCollector instance.

    Raises:
        ValueError: If server type is unknown.
    """
    if server_type not in COLLECTORS:
        available = ", ".join(COLLECTORS.keys())
        raise ValueError(f"Unknown server type '{server_type}'. Available: {available}")

    collector_class = COLLECTORS[server_type]
    return collector_class(base_url, config)


def collect_and_save(
    server_type: str,
    base_url: str,
    output_path: Path,
    config: CollectionConfig | None = None,
) -> CollectionResult:
    """Collect telemetry and save to file.

    Args:
        server_type: Type of inference server.
        base_url: URL of the server.
        output_path: Path to save telemetry JSON.
        config: Optional collection configuration.

    Returns:
        CollectionResult with collected data.
    """
    collector = get_collector(server_type, base_url, config)
    result = collector.collect()

    if result.success and result.aggregated:
        data = {
            "source": result.aggregated.source,
            "memory_gb": result.aggregated.memory_gb,
            "throughput_toks_per_sec": result.aggregated.throughput_toks_per_sec,
            "latency_ms_per_request": result.aggregated.latency_ms_per_request,
            "tokens_per_request": result.aggregated.tokens_per_request,
            "parameters_b": result.aggregated.parameters_b,
            "power_watts": result.aggregated.power_watts,
            "requests_per_day": result.aggregated.requests_per_day,
            "timestamp": result.aggregated.timestamp,
            "raw_metrics": result.aggregated.raw_metrics,
            "collection_metadata": result.metadata,
        }
        output_path.write_text(json.dumps(data, indent=2))
        logger.info("Telemetry saved to %s", output_path)

    return result
