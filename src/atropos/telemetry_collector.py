"""Active telemetry collection from running inference servers.

This module provides collectors that can query metrics from running
inference servers like vLLM, TGI, Triton, etc.
"""

from __future__ import annotations

import json
import time
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .telemetry import TelemetryData


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


class TelemetryCollector(ABC):
    """Abstract base class for telemetry collectors."""

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

    def _make_request(
        self, endpoint: str, method: str = "GET", data: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Make HTTP request to the inference server.

        Args:
            endpoint: API endpoint (relative to base_url).
            method: HTTP method.
            data: Request body for POST requests.

        Returns:
            Parsed JSON response.

        Raises:
            ConnectionError: If request fails.
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        req = urllib.request.Request(url, method=method)
        req.add_header("Content-Type", "application/json")

        if data:
            req.data = json.dumps(data).encode("utf-8")

        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                result: dict[str, Any] = json.loads(response.read().decode("utf-8"))
                return result
        except urllib.error.URLError as e:
            raise ConnectionError(f"Failed to connect to {url}: {e}") from e


class VLLMCollector(TelemetryCollector):
    """Collector for vLLM inference server metrics."""

    def check_health(self) -> bool:
        """Check vLLM health endpoint."""
        try:
            response = self._make_request("/health")
            return response.get("status") == "healthy"
        except Exception:
            return False

    def collect(self) -> CollectionResult:
        """Collect telemetry from vLLM server.

        Uses vLLM's /metrics endpoint for Prometheus metrics
        and optionally runs a benchmark workload.
        """
        try:
            # Check health first
            if not self.check_health():
                return CollectionResult(
                    success=False,
                    error_message="vLLM server is not healthy",
                )

            # Run warmup
            self._run_warmup()

            # Collect metrics over time
            samples: list[TelemetryData] = []
            start_time = time.time()

            while time.time() - start_time < self.config.collection_duration_sec:
                sample = self._collect_sample()
                if sample:
                    samples.append(sample)
                time.sleep(self.config.sampling_interval_sec)

            if not samples:
                return CollectionResult(
                    success=False,
                    error_message="No samples collected",
                )

            # Aggregate samples
            aggregated = self._aggregate_samples(samples)

            return CollectionResult(
                success=True,
                samples=samples,
                aggregated=aggregated,
                metadata={
                    "collection_duration_sec": self.config.collection_duration_sec,
                    "sample_count": len(samples),
                    "server_type": "vllm",
                },
            )

        except Exception as e:
            return CollectionResult(
                success=False,
                error_message=str(e),
            )

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
            except Exception:
                pass  # Warmup failures are ok

    def _collect_sample(self) -> TelemetryData | None:
        """Collect a single telemetry sample."""
        try:
            # Try to get metrics from /metrics endpoint (Prometheus format)
            metrics_url = f"{self.base_url}/metrics"
            req = urllib.request.Request(metrics_url)

            with urllib.request.urlopen(req, timeout=10) as response:
                metrics_text = response.read().decode("utf-8")
                return self._parse_prometheus_metrics(metrics_text)

        except Exception:
            # Fallback: estimate from a single completion request
            return self._estimate_from_request()

    def _parse_prometheus_metrics(self, metrics_text: str) -> TelemetryData | None:
        """Parse Prometheus-style metrics from vLLM."""
        metrics: dict[str, float] = {}

        for line in metrics_text.split("\n"):
            # Look for key metrics
            if "vllm:gpu_cache_usage_perc" in line and not line.startswith("#"):
                try:
                    value = float(line.split()[-1])
                    metrics["gpu_cache_usage"] = value
                except (ValueError, IndexError):
                    pass

            elif "vllm:num_requests_running" in line and not line.startswith("#"):
                try:
                    value = float(line.split()[-1])
                    metrics["requests_running"] = value
                except (ValueError, IndexError):
                    pass

            elif "vllm:time_per_output_token_seconds" in line and not line.startswith("#"):
                try:
                    value = float(line.split()[-1])
                    metrics["time_per_token_ms"] = value * 1000
                except (ValueError, IndexError):
                    pass

        # Calculate throughput from time_per_token
        time_per_token_ms = metrics.get("time_per_token_ms", 0)
        throughput = 1000 / time_per_token_ms if time_per_token_ms > 0 else 0

        return TelemetryData(
            source="vllm",
            memory_gb=metrics.get("gpu_cache_usage", 0) * 80,  # Rough estimate
            throughput_toks_per_sec=throughput,
            latency_ms_per_request=time_per_token_ms * self.config.max_tokens,
            tokens_per_request=float(self.config.max_tokens),
            timestamp=str(time.time()),
            raw_metrics=metrics,
        )

    def _estimate_from_request(self) -> TelemetryData | None:
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

            elapsed_ms = (time.time() - start_time) * 1000
            tokens_generated = response.get("usage", {}).get("completion_tokens", 0)

            throughput = (tokens_generated / elapsed_ms) * 1000 if elapsed_ms > 0 else 0

            return TelemetryData(
                source="vllm",
                memory_gb=0.0,  # Can't estimate from request alone
                throughput_toks_per_sec=throughput,
                latency_ms_per_request=elapsed_ms,
                tokens_per_request=float(tokens_generated or self.config.max_tokens),
                timestamp=str(time.time()),
                raw_metrics={"response": response},
            )

        except Exception:
            return None

    def _aggregate_samples(self, samples: list[TelemetryData]) -> TelemetryData:
        """Aggregate multiple samples into a single TelemetryData."""
        if not samples:
            raise ValueError("No samples to aggregate")

        n = len(samples)
        avg_memory = sum(s.memory_gb for s in samples) / n
        avg_throughput = sum(s.throughput_toks_per_sec for s in samples) / n
        avg_latency = sum(s.latency_ms_per_request for s in samples) / n
        avg_tokens = sum(s.tokens_per_request for s in samples) / n

        return TelemetryData(
            source="vllm",
            memory_gb=avg_memory,
            throughput_toks_per_sec=avg_throughput,
            latency_ms_per_request=avg_latency,
            tokens_per_request=avg_tokens,
            timestamp=str(time.time()),
            raw_metrics={"sample_count": n},
        )


class TGICollector(TelemetryCollector):
    """Collector for Text Generation Inference (TGI) server metrics."""

    def check_health(self) -> bool:
        """Check TGI health endpoint."""
        try:
            self._make_request("/health")
            return True
        except Exception:
            return False

    def collect(self) -> CollectionResult:
        """Collect telemetry from TGI server."""
        try:
            if not self.check_health():
                return CollectionResult(
                    success=False,
                    error_message="TGI server is not healthy",
                )

            # TGI has a /metrics endpoint with Prometheus metrics
            samples: list[TelemetryData] = []
            start_time = time.time()

            while time.time() - start_time < self.config.collection_duration_sec:
                sample = self._collect_sample()
                if sample:
                    samples.append(sample)
                time.sleep(self.config.sampling_interval_sec)

            if not samples:
                return CollectionResult(
                    success=False,
                    error_message="No samples collected",
                )

            aggregated = self._aggregate_samples(samples)

            return CollectionResult(
                success=True,
                samples=samples,
                aggregated=aggregated,
                metadata={
                    "collection_duration_sec": self.config.collection_duration_sec,
                    "sample_count": len(samples),
                    "server_type": "tgi",
                },
            )

        except Exception as e:
            return CollectionResult(
                success=False,
                error_message=str(e),
            )

    def _collect_sample(self) -> TelemetryData | None:
        """Collect a single telemetry sample from TGI."""
        try:
            metrics_url = f"{self.base_url}/metrics"
            req = urllib.request.Request(metrics_url)

            with urllib.request.urlopen(req, timeout=10) as response:
                metrics_text = response.read().decode("utf-8")
                return self._parse_prometheus_metrics(metrics_text)

        except Exception:
            return None

    def _parse_prometheus_metrics(self, metrics_text: str) -> TelemetryData | None:
        """Parse Prometheus-style metrics from TGI."""
        metrics: dict[str, float] = {}

        for line in metrics_text.split("\n"):
            if "tgi_batch_current_size" in line and not line.startswith("#"):
                try:
                    value = float(line.split()[-1])
                    metrics["batch_size"] = value
                except (ValueError, IndexError):
                    pass

            elif "tgi_queue_size" in line and not line.startswith("#"):
                try:
                    value = float(line.split()[-1])
                    metrics["queue_size"] = value
                except (ValueError, IndexError):
                    pass

        return TelemetryData(
            source="tgi",
            memory_gb=0.0,  # TGI doesn't expose memory directly
            throughput_toks_per_sec=0.0,  # Would need to calculate from rate
            latency_ms_per_request=0.0,
            tokens_per_request=float(self.config.max_tokens),
            timestamp=str(time.time()),
            raw_metrics=metrics,
        )

    def _aggregate_samples(self, samples: list[TelemetryData]) -> TelemetryData:
        """Aggregate multiple samples."""
        n = len(samples)
        return TelemetryData(
            source="tgi",
            memory_gb=sum(s.memory_gb for s in samples) / n,
            throughput_toks_per_sec=sum(s.throughput_toks_per_sec for s in samples) / n,
            latency_ms_per_request=sum(s.latency_ms_per_request for s in samples) / n,
            tokens_per_request=sum(s.tokens_per_request for s in samples) / n,
            timestamp=str(time.time()),
            raw_metrics={"sample_count": n},
        )


class TritonCollector(TelemetryCollector):
    """Collector for NVIDIA Triton Inference Server metrics."""

    def check_health(self) -> bool:
        """Check Triton health endpoint."""
        try:
            self._make_request("/v2/health/ready")
            return True
        except Exception:
            return False

    def collect(self) -> CollectionResult:
        """Collect telemetry from Triton server."""
        try:
            if not self.check_health():
                return CollectionResult(
                    success=False,
                    error_message="Triton server is not healthy",
                )

            # Get model statistics
            stats = self._make_request("/v2/models/stats")

            samples: list[TelemetryData] = []
            start_time = time.time()

            while time.time() - start_time < self.config.collection_duration_sec:
                sample = self._collect_sample(stats)
                if sample:
                    samples.append(sample)
                time.sleep(self.config.sampling_interval_sec)

            if not samples:
                return CollectionResult(
                    success=False,
                    error_message="No samples collected",
                )

            aggregated = self._aggregate_samples(samples)

            return CollectionResult(
                success=True,
                samples=samples,
                aggregated=aggregated,
                metadata={
                    "collection_duration_sec": self.config.collection_duration_sec,
                    "sample_count": len(samples),
                    "server_type": "triton",
                },
            )

        except Exception as e:
            return CollectionResult(
                success=False,
                error_message=str(e),
            )

    def _collect_sample(self, model_stats: dict[str, Any]) -> TelemetryData | None:
        """Collect a single telemetry sample from Triton."""
        try:
            # Parse model statistics
            model_name = list(model_stats.get("model_stats", {}).keys())[0]
            stats = model_stats["model_stats"][model_name]

            inference_stats = stats.get("inference_stats", {})
            success_count = inference_stats.get("success", {}).get("count", 0)
            compute_time = inference_stats.get("compute_infer", {}).get("ns", 0) / 1e6  # ms

            return TelemetryData(
                source="triton",
                memory_gb=0.0,  # Would need GPU metrics
                throughput_toks_per_sec=float(success_count),  # Simplified
                latency_ms_per_request=compute_time,
                tokens_per_request=float(self.config.max_tokens),
                timestamp=str(time.time()),
                raw_metrics=stats,
            )

        except Exception:
            return None

    def _aggregate_samples(self, samples: list[TelemetryData]) -> TelemetryData:
        """Aggregate multiple samples."""
        n = len(samples)
        return TelemetryData(
            source="triton",
            memory_gb=sum(s.memory_gb for s in samples) / n,
            throughput_toks_per_sec=sum(s.throughput_toks_per_sec for s in samples) / n,
            latency_ms_per_request=sum(s.latency_ms_per_request for s in samples) / n,
            tokens_per_request=sum(s.tokens_per_request for s in samples) / n,
            timestamp=str(time.time()),
            raw_metrics={"sample_count": n},
        )


# Registry of collectors
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
        # Save aggregated telemetry
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
        print(f"Telemetry saved to {output_path}")

    return result
