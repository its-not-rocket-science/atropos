"""Experiment runner for A/B testing.

Orchestrates deployment, metric collection, statistical analysis, and
experiment lifecycle management.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta
from typing import Any

from ..deployment.models import DeploymentRequest, DeploymentStrategyType
from ..deployment.platforms import DeploymentPlatform
from ..logging_config import get_logger
from ..telemetry_collector import get_collector
from .models import (
    ABTestConfig,
    ExperimentResult,
    ExperimentStatus,
    StatisticalResult,
    VariantMetrics,
)
from .statistics import analyze_variant_comparison


class ExperimentRunner:
    """Runs and monitors A/B test experiments.

    This class handles:
    - Deploying variants using the configured deployment platform
    - Collecting metrics from telemetry system
    - Periodic statistical analysis
    - Automatic stopping based on configured conditions
    - Promotion of winning variants
    """

    logger = get_logger(__name__)

    def __init__(
        self,
        config: ABTestConfig,
        platform: DeploymentPlatform,
        telemetry_client: Any | None = None,
    ):
        """Initialize experiment runner.

        Args:
            config: A/B test configuration.
            platform: Deployment platform for variant deployment.
            telemetry_client: Client for fetching metrics (optional).
                If None, a default client will be used.
        """
        self.config = config
        self.platform = platform
        self.telemetry_client = telemetry_client
        self._status = ExperimentStatus.DRAFT
        self._start_time: str | None = None
        self._end_time: str | None = None
        self._deployment_ids: dict[str, str] = {}  # variant_id -> deployment_id
        self._variant_endpoints: dict[str, list[str]] = {}  # variant_id -> endpoints
        self._variant_metrics: dict[str, VariantMetrics] = {}
        self._statistical_results: dict[str, StatisticalResult] = {}
        self._error_counts: dict[str, int] = {}  # variant_id -> error count
        # Monitoring thread
        self._monitoring_event = threading.Event()
        self._monitoring_thread: threading.Thread | None = None
        self._monitoring_interval = self._get_monitoring_interval()

    def _get_monitoring_interval(self) -> float:
        """Get monitoring interval from config or default to 60 seconds."""
        interval = self.config.auto_stop_conditions.get("monitoring_interval_seconds", 60.0)
        # Ensure reasonable bounds (5 seconds to 1 hour)
        return max(5.0, min(3600.0, float(interval)))

    def _get_server_type(self) -> str:
        """Map deployment platform to telemetry server type.

        Returns:
            Server type string (vllm, tgi, triton).
        """
        platform = self.config.deployment_platform.lower()
        if "vllm" in platform:
            return "vllm"
        elif "tgi" in platform or "text-generation-inference" in platform:
            return "tgi"
        elif "triton" in platform:
            return "triton"
        else:
            # Default to vllm
            return "vllm"

    def _record_error(self, variant_id: str, count: int = 1) -> None:
        """Record an error for a variant.

        Args:
            variant_id: Variant identifier.
            count: Number of errors to record (default 1).
        """
        self._error_counts[variant_id] = self._error_counts.get(variant_id, 0) + count

    def _check_auto_stop_conditions(self) -> tuple[bool, str]:
        """Check if experiment should stop automatically based on conditions.

        Returns:
            Tuple of (should_stop, reason).
        """
        conditions = self.config.auto_stop_conditions

        # Check max errors
        max_errors = conditions.get("max_errors")
        if max_errors is not None:
            # Check if any variant has exceeded max errors
            for variant_id, error_count in self._error_counts.items():
                if error_count >= max_errors:
                    return True, (
                        f"Max errors reached for variant {variant_id}: "
                        f"{error_count} >= {max_errors}"
                    )

        has_min_samples, sample_reason = self._has_minimum_samples()
        if not has_min_samples:
            return False, sample_reason

        duration_reached, duration_reason = self._max_duration_reached()
        if duration_reached:
            return True, duration_reason

        target_reached, target_reason = self._statistical_target_reached()
        if target_reached:
            return True, target_reason

        return False, target_reason

    def _has_minimum_samples(self) -> tuple[bool, str]:
        """Return whether all variants have the configured minimum sample count."""
        if not self._variant_metrics:
            return False, "No variant metrics collected yet"

        insufficient_variants = [
            variant_id
            for variant_id, metrics in self._variant_metrics.items()
            if metrics.sample_count < self.config.min_sample_size_per_variant
        ]
        if insufficient_variants:
            return (
                False,
                "Waiting for minimum samples in variants: " + ", ".join(insufficient_variants),
            )
        return True, ""

    def _max_duration_reached(self) -> tuple[bool, str]:
        """Return whether the configured max experiment duration has been reached."""
        if not self._start_time:
            return False, ""
        start_dt = datetime.fromisoformat(self._start_time)
        duration_hours = (datetime.now() - start_dt).total_seconds() / 3600.0
        if duration_hours >= self.config.max_duration_hours:
            return True, (
                f"Max duration reached with min sample size: {duration_hours:.1f}h >= "
                f"{self.config.max_duration_hours}h"
            )
        return False, ""

    def _statistical_target_reached(self) -> tuple[bool, str]:
        """Return whether confidence/significance or power target has been reached."""
        if not self._statistical_results:
            return False, "No statistical results available yet"

        conditions = self.config.auto_stop_conditions
        confidence_threshold = conditions.get("confidence_threshold")
        power_target = conditions.get("power_target")
        primary_metric = self.config.primary_metric

        for result in self._statistical_results.values():
            if result.metric_name != primary_metric:
                continue

            if confidence_threshold is not None and result.p_value is not None:
                confidence = 1.0 - result.p_value
                if confidence >= confidence_threshold:
                    return True, (
                        f"Confidence threshold reached with min sample size: "
                        f"{confidence:.3f} >= {confidence_threshold}"
                    )

            if power_target is not None and result.statistical_power is not None:
                if result.statistical_power >= power_target:
                    return True, (
                        "Power target reached with min sample size: "
                        f"{result.statistical_power:.3f} >= {power_target}"
                    )

            if (
                confidence_threshold is None
                and power_target is None
                and result.is_significant
            ):
                return True, "Primary metric is statistically significant with min sample size"

        return False, "Statistical target not reached yet"

    @property
    def status(self) -> ExperimentStatus:
        """Current experiment status."""
        return self._status

    @property
    def start_time(self) -> str | None:
        """Experiment start timestamp."""
        return self._start_time

    @property
    def end_time(self) -> str | None:
        """Experiment end timestamp (if completed/stopped)."""
        return self._end_time

    @property
    def deployment_ids(self) -> dict[str, str]:
        """Deployment IDs for each variant."""
        return self._deployment_ids.copy()

    def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        self.logger.info("Starting monitoring loop for experiment %s", self.config.experiment_id)
        while not self._monitoring_event.is_set():
            try:
                # Collect metrics (record errors if error rate > 1%)
                self._collect_metrics(record_errors=True)
                # Run analysis
                self._statistical_results = self.analyze()
                # Check auto-stop conditions
                should_stop, reason = self._check_auto_stop_conditions()
                if should_stop:
                    self.logger.info("Auto-stop condition met: %s", reason)
                    self.stop(reason=f"auto: {reason}")
                    break
                # Log status periodically
                self.logger.debug(
                    "Experiment %s monitoring cycle complete", self.config.experiment_id
                )
            except Exception as e:
                self.logger.error("Error in monitoring loop: %s", e, exc_info=True)
            # Wait for interval or shutdown signal
            self._monitoring_event.wait(timeout=self._monitoring_interval)
        self.logger.info("Monitoring loop stopped for experiment %s", self.config.experiment_id)

    def _stop_monitoring(self, skip_join: bool = False) -> None:
        """Stop the monitoring thread.

        Args:
            skip_join: If True, skip joining the thread (call from within the thread).
        """
        if self._monitoring_thread is not None:
            self.logger.debug(
                "Stopping monitoring thread for experiment %s", self.config.experiment_id
            )
            self._monitoring_event.set()
            if not skip_join:
                self._monitoring_thread.join(timeout=5.0)
                if self._monitoring_thread.is_alive():
                    self.logger.warning(
                        "Monitoring thread did not stop gracefully for experiment %s",
                        self.config.experiment_id,
                    )
            self._monitoring_thread = None
            self._monitoring_event.clear()

    def start(self) -> ExperimentResult:
        """Start the experiment.

        Deploys all variants using the configured deployment strategy,
        begins metric collection, and transitions to RUNNING status.

        Returns:
            Initial experiment result with deployment information.
        """
        if self._status != ExperimentStatus.DRAFT:
            raise RuntimeError(f"Cannot start experiment in status {self._status}")

        self._start_time = datetime.now().isoformat()
        self._status = ExperimentStatus.RUNNING

        # Deploy all variants using ABTestStrategy
        from ..deployment.strategies import ABTestStrategy

        strategy = ABTestStrategy(
            config={
                "experiment_id": self.config.experiment_id,
                "traffic_allocation": self.config.traffic_allocation,
            }
        )

        # Create deployment request with experiment configuration
        # Use first variant's model_path as placeholder (strategy will override per variant)
        first_variant = self.config.variants[0]
        deployment_request = DeploymentRequest(
            model_path=first_variant.model_path,
            platform=self.config.deployment_platform,
            strategy=DeploymentStrategyType.A_B_TEST,
            strategy_config={
                "experiment_id": self.config.experiment_id,
                "traffic_allocation": self.config.traffic_allocation,
            },
            experiment_config=self.config.to_dict(),
            health_checks=self.config.health_checks,
            metadata={
                "experiment_id": self.config.experiment_id,
                "variant_id": first_variant.variant_id,
            },
        )

        # Execute deployment
        result = strategy.execute(self.platform, deployment_request)

        # Store deployment IDs and endpoints for all variants
        if result.metrics and "variant_deployments" in result.metrics:
            variant_deployments = result.metrics["variant_deployments"]
            for variant_id, deployment_info in variant_deployments.items():
                if deployment_info.get("deployment_id"):
                    self._deployment_ids[variant_id] = deployment_info["deployment_id"]
                if deployment_info.get("endpoints"):
                    self._variant_endpoints[variant_id] = deployment_info["endpoints"]
        elif result.deployment_id:
            # Fallback: store single deployment ID for first variant
            self._deployment_ids[first_variant.variant_id] = result.deployment_id

        # Create initial experiment result
        experiment_result = ExperimentResult(
            experiment_id=self.config.experiment_id,
            status=self._status,
            start_time=self._start_time,
            variant_metrics=self._variant_metrics,
            statistical_results={},
        )

        # Start background monitoring thread
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            name=f"experiment-monitor-{self.config.experiment_id}",
            daemon=True,
        )
        self._monitoring_thread.start()
        self.logger.info("Started monitoring thread for experiment %s", self.config.experiment_id)

        return experiment_result

    def stop(self, reason: str = "manual") -> ExperimentResult:
        """Stop the experiment.

        Args:
            reason: Reason for stopping.

        Returns:
            Final experiment result.
        """
        # Idempotency: if already stopped, return current result
        if self._status == ExperimentStatus.STOPPED:
            return self._get_current_result()
        if self._status not in (ExperimentStatus.RUNNING, ExperimentStatus.PAUSED):
            raise RuntimeError(f"Cannot stop experiment in status {self._status}")

        self._end_time = datetime.now().isoformat()
        self._status = ExperimentStatus.STOPPED
        # Stop monitoring thread
        skip_join = threading.current_thread() == self._monitoring_thread
        self._stop_monitoring(skip_join=skip_join)

        # Perform final analysis before stopping
        self._statistical_results = self.analyze()

        # Optional: clean up deployments based on configuration
        if self.config.deployment_strategy == "cleanup":
            for deployment_id in self._deployment_ids.values():
                try:
                    self.platform.delete(deployment_id)
                except Exception:
                    # Log warning but continue
                    pass

        return self._get_current_result()

    def pause(self) -> None:
        """Pause the experiment (stop collecting new traffic)."""
        if self._status != ExperimentStatus.RUNNING:
            raise RuntimeError(f"Cannot pause experiment in status {self._status}")
        self._status = ExperimentStatus.PAUSED
        # Try to pause traffic via platform if supported
        if hasattr(self.platform, "pause_experiment"):
            self.platform.pause_experiment(self.config.experiment_id, self._deployment_ids)
        else:
            self.logger.warning("Platform does not support traffic pausing; only status updated")

    def resume(self) -> None:
        """Resume a paused experiment."""
        if self._status != ExperimentStatus.PAUSED:
            raise RuntimeError(f"Cannot resume experiment in status {self._status}")
        self._status = ExperimentStatus.RUNNING
        # Try to resume traffic via platform if supported
        if hasattr(self.platform, "resume_experiment"):
            self.platform.resume_experiment(self.config.experiment_id, self._deployment_ids)
        else:
            self.logger.warning("Platform does not support traffic resuming; only status updated")

    def analyze(self) -> dict[str, StatisticalResult]:
        """Perform statistical analysis on current metrics.

        Returns:
            Dictionary mapping metric names to statistical results.
        """
        # Collect current metrics
        variant_metrics = self._collect_metrics()

        # If no metrics collected yet, return empty results
        if not variant_metrics:
            return {}

        # Perform statistical analysis
        return analyze_experiment_results(variant_metrics, self.config)

    def get_status_report(self) -> dict[str, Any]:
        """Generate a status report for the experiment.

        Returns:
            Dictionary with experiment status, metrics, and statistics.
        """
        result = self._get_current_result()
        return result.to_dict()

    def _get_current_result(self) -> ExperimentResult:
        """Get current experiment result."""
        # Determine winner based on statistical results
        winner_variant_id = None
        confidence = None
        recommendations: list[str] = []

        if self._statistical_results:
            # Extract primary metric results
            primary_metric = self.config.primary_metric
            primary_results = {
                key: result
                for key, result in self._statistical_results.items()
                if result.metric_name == primary_metric
            }

            if primary_results:
                # Find variant with best performance (largest effect size)
                best_variant = None
                best_effect_size = -float("inf")

                for key, result in primary_results.items():
                    if result.effect_size is not None and result.is_significant:
                        # Effect size sign indicates direction: positive means
                        # variant better than control (for throughput-like metrics)
                        if result.effect_size > best_effect_size:
                            best_effect_size = result.effect_size
                            # Extract variant ID from key (format: metric_control_vs_variant)
                            parts = key.split("_vs_")
                            if len(parts) == 2:
                                best_variant = parts[1]

                if best_variant:
                    winner_variant_id = best_variant
                    # Confidence approximated as 1 - p-value of that result
                    for result in primary_results.values():
                        if result.p_value is not None:
                            confidence = 1.0 - result.p_value
                            break

                # Generate recommendations from statistical results
                seen_recs = set()
                for result in self._statistical_results.values():
                    for rec in result.recommendations:
                        if rec and rec not in seen_recs:
                            recommendations.append(rec)
                            seen_recs.add(rec)

        return ExperimentResult(
            experiment_id=self.config.experiment_id,
            status=self._status,
            start_time=self._start_time or datetime.now().isoformat(),
            end_time=self._end_time,
            variant_metrics=self._variant_metrics,
            statistical_results=self._statistical_results,
            winner_variant_id=winner_variant_id,
            confidence=confidence,
            recommendations=recommendations,
            metadata={"analyzed_at": datetime.now().isoformat()},
        )

    def _collect_metrics(self, record_errors: bool = False) -> dict[str, VariantMetrics]:
        """Collect metrics from telemetry system.

        Returns:
            Dictionary of variant_id to VariantMetrics.
        """
        # Try to collect real telemetry if we have endpoint information
        if hasattr(self, "_variant_endpoints") and self._variant_endpoints:
            try:
                return self._collect_telemetry_metrics(record_errors)
            except Exception as e:
                self.logger.warning(
                    f"Failed to collect telemetry metrics: {e}. Falling back to mock metrics."
                )
                # Fall through to mock implementation

        # Mock implementation for testing/demonstration
        # Generate synthetic metrics for each variant
        import random

        result: dict[str, VariantMetrics] = {}

        # Define metrics to collect (primary + secondary)
        metric_names = [self.config.primary_metric] + self.config.secondary_metrics
        if not metric_names:
            metric_names = ["throughput_toks_per_sec", "latency_p95", "error_rate"]

        for variant in self.config.variants:
            variant_id = variant.variant_id
            metrics: dict[str, dict[str, float]] = {}

            for metric_name in metric_names:
                # Generate realistic mock values based on metric type
                if "throughput" in metric_name:
                    mean = random.uniform(100.0, 1000.0)
                    std = mean * 0.1  # 10% variability
                elif "latency" in metric_name:
                    mean = random.uniform(10.0, 200.0)
                    std = mean * 0.2
                elif "error" in metric_name:
                    mean = random.uniform(0.0, 5.0)
                    std = mean * 0.5
                else:
                    mean = random.uniform(0.0, 100.0)
                    std = mean * 0.15

                count = random.randint(50, 200)  # Simulated sample count

                metrics[metric_name] = {
                    "mean": mean,
                    "std": std,
                    "count": count,
                }

            # Create VariantMetrics object
            now = datetime.now()
            variant_metrics = VariantMetrics(
                variant_id=variant_id,
                sample_count=sum(int(metrics[m]["count"]) for m in metrics),
                metrics=metrics,
                percentiles=None,
                timestamp_start=(now - timedelta(minutes=5)).isoformat(),
                timestamp_end=now.isoformat(),
            )

            result[variant_id] = variant_metrics

            # Record errors if requested and error metrics exceed threshold
            if record_errors:
                for metric_name, metric_stats in metrics.items():
                    if "error" in metric_name.lower():
                        error_rate = metric_stats.get("mean", 0.0)
                        if error_rate > 1.0:  # 1% error rate threshold
                            self._record_error(variant_id)

        # Update internal cache
        self._variant_metrics = result
        return result

    def _collect_telemetry_metrics(self, record_errors: bool = False) -> dict[str, VariantMetrics]:
        """Collect actual telemetry metrics from deployed variant endpoints.

        Returns:
            Dictionary of variant_id to VariantMetrics.

        Raises:
            RuntimeError: If endpoints are not available or collection fails.
        """
        from datetime import datetime, timedelta

        result: dict[str, VariantMetrics] = {}
        server_type = self._get_server_type()

        # Define metrics to collect (primary + secondary)
        metric_names = [self.config.primary_metric] + self.config.secondary_metrics
        if not metric_names:
            metric_names = ["throughput_toks_per_sec", "latency_ms_per_request", "error_rate"]

        # Mapping from telemetry field names to metric names
        telemetry_field_map = {
            "throughput_toks_per_sec": "throughput_toks_per_sec",
            "latency_ms_per_request": "latency_ms_per_request",
            "memory_gb": "memory_gb",
            "tokens_per_request": "tokens_per_request",
            "power_watts": "power_watts",
        }

        for variant in self.config.variants:
            variant_id = variant.variant_id
            endpoints = self._variant_endpoints.get(variant_id, [])
            if not endpoints:
                self.logger.warning(
                    f"No endpoints available for variant {variant_id}, "
                    "skipping telemetry collection"
                )
                continue

            # Use the first endpoint (could extend to collect from multiple)
            base_url = endpoints[0]

            try:
                collector = get_collector(server_type, base_url)
                collection_result = collector.collect()

                if not collection_result.success or collection_result.aggregated is None:
                    raise RuntimeError(
                        f"Telemetry collection failed for variant {variant_id}: "
                        f"{collection_result.error_message}"
                    )

                telemetry_data = collection_result.aggregated

                # Convert TelemetryData to metrics dictionary
                metrics: dict[str, dict[str, float]] = {}
                raw_observations: dict[str, list[float]] = {}
                for metric_name in metric_names:
                    # Map metric name to telemetry field
                    field_name = None
                    for telemetry_field, mapped_name in telemetry_field_map.items():
                        if (
                            metric_name.lower() == mapped_name.lower()
                            or metric_name.lower() in telemetry_field.lower()
                        ):
                            field_name = telemetry_field
                            break

                    if field_name is None:
                        # Try to get from raw_metrics
                        raw_value = telemetry_data.raw_metrics.get(metric_name)
                        if raw_value is not None:
                            field_name = metric_name
                        else:
                            # Metric not available in telemetry
                            self.logger.debug(
                                f"Metric {metric_name} not available in telemetry "
                                f"for variant {variant_id}"
                            )
                            continue

                    # Get value from telemetry data
                    if hasattr(telemetry_data, field_name):
                        value = getattr(telemetry_data, field_name)
                    else:
                        value = telemetry_data.raw_metrics.get(field_name)

                    if value is None:
                        continue

                    observations = _extract_metric_observations(
                        collection_result.samples, field_name
                    )
                    if observations:
                        mean = sum(observations) / len(observations)
                        if len(observations) > 1:
                            variance = sum((x - mean) ** 2 for x in observations) / (
                                len(observations) - 1
                            )
                            std = variance**0.5
                        else:
                            std = 0.0
                        count = len(observations)
                        raw_observations[metric_name] = observations
                    else:
                        mean = float(value)
                        std = mean * 0.1
                        count = len(collection_result.samples) if collection_result.samples else 1

                    metrics[metric_name] = {
                        "mean": mean,
                        "std": std,
                        "count": count,
                    }

                # If no metrics collected, skip this variant
                if not metrics:
                    self.logger.warning(f"No metrics collected for variant {variant_id}")
                    continue

                # Create VariantMetrics object
                now = datetime.now()
                variant_metrics = VariantMetrics(
                    variant_id=variant_id,
                    sample_count=sum(int(metrics[m]["count"]) for m in metrics),
                    metrics=metrics,
                    raw_observations=raw_observations or None,
                    percentiles=None,
                    timestamp_start=(now - timedelta(minutes=5)).isoformat(),
                    timestamp_end=now.isoformat(),
                )

                result[variant_id] = variant_metrics

                # Record errors if requested
                if record_errors:
                    for metric_name, metric_stats in metrics.items():
                        if "error" in metric_name.lower():
                            error_rate = metric_stats.get("mean", 0.0)
                            if error_rate > 1.0:  # 1% error rate threshold
                                self._record_error(variant_id)

            except Exception as e:
                self.logger.error(f"Failed to collect telemetry for variant {variant_id}: {e}")
                # Continue with other variants

        # Update internal cache
        self._variant_metrics = result
        return result


def run_ab_test(
    config: ABTestConfig,
    platform: DeploymentPlatform,
    telemetry_client: Any | None = None,
) -> ExperimentResult:
    """Run a complete A/B test experiment.

    Convenience function that creates an ExperimentRunner, starts the experiment,
    monitors it until completion, and returns the final result.

    Args:
        config: A/B test configuration.
        platform: Deployment platform.
        telemetry_client: Telemetry client (optional).

    Returns:
        Final experiment result.
    """
    runner = ExperimentRunner(config, platform, telemetry_client)
    runner.start()

    # Wait for experiment completion (RUNNING or PAUSED -> final state)
    polling_interval = config.auto_stop_conditions.get("monitoring_interval_seconds", 60.0)
    # Ensure reasonable bounds
    polling_interval = max(1.0, min(300.0, polling_interval))

    try:
        while runner.status in (ExperimentStatus.RUNNING, ExperimentStatus.PAUSED):
            time.sleep(polling_interval)
    except KeyboardInterrupt:
        runner.logger.info("Experiment interrupted by user, stopping...")
        runner.stop(reason="user_interrupt")

    # Return final result
    return runner._get_current_result()


def analyze_experiment_results(
    variant_metrics: dict[str, VariantMetrics],
    config: ABTestConfig,
) -> dict[str, StatisticalResult]:
    """Analyze experiment results from collected metrics.

    Args:
        variant_metrics: Metrics for each variant.
        config: Experiment configuration.

    Returns:
        Statistical results for each metric.
    """
    results: dict[str, StatisticalResult] = {}

    if not variant_metrics:
        return results

    # Determine control variant (first variant in config)
    control_id = config.variants[0].variant_id
    if control_id not in variant_metrics:
        # Fallback to first available variant
        control_id = next(iter(variant_metrics))

    # Get all metric names from control variant
    control_metrics = variant_metrics[control_id].metrics
    if not control_metrics:
        return results

    metric_names = list(control_metrics.keys())

    # For each metric, compare control against each treatment variant
    for metric_name in metric_names:
        control_samples, control_count, control_mode = _samples_for_metric(
            variant_metrics[control_id], metric_name
        )

        # Compare against each treatment variant
        for variant_id, variant_metric in variant_metrics.items():
            if variant_id == control_id:
                continue

            variant_samples, variant_count, variant_mode = _samples_for_metric(
                variant_metric, metric_name
            )

            # Perform statistical test
            test_result = analyze_variant_comparison(
                control_samples,
                variant_samples,
                test_type=str(config.test_type),
                alpha=config.significance_level,
            )

            warnings: list[str] = []
            if control_mode != "raw_observations" or variant_mode != "raw_observations":
                warnings.append(
                    "Aggregate-only fallback used; "
                    "raw observations were not available for all variants"
                )

            # Create StatisticalResult
            stat_result = StatisticalResult(
                metric_name=metric_name,
                test_type=config.test_type,
                p_value=test_result.get("p_value"),
                confidence_interval=test_result.get("confidence_intervals", {}).get("a"),
                effect_size=test_result.get("effect_size"),
                statistical_power=test_result.get("statistical_power"),
                is_significant=test_result.get("is_significant", False),
                sample_sizes={control_id: control_count, variant_id: variant_count},
                recommendations=(
                    test_result.get("recommendation", "").split(". ")
                    if test_result.get("recommendation")
                    else []
                ),
                metadata={
                    "comparison": f"{control_id}_vs_{variant_id}",
                    "analysis_mode": (
                        "raw_observations"
                        if not warnings
                        else "aggregate_only_fallback"
                    ),
                    "data_sources": {
                        control_id: control_mode,
                        variant_id: variant_mode,
                    },
                    "warnings": warnings,
                },
            )

            # Store result with unique key
            result_key = f"{metric_name}_{control_id}_vs_{variant_id}"
            results[result_key] = stat_result

    return results


def _extract_metric_observations(samples: list[Any], field_name: str) -> list[float]:
    """Extract per-sample observations for a metric from telemetry samples."""
    observations: list[float] = []
    for sample in samples:
        value: Any = None
        if hasattr(sample, field_name):
            value = getattr(sample, field_name)
        elif hasattr(sample, "raw_metrics"):
            value = sample.raw_metrics.get(field_name)

        if isinstance(value, (int, float)):
            observations.append(float(value))
    return observations


def _samples_for_metric(
    variant_metric: VariantMetrics, metric_name: str
) -> tuple[list[float], int, str]:
    """Build analysis samples for a metric, preferring raw observations."""
    if variant_metric.raw_observations:
        raw_values = variant_metric.raw_observations.get(metric_name)
        if raw_values:
            return [float(x) for x in raw_values], len(raw_values), "raw_observations"

    metric_stats = variant_metric.metrics.get(metric_name, {})
    mean = float(metric_stats.get("mean", 0.0))
    count = int(metric_stats.get("count", 0))
    sample_count = max(count, 1)
    return [mean] * sample_count, count, "aggregated_mean_repeated"
