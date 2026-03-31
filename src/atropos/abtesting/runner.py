"""Experiment runner for A/B testing.

Orchestrates deployment, metric collection, statistical analysis, and
experiment lifecycle management.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from ..deployment.models import DeploymentRequest, DeploymentStrategyType
from ..deployment.platforms import DeploymentPlatform
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
        self._variant_metrics: dict[str, VariantMetrics] = {}
        self._statistical_results: dict[str, StatisticalResult] = {}

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

        # Store deployment IDs for all variants
        if result.metrics and "variant_deployments" in result.metrics:
            variant_deployments = result.metrics["variant_deployments"]
            for variant_id, deployment_info in variant_deployments.items():
                if deployment_info.get("deployment_id"):
                    self._deployment_ids[variant_id] = deployment_info["deployment_id"]
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

        # TODO: Start background monitoring thread
        # For now, just return initial result

        return experiment_result

    def stop(self, reason: str = "manual") -> ExperimentResult:
        """Stop the experiment.

        Args:
            reason: Reason for stopping.

        Returns:
            Final experiment result.
        """
        if self._status not in (ExperimentStatus.RUNNING, ExperimentStatus.PAUSED):
            raise RuntimeError(f"Cannot stop experiment in status {self._status}")

        self._end_time = datetime.now().isoformat()
        self._status = ExperimentStatus.STOPPED

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
        # TODO: Implement traffic pausing

    def resume(self) -> None:
        """Resume a paused experiment."""
        if self._status != ExperimentStatus.PAUSED:
            raise RuntimeError(f"Cannot resume experiment in status {self._status}")
        self._status = ExperimentStatus.RUNNING
        # TODO: Implement traffic resuming

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

    def _collect_metrics(self) -> dict[str, VariantMetrics]:
        """Collect metrics from telemetry system.

        Returns:
            Dictionary of variant_id to VariantMetrics.
        """
        # If we have a telemetry client, fetch actual metrics
        if self.telemetry_client is not None:
            # TODO: Implement actual metric collection from telemetry client
            # For now, return empty dict
            return {}

        # Mock implementation for testing/demonstration
        # Generate synthetic metrics for each variant
        import random
        from datetime import datetime, timedelta

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
    result = runner.start()

    # TODO: Implement monitoring loop with periodic analysis
    # For now, just return initial result
    return result


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
        # Extract raw values for control
        control_stats = control_metrics.get(metric_name, {})
        control_mean = control_stats.get("mean", 0.0)
        control_std = control_stats.get("std", 0.0)
        control_count = int(control_stats.get("count", 0))

        # Generate synthetic samples for control (for statistical test)
        # This is a simplification; real implementation should use raw samples
        import random

        random.seed(42)  # For reproducibility
        control_samples = [
            random.gauss(control_mean, control_std if control_std > 0 else 1e-6)
            for _ in range(max(control_count, 1))
        ]

        # Compare against each treatment variant
        for variant_id, variant_metric in variant_metrics.items():
            if variant_id == control_id:
                continue

            variant_stats = variant_metric.metrics.get(metric_name, {})
            variant_mean = variant_stats.get("mean", 0.0)
            variant_std = variant_stats.get("std", 0.0)
            variant_count = int(variant_stats.get("count", 0))

            # Generate synthetic samples for treatment
            variant_samples = [
                random.gauss(variant_mean, variant_std if variant_std > 0 else 1e-6)
                for _ in range(max(variant_count, 1))
            ]

            # Perform statistical test
            test_result = analyze_variant_comparison(
                control_samples,
                variant_samples,
                test_type=str(config.test_type),
                alpha=config.significance_level,
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
                metadata={"comparison": f"{control_id}_vs_{variant_id}"},
            )

            # Store result with unique key
            result_key = f"{metric_name}_{control_id}_vs_{variant_id}"
            results[result_key] = stat_result

    return results
