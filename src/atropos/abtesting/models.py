"""Data models for A/B testing framework.

Provides configuration, variant, experiment result, and statistical result
classes for running controlled experiments with model variants.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any


class ExperimentStatus(Enum):
    """Status of an A/B test experiment."""

    DRAFT = auto()  # Experiment configured but not started
    RUNNING = auto()  # Experiment is active and collecting metrics
    PAUSED = auto()  # Experiment paused (no new traffic)
    COMPLETED = auto()  # Experiment finished normally
    STOPPED = auto()  # Experiment stopped early (manually or by condition)
    FAILED = auto()  # Experiment failed (deployment errors, etc.)

    def __str__(self) -> str:
        return self.name.lower()


class StatisticalTestType(Enum):
    """Type of statistical test to use for comparison."""

    T_TEST = auto()  # Independent two-sample t-test
    MANN_WHITNEY = auto()  # Non-parametric rank test
    CHI_SQUARED = auto()  # For proportion/categorical metrics
    BAYESIAN = auto()  # Bayesian inference
    SEQUENTIAL = auto()  # Sequential probability ratio test

    def __str__(self) -> str:
        return self.name.lower().replace("_", "-")


@dataclass(frozen=True)
class Variant:
    """A model variant in an A/B test experiment.

    Attributes:
        variant_id: Unique identifier for this variant.
        name: Human-readable name (e.g., "pruned-70%", "quantized-int8").
        model_path: Path to model (local or remote).
        deployment_config: Optional deployment configuration override.
            If None, uses experiment-level deployment config.
        traffic_weight: Proportion of traffic to allocate (0.0-1.0).
            Actual traffic may be normalized across all variants.
        description: Optional description of the variant.
        metadata: Additional variant-specific metadata.
    """

    variant_id: str
    name: str
    model_path: str
    deployment_config: dict[str, Any] | None = None
    traffic_weight: float = 1.0
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "variant_id": self.variant_id,
            "name": self.name,
            "model_path": self.model_path,
            "deployment_config": self.deployment_config,
            "traffic_weight": self.traffic_weight,
            "description": self.description,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class ABTestConfig:
    """Configuration for an A/B test experiment.

    Attributes:
        experiment_id: Unique identifier for the experiment.
        name: Human-readable experiment name.
        variants: List of variants to test.
        primary_metric: Name of the primary metric to optimize.
            Must match telemetry metric names.
        secondary_metrics: List of secondary metrics to monitor.
        traffic_allocation: Global traffic allocation for experiment (0.0-1.0).
            Allows gradual ramp-up of experiment traffic.
        significance_level: Alpha level for statistical tests (default 0.05).
        statistical_power: Desired power (1 - beta) for sample size calculation.
        test_type: Statistical test to use for comparison.
        min_sample_size_per_variant: Minimum samples required before evaluation.
        max_duration_hours: Maximum experiment duration before auto-stop.
        auto_stop_conditions: Conditions for automatic stopping.
            e.g., {"confidence_threshold": 0.95, "max_errors": 5}
        deployment_platform: Platform to use for variant deployment.
        deployment_strategy: Strategy for deploying variants.
        health_checks: Health check configuration for deployments.
        metadata: Additional experiment metadata.
        created_at: Timestamp when configuration was created.
        updated_at: Timestamp when configuration was last updated.
    """

    experiment_id: str
    name: str
    variants: list[Variant]
    primary_metric: str
    secondary_metrics: list[str] = field(default_factory=list)
    traffic_allocation: float = 1.0
    significance_level: float = 0.05
    statistical_power: float = 0.8
    test_type: StatisticalTestType = StatisticalTestType.T_TEST
    min_sample_size_per_variant: int = 100
    max_duration_hours: float = 168.0  # 1 week
    auto_stop_conditions: dict[str, Any] = field(default_factory=dict)
    deployment_platform: str = "vllm"
    deployment_strategy: str = "immediate"
    health_checks: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "variants": [v.to_dict() for v in self.variants],
            "primary_metric": self.primary_metric,
            "secondary_metrics": self.secondary_metrics,
            "traffic_allocation": self.traffic_allocation,
            "significance_level": self.significance_level,
            "statistical_power": self.statistical_power,
            "test_type": str(self.test_type),
            "min_sample_size_per_variant": self.min_sample_size_per_variant,
            "max_duration_hours": self.max_duration_hours,
            "auto_stop_conditions": self.auto_stop_conditions,
            "deployment_platform": self.deployment_platform,
            "deployment_strategy": self.deployment_strategy,
            "health_checks": self.health_checks,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass(frozen=True)
class VariantMetrics:
    """Aggregated metrics for a single variant.

    Attributes:
        variant_id: Variant identifier.
        sample_count: Number of samples collected.
        metrics: Dictionary of metric name to aggregated values.
            Expected structure: {"mean": float, "std": float, "count": int}
        percentiles: Optional percentile distribution (e.g., p50, p95, p99).
        timestamp_start: Start of metric collection period.
        timestamp_end: End of metric collection period.
    """

    variant_id: str
    sample_count: int
    metrics: dict[str, dict[str, float]]  # metric_name -> {mean, std, count, ...}
    percentiles: dict[str, dict[str, float]] | None = None
    timestamp_start: str | None = None
    timestamp_end: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result: dict[str, Any] = {
            "variant_id": self.variant_id,
            "sample_count": self.sample_count,
            "metrics": self.metrics,
        }
        if self.percentiles is not None:
            result["percentiles"] = self.percentiles
        if self.timestamp_start is not None:
            result["timestamp_start"] = self.timestamp_start
        if self.timestamp_end is not None:
            result["timestamp_end"] = self.timestamp_end
        return result


@dataclass(frozen=True)
class StatisticalResult:
    """Statistical test results for comparing variants.

    Attributes:
        metric_name: Name of metric being compared.
        test_type: Type of statistical test used.
        p_value: Calculated p-value from statistical test.
        confidence_interval: Tuple of (lower, upper) confidence bounds.
        effect_size: Standardized effect size (e.g., Cohen's d).
        statistical_power: Actual power of the test given sample sizes.
        is_significant: Whether result is statistically significant.
        sample_sizes: Dictionary of variant_id to sample count.
        recommendations: List of recommendation strings based on results.
        metadata: Additional statistical metadata.
    """

    metric_name: str
    test_type: StatisticalTestType
    p_value: float | None
    confidence_interval: tuple[float, float] | None
    effect_size: float | None
    statistical_power: float | None
    is_significant: bool
    sample_sizes: dict[str, int]
    recommendations: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "metric_name": self.metric_name,
            "test_type": str(self.test_type),
            "p_value": round(self.p_value, 6) if self.p_value is not None else None,
            "confidence_interval": (
                [round(self.confidence_interval[0], 6), round(self.confidence_interval[1], 6)]
                if self.confidence_interval is not None
                else None
            ),
            "effect_size": round(self.effect_size, 6) if self.effect_size is not None else None,
            "statistical_power": (
                round(self.statistical_power, 6) if self.statistical_power is not None else None
            ),
            "is_significant": self.is_significant,
            "sample_sizes": self.sample_sizes,
            "recommendations": self.recommendations,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class ExperimentResult:
    """Results of an A/B test experiment.

    Attributes:
        experiment_id: Experiment identifier.
        status: Current experiment status.
        start_time: Experiment start timestamp.
        end_time: Experiment end timestamp (if completed/stopped).
        variant_metrics: Dictionary of variant_id to VariantMetrics.
        statistical_results: Dictionary of metric_name to StatisticalResult.
        winner_variant_id: Variant ID of winning variant (if determined).
        confidence: Confidence in winner selection (0.0-1.0).
        recommendations: List of actionable recommendations.
        metadata: Additional result metadata.
    """

    experiment_id: str
    status: ExperimentStatus
    start_time: str
    variant_metrics: dict[str, VariantMetrics]
    statistical_results: dict[str, StatisticalResult]
    end_time: str | None = None
    winner_variant_id: str | None = None
    confidence: float | None = None
    recommendations: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_hours(self) -> float | None:
        """Calculate experiment duration in hours."""
        if self.end_time:
            start = datetime.fromisoformat(self.start_time)
            end = datetime.fromisoformat(self.end_time)
            return (end - start).total_seconds() / 3600.0
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result: dict[str, Any] = {
            "experiment_id": self.experiment_id,
            "status": str(self.status),
            "start_time": self.start_time,
            "variant_metrics": {vid: vm.to_dict() for vid, vm in self.variant_metrics.items()},
            "statistical_results": {
                metric: sr.to_dict() for metric, sr in self.statistical_results.items()
            },
            "recommendations": self.recommendations,
            "metadata": self.metadata,
        }
        if self.end_time is not None:
            result["end_time"] = self.end_time
        if self.winner_variant_id is not None:
            result["winner_variant_id"] = self.winner_variant_id
        if self.confidence is not None:
            result["confidence"] = round(self.confidence, 6)
        if self.duration_hours is not None:
            result["duration_hours"] = round(self.duration_hours, 3)
        return result
