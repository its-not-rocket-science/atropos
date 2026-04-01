"""Basic cost anomaly detection for optimization outcomes.

Provides statistical anomaly detection for cost projections and optimization
results, flagging outcomes that deviate significantly from expected baselines.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from ..models import OptimizationOutcome


@dataclass
class Anomaly:
    """Represents a detected anomaly in optimization results."""

    metric: str
    value: float
    baseline_mean: float
    baseline_std: float
    z_score: float
    threshold: float = 3.0  # Default: 3 sigma
    is_anomaly: bool = True
    description: str = ""

    def __post_init__(self) -> None:
        """Generate description if not provided."""
        if not self.description:
            self.description = (
                f"{self.metric} = {self.value:.2f} "
                f"(z-score: {self.z_score:.2f}, "
                f"baseline: {self.baseline_mean:.2f} +/-{self.baseline_std:.2f})"
            )


@dataclass
class AnomalyDetectionResult:
    """Results of anomaly detection on an optimization outcome."""

    outcome: OptimizationOutcome
    anomalies: list[Anomaly]
    has_anomalies: bool = False
    threshold: float = 3.0

    def __post_init__(self) -> None:
        """Set has_anomalies based on anomalies list."""
        self.has_anomalies = len(self.anomalies) > 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "scenario_name": self.outcome.scenario_name,
            "strategy_name": self.outcome.strategy_name,
            "has_anomalies": self.has_anomalies,
            "threshold": self.threshold,
            "anomalies": [
                {
                    "metric": a.metric,
                    "value": a.value,
                    "z_score": a.z_score,
                    "baseline_mean": a.baseline_mean,
                    "baseline_std": a.baseline_std,
                    "description": a.description,
                }
                for a in self.anomalies
            ],
            "summary_metrics": {
                "annual_savings_usd": self.outcome.annual_total_savings_usd,
                "break_even_months": self.outcome.break_even_years * 12
                if self.outcome.break_even_years
                else None,
                "total_co2e_saved_kg": self.outcome.annual_co2e_savings_kg,
            },
        }

    def to_markdown(self) -> str:
        """Generate markdown report of anomalies."""
        lines = [
            "# Cost Anomaly Detection Report",
            "",
            f"**Scenario**: {self.outcome.scenario_name}",
            f"**Strategy**: {self.outcome.strategy_name}",
            f"**Threshold**: {self.threshold} (z-score)",
            "",
        ]

        if not self.has_anomalies:
            lines.extend(["[OK] **No anomalies detected**", ""])
        else:
            lines.extend([f"[WARNING] **{len(self.anomalies)} anomaly(ies) detected**", ""])
            lines.append("| Metric | Value | Z-Score | Baseline (mean +/- std) |")
            lines.append("|--------|-------|---------|------------------|")
            for anomaly in self.anomalies:
                lines.append(
                    f"| {anomaly.metric} | {anomaly.value:.2f} | {anomaly.z_score:.2f} | "
                    f"{anomaly.baseline_mean:.2f} +/-{anomaly.baseline_std:.2f} |"
                )
            lines.append("")

        # Add summary metrics
        break_even_text = (
            f"{self.outcome.break_even_years * 12:.1f} months"
            if self.outcome.break_even_years
            else "never"
        )
        lines.extend(
            [
                "## Summary Metrics",
                "",
                f"- **Annual Savings**: ${self.outcome.annual_total_savings_usd:,.2f}",
                f"- **Break-even Period**: {break_even_text}",
                f"- **CO2e Saved**: {self.outcome.annual_co2e_savings_kg:,.0f} kg",
                "",
            ]
        )

        return "\n".join(lines)


class CostAnomalyDetector:
    """Detects anomalies in optimization cost projections.

    Uses statistical baselines (mean and standard deviation) of historical
    optimization outcomes to identify results that deviate significantly.
    """

    def __init__(
        self,
        baseline_data: list[OptimizationOutcome] | None = None,
        threshold: float = 3.0,
    ):
        """Initialize detector with optional baseline data.

        Args:
            baseline_data: list of historical OptimizationOutcome instances
                used to compute statistical baselines. If None, uses built-in
                conservative defaults.
            threshold: Z-score threshold for anomaly detection (default: 3.0).
        """
        self.threshold = threshold
        self.baselines: dict[str, dict[str, float]] = {}

        if baseline_data:
            self._compute_baselines(baseline_data)
        else:
            # Conservative defaults based on typical optimization results
            self._set_default_baselines()

    def _set_default_baselines(self) -> None:
        """Set conservative default baselines for common metrics."""
        # These defaults represent typical ranges for successful optimizations
        # Based on empirical data from pruning/quantization experiments
        self.baselines = {
            "annual_savings_usd": {"mean": 5000.0, "std": 10000.0},
            "break_even_months": {"mean": 6.0, "std": 4.0},
            "total_co2e_saved_kg": {"mean": 1000.0, "std": 2000.0},
        }

    def _compute_baselines(self, outcomes: list[OptimizationOutcome]) -> None:
        """Compute statistical baselines from historical outcomes.

        Args:
            outcomes: list of historical OptimizationOutcome instances.
        """
        if not outcomes:
            self._set_default_baselines()
            return

        # Collect metrics across all outcomes
        metrics: dict[str, list[float]] = {
            "annual_savings_usd": [],
            "break_even_months": [],
            "total_co2e_saved_kg": [],
        }

        for outcome in outcomes:
            metrics["annual_savings_usd"].append(outcome.annual_total_savings_usd)
            if outcome.break_even_years is not None:
                metrics["break_even_months"].append(outcome.break_even_years * 12)
            metrics["total_co2e_saved_kg"].append(outcome.annual_co2e_savings_kg)

        # Compute mean and standard deviation for each metric
        for metric_name, values in metrics.items():
            if values:
                mean = sum(values) / len(values)
                if len(values) > 1:
                    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
                    std = math.sqrt(variance)
                else:
                    std = 0.0
                self.baselines[metric_name] = {"mean": mean, "std": max(std, 0.001)}
            else:
                # Fall back to defaults if no data
                self._set_default_baselines()
                break

    def detect(self, outcome: OptimizationOutcome) -> AnomalyDetectionResult:
        """Detect anomalies in a single optimization outcome.

        Args:
            outcome: The optimization outcome to analyze.

        Returns:
            AnomalyDetectionResult with any detected anomalies.
        """
        anomalies: list[Anomaly] = []

        # Check key financial and environmental metrics
        self._check_metric(
            outcome,
            "annual_savings_usd",
            outcome.annual_total_savings_usd,
            anomalies,
        )
        # Convert break-even years to months for reporting
        break_even_months = (
            outcome.break_even_years * 12 if outcome.break_even_years else float("inf")
        )
        self._check_metric(
            outcome,
            "break_even_months",
            break_even_months,
            anomalies,
        )
        self._check_metric(
            outcome,
            "total_co2e_saved_kg",
            outcome.annual_co2e_savings_kg,
            anomalies,
        )

        return AnomalyDetectionResult(
            outcome=outcome,
            anomalies=anomalies,
            threshold=self.threshold,
        )

    def _check_metric(
        self,
        outcome: OptimizationOutcome,
        metric_name: str,
        value: float,
        anomalies: list[Anomaly],
    ) -> None:
        """Check a single metric for anomalies and add to list if detected.

        Args:
            outcome: The optimization outcome (for context).
            metric_name: Name of the metric being checked.
            value: Current value of the metric.
            anomalies: list to append anomaly to if detected.
        """
        if metric_name not in self.baselines:
            return

        baseline = self.baselines[metric_name]
        mean = baseline["mean"]
        std = baseline["std"]

        # Avoid division by zero
        if std == 0:
            std = 0.001

        z_score = abs((value - mean) / std) if std > 0 else 0.0

        if z_score > self.threshold:
            anomalies.append(
                Anomaly(
                    metric=metric_name,
                    value=value,
                    baseline_mean=mean,
                    baseline_std=std,
                    z_score=z_score,
                    threshold=self.threshold,
                    description=(
                        f"{metric_name} = {value:.2f} "
                        f"(z-score: {z_score:.2f}, "
                        f"expected: {mean:.2f} +/-{std:.2f})"
                    ),
                )
            )

    @classmethod
    def load_baselines_from_file(cls, path: Path) -> CostAnomalyDetector:
        """Create detector with baselines loaded from JSON file.

        Args:
            path: Path to JSON file containing baseline statistics.

        Returns:
            CostAnomalyDetector initialized with loaded baselines.
        """
        with open(path) as f:
            data = json.load(f)

        detector = cls()
        detector.baselines = data.get("baselines", {})
        detector.threshold = data.get("threshold", 3.0)
        return detector

    def save_baselines_to_file(self, path: Path) -> None:
        """Save current baselines to JSON file.

        Args:
            path: Path to save JSON file.
        """
        data = {
            "baselines": self.baselines,
            "threshold": self.threshold,
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "outcome_count": "unknown",
            },
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


def detect_anomalies(
    outcome: OptimizationOutcome,
    baseline_data: list[OptimizationOutcome] | None = None,
    threshold: float = 3.0,
) -> AnomalyDetectionResult:
    """Convenience function for detecting anomalies in an optimization outcome.

    Args:
        outcome: Optimization outcome to analyze.
        baseline_data: Optional historical data for statistical baselines.
        threshold: Z-score threshold for anomaly detection.

    Returns:
        AnomalyDetectionResult with detected anomalies.
    """
    detector = CostAnomalyDetector(baseline_data, threshold)
    return detector.detect(outcome)
