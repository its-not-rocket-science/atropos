"""Models for validation results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class MeasuredMetrics:
    """Metrics measured from actual model execution.

    Attributes:
        model_name: Name of the model tested.
        parameters_b: Actual parameter count in billions.
        memory_gb: Peak memory usage in GB.
        throughput_toks_per_sec: Measured token throughput.
        latency_ms_per_request: Average latency per request.
        power_watts: Average power consumption (if available).
        batch_size: Batch size used for measurement.
    """

    model_name: str
    parameters_b: float
    memory_gb: float
    throughput_toks_per_sec: float
    latency_ms_per_request: float
    batch_size: int = 1
    power_watts: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result: dict[str, Any] = {
            "model_name": self.model_name,
            "parameters_b": self.parameters_b,
            "memory_gb": self.memory_gb,
            "throughput_toks_per_sec": self.throughput_toks_per_sec,
            "latency_ms_per_request": self.latency_ms_per_request,
            "batch_size": self.batch_size,
        }
        if self.power_watts is not None:
            result["power_watts"] = self.power_watts
        return result


@dataclass(frozen=True)
class ComparisonMetric:
    """Comparison between projected and measured values.

    Attributes:
        name: Metric name.
        projected: Atropos projected value.
        measured: Actually measured value.
        variance_pct: Percentage difference.
    """

    name: str
    projected: float
    measured: float
    unit: str
    variance_pct: float

    @property
    def is_accurate(self, tolerance_pct: float = 15.0) -> bool:
        """Check if variance is within acceptable tolerance."""
        return abs(self.variance_pct) <= tolerance_pct

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "projected": self.projected,
            "measured": self.measured,
            "unit": self.unit,
            "variance_pct": round(self.variance_pct, 2),
            "is_accurate": self.is_accurate,
        }


@dataclass
class ValidationResult:
    """Complete validation result comparing Atropos vs real measurements.

    Attributes:
        scenario_name: Name of the Atropos scenario used.
        strategy_name: Name of the optimization strategy.
        baseline_metrics: Measured metrics for baseline model.
        optimized_metrics: Measured metrics for optimized/pruned model.
        comparisons: List of metric comparisons.
        atropos_outcome: Atropos projected outcome.
        overall_assessment: Summary of validation.
    """

    scenario_name: str
    strategy_name: str
    baseline_metrics: MeasuredMetrics
    optimized_metrics: MeasuredMetrics
    comparisons: list[ComparisonMetric] = field(default_factory=list)
    atropos_savings_pct: float = 0.0
    measured_savings_pct: float = 0.0
    overall_assessment: str = ""

    @property
    def savings_accuracy(self) -> float:
        """Calculate how accurate the savings projection was."""
        if self.atropos_savings_pct == 0:
            return 0.0
        return (
            1
            - abs(self.atropos_savings_pct - self.measured_savings_pct)
            / self.atropos_savings_pct
        ) * 100

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "scenario_name": self.scenario_name,
            "strategy_name": self.strategy_name,
            "baseline_metrics": self.baseline_metrics.to_dict(),
            "optimized_metrics": self.optimized_metrics.to_dict(),
            "comparisons": [c.to_dict() for c in self.comparisons],
            "atropos_savings_pct": round(self.atropos_savings_pct, 2),
            "measured_savings_pct": round(self.measured_savings_pct, 2),
            "savings_accuracy": round(self.savings_accuracy, 2),
            "overall_assessment": self.overall_assessment,
        }

    def to_markdown(self) -> str:
        """Generate markdown report."""
        lines = [
            "# Atropos Validation Report",
            "",
            f"**Scenario:** {self.scenario_name}",
            f"**Strategy:** {self.strategy_name}",
            "",
            "## Measured Metrics",
            "",
            "### Baseline (Unoptimized)",
            f"- Model: {self.baseline_metrics.model_name}",
            f"- Parameters: {self.baseline_metrics.parameters_b:.2f}B",
            f"- Memory: {self.baseline_metrics.memory_gb:.2f} GB",
            f"- Throughput: {self.baseline_metrics.throughput_toks_per_sec:.1f} tok/s",
            f"- Latency: {self.baseline_metrics.latency_ms_per_request:.1f} ms/req",
            "",
            "### Optimized (Pruned)",
            f"- Model: {self.optimized_metrics.model_name}",
            f"- Parameters: {self.optimized_metrics.parameters_b:.2f}B",
            f"- Memory: {self.optimized_metrics.memory_gb:.2f} GB",
            f"- Throughput: {self.optimized_metrics.throughput_toks_per_sec:.1f} tok/s",
            f"- Latency: {self.optimized_metrics.latency_ms_per_request:.1f} ms/req",
            "",
            "## Comparison: Atropos Projected vs Measured",
            "",
            "| Metric | Projected | Measured | Variance |",
            "|--------|-----------|----------|----------|",
        ]

        for comp in self.comparisons:
            variance_str = f"{comp.variance_pct:+.1f}%"
            lines.append(
                f"| {comp.name} | {comp.projected:.2f} {comp.unit} | "
                f"{comp.measured:.2f} {comp.unit} | {variance_str} |"
            )

        lines.extend([
            "",
            "## Savings Analysis",
            f"- Atropos projected savings: {self.atropos_savings_pct:.1f}%",
            f"- Measured actual savings: {self.measured_savings_pct:.1f}%",
            f"- Projection accuracy: {self.savings_accuracy:.1f}%",
            "",
            "## Assessment",
            self.overall_assessment,
        ])

        return "\n".join(lines)
