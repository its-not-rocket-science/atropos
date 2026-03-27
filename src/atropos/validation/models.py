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
        gpu_count: Number of GPUs used for measurement (default 1).
        parallel_strategy: Parallelization strategy ("data", "model", "layer") (default "data").
        scaling_efficiency: Scaling efficiency as fraction of ideal linear scaling (optional).
        communication_overhead_ms: Communication overhead per iteration in milliseconds (optional).
        per_gpu_memory_gb: Peak memory usage per GPU in GB (list length equals
            gpu_count) (optional).
    """

    model_name: str
    parameters_b: float
    memory_gb: float
    throughput_toks_per_sec: float
    latency_ms_per_request: float
    batch_size: int = 1
    power_watts: float | None = None
    gpu_count: int = 1
    parallel_strategy: str = "data"
    scaling_efficiency: float | None = None
    communication_overhead_ms: float | None = None
    per_gpu_memory_gb: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result: dict[str, Any] = {
            "model_name": self.model_name,
            "parameters_b": self.parameters_b,
            "memory_gb": self.memory_gb,
            "throughput_toks_per_sec": self.throughput_toks_per_sec,
            "latency_ms_per_request": self.latency_ms_per_request,
            "batch_size": self.batch_size,
            "gpu_count": self.gpu_count,
            "parallel_strategy": self.parallel_strategy,
        }
        if self.power_watts is not None:
            result["power_watts"] = self.power_watts
        if self.scaling_efficiency is not None:
            result["scaling_efficiency"] = self.scaling_efficiency
        if self.communication_overhead_ms is not None:
            result["communication_overhead_ms"] = self.communication_overhead_ms
        if self.per_gpu_memory_gb:
            result["per_gpu_memory_gb"] = self.per_gpu_memory_gb
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
            1 - abs(self.atropos_savings_pct - self.measured_savings_pct) / self.atropos_savings_pct
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
            f"- GPU count: {self.baseline_metrics.gpu_count}",
            f"- Parallel strategy: {self.baseline_metrics.parallel_strategy}",
        ]
        if self.baseline_metrics.scaling_efficiency is not None:
            lines.append(f"- Scaling efficiency: {self.baseline_metrics.scaling_efficiency:.1%}")
        if self.baseline_metrics.communication_overhead_ms is not None:
            lines.append(
                f"- Communication overhead: "
                f"{self.baseline_metrics.communication_overhead_ms:.1f} ms"
            )
        if self.baseline_metrics.per_gpu_memory_gb:
            mem_str = ", ".join(
                f"{gpu_mem:.2f} GB" for gpu_mem in self.baseline_metrics.per_gpu_memory_gb
            )
            lines.append(f"- Per-GPU memory: [{mem_str}]")
        lines.append("")

        lines.append("### Optimized (Pruned)")
        lines.append(f"- Model: {self.optimized_metrics.model_name}")
        lines.append(f"- Parameters: {self.optimized_metrics.parameters_b:.2f}B")
        lines.append(f"- Memory: {self.optimized_metrics.memory_gb:.2f} GB")
        lines.append(f"- Throughput: {self.optimized_metrics.throughput_toks_per_sec:.1f} tok/s")
        lines.append(f"- Latency: {self.optimized_metrics.latency_ms_per_request:.1f} ms/req")
        lines.append(f"- GPU count: {self.optimized_metrics.gpu_count}")
        lines.append(f"- Parallel strategy: {self.optimized_metrics.parallel_strategy}")
        if self.optimized_metrics.scaling_efficiency is not None:
            lines.append(f"- Scaling efficiency: {self.optimized_metrics.scaling_efficiency:.1%}")
        if self.optimized_metrics.communication_overhead_ms is not None:
            lines.append(
                f"- Communication overhead: "
                f"{self.optimized_metrics.communication_overhead_ms:.1f} ms"
            )
        if self.optimized_metrics.per_gpu_memory_gb:
            mem_str = ", ".join(
                f"{gpu_mem:.2f} GB" for gpu_mem in self.optimized_metrics.per_gpu_memory_gb
            )
            lines.append(f"- Per-GPU memory: [{mem_str}]")
        lines.append("")

        lines.extend(
            [
                "## Comparison: Atropos Projected vs Measured",
                "",
                "| Metric | Projected | Measured | Variance |",
                "|--------|-----------|----------|----------|",
            ]
        )

        for comp in self.comparisons:
            variance_str = f"{comp.variance_pct:+.1f}%"
            lines.append(
                f"| {comp.name} | {comp.projected:.2f} {comp.unit} | "
                f"{comp.measured:.2f} {comp.unit} | {variance_str} |"
            )

        lines.extend(
            [
                "",
                "## Savings Analysis",
                f"- Atropos projected savings: {self.atropos_savings_pct:.1f}%",
                f"- Measured actual savings: {self.measured_savings_pct:.1f}%",
                f"- Projection accuracy: {self.savings_accuracy:.1f}%",
                "",
                "## Assessment",
                self.overall_assessment,
            ]
        )

        return "\n".join(lines)
