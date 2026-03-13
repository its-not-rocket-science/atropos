"""Scenario calibration from real serving traces.

Compares projected scenario parameters against actual telemetry data
to identify variance and recommend calibration adjustments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .models import DeploymentScenario
    from .telemetry import TelemetryData


@dataclass(frozen=True)
class CalibrationMetric:
    """Single metric calibration result.

    Attributes:
        name: Metric name (e.g., 'throughput', 'memory').
        projected: Projected/estimated value from scenario.
        actual: Actual measured value from telemetry.
        unit: Unit of measurement.
        variance_pct: Percentage difference ((actual - projected) / projected * 100).
        confidence: Confidence level (high/medium/low) based on variance magnitude.
    """

    name: str
    projected: float
    actual: float
    unit: str
    variance_pct: float
    confidence: str  # "high", "medium", "low"

    def is_within_tolerance(self, tolerance_pct: float = 10.0) -> bool:
        """Check if variance is within acceptable tolerance."""
        return abs(self.variance_pct) <= tolerance_pct


@dataclass(frozen=True)
class CalibrationResult:
    """Complete calibration analysis result.

    Attributes:
        scenario_name: Name of the scenario being calibrated.
        telemetry_source: Source of telemetry data.
        metrics: List of calibrated metrics.
        overall_confidence: Overall confidence based on metric variances.
        recommendations: List of calibration recommendations.
    """

    scenario_name: str
    telemetry_source: str
    metrics: list[CalibrationMetric]
    overall_confidence: str
    recommendations: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert calibration result to dictionary."""
        return {
            "scenario_name": self.scenario_name,
            "telemetry_source": self.telemetry_source,
            "metrics": [
                {
                    "name": m.name,
                    "projected": m.projected,
                    "actual": m.actual,
                    "unit": m.unit,
                    "variance_pct": round(m.variance_pct, 2),
                    "confidence": m.confidence,
                    "within_tolerance_10pct": m.is_within_tolerance(10.0),
                }
                for m in self.metrics
            ],
            "overall_confidence": self.overall_confidence,
            "recommendations": self.recommendations,
        }


def _calculate_confidence(variance_pct: float) -> str:
    """Determine confidence level based on variance percentage.

    Args:
        variance_pct: Percentage difference between actual and projected.

    Returns:
        Confidence level: "high" (<10%), "medium" (10-25%), "low" (>25%).
    """
    abs_variance = abs(variance_pct)
    if abs_variance < 10.0:
        return "high"
    elif abs_variance < 25.0:
        return "medium"
    else:
        return "low"


def _compute_variance(projected: float, actual: float) -> float:
    """Compute percentage variance between projected and actual.

    Args:
        projected: Projected/estimated value.
        actual: Actual measured value.

    Returns:
        Percentage variance ((actual - projected) / projected * 100).
    """
    if projected == 0:
        return float("inf") if actual != 0 else 0.0
    return ((actual - projected) / projected) * 100


def calibrate_scenario(
    scenario: DeploymentScenario,
    telemetry: TelemetryData,
    tolerance_pct: float = 10.0,
) -> CalibrationResult:
    """Calibrate a scenario against real telemetry data.

    Compares projected scenario parameters against actual telemetry
    measurements to identify variance and generate recommendations.

    Args:
        scenario: Deployment scenario with projected parameters.
        telemetry: Actual telemetry data from serving traces.
        tolerance_pct: Acceptable variance tolerance percentage.

    Returns:
        CalibrationResult with variance analysis and recommendations.
    """
    metrics: list[CalibrationMetric] = []

    # Memory calibration
    memory_variance = _compute_variance(scenario.memory_gb, telemetry.memory_gb)
    metrics.append(
        CalibrationMetric(
            name="memory",
            projected=scenario.memory_gb,
            actual=telemetry.memory_gb,
            unit="GB",
            variance_pct=memory_variance,
            confidence=_calculate_confidence(memory_variance),
        )
    )

    # Throughput calibration
    throughput_variance = _compute_variance(
        scenario.throughput_toks_per_sec, telemetry.throughput_toks_per_sec
    )
    metrics.append(
        CalibrationMetric(
            name="throughput",
            projected=scenario.throughput_toks_per_sec,
            actual=telemetry.throughput_toks_per_sec,
            unit="tok/s",
            variance_pct=throughput_variance,
            confidence=_calculate_confidence(throughput_variance),
        )
    )

    # Power calibration (if telemetry has power data)
    if telemetry.power_watts is not None:
        power_variance = _compute_variance(scenario.power_watts, telemetry.power_watts)
        metrics.append(
            CalibrationMetric(
                name="power",
                projected=scenario.power_watts,
                actual=telemetry.power_watts,
                unit="W",
                variance_pct=power_variance,
                confidence=_calculate_confidence(power_variance),
            )
        )

    # Tokens per request calibration
    token_variance = _compute_variance(scenario.tokens_per_request, telemetry.tokens_per_request)
    metrics.append(
        CalibrationMetric(
            name="tokens_per_request",
            projected=scenario.tokens_per_request,
            actual=telemetry.tokens_per_request,
            unit="tokens",
            variance_pct=token_variance,
            confidence=_calculate_confidence(token_variance),
        )
    )

    # Calculate overall confidence
    confidence_scores = {"high": 3, "medium": 2, "low": 1}
    total_score = sum(confidence_scores[m.confidence] for m in metrics)
    avg_score = total_score / len(metrics) if metrics else 0

    if avg_score >= 2.5:
        overall_confidence = "high"
    elif avg_score >= 1.5:
        overall_confidence = "medium"
    else:
        overall_confidence = "low"

    # Generate recommendations
    recommendations = _generate_recommendations(metrics, tolerance_pct)

    return CalibrationResult(
        scenario_name=scenario.name,
        telemetry_source=telemetry.source,
        metrics=metrics,
        overall_confidence=overall_confidence,
        recommendations=recommendations,
    )


def _generate_recommendations(metrics: list[CalibrationMetric], tolerance_pct: float) -> list[str]:
    """Generate calibration recommendations based on metric variances.

    Args:
        metrics: List of calibrated metrics.
        tolerance_pct: Acceptable variance tolerance percentage.

    Returns:
        List of recommendation strings.
    """
    recommendations: list[str] = []

    # Identify significantly miscalibrated metrics
    high_variance_metrics = [m for m in metrics if abs(m.variance_pct) > tolerance_pct * 2]
    medium_variance_metrics = [
        m for m in metrics if tolerance_pct < abs(m.variance_pct) <= tolerance_pct * 2
    ]

    # Overall assessment
    if not high_variance_metrics and not medium_variance_metrics:
        recommendations.append(
            "Scenario is well-calibrated. All metrics within acceptable tolerance."
        )
        return recommendations

    # Specific metric recommendations
    for metric in high_variance_metrics:
        direction = "higher" if metric.variance_pct > 0 else "lower"
        recommendations.append(
            f"Significant variance in {metric.name}: "
            f"actual is {abs(metric.variance_pct):.1f}% {direction} than projected. "
            f"Consider updating scenario {metric.name} to {metric.actual:.2f} {metric.unit}."
        )

    for metric in medium_variance_metrics:
        direction = "higher" if metric.variance_pct > 0 else "lower"
        recommendations.append(
            f"Moderate variance in {metric.name}: "
            f"actual is {abs(metric.variance_pct):.1f}% {direction} than projected. "
            f"Monitor this metric for drift."
        )

    # Calibration advice based on patterns
    throughput_metric = next((m for m in metrics if m.name == "throughput"), None)
    memory_metric = next((m for m in metrics if m.name == "memory"), None)

    if throughput_metric and memory_metric:
        if throughput_metric.variance_pct < -20 and memory_metric.variance_pct > 20:
            recommendations.append(
                "Pattern detected: Lower throughput with higher memory usage suggests "
                "potential batching inefficiency or memory pressure. Consider reducing batch_size."
            )
        elif throughput_metric.variance_pct > 20 and memory_metric.variance_pct < -20:
            recommendations.append(
                "Pattern detected: Higher throughput with lower memory suggests "
                "model may be more efficient than projected. Consider increasing batch_size."
            )

    return recommendations


def generate_calibration_report(result: CalibrationResult, format: str = "markdown") -> str:
    """Generate a calibration report in the specified format.

    Args:
        result: Calibration analysis result.
        format: Output format ("markdown" or "json").

    Returns:
        Formatted calibration report.
    """
    if format == "json":
        import json

        return json.dumps(result.to_dict(), indent=2)

    # Markdown format
    lines = [
        "# Scenario Calibration Report",
        "",
        f"**Scenario:** {result.scenario_name}",
        f"**Telemetry Source:** {result.telemetry_source}",
        f"**Overall Confidence:** {result.overall_confidence.upper()}",
        "",
        "## Metric Calibration",
        "",
        "| Metric | Projected | Actual | Variance | Confidence |",
        "|--------|-----------|--------|----------|------------|",
    ]

    for metric in result.metrics:
        variance_str = f"{metric.variance_pct:+.1f}%"
        lines.append(
            f"| {metric.name} | {metric.projected:.2f} {metric.unit} | "
            f"{metric.actual:.2f} {metric.unit} | {variance_str} | "
            f"{metric.confidence.upper()} |"
        )

    lines.extend(
        [
            "",
            "## Recommendations",
            "",
        ]
    )

    if result.recommendations:
        for rec in result.recommendations:
            lines.append(f"- {rec}")
    else:
        lines.append("No recommendations - scenario is well-calibrated.")

    lines.append("")
    return "\n".join(lines)
