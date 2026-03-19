"""Visualization module for pruning result JSON reports.

Generates interactive charts using Plotly for framework comparison,
quality/speed trade-off analysis, pruning reports, and validation reports.

Supports HTML and optional PNG export (requires kaleido).
"""

from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Literal

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    go = None  # type: ignore[assignment]
    make_subplots = None  # type: ignore[assignment]

# Optional PNG export via kaleido
HAS_KALEIDO = importlib.util.find_spec("kaleido") is not None


class ReportType(Enum):
    """Supported report types."""

    FRAMEWORK_COMPARISON = "framework-comparison"
    TRADEOFF_ANALYSIS = "tradeoff-analysis"
    PRUNING_REPORT = "pruning-report"
    VALIDATION_REPORT = "validation-report"
    UNKNOWN = "unknown"


@dataclass
class FrameworkComparisonResult:
    """Lightweight representation of a single framework comparison result."""

    model: str
    framework: str
    status: str
    target_sparsity: float
    achieved_sparsity: float
    original_params: int
    pruned_params: int
    pruning_time_sec: float
    output_path: str
    error_message: str
    timestamp: str

    @property
    def parameter_reduction_fraction(self) -> float:
        """Calculate parameter reduction fraction."""
        if self.original_params == 0:
            return 0.0
        return (self.original_params - self.pruned_params) / self.original_params

    @property
    def sparsity_error(self) -> float:
        """Absolute difference between target and achieved sparsity."""
        return abs(self.target_sparsity - self.achieved_sparsity)


@dataclass
class FrameworkComparisonReport:
    """Complete framework comparison report."""

    total_tests: int
    successful: int
    failed: int
    duration_sec: float
    start_time: str
    end_time: str
    config: dict[str, Any]
    results: list[FrameworkComparisonResult]


@dataclass
class TradeOffResult:
    """Lightweight representation of a single trade-off analysis result."""

    model: str
    framework: str
    status: str
    target_sparsity: float
    achieved_sparsity: float
    original_params: int
    pruned_params: int
    pruning_time_sec: float
    pruning_output_path: str
    baseline_perplexity: float | None
    pruned_perplexity: float | None
    perplexity_change_pct: float | None
    baseline_completion_score: float | None
    pruned_completion_score: float | None
    completion_score_change_pct: float | None
    baseline_inference_time_ms: float | None
    pruned_inference_time_ms: float | None
    inference_speedup_pct: float | None
    quality_speed_ratio: float | None
    error_message: str
    timestamp: str

    @property
    def parameter_reduction_fraction(self) -> float:
        if self.original_params == 0:
            return 0.0
        return (self.original_params - self.pruned_params) / self.original_params

    @property
    def quality_degradation_pct(self) -> float | None:
        # Average of perplexity change and completion score change (both positive = degradation)
        if self.perplexity_change_pct is not None and self.completion_score_change_pct is not None:
            return (self.perplexity_change_pct + self.completion_score_change_pct) / 2
        elif self.perplexity_change_pct is not None:
            return self.perplexity_change_pct
        elif self.completion_score_change_pct is not None:
            return self.completion_score_change_pct
        else:
            return None


@dataclass
class TradeOffReport:
    """Complete trade-off analysis report."""

    total_tests: int
    successful: int
    failed: int
    duration_sec: float
    start_time: str
    end_time: str
    config: dict[str, Any]
    results: list[TradeOffResult]


@dataclass
class PruningReportResult:
    """Lightweight representation of a single pruning report result."""

    model_id: str
    strategy: str
    status: str
    original_params: int
    pruned_params: int
    target_sparsity: float
    actual_sparsity: float
    pruning_time_sec: float
    output_path: str
    error_message: str
    timestamp: str


@dataclass
class PruningReport:
    """Complete pruning report."""

    total_models: int
    successful: int
    failed: int
    duration_sec: float
    output_dir: str
    start_time: str
    end_time: str
    results: list[PruningReportResult]


@dataclass
class ValidationResult:
    """Lightweight representation of a single validation result."""

    model: str
    strategy: str
    original_path: str
    pruned_path: str
    passed: bool
    metrics: dict[str, Any]
    errors: list[str]
    perplexity_passed: bool
    generation_passed: bool


@dataclass
class ValidationReport:
    """Complete validation report."""

    summary: dict[str, int]
    criteria: dict[str, float]
    results: list[ValidationResult]


def detect_report_type(data: dict[str, Any]) -> ReportType:
    """Detect report type based on JSON structure.

    Args:
        data: Parsed JSON data

    Returns:
        ReportType enum
    """
    # Validation report has summary and criteria
    if "summary" in data and "criteria" in data:
        return ReportType.VALIDATION_REPORT

    # Check for results
    if "results" in data:
        results = data.get("results", [])
        if results and isinstance(results[0], dict):
            first = results[0]
            # Pruning report has output_dir at top level
            if "output_dir" in data:
                return ReportType.PRUNING_REPORT
            # Trade-off analysis has baseline_perplexity
            elif "baseline_perplexity" in first:
                return ReportType.TRADEOFF_ANALYSIS
            # Framework comparison has framework but not baseline_perplexity
            elif "framework" in first:
                return ReportType.FRAMEWORK_COMPARISON

    return ReportType.UNKNOWN


def load_framework_comparison(data: dict[str, Any]) -> FrameworkComparisonReport:
    """Load framework comparison data into typed structure."""
    results = []
    for r in data.get("results", []):
        result = FrameworkComparisonResult(
            model=r.get("model", ""),
            framework=r.get("framework", ""),
            status=r.get("status", "failed"),
            target_sparsity=r.get("target_sparsity", 0.0),
            achieved_sparsity=r.get("achieved_sparsity", 0.0),
            original_params=r.get("original_params", 0),
            pruned_params=r.get("pruned_params", 0),
            pruning_time_sec=r.get("pruning_time_sec", 0.0),
            output_path=r.get("output_path", ""),
            error_message=r.get("error_message", ""),
            timestamp=r.get("timestamp", ""),
        )
        results.append(result)

    return FrameworkComparisonReport(
        total_tests=data.get("total_tests", 0),
        successful=data.get("successful", 0),
        failed=data.get("failed", 0),
        duration_sec=data.get("duration_sec", 0.0),
        start_time=data.get("start_time", ""),
        end_time=data.get("end_time", ""),
        config=data.get("config", {}),
        results=results,
    )


def load_tradeoff_analysis(data: dict[str, Any]) -> TradeOffReport:
    """Load trade-off analysis data into typed structure."""
    results = []
    for r in data.get("results", []):
        result = TradeOffResult(
            model=r.get("model", ""),
            framework=r.get("framework", ""),
            status=r.get("status", "failed"),
            target_sparsity=r.get("target_sparsity", 0.0),
            achieved_sparsity=r.get("achieved_sparsity", 0.0),
            original_params=r.get("original_params", 0),
            pruned_params=r.get("pruned_params", 0),
            pruning_time_sec=r.get("pruning_time_sec", 0.0),
            pruning_output_path=r.get("pruning_output_path", ""),
            baseline_perplexity=r.get("baseline_perplexity"),
            pruned_perplexity=r.get("pruned_perplexity"),
            perplexity_change_pct=r.get("perplexity_change_pct"),
            baseline_completion_score=r.get("baseline_completion_score"),
            pruned_completion_score=r.get("pruned_completion_score"),
            completion_score_change_pct=r.get("completion_score_change_pct"),
            baseline_inference_time_ms=r.get("baseline_inference_time_ms"),
            pruned_inference_time_ms=r.get("pruned_inference_time_ms"),
            inference_speedup_pct=r.get("inference_speedup_pct"),
            quality_speed_ratio=r.get("quality_speed_ratio"),
            error_message=r.get("error_message", ""),
            timestamp=r.get("timestamp", ""),
        )
        results.append(result)

    return TradeOffReport(
        total_tests=data.get("total_tests", 0),
        successful=data.get("successful", 0),
        failed=data.get("failed", 0),
        duration_sec=data.get("duration_sec", 0.0),
        start_time=data.get("start_time", ""),
        end_time=data.get("end_time", ""),
        config=data.get("config", {}),
        results=results,
    )


def load_pruning_report(data: dict[str, Any]) -> PruningReport:
    """Load pruning report data into typed structure."""
    results = []
    for r in data.get("results", []):
        result = PruningReportResult(
            model_id=r.get("model_id", ""),
            strategy=r.get("strategy", ""),
            status=r.get("status", "failed"),
            original_params=r.get("original_params", 0),
            pruned_params=r.get("pruned_params", 0),
            target_sparsity=r.get("target_sparsity", 0.0),
            actual_sparsity=r.get("actual_sparsity", 0.0),
            pruning_time_sec=r.get("pruning_time_sec", 0.0),
            output_path=r.get("output_path", ""),
            error_message=r.get("error_message", ""),
            timestamp=r.get("timestamp", ""),
        )
        results.append(result)

    return PruningReport(
        total_models=data.get("total_models", 0),
        successful=data.get("successful", 0),
        failed=data.get("failed", 0),
        duration_sec=data.get("duration_sec", 0.0),
        output_dir=data.get("output_dir", ""),
        start_time=data.get("start_time", ""),
        end_time=data.get("end_time", ""),
        results=results,
    )


def load_validation_report(data: dict[str, Any]) -> ValidationReport:
    """Load validation report data into typed structure."""
    results = []
    for r in data.get("results", []):
        result = ValidationResult(
            model=r.get("model", ""),
            strategy=r.get("strategy", ""),
            original_path=r.get("original_path", ""),
            pruned_path=r.get("pruned_path", ""),
            passed=r.get("passed", False),
            metrics=r.get("metrics", {}),
            errors=r.get("errors", []),
            perplexity_passed=r.get("perplexity_passed", False),
            generation_passed=r.get("generation_passed", False),
        )
        results.append(result)

    return ValidationReport(
        summary=data.get("summary", {}),
        criteria=data.get("criteria", {}),
        results=results,
    )


def create_framework_comparison_figures(
    report: FrameworkComparisonReport,
) -> dict[str, go.Figure]:
    """Create Plotly figures for framework comparison report.

    Returns:
        Dictionary mapping figure names to Plotly Figure objects
    """
    if go is None:
        return {}

    figures = {}

    # 1. Sparsity accuracy bar chart
    fig_sparsity = go.Figure()

    # Group by framework
    frameworks = sorted({r.framework for r in report.results})
    models = sorted({r.model for r in report.results})

    for framework in frameworks:
        framework_results = [r for r in report.results if r.framework == framework]
        target_sparsities = [r.target_sparsity * 100 for r in framework_results]
        achieved_sparsities = [r.achieved_sparsity * 100 for r in framework_results]

        # Use model names as x-axis labels
        x_labels = [r.model for r in framework_results]

        fig_sparsity.add_trace(
            go.Bar(
                name=f"{framework} (target)",
                x=x_labels,
                y=target_sparsities,
                marker_color="#95a5a6",
                opacity=0.7,
                text=[f"{t:.1f}%" for t in target_sparsities],
                textposition="auto",
            )
        )

        fig_sparsity.add_trace(
            go.Bar(
                name=f"{framework} (achieved)",
                x=x_labels,
                y=achieved_sparsities,
                marker_color="#2ecc71",
                text=[f"{a:.1f}%" for a in achieved_sparsities],
                textposition="auto",
            )
        )

    fig_sparsity.update_layout(
        barmode="group",
        title="Sparsity: Target vs Achieved",
        xaxis_title="Model",
        yaxis_title="Sparsity (%)",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=400,
    )
    figures["sparsity_accuracy"] = fig_sparsity

    # 2. Pruning time per framework (average across models)
    fig_time = go.Figure()

    avg_times = []
    success_rates = []
    for framework in frameworks:
        framework_results = [r for r in report.results if r.framework == framework]
        successful = [r for r in framework_results if r.status == "success"]
        if successful:
            avg_time = sum(r.pruning_time_sec for r in successful) / len(successful)
            success_rate = len(successful) / len(framework_results)
        else:
            avg_time = 0.0
            success_rate = 0.0
        avg_times.append(avg_time)
        success_rates.append(success_rate)

    fig_time.add_trace(
        go.Bar(
            name="Avg Pruning Time",
            x=frameworks,
            y=avg_times,
            marker_color="#3498db",
            text=[f"{t:.1f}s" for t in avg_times],
            textposition="auto",
        )
    )

    # Add success rate as line on secondary y-axis
    fig_time.add_trace(
        go.Scatter(
            name="Success Rate",
            x=frameworks,
            y=[sr * 100 for sr in success_rates],
            yaxis="y2",
            mode="lines+markers",
            line=dict(color="#e74c3c", width=3),
            marker=dict(size=10),
            text=[f"{sr:.1%}" for sr in success_rates],
            textposition="top center",
        )
    )

    fig_time.update_layout(
        title="Pruning Time and Success Rate by Framework",
        xaxis_title="Framework",
        yaxis_title="Average Time (seconds)",
        yaxis2=dict(
            title="Success Rate (%)",
            overlaying="y",
            side="right",
            range=[0, 100],
        ),
        showlegend=True,
        height=400,
    )
    figures["pruning_time"] = fig_time

    # 3. Parameter reduction heatmap (if multiple models and frameworks)
    if len(models) > 1 and len(frameworks) > 1:
        # Create matrix of parameter reduction fractions
        matrix = []
        for model in models:
            row: list[float | None] = []
            for framework in frameworks:
                result = next(
                    (r for r in report.results if r.model == model and r.framework == framework),
                    None,
                )
                if result and result.status == "success":
                    row.append(result.parameter_reduction_fraction * 100)
                else:
                    row.append(None)
            matrix.append(row)

        fig_heatmap = go.Figure(
            data=go.Heatmap(
                z=matrix,
                x=frameworks,
                y=models,
                colorscale="Viridis",
                text=[[f"{v:.1f}%" if v is not None else "N/A" for v in row] for row in matrix],
                texttemplate="%{text}",
                textfont={"size": 12},
            )
        )

        fig_heatmap.update_layout(
            title="Parameter Reduction (%)",
            xaxis_title="Framework",
            yaxis_title="Model",
            height=400,
        )
        figures["parameter_reduction_heatmap"] = fig_heatmap

    # 4. Success/failure stacked bar chart
    fig_status = go.Figure()

    for framework in frameworks:
        framework_results = [r for r in report.results if r.framework == framework]
        success_count = sum(1 for r in framework_results if r.status == "success")
        failed_count = len(framework_results) - success_count

        fig_status.add_trace(
            go.Bar(
                name=framework,
                x=[framework],
                y=[success_count],
                marker_color="#2ecc71",
                text=f"{success_count}",
                textposition="inside",
            )
        )

        if failed_count > 0:
            fig_status.add_trace(
                go.Bar(
                    name=framework,
                    x=[framework],
                    y=[failed_count],
                    marker_color="#e74c3c",
                    text=f"{failed_count}",
                    textposition="inside",
                    base=[success_count],  # Stack on top of success
                    showlegend=False,
                )
            )

    fig_status.update_layout(
        barmode="stack",
        title="Success vs Failure Count by Framework",
        xaxis_title="Framework",
        yaxis_title="Count",
        showlegend=False,
        height=400,
    )
    figures["success_failure"] = fig_status

    return figures


def create_tradeoff_analysis_figures(
    report: TradeOffReport,
) -> dict[str, go.Figure]:
    """Create Plotly figures for trade-off analysis report."""
    if go is None:
        return {}

    figures = {}
    successful_results = [r for r in report.results if r.status == "success"]
    if not successful_results:
        return {}

    # 1. Quality vs Speed scatter plot
    fig_scatter = go.Figure()

    frameworks = sorted({r.framework for r in successful_results})
    colors = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12", "#9b59b6"]

    for i, framework in enumerate(frameworks):
        framework_results = [r for r in successful_results if r.framework == framework]

        # Calculate quality degradation (positive is bad) and speed improvement (positive is good)
        x_vals = []
        y_vals = []
        sizes = []
        texts = []

        for result in framework_results:
            # Quality degradation: perplexity increase + completion score decrease
            qual_degradation = result.quality_degradation_pct
            if qual_degradation is None:
                continue

            # Speed improvement: negative inference_speedup_pct means slowdown
            speed_improvement = -result.inference_speedup_pct if result.inference_speedup_pct else 0
            if speed_improvement is None:
                continue

            x_vals.append(speed_improvement)  # X: speed improvement (positive = faster)
            y_vals.append(qual_degradation)  # Y: quality degradation (positive = worse)
            sizes.append(result.parameter_reduction_fraction * 50 + 10)  # Scale for visibility
            texts.append(
                f"{result.model}<br>"
                f"Quality: {qual_degradation:.1f}% worse<br>"
                f"Speed: {speed_improvement:.1f}% faster<br>"
                f"Sparsity: {result.achieved_sparsity:.1%}"
            )

        if x_vals:
            fig_scatter.add_trace(
                go.Scatter(
                    name=framework,
                    x=x_vals,
                    y=y_vals,
                    mode="markers",
                    marker=dict(
                        size=sizes,
                        color=colors[i % len(colors)],
                        line=dict(width=1, color="white"),
                    ),
                    text=texts,
                    hoverinfo="text",
                )
            )

    fig_scatter.update_layout(
        title="Quality vs Speed Trade-off",
        xaxis_title="Speed Improvement (%) (positive = faster)",
        yaxis_title="Quality Degradation (%) (positive = worse)",
        showlegend=True,
        hovermode="closest",
        height=500,
    )

    # Add quadrant lines
    fig_scatter.add_hline(y=0, line=dict(color="gray", width=1, dash="dash"))
    fig_scatter.add_vline(x=0, line=dict(color="gray", width=1, dash="dash"))

    # Add quadrant labels
    fig_scatter.add_annotation(
        x=0.05,
        y=0.05,
        xref="paper",
        yref="paper",
        text="Better quality, faster",
        showarrow=False,
        font=dict(color="green", size=10),
    )

    figures["quality_speed_scatter"] = fig_scatter

    # 2. Perplexity change per framework (grouped bar)
    fig_perplexity = go.Figure()

    for framework in frameworks:
        framework_results = [r for r in successful_results if r.framework == framework]
        models = [r.model for r in framework_results]
        perplexity_changes = []

        for result in framework_results:
            if result.perplexity_change_pct is not None:
                # Positive perplexity change means degradation (worse)
                perplexity_changes.append(result.perplexity_change_pct)
            else:
                perplexity_changes.append(0)

        color = "#e74c3c" if any(p > 0 for p in perplexity_changes) else "#2ecc71"

        fig_perplexity.add_trace(
            go.Bar(
                name=framework,
                x=models,
                y=perplexity_changes,
                marker_color=color,
                text=[f"{p:+.1f}%" for p in perplexity_changes],
                textposition="auto",
            )
        )

    fig_perplexity.update_layout(
        barmode="group",
        title="Perplexity Change by Framework",
        xaxis_title="Model",
        yaxis_title="Perplexity Change (%)",
        showlegend=True,
        height=400,
    )
    figures["perplexity_change"] = fig_perplexity

    # 3. Quality/Speed ratio bar chart
    fig_ratio = go.Figure()

    for framework in frameworks:
        framework_results = [r for r in successful_results if r.framework == framework]
        avg_ratio = None
        ratios = []

        for result in framework_results:
            if result.quality_speed_ratio is not None:
                ratios.append(result.quality_speed_ratio)

        if ratios:
            avg_ratio = sum(ratios) / len(ratios)

        if avg_ratio is not None:
            color = "#e74c3c" if avg_ratio > 1 else "#2ecc71"
            fig_ratio.add_trace(
                go.Bar(
                    name=framework,
                    x=[framework],
                    y=[avg_ratio],
                    marker_color=color,
                    text=f"{avg_ratio:.2f}",
                    textposition="auto",
                )
            )

    if fig_ratio.data:
        fig_ratio.update_layout(
            title="Average Quality/Speed Ratio by Framework",
            xaxis_title="Framework",
            yaxis_title="Ratio (Quality degradation / Speed improvement)",
            showlegend=False,
            height=400,
        )
        figures["quality_speed_ratio"] = fig_ratio

    return figures


def create_pruning_report_figures(
    report: PruningReport,
) -> dict[str, go.Figure]:
    """Create Plotly figures for pruning report."""
    if go is None:
        return {}

    figures = {}
    successful_results = [r for r in report.results if r.status == "success"]
    if not successful_results:
        return {}

    # 1. Target vs Actual sparsity (grouped by model and strategy)
    fig_sparsity = go.Figure()

    strategies = sorted({r.strategy for r in successful_results})
    models = sorted({r.model_id for r in successful_results})

    # Create grouped bars
    for strategy in strategies:
        strategy_results = [r for r in successful_results if r.strategy == strategy]
        target_sparsities = [r.target_sparsity * 100 for r in strategy_results]
        actual_sparsities = [r.actual_sparsity * 100 for r in strategy_results]
        model_labels = [r.model_id for r in strategy_results]

        fig_sparsity.add_trace(
            go.Bar(
                name=f"{strategy} (target)",
                x=model_labels,
                y=target_sparsities,
                marker_color="#95a5a6",
                opacity=0.7,
                text=[f"{t:.1f}%" for t in target_sparsities],
                textposition="auto",
            )
        )

        fig_sparsity.add_trace(
            go.Bar(
                name=f"{strategy} (actual)",
                x=model_labels,
                y=actual_sparsities,
                marker_color="#3498db",
                text=[f"{a:.1f}%" for a in actual_sparsities],
                textposition="auto",
            )
        )

    fig_sparsity.update_layout(
        barmode="group",
        title="Target vs Actual Sparsity",
        xaxis_title="Model",
        yaxis_title="Sparsity (%)",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=400,
    )
    figures["sparsity_target_vs_actual"] = fig_sparsity

    # 2. Pruning time per model (stacked by strategy)
    fig_time = go.Figure()

    for model in models:
        model_results = [r for r in successful_results if r.model_id == model]
        total_time = sum(r.pruning_time_sec for r in model_results)

        # Create stacked bar segments for each strategy
        cumulative = 0.0
        for strategy in strategies:
            strategy_result = next(
                (r for r in model_results if r.strategy == strategy),
                None,
            )
            if strategy_result:
                fig_time.add_trace(
                    go.Bar(
                        name=strategy,
                        x=[model],
                        y=[strategy_result.pruning_time_sec],
                        marker_color="#3498db",
                        text=f"{strategy_result.pruning_time_sec:.1f}s",
                        textposition="inside",
                        base=cumulative,
                        showlegend=(model == models[0]),  # Only show legend for first model
                    )
                )
                cumulative += strategy_result.pruning_time_sec

        # Add total time annotation
        if total_time > 0:
            fig_time.add_annotation(
                x=model,
                y=total_time * 1.05,
                text=f"Total: {total_time:.1f}s",
                showarrow=False,
                font=dict(size=10),
            )

    fig_time.update_layout(
        barmode="stack",
        title="Pruning Time by Model (Stacked by Strategy)",
        xaxis_title="Model",
        yaxis_title="Time (seconds)",
        showlegend=True,
        height=400,
    )
    figures["pruning_time_stacked"] = fig_time

    # 3. Parameter reduction across strategies
    fig_reduction = go.Figure()

    for strategy in strategies:
        strategy_results = [r for r in successful_results if r.strategy == strategy]
        reductions = []
        model_labels = []

        for result in strategy_results:
            numerator = result.original_params - result.pruned_params
            reduction = numerator / result.original_params * 100
            reductions.append(reduction)
            model_labels.append(result.model_id)

        fig_reduction.add_trace(
            go.Bar(
                name=strategy,
                x=model_labels,
                y=reductions,
                marker_color="#2ecc71",
                text=[f"{r:.1f}%" for r in reductions],
                textposition="auto",
            )
        )

    fig_reduction.update_layout(
        barmode="group",
        title="Parameter Reduction by Strategy",
        xaxis_title="Model",
        yaxis_title="Parameter Reduction (%)",
        showlegend=True,
        height=400,
    )
    figures["parameter_reduction"] = fig_reduction

    return figures


def create_validation_report_figures(
    report: ValidationReport,
) -> dict[str, go.Figure]:
    """Create Plotly figures for validation report."""
    if go is None:
        return {}

    figures = {}

    # 1. Perplexity increase vs generation similarity scatter
    fig_scatter = go.Figure()

    colors = {"passed": "#2ecc71", "failed": "#e74c3c"}

    for result in report.results:
        perplexity_increase = result.metrics.get("perplexity_increase_pct", 0)
        avg_similarity = result.metrics.get("avg_generation_similarity", 0) * 100

        fig_scatter.add_trace(
            go.Scatter(
                name=f"{result.model} - {result.strategy}",
                x=[perplexity_increase],
                y=[avg_similarity],
                mode="markers",
                marker=dict(
                    size=20,
                    color=colors["passed"] if result.passed else colors["failed"],
                    line=dict(width=1, color="white"),
                ),
                text=(
                    f"{result.model} - {result.strategy}<br>"
                    f"Passed: {result.passed}<br>"
                    f"Perplexity increase: {perplexity_increase:.1f}%<br>"
                    f"Avg similarity: {avg_similarity:.1f}%"
                ),
                hoverinfo="text",
            )
        )

    # Add thresholds
    perplexity_tolerance = report.criteria.get("perplexity_tolerance_pct", 20)
    similarity_threshold = report.criteria.get("generation_similarity_threshold", 0.7) * 100

    fig_scatter.add_hline(
        y=similarity_threshold,
        line=dict(color="gray", width=2, dash="dash"),
        annotation_text=f"Similarity threshold: {similarity_threshold:.0f}%",
        annotation_position="bottom right",
    )

    fig_scatter.add_vline(
        x=perplexity_tolerance,
        line=dict(color="gray", width=2, dash="dash"),
        annotation_text=f"Perplexity tolerance: {perplexity_tolerance:.0f}%",
        annotation_position="top right",
    )

    fig_scatter.update_layout(
        title="Validation Results: Perplexity vs Generation Similarity",
        xaxis_title="Perplexity Increase (%)",
        yaxis_title="Average Generation Similarity (%)",
        showlegend=True,
        hovermode="closest",
        height=500,
    )
    figures["validation_scatter"] = fig_scatter

    # 2. Pass/fail heatmap by model and strategy
    models = sorted({r.model for r in report.results})
    strategies = sorted({r.strategy for r in report.results})

    if len(models) > 1 and len(strategies) > 1:
        matrix = []
        for model in models:
            row: list[int | None] = []
            for strategy in strategies:
                found_result = next(
                    (r for r in report.results if r.model == model and r.strategy == strategy),
                    None,
                )
                if found_result:
                    # 1 for passed, 0 for failed
                    row.append(1 if found_result.passed else 0)
                else:
                    row.append(None)
            matrix.append(row)

        fig_heatmap = go.Figure(
            data=go.Heatmap(
                z=matrix,
                x=strategies,
                y=models,
                colorscale=[[0, "#e74c3c"], [1, "#2ecc71"]],
                text=[
                    ["N/A" if v is None else {1: "PASS", 0: "FAIL"}[v] for v in row]
                    for row in matrix
                ],
                texttemplate="%{text}",
                textfont={"size": 14, "color": "white"},
                showscale=False,
            )
        )

        fig_heatmap.update_layout(
            title="Pass/Fail Status by Model and Strategy",
            xaxis_title="Strategy",
            yaxis_title="Model",
            height=400,
        )
        figures["pass_fail_heatmap"] = fig_heatmap

    # 3. Summary pie chart
    summary = report.summary
    if summary:
        fig_pie = go.Figure(
            data=[
                go.Pie(
                    labels=["Passed", "Failed"],
                    values=[summary.get("passed", 0), summary.get("failed", 0)],
                    marker_colors=["#2ecc71", "#e74c3c"],
                    textinfo="label+percent+value",
                    hole=0.3,
                )
            ]
        )

        fig_pie.update_layout(
            title="Validation Summary",
            height=400,
        )
        figures["summary_pie"] = fig_pie

    return figures


def save_figures(
    figures: dict[str, go.Figure],
    output_dir: Path,
    prefix: str = "",
    formats: list[Literal["html", "png"]] | None = None,
) -> list[Path]:
    """Save figures to files in the specified formats.

    Args:
        figures: Dictionary mapping figure names to Plotly Figure objects
        output_dir: Directory to save figures
        prefix: Optional prefix for filenames
        formats: List of formats to save ("html", "png")

    Returns:
        List of saved file paths
    """
    if formats is None:
        formats = ["html"]
    saved_paths = []
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, fig in figures.items():
        safe_name = name.lower().replace(" ", "_").replace("/", "_")
        filename = f"{prefix}_{safe_name}" if prefix else safe_name

        for fmt in formats:
            if fmt == "html":
                filepath = output_dir / f"{filename}.html"
                fig.write_html(str(filepath))
                saved_paths.append(filepath)
            elif fmt == "png":
                if HAS_KALEIDO:
                    filepath = output_dir / f"{filename}.png"
                    fig.write_image(str(filepath))
                    saved_paths.append(filepath)
                else:
                    print(f"Warning: Skipping PNG export for {name} - kaleido not installed")

    return saved_paths


def visualize_json(
    input_path: Path | str,
    output_dir: Path | str = ".",
    formats: list[Literal["html", "png"]] | None = None,
    report_type: str = "auto",
) -> None:
    """Main entry point: load JSON report and generate visualizations.

    Args:
        input_path: Path to JSON report file
        output_dir: Directory to save visualizations
        formats: List of formats to generate ("html", "png")
        report_type: Report type ("auto", "framework-comparison", "tradeoff-analysis",
                   "pruning-report", "validation-report")

    Raises:
        FileNotFoundError: If input_path doesn't exist
        ValueError: If report type cannot be determined or is unsupported
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    if formats is None:
        formats = ["html"]

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Load JSON data
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    # Determine report type
    if report_type == "auto":
        detected = detect_report_type(data)
        if detected == ReportType.UNKNOWN:
            raise ValueError(f"Could not detect report type from {input_path}")
        report_type = detected.value
    else:
        try:
            detected = ReportType(report_type)
        except ValueError:
            raise ValueError(f"Invalid report type: {report_type}") from None

    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    vis_dir = output_dir / "visualizations" / f"{report_type}_{timestamp}"

    # Load data and generate figures based on type
    figures = {}

    if report_type == ReportType.FRAMEWORK_COMPARISON.value:
        figures = create_framework_comparison_figures(load_framework_comparison(data))
    elif report_type == ReportType.TRADEOFF_ANALYSIS.value:
        figures = create_tradeoff_analysis_figures(load_tradeoff_analysis(data))
    elif report_type == ReportType.PRUNING_REPORT.value:
        figures = create_pruning_report_figures(load_pruning_report(data))
    elif report_type == ReportType.VALIDATION_REPORT.value:
        figures = create_validation_report_figures(load_validation_report(data))
    else:
        raise ValueError(f"Unsupported report type: {report_type}")

    # Check if Plotly is available
    if go is None:
        print("Warning: Plotly not installed. Skipping visualization generation.")
        return

    if not figures:
        print("Warning: No figures generated. The report may contain no successful results.")
        return

    # Save figures
    saved = save_figures(figures, vis_dir, prefix=report_type, formats=formats)

    print(f"Generated {len(saved)} visualization files in {vis_dir}")
    for path in saved:
        print(f"  - {path.name}")
