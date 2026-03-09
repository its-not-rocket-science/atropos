"""UI component builders for the Atropos dashboard."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dash import html

try:
    import plotly.graph_objects as go
except ImportError:
    go = None

if TYPE_CHECKING:
    from ..models import DeploymentScenario, OptimizationOutcome, OptimizationStrategy


def create_results_display(outcome: OptimizationOutcome) -> html.Div:
    """Create the main results display."""
    break_even_text = (
        f"{outcome.break_even_years:.1f} years"
        if outcome.break_even_years
        else "Never"
    )

    # Calculate percentage changes
    mem_pct = (
        (outcome.baseline_memory_gb - outcome.optimized_memory_gb)
        / outcome.baseline_memory_gb * 100
    )
    thr_diff = (
        outcome.optimized_throughput_toks_per_sec
        - outcome.baseline_throughput_toks_per_sec
    )
    thr_pct = thr_diff / outcome.baseline_throughput_toks_per_sec * 100
    pwr_pct = (
        (outcome.baseline_power_watts - outcome.optimized_power_watts)
        / outcome.baseline_power_watts * 100
    )
    nrg_diff = (
        outcome.baseline_energy_wh_per_request
        - outcome.optimized_energy_wh_per_request
    )
    nrg_pct = nrg_diff / outcome.baseline_energy_wh_per_request * 100

    # Quality risk color mapping
    risk_bg = {
        "low": "#d4edda",
        "medium": "#fff3cd",
        "high": "#f8d7da",
    }.get(outcome.quality_risk, "#f8d7da")
    risk_color = {
        "low": "#155724",
        "medium": "#856404",
        "high": "#721c24",
    }.get(outcome.quality_risk, "#721c24")

    return html.Div(
        [
            html.Div(
                [
                    create_metric_card(
                        "Annual Savings",
                        f"${outcome.annual_total_savings_usd:,.0f}",
                        "#2ecc71",
                    ),
                    create_metric_card(
                        "Break-even Time",
                        break_even_text,
                        "#3498db",
                    ),
                    create_metric_card(
                        "CO₂e Savings",
                        f"{outcome.annual_co2e_savings_kg:,.0f} kg/year",
                        "#27ae60",
                    ),
                ],
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(3, 1fr)",
                    "gap": "15px",
                    "marginBottom": "25px",
                },
            ),
            html.Hr(style={"margin": "20px 0", "border": "none", "borderTop": "1px solid #ecf0f1"}),
            html.H4("Optimized Metrics", style={"color": "#2c3e50", "marginBottom": "15px"}),
            html.Div(
                [
                    create_metric_row(
                        "Memory",
                        f"{outcome.baseline_memory_gb:.1f} GB",
                        f"{outcome.optimized_memory_gb:.1f} GB",
                        f"{mem_pct:.1f}%",
                    ),
                    create_metric_row(
                        "Throughput",
                        f"{outcome.baseline_throughput_toks_per_sec:.1f} tok/s",
                        f"{outcome.optimized_throughput_toks_per_sec:.1f} tok/s",
                        f"{thr_pct:+.1f}%",
                    ),
                    create_metric_row(
                        "Power",
                        f"{outcome.baseline_power_watts:.0f} W",
                        f"{outcome.optimized_power_watts:.0f} W",
                        f"{pwr_pct:.1f}%",
                    ),
                    create_metric_row(
                        "Energy/Request",
                        f"{outcome.baseline_energy_wh_per_request:.2f} Wh",
                        f"{outcome.optimized_energy_wh_per_request:.2f} Wh",
                        f"{nrg_pct:.1f}%",
                    ),
                ],
                style={"display": "grid", "gap": "10px"},
            ),
            html.Hr(style={"margin": "20px 0", "border": "none", "borderTop": "1px solid #ecf0f1"}),
            html.H4("Annual Cost Breakdown", style={"color": "#2c3e50", "marginBottom": "15px"}),
            html.Div(
                [
                    create_cost_row(
                        "Total Cost",
                        outcome.baseline_annual_total_cost_usd,
                        outcome.optimized_annual_total_cost_usd,
                    ),
                    create_cost_row(
                        "Energy Cost",
                        outcome.baseline_annual_energy_cost_usd,
                        outcome.optimized_annual_energy_cost_usd,
                    ),
                ],
                style={"display": "grid", "gap": "10px"},
            ),
            html.Div(
                f"Quality Risk: {outcome.quality_risk.upper()}",
                style={
                    "marginTop": "20px",
                    "padding": "10px",
                    "backgroundColor": risk_bg,
                    "borderRadius": "4px",
                    "textAlign": "center",
                    "fontWeight": "bold",
                    "color": risk_color,
                },
            ),
        ]
    )


def create_metric_card(title: str, value: str, color: str) -> html.Div:
    """Create a metric card."""
    return html.Div(
        [
            html.Div(
                title,
                style={
                    "fontSize": "12px",
                    "color": "#7f8c8d",
                    "textTransform": "uppercase",
                    "marginBottom": "5px",
                },
            ),
            html.Div(
                value,
                style={
                    "fontSize": "24px",
                    "fontWeight": "bold",
                    "color": color,
                },
            ),
        ],
        style={
            "backgroundColor": "#f8f9fa",
            "padding": "15px",
            "borderRadius": "6px",
            "textAlign": "center",
            "borderLeft": f"4px solid {color}",
        },
    )


def create_metric_row(
    label: str, baseline: str, optimized: str, change: str
) -> html.Div:
    """Create a metric comparison row."""
    return html.Div(
        [
            html.Div(label, style={"fontWeight": "bold", "color": "#34495e", "width": "100px"}),
            html.Div(baseline, style={"color": "#7f8c8d", "width": "100px"}),
            html.Div("→", style={"color": "#bdc3c7", "margin": "0 10px"}),
            html.Div(
                optimized,
                style={"fontWeight": "bold", "color": "#2ecc71", "width": "100px"},
            ),
            html.Div(
                change,
                style={
                    "marginLeft": "auto",
                    "padding": "2px 8px",
                    "backgroundColor": "#d4edda",
                    "borderRadius": "4px",
                    "fontSize": "12px",
                    "color": "#155724",
                },
            ),
        ],
        style={
            "display": "flex",
            "alignItems": "center",
            "padding": "8px 12px",
            "backgroundColor": "#f8f9fa",
            "borderRadius": "4px",
        },
    )


def create_cost_row(label: str, baseline: float, optimized: float) -> html.Div:
    """Create a cost comparison row."""
    savings = baseline - optimized
    return html.Div(
        [
            html.Div(label, style={"fontWeight": "bold", "color": "#34495e", "width": "120px"}),
            html.Div(f"${baseline:,.0f}", style={"color": "#7f8c8d", "width": "100px"}),
            html.Div("→", style={"color": "#bdc3c7", "margin": "0 10px"}),
            html.Div(
                f"${optimized:,.0f}",
                style={"fontWeight": "bold", "color": "#2ecc71", "width": "100px"},
            ),
            html.Div(
                f"Save ${savings:,.0f}",
                style={
                    "marginLeft": "auto",
                    "padding": "2px 8px",
                    "backgroundColor": "#d4edda",
                    "borderRadius": "4px",
                    "fontSize": "12px",
                    "color": "#155724",
                },
            ),
        ],
        style={
            "display": "flex",
            "alignItems": "center",
            "padding": "8px 12px",
            "backgroundColor": "#f8f9fa",
            "borderRadius": "4px",
        },
    )


def create_scenario_summary(scenario: DeploymentScenario) -> html.Div:
    """Create a scenario summary display."""
    return html.Div(
        [
            html.Div(f"Model: {scenario.parameters_b}B parameters"),
            html.Div(f"Memory: {scenario.memory_gb} GB"),
            html.Div(f"Throughput: {scenario.throughput_toks_per_sec} tok/s"),
            html.Div(f"Requests: {scenario.requests_per_day:,}/day"),
        ]
    )


def create_strategy_summary(strategy: OptimizationStrategy) -> html.Div:
    """Create a strategy summary display."""
    return html.Div(
        [
            html.Div(f"Memory reduction: {strategy.memory_reduction_fraction * 100:.0f}%"),
            html.Div(f"Throughput gain: {strategy.throughput_improvement_fraction * 100:.0f}%"),
            html.Div(f"Power reduction: {strategy.power_reduction_fraction * 100:.0f}%"),
            html.Div(f"Quality risk: {strategy.quality_risk}"),
        ]
    )


def create_comparison_chart(outcome: OptimizationOutcome) -> dict[str, Any]:
    """Create a before/after comparison chart."""
    if go is None:
        return {}

    categories = ["Memory\n(GB)", "Throughput\n(tok/s)", "Power\n(W)", "Energy/Req\n(Wh)"]
    baseline_values = [
        outcome.baseline_memory_gb,
        outcome.baseline_throughput_toks_per_sec,
        outcome.baseline_power_watts,
        outcome.baseline_energy_wh_per_request,
    ]
    optimized_values = [
        outcome.optimized_memory_gb,
        outcome.optimized_throughput_toks_per_sec,
        outcome.optimized_power_watts,
        outcome.optimized_energy_wh_per_request,
    ]

    # Normalize throughput to show improvement direction consistently
    baseline_normalized = [
        baseline_values[0] / max(baseline_values[0], 0.1) * 100,
        baseline_values[1] / max(baseline_values[1], 0.1) * 100,
        baseline_values[2] / max(baseline_values[2], 0.1) * 100,
        baseline_values[3] / max(baseline_values[3], 0.1) * 100,
    ]
    optimized_normalized = [
        optimized_values[0] / max(baseline_values[0], 0.1) * 100,
        optimized_values[1] / max(baseline_values[1], 0.1) * 100,
        optimized_values[2] / max(baseline_values[2], 0.1) * 100,
        optimized_values[3] / max(baseline_values[3], 0.1) * 100,
    ]

    # For throughput, higher is better, so invert for consistent "lower is better" visualization
    baseline_normalized[1] = 100
    optimized_normalized[1] = (optimized_values[1] / baseline_values[1]) * 100

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            name="Baseline",
            x=categories,
            y=baseline_normalized,
            marker_color="#95a5a6",
            text=[f"{v:.1f}" for v in baseline_values],
            textposition="auto",
        )
    )

    fig.add_trace(
        go.Bar(
            name="Optimized",
            x=categories,
            y=optimized_normalized,
            marker_color="#2ecc71",
            text=[f"{v:.1f}" for v in optimized_values],
            textposition="auto",
        )
    )

    fig.update_layout(
        barmode="group",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=40, b=40, l=40, r=20),
        height=300,
        yaxis=dict(title="Relative Value (%)"),
    )

    return fig  # type: ignore[no-any-return]  # type: ignore[no-any-return]


def create_savings_breakdown_chart(outcome: OptimizationOutcome) -> dict[str, Any]:
    """Create a savings breakdown pie chart."""
    if go is None:
        return {}

    energy_savings = (
        outcome.baseline_annual_energy_cost_usd - outcome.optimized_annual_energy_cost_usd
    )
    hardware_savings = (
        outcome.baseline_annual_total_cost_usd
        - outcome.baseline_annual_energy_cost_usd
        - (outcome.optimized_annual_total_cost_usd - outcome.optimized_annual_energy_cost_usd)
    )

    fig = go.Figure(
        data=[
            go.Pie(
                labels=["Energy Savings", "Hardware Savings"],
                values=[max(0, energy_savings), max(0, hardware_savings)],
                marker_colors=["#3498db", "#2ecc71"],
                textinfo="label+percent+value",
                texttemplate="%{label}<br>$%{value:,.0f}<br>(%{percent})",
                hole=0.4,
            )
        ]
    )

    fig.update_layout(
        showlegend=False,
        margin=dict(t=20, b=20, l=20, r=20),
        height=300,
        annotations=[
            dict(
                text=f"${outcome.annual_total_savings_usd:,.0f}",
                x=0.5,
                y=0.5,
                font_size=20,
                showarrow=False,
            )
        ],
    )

    return fig  # type: ignore[no-any-return]
