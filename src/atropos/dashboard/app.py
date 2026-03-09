"""Main Dash application for Atropos web dashboard."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import dash
from dash import html
from dash.dependencies import Input, Output, State

from ..calculations import combine_strategies, estimate_outcome
from ..carbon_presets import get_carbon_intensity
from ..core.uncertainty import ParameterDistribution, run_monte_carlo
from ..models import DeploymentScenario, OptimizationStrategy
from ..presets import QUANTIZATION_BONUS, SCENARIOS, STRATEGIES
from . import components, layouts

if TYPE_CHECKING:
    pass


def create_app() -> dash.Dash:
    """Create and configure the Dash application."""
    app = dash.Dash(
        __name__,
        title="Atropos - ROI Calculator",
        suppress_callback_exceptions=True,
    )

    app.layout = layouts.create_main_layout()

    register_callbacks(app)

    return app


def register_callbacks(app: dash.Dash) -> None:
    """Register all Dash callbacks."""

    @app.callback(
        Output("results-container", "children"),
        Output("comparison-chart", "figure"),
        Output("savings-breakdown-chart", "figure"),
        Output("monte-carlo-chart", "figure"),
        Input("calculate-button", "n_clicks"),
        State("preset-selector", "value"),
        State("strategy-selector", "value"),
        State("with-quantization", "value"),
        State("region-selector", "value"),
        State("custom-params", "data"),
        prevent_initial_call=True,
    )
    def update_results(
        n_clicks: int,
        preset_name: str,
        strategy_name: str,
        with_quantization: list[str],
        region: str,
        custom_params: dict[str, Any] | None,
    ) -> tuple[Any, dict[str, Any], dict[str, Any], dict[str, Any]]:
        """Update all results when calculate button is clicked."""
        if not n_clicks:
            return (
                html.Div("Click Calculate to see results"),
                {},
                {},
                {},
            )

        try:
            # Get scenario
            if preset_name == "custom" and custom_params:
                scenario = DeploymentScenario(**custom_params)
            elif preset_name in SCENARIOS:
                scenario = SCENARIOS[preset_name]
            else:
                scenario = SCENARIOS["medium-coder"]

            # Get strategy
            strategy = STRATEGIES.get(strategy_name, STRATEGIES["structured_pruning"])

            # Apply quantization if selected
            if with_quantization and "quantization" in with_quantization:
                strategy = combine_strategies(strategy, QUANTIZATION_BONUS)

            # Get carbon intensity for region
            grid_co2e = get_carbon_intensity(region) if region else 0.35

            # Calculate outcome
            outcome = estimate_outcome(
                scenario, strategy, grid_co2e_kg_per_kwh=grid_co2e
            )

            # Create results display
            results = components.create_results_display(outcome)

            # Create charts
            comparison_fig = components.create_comparison_chart(outcome)
            breakdown_fig = components.create_savings_breakdown_chart(outcome)

            # Run Monte Carlo for distribution
            mc_fig = create_monte_carlo_distribution(scenario, strategy, grid_co2e)

            return results, comparison_fig, breakdown_fig, mc_fig

        except Exception as e:
            error_msg = html.Div(
                f"Error: {e}",
                style={"color": "red", "padding": "20px"},
            )
            return error_msg, {}, {}, {}

    @app.callback(
        Output("custom-params", "data"),
        Input("param-parameters-b", "value"),
        Input("param-memory-gb", "value"),
        Input("param-throughput", "value"),
        Input("param-power", "value"),
        Input("param-requests", "value"),
        Input("param-tokens", "value"),
        Input("param-electricity-cost", "value"),
        Input("param-hardware-cost", "value"),
        Input("param-project-cost", "value"),
        State("custom-name", "value"),
    )
    def update_custom_params(
        parameters_b: float,
        memory_gb: float,
        throughput: float,
        power: float,
        requests: int,
        tokens: int,
        electricity_cost: float,
        hardware_cost: float,
        project_cost: float,
        name: str,
    ) -> dict[str, Any]:
        """Update custom parameters store."""
        return {
            "name": name or "custom-scenario",
            "parameters_b": parameters_b,
            "memory_gb": memory_gb,
            "throughput_toks_per_sec": throughput,
            "power_watts": power,
            "requests_per_day": requests,
            "tokens_per_request": tokens,
            "electricity_cost_per_kwh": electricity_cost,
            "annual_hardware_cost_usd": hardware_cost,
            "one_time_project_cost_usd": project_cost,
        }

    @app.callback(
        Output("preset-params-display", "children"),
        Input("preset-selector", "value"),
    )
    def update_preset_display(preset_name: str) -> Any:
        """Display preset scenario parameters."""
        if preset_name == "custom":
            return html.Div("Enter custom parameters below")

        if preset_name not in SCENARIOS:
            return html.Div("Select a preset")

        scenario = SCENARIOS[preset_name]
        return components.create_scenario_summary(scenario)

    @app.callback(
        Output("strategy-details", "children"),
        Input("strategy-selector", "value"),
    )
    def update_strategy_details(strategy_name: str) -> Any:
        """Display strategy details."""
        if strategy_name not in STRATEGIES:
            return html.Div("Select a strategy")

        strategy = STRATEGIES[strategy_name]
        return components.create_strategy_summary(strategy)

    @app.callback(
        Output("download-scenario", "data"),
        Input("export-button", "n_clicks"),
        State("preset-selector", "value"),
        State("custom-params", "data"),
        prevent_initial_call=True,
    )
    def export_scenario(
        n_clicks: int, preset_name: str, custom_params: dict[str, Any] | None
    ) -> dict[str, str]:
        """Export scenario as YAML."""
        if not n_clicks:
            raise dash.exceptions.PreventUpdate

        if preset_name == "custom" and custom_params:
            data = custom_params
        elif preset_name in SCENARIOS:
            scenario = SCENARIOS[preset_name]
            data = {
                "name": scenario.name,
                "parameters_b": scenario.parameters_b,
                "memory_gb": scenario.memory_gb,
                "throughput_toks_per_sec": scenario.throughput_toks_per_sec,
                "power_watts": scenario.power_watts,
                "requests_per_day": scenario.requests_per_day,
                "tokens_per_request": scenario.tokens_per_request,
                "electricity_cost_per_kwh": scenario.electricity_cost_per_kwh,
                "annual_hardware_cost_usd": scenario.annual_hardware_cost_usd,
                "one_time_project_cost_usd": scenario.one_time_project_cost_usd,
            }
        else:
            data = {}

        return {
            "content": json.dumps(data, indent=2),
            "filename": f"{data.get('name', 'scenario')}.json",
        }


def create_monte_carlo_distribution(
    scenario: DeploymentScenario,
    strategy: OptimizationStrategy,
    grid_co2e: float,
    num_simulations: int = 500,
) -> dict[str, Any]:
    """Create Monte Carlo simulation distribution chart."""
    try:
        import plotly.graph_objects as go

        distributions = [
            ParameterDistribution(
                param_name="memory_reduction_fraction",
                distribution="normal",
                std_dev=0.05,
            ),
            ParameterDistribution(
                param_name="throughput_improvement_fraction",
                distribution="normal",
                std_dev=0.1,
            ),
        ]

        def estimator(scen: DeploymentScenario, strat: OptimizationStrategy) -> Any:
            return estimate_outcome(
                scen, strat, grid_co2e_kg_per_kwh=grid_co2e
            )

        result = run_monte_carlo(
            scenario, strategy, distributions, estimator, num_simulations, seed=42
        )

        # Create histogram of savings
        savings_values = [o.annual_total_savings_usd for o in result.all_outcomes]

        fig = go.Figure(
            data=[
                go.Histogram(
                    x=savings_values,
                    nbinsx=30,
                    name="Annual Savings Distribution",
                    marker_color="#2ecc71",
                    opacity=0.75,
                )
            ]
        )

        fig.update_layout(
            title="Monte Carlo Simulation: Annual Savings Distribution",
            xaxis_title="Annual Savings (USD)",
            yaxis_title="Frequency",
            showlegend=False,
            margin={"t": 40, "b": 40, "l": 40, "r": 20},
            height=300,
        )

        # Add vertical lines for mean and percentiles
        fig.add_vline(
            x=result.savings_mean,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean: ${result.savings_mean:,.0f}",
        )

        return fig  # type: ignore[no-any-return]

    except Exception:
        return {}


def run_dashboard(host: str = "127.0.0.1", port: int = 8050, debug: bool = False) -> None:
    """Run the dashboard server."""
    app = create_app()
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    run_dashboard(debug=True)
