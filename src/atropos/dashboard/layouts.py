"""Layout definitions for the Atropos dashboard."""

from __future__ import annotations

from typing import TYPE_CHECKING

from dash import dcc, html

from ..carbon_presets import CARBON_PRESETS, list_regions
from ..presets import SCENARIOS, STRATEGIES

if TYPE_CHECKING:
    pass


def create_main_layout() -> html.Div:
    """Create the main dashboard layout."""
    return html.Div(
        [
            create_header(),
            create_content(),
            create_footer(),
            # Store for custom parameters
            dcc.Store(id="custom-params", data={}),
            # Download component
            dcc.Download(id="download-scenario"),
        ],
        style={
            "fontFamily": "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
            "maxWidth": "1400px",
            "margin": "0 auto",
            "padding": "20px",
            "backgroundColor": "#f5f5f5",
            "minHeight": "100vh",
        },
    )


def create_header() -> html.Div:
    """Create the dashboard header."""
    return html.Div(
        [
            html.H1(
                "Atropos",
                style={
                    "margin": "0",
                    "color": "#2c3e50",
                    "fontSize": "32px",
                },
            ),
            html.P(
                "ROI Calculator for LLM Pruning & Optimization",
                style={
                    "margin": "5px 0 0 0",
                    "color": "#7f8c8d",
                    "fontSize": "16px",
                },
            ),
        ],
        style={
            "backgroundColor": "white",
            "padding": "20px 30px",
            "borderRadius": "8px",
            "marginBottom": "20px",
            "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
        },
    )


def create_content() -> html.Div:
    """Create the main content area."""
    return html.Div(
        [
            create_input_panel(),
            create_results_panel(),
        ],
        style={
            "display": "grid",
            "gridTemplateColumns": "350px 1fr",
            "gap": "20px",
        },
    )


def create_input_panel() -> html.Div:
    """Create the input configuration panel."""
    return html.Div(
        [
            html.H2(
                "Configuration",
                style={"marginTop": "0", "fontSize": "20px", "color": "#2c3e50"},
            ),
            create_scenario_section(),
            create_strategy_section(),
            create_region_section(),
            create_action_buttons(),
        ],
        style={
            "backgroundColor": "white",
            "padding": "25px",
            "borderRadius": "8px",
            "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
            "height": "fit-content",
        },
    )


def create_scenario_section() -> html.Div:
    """Create the scenario selection section."""
    preset_options = [{"label": "Custom", "value": "custom"}] + [
        {"label": name.replace("-", " ").title(), "value": name}
        for name in sorted(SCENARIOS.keys())
    ]

    return html.Div(
        [
            html.H3(
                "Scenario",
                style={"fontSize": "16px", "color": "#34495e", "marginBottom": "10px"},
            ),
            html.Label("Preset:"),
            dcc.Dropdown(
                id="preset-selector",
                options=preset_options,
                value="medium-coder",
                clearable=False,
                style={"marginBottom": "15px"},
            ),
            html.Div(
                id="preset-params-display",
                style={
                    "marginBottom": "15px",
                    "padding": "10px",
                    "backgroundColor": "#ecf0f1",
                    "borderRadius": "4px",
                    "fontSize": "12px",
                },
            ),
            create_custom_params_section(),
        ],
        style={"marginBottom": "25px"},
    )


def create_custom_params_section() -> html.Div:
    """Create custom parameters input section."""
    return html.Div(
        [
            html.Label("Scenario Name:"),
            dcc.Input(
                id="custom-name",
                type="text",
                value="my-scenario",
                style={"width": "100%", "marginBottom": "10px"},
            ),
            html.Label("Model Size (B parameters):"),
            dcc.Slider(
                id="param-parameters-b",
                min=1,
                max=100,
                step=1,
                value=7,
                marks={1: "1B", 50: "50B", 100: "100B"},
            ),
            html.Label("Memory (GB):"),
            dcc.Slider(
                id="param-memory-gb",
                min=4,
                max=80,
                step=2,
                value=14,
                marks={4: "4", 40: "40", 80: "80"},
            ),
            html.Label("Throughput (tok/s):"),
            dcc.Slider(
                id="param-throughput",
                min=10,
                max=200,
                step=5,
                value=40,
                marks={10: "10", 100: "100", 200: "200"},
            ),
            html.Label("Power (W):"),
            dcc.Slider(
                id="param-power",
                min=100,
                max=1000,
                step=50,
                value=320,
                marks={100: "100", 500: "500", 1000: "1000"},
            ),
            html.Label("Requests/Day:"),
            dcc.Slider(
                id="param-requests",
                min=1000,
                max=500000,
                step=1000,
                value=50000,
                marks={1000: "1K", 250000: "250K", 500000: "500K"},
            ),
            html.Label("Tokens/Request:"),
            dcc.Slider(
                id="param-tokens",
                min=100,
                max=8000,
                step=100,
                value=1200,
                marks={100: "100", 4000: "4K", 8000: "8K"},
            ),
            html.Label("Electricity Cost ($/kWh):"),
            dcc.Slider(
                id="param-electricity-cost",
                min=0.05,
                max=0.50,
                step=0.01,
                value=0.15,
                marks={0.05: "$0.05", 0.25: "$0.25", 0.50: "$0.50"},
            ),
            html.Label("Annual Hardware Cost ($):"),
            dcc.Slider(
                id="param-hardware-cost",
                min=5000,
                max=100000,
                step=5000,
                value=24000,
                marks={5000: "$5K", 50000: "$50K", 100000: "$100K"},
            ),
            html.Label("Project Cost ($):"),
            dcc.Slider(
                id="param-project-cost",
                min=5000,
                max=100000,
                step=5000,
                value=27000,
                marks={5000: "$5K", 50000: "$50K", 100000: "$100K"},
            ),
        ],
        id="custom-params-section",
        style={"marginTop": "15px"},
    )


def create_strategy_section() -> html.Div:
    """Create the strategy selection section."""
    strategy_options = [
        {"label": name.replace("_", " ").title(), "value": name}
        for name in sorted(STRATEGIES.keys())
    ]

    return html.Div(
        [
            html.H3(
                "Strategy",
                style={"fontSize": "16px", "color": "#34495e", "marginBottom": "10px"},
            ),
            html.Label("Optimization Strategy:"),
            dcc.Dropdown(
                id="strategy-selector",
                options=strategy_options,
                value="structured_pruning",
                clearable=False,
                style={"marginBottom": "10px"},
            ),
            html.Div(
                id="strategy-details",
                style={
                    "marginBottom": "15px",
                    "padding": "10px",
                    "backgroundColor": "#ecf0f1",
                    "borderRadius": "4px",
                    "fontSize": "12px",
                },
            ),
            dcc.Checklist(
                id="with-quantization",
                options=[{"label": " Include Quantization", "value": "quantization"}],
                value=[],
                style={"marginBottom": "15px"},
            ),
        ],
        style={"marginBottom": "25px"},
    )


def create_region_section() -> html.Div:
    """Create the region selection section."""
    regions = list_regions()
    region_options = [
        {"label": f"{code} - {CARBON_PRESETS[code].region_name}", "value": code}
        for code in regions
    ]

    return html.Div(
        [
            html.H3(
                "Region",
                style={"fontSize": "16px", "color": "#34495e", "marginBottom": "10px"},
            ),
            html.Label("Grid Carbon Intensity:"),
            dcc.Dropdown(
                id="region-selector",
                options=region_options,
                value="US",
                clearable=False,
                style={"marginBottom": "15px"},
            ),
        ],
        style={"marginBottom": "25px"},
    )


def create_action_buttons() -> html.Div:
    """Create action buttons."""
    return html.Div(
        [
            html.Button(
                "Calculate ROI",
                id="calculate-button",
                style={
                    "width": "100%",
                    "padding": "12px",
                    "backgroundColor": "#3498db",
                    "color": "white",
                    "border": "none",
                    "borderRadius": "4px",
                    "cursor": "pointer",
                    "fontSize": "16px",
                    "fontWeight": "bold",
                    "marginBottom": "10px",
                },
            ),
            html.Button(
                "Export Scenario",
                id="export-button",
                style={
                    "width": "100%",
                    "padding": "10px",
                    "backgroundColor": "#95a5a6",
                    "color": "white",
                    "border": "none",
                    "borderRadius": "4px",
                    "cursor": "pointer",
                    "fontSize": "14px",
                },
            ),
        ]
    )


def create_results_panel() -> html.Div:
    """Create the results display panel."""
    return html.Div(
        [
            html.H2(
                "Results",
                style={"marginTop": "0", "fontSize": "20px", "color": "#2c3e50"},
            ),
            html.Div(
                id="results-container",
                children=html.Div(
                    "Click 'Calculate ROI' to see results",
                    style={
                        "textAlign": "center",
                        "padding": "60px 20px",
                        "color": "#95a5a6",
                        "fontSize": "18px",
                    },
                ),
                style={
                    "backgroundColor": "white",
                    "padding": "25px",
                    "borderRadius": "8px",
                    "marginBottom": "20px",
                    "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
                },
            ),
            html.Div(
                [
                    html.H3(
                        "Comparison",
                        style={"fontSize": "16px", "color": "#34495e", "marginBottom": "10px"},
                    ),
                    dcc.Graph(
                        id="comparison-chart",
                        config={"displayModeBar": False},
                        style={"height": "300px"},
                    ),
                ],
                style={
                    "backgroundColor": "white",
                    "padding": "20px",
                    "borderRadius": "8px",
                    "marginBottom": "20px",
                    "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
                },
            ),
            html.Div(
                [
                    html.H3(
                        "Savings Breakdown",
                        style={"fontSize": "16px", "color": "#34495e", "marginBottom": "10px"},
                    ),
                    dcc.Graph(
                        id="savings-breakdown-chart",
                        config={"displayModeBar": False},
                        style={"height": "300px"},
                    ),
                ],
                style={
                    "backgroundColor": "white",
                    "padding": "20px",
                    "borderRadius": "8px",
                    "marginBottom": "20px",
                    "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
                },
            ),
            html.Div(
                [
                    html.H3(
                        "Uncertainty Analysis",
                        style={"fontSize": "16px", "color": "#34495e", "marginBottom": "10px"},
                    ),
                    dcc.Graph(
                        id="monte-carlo-chart",
                        config={"displayModeBar": False},
                        style={"height": "300px"},
                    ),
                ],
                style={
                    "backgroundColor": "white",
                    "padding": "20px",
                    "borderRadius": "8px",
                    "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
                },
            ),
        ]
    )


def create_footer() -> html.Div:
    """Create the dashboard footer."""
    return html.Div(
        [
            html.P(
                "Atropos - Named after the Fate who cuts the thread",
                style={"margin": "0", "fontSize": "12px", "color": "#95a5a6"},
            ),
        ],
        style={
            "textAlign": "center",
            "padding": "20px",
            "marginTop": "20px",
        },
    )
