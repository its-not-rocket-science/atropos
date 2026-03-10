"""Command-line interface for Atropos."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .batch import batch_process
from .calculations import combine_strategies, estimate_outcome
from .calibration import calibrate_scenario, generate_calibration_report
from .carbon_presets import CARBON_PRESETS, get_carbon_intensity, list_regions
from .config import AtroposConfig
from .core.calculator import ROICalculator
from .core.uncertainty import ParameterDistribution
from .integrations import TRACKERS, get_tracker, run_to_scenario
from .io import csv_to_markdown, export_to_csv, load_scenario, render_report
from .models import DeploymentScenario
from .pipeline import PipelineConfig, run_pipeline
from .presets import QUANTIZATION_BONUS, SCENARIOS, STRATEGIES
from .reporting import generate_comparison_json, generate_comparison_table
from .telemetry import (
    PARSERS,
    get_parser,
    telemetry_to_scenario,
    validate_telemetry,
)
from .validation import run_validation


def build_parser() -> argparse.ArgumentParser:
    """Build and configure the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        prog="atropos",
        description=(
            "Estimate the ROI of pruning and related optimizations for coding LLM deployments."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list-presets", help="List built-in deployment and strategy presets.")

    preset_parser = subparsers.add_parser("preset", help="Run a built-in scenario preset.")
    preset_parser.add_argument("name", choices=sorted(SCENARIOS.keys()))
    _add_strategy_args(preset_parser)
    preset_parser.add_argument(
        "--region", help="Region for carbon intensity (ISO code or cloud region like us-east-1)"
    )

    scenario_parser = subparsers.add_parser("scenario", help="Run a scenario from a YAML file.")
    scenario_parser.add_argument("path", type=Path)
    _add_strategy_args(scenario_parser)
    scenario_parser.add_argument(
        "--region", help="Region for carbon intensity (ISO code or cloud region like us-east-1)"
    )

    compare_parser = subparsers.add_parser("compare", help="Compare multiple strategies.")
    compare_parser.add_argument("scenario", help="Scenario name or path to YAML")
    compare_parser.add_argument(
        "--strategies", nargs="+", required=True, choices=sorted(STRATEGIES.keys())
    )
    compare_parser.add_argument("--with-quantization", action="store_true")
    compare_parser.add_argument("--format", choices=["text", "markdown", "json"], default="text")
    compare_parser.add_argument(
        "--sort-by", choices=["savings", "breakeven", "risk"], default="savings"
    )
    compare_parser.add_argument("--ascending", action="store_true", help="Sort in ascending order")
    compare_parser.add_argument("--output", "-o", type=Path)
    compare_parser.add_argument(
        "--region", help="Region for carbon intensity (ISO code or cloud region like us-east-1)"
    )

    batch_parser = subparsers.add_parser("batch", help="Process multiple scenario files.")
    batch_parser.add_argument("directory", type=Path)
    batch_parser.add_argument(
        "--strategies", nargs="+", required=True, choices=sorted(STRATEGIES.keys())
    )
    batch_parser.add_argument("--with-quantization", action="store_true")
    batch_parser.add_argument("--output", "-o", type=Path, required=True)

    sensitivity_parser = subparsers.add_parser("sensitivity", help="Run sensitivity analysis.")
    sensitivity_parser.add_argument("scenario", help="Scenario name or path to YAML")
    sensitivity_parser.add_argument("--strategy", required=True, choices=sorted(STRATEGIES.keys()))
    sensitivity_parser.add_argument(
        "--param",
        required=True,
        choices=[
            "memory_reduction_fraction",
            "throughput_improvement_fraction",
            "power_reduction_fraction",
        ],
    )
    sensitivity_parser.add_argument("--variations", type=int, default=5)
    sensitivity_parser.add_argument("--step", type=float, default=0.1)
    sensitivity_parser.add_argument("--output", "-o", type=Path)
    sensitivity_parser.add_argument("--format", choices=["text", "csv", "json"], default="text")

    mc_parser = subparsers.add_parser("monte-carlo", help="Run Monte Carlo uncertainty analysis.")
    mc_parser.add_argument("scenario", help="Scenario name or path to YAML")
    mc_parser.add_argument("--strategy", required=True, choices=sorted(STRATEGIES.keys()))
    mc_parser.add_argument(
        "--params",
        nargs="+",
        default=["memory_reduction_fraction", "throughput_improvement_fraction"],
        help=(
            "Parameters to vary "
            "(default: memory_reduction_fraction throughput_improvement_fraction)"
        ),
    )
    mc_parser.add_argument(
        "--distribution",
        choices=["normal", "uniform", "triangular"],
        default="normal",
        help="Distribution type for parameter variation",
    )
    mc_parser.add_argument(
        "--std-dev",
        type=float,
        default=0.1,
        help="Standard deviation for normal distribution (as fraction of mean)",
    )
    mc_parser.add_argument(
        "--range",
        type=float,
        default=0.2,
        dest="range_fraction",
        help="Range for uniform/triangular distribution (as +/- fraction of mean)",
    )
    mc_parser.add_argument(
        "--simulations", type=int, default=1000, help="Number of simulations to run"
    )
    mc_parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    mc_parser.add_argument("--output", "-o", type=Path, help="Output file path")
    mc_parser.add_argument(
        "--format", choices=["text", "json", "csv"], default="text", help="Output format"
    )

    csv_md_parser = subparsers.add_parser(
        "csv-to-markdown", help="Convert CSV results to markdown report."
    )
    csv_md_parser.add_argument("input", type=Path, help="Path to CSV file")
    csv_md_parser.add_argument("--output", "-o", type=Path, help="Output markdown file path")

    telemetry_parser = subparsers.add_parser(
        "import-telemetry", help="Import benchmark telemetry to create a scenario."
    )
    telemetry_parser.add_argument("input", type=Path, help="Path to telemetry file")
    telemetry_parser.add_argument(
        "--format",
        choices=list(PARSERS.keys()),
        required=True,
        help="Telemetry format",
    )
    telemetry_parser.add_argument("--name", required=True, help="Scenario name")
    telemetry_parser.add_argument(
        "--mapping",
        type=str,
        help='Field mapping as JSON (e.g., \'{"memory_gb": "gpu_memory"}\')',
    )
    telemetry_parser.add_argument(
        "--electricity-cost", type=float, default=0.15, help="Electricity cost per kWh"
    )
    telemetry_parser.add_argument(
        "--hardware-cost", type=float, default=24000.0, help="Annual hardware cost in USD"
    )
    telemetry_parser.add_argument(
        "--project-cost", type=float, default=27000.0, help="One-time project cost in USD"
    )
    telemetry_parser.add_argument("--requests-per-day", type=int, help="Expected requests per day")
    telemetry_parser.add_argument("--output", "-o", type=Path, help="Output YAML file path")
    telemetry_parser.add_argument(
        "--preview", action="store_true", help="Preview scenario params without saving"
    )

    exp_parser = subparsers.add_parser(
        "import-experiment", help="Import scenario from experiment tracker (wandb/mlflow)."
    )
    exp_parser.add_argument(
        "--tracker",
        choices=list(TRACKERS.keys()),
        required=True,
        help="Experiment tracker type",
    )
    exp_parser.add_argument("--run-id", help="Specific run ID to import")
    exp_parser.add_argument("--experiment", help="Experiment/project name")
    exp_parser.add_argument("--entity", help="Entity/team name (wandb)")
    exp_parser.add_argument("--project", help="Project name (wandb)")
    exp_parser.add_argument("--limit", type=int, default=1, help="Number of runs to import")
    exp_parser.add_argument("--api-key", help="API key for authentication")
    exp_parser.add_argument("--host", help="Tracker host URL")
    exp_parser.add_argument("--name", help="Scenario name (or use run ID)")
    exp_parser.add_argument(
        "--electricity-cost", type=float, default=0.15, help="Electricity cost per kWh"
    )
    exp_parser.add_argument(
        "--hardware-cost", type=float, default=24000.0, help="Annual hardware cost in USD"
    )
    exp_parser.add_argument(
        "--project-cost", type=float, default=27000.0, help="One-time project cost in USD"
    )
    exp_parser.add_argument("--requests-per-day", type=int, help="Expected requests per day")
    exp_parser.add_argument("--output", "-o", type=Path, help="Output YAML file path")
    exp_parser.add_argument(
        "--preview", action="store_true", help="Preview scenario params without saving"
    )

    dashboard_parser = subparsers.add_parser(
        "dashboard", help="Launch the interactive web dashboard."
    )
    dashboard_parser.add_argument(
        "--host", default="127.0.0.1", help="Host to bind the dashboard server"
    )
    dashboard_parser.add_argument(
        "--port", type=int, default=8050, help="Port to run the dashboard server"
    )
    dashboard_parser.add_argument(
        "--debug", action="store_true", help="Run in debug mode"
    )

    carbon_parser = subparsers.add_parser(
        "list-carbon-presets", help="List available carbon intensity presets."
    )
    carbon_parser.add_argument(
        "--region", help="Show details for specific region (ISO code or cloud region)"
    )
    carbon_parser.add_argument(
        "--format", choices=["text", "json"], default="text", help="Output format"
    )

    calibrate_parser = subparsers.add_parser(
        "calibrate",
        help="Calibrate scenario parameters against real telemetry data.",
    )
    calibrate_parser.add_argument(
        "scenario", help="Scenario name (preset) or path to YAML file"
    )
    calibrate_parser.add_argument(
        "telemetry", type=Path, help="Path to telemetry file (JSON, CSV, or log)"
    )
    calibrate_parser.add_argument(
        "--parser", choices=list(PARSERS.keys()), help="Telemetry parser type"
    )
    calibrate_parser.add_argument(
        "--tolerance",
        type=float,
        default=10.0,
        help="Acceptable variance tolerance percentage (default: 10)",
    )
    calibrate_parser.add_argument(
        "--format",
        choices=["markdown", "json"],
        default="markdown",
        help="Output format",
    )
    calibrate_parser.add_argument(
        "--output", "-o", type=Path, help="Output file path (default: stdout)"
    )

    # Pipeline command
    pipeline_parser = subparsers.add_parser(
        "pipeline",
        help="Run the Atropos optimization pipeline.",
    )
    pipeline_parser.add_argument(
        "scenario", help="Scenario name (preset) or path to YAML file"
    )
    pipeline_parser.add_argument(
        "--config", "-c", type=Path, required=True,
        help="Pipeline configuration YAML file"
    )
    pipeline_parser.add_argument(
        "--strategy", choices=sorted(STRATEGIES.keys()),
        default="structured_pruning",
        help="Optimization strategy to use"
    )
    pipeline_parser.add_argument(
        "--region", help="Region for carbon intensity (ISO code or cloud region)"
    )
    pipeline_parser.add_argument(
        "--dry-run", action="store_true",
        help="Simulate pipeline without actual execution"
    )
    pipeline_parser.add_argument(
        "--output", "-o", type=Path,
        help="Output JSON file for pipeline results"
    )

    # Pipeline config validation command
    pipeline_config_parser = subparsers.add_parser(
        "validate-pipeline-config",
        help="Validate a pipeline configuration file.",
    )
    pipeline_config_parser.add_argument(
        "config", type=Path, help="Pipeline configuration YAML file"
    )

    # Validation command (test against real models)
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate Atropos projections against real neural networks.",
    )
    validate_parser.add_argument(
        "scenario", help="Scenario name (preset) or path to YAML file"
    )
    validate_parser.add_argument(
        "--strategy", choices=sorted(STRATEGIES.keys()),
        default="structured_pruning",
        help="Optimization strategy to test"
    )
    validate_parser.add_argument(
        "--model", help="HuggingFace model name to test (auto-selected if not provided)"
    )
    validate_parser.add_argument(
        "--device", choices=["cpu", "cuda"], default="cpu",
        help="Device to run validation on"
    )
    validate_parser.add_argument(
        "--pruning-method", default="magnitude",
        choices=["magnitude", "random", "structured"],
        help="Pruning method to apply"
    )
    validate_parser.add_argument(
        "--format", choices=["markdown", "json"], default="markdown",
        help="Output format"
    )
    validate_parser.add_argument(
        "--output", "-o", type=Path, help="Output file path (default: stdout)"
    )

    return parser


def _add_strategy_args(parser: argparse.ArgumentParser) -> None:
    """Add common strategy-related arguments to a parser."""
    parser.add_argument(
        "--strategy", choices=sorted(STRATEGIES.keys()), default="structured_pruning"
    )
    parser.add_argument("--with-quantization", action="store_true")
    parser.add_argument("--report", choices=["text", "json", "markdown", "html"], default="text")


def _load_scenario_input(scenario_input: str) -> tuple[str, DeploymentScenario]:
    """Load scenario from preset name or YAML file path."""
    path = Path(scenario_input)
    if path.exists():
        scenario = load_scenario(path)
        return scenario.name, scenario
    if scenario_input not in SCENARIOS:
        raise KeyError(f"Unknown scenario '{scenario_input}'")
    return scenario_input, SCENARIOS[scenario_input]


def main(argv: Sequence[str] | None = None) -> int:
    """Main entry point for the CLI.

    Args:
        argv: Command-line arguments. If None, uses sys.argv.

    Returns:
        Exit code (0 for success, 1 for user errors, 2 for unexpected errors).
    """
    try:
        parser = build_parser()
        args = parser.parse_args(argv)

        if args.command == "list-presets":
            print("Deployment Scenarios:")
            for name in sorted(SCENARIOS):
                scenario = SCENARIOS[name]
                print(f"  - {name}: {scenario.parameters_b}B params, {scenario.memory_gb:.0f}GB")
            print("\nOptimization Strategies:")
            for name in sorted(STRATEGIES):
                strategy = STRATEGIES[name]
                print(
                    f"  - {name}: {strategy.throughput_improvement_fraction * 100:.0f}% "
                    f"throughput, {strategy.memory_reduction_fraction * 100:.0f}% memory, "
                    f"risk={strategy.quality_risk}"
                )
            return 0

        if args.command == "preset":
            scenario = SCENARIOS[args.name]
            strategy = STRATEGIES[args.strategy]
            if args.with_quantization:
                strategy = combine_strategies(strategy, QUANTIZATION_BONUS)
            grid_co2e = get_carbon_intensity(args.region) if args.region else 0.35
            outcome = estimate_outcome(scenario, strategy, grid_co2e_kg_per_kwh=grid_co2e)
            print(render_report(outcome, args.report))
            return 0

        if args.command == "scenario":
            scenario = load_scenario(args.path)
            strategy = STRATEGIES[args.strategy]
            if args.with_quantization:
                strategy = combine_strategies(strategy, QUANTIZATION_BONUS)
            grid_co2e = get_carbon_intensity(args.region) if args.region else 0.35
            outcome = estimate_outcome(scenario, strategy, grid_co2e_kg_per_kwh=grid_co2e)
            print(render_report(outcome, args.report))
            return 0

        if args.command == "compare":
            scenario_name, scenario = _load_scenario_input(args.scenario)
            grid_co2e = get_carbon_intensity(args.region) if args.region else 0.35
            config = AtroposConfig(grid_co2e_factor=grid_co2e)
            calculator = ROICalculator(config=config)
            calculator.register_scenario(scenario)
            for name in args.strategies:
                calculator.register_strategy(STRATEGIES[name])
            results = calculator.compare_strategies(
                scenario_name, args.strategies, args.with_quantization
            )

            # Sort results
            outcomes = list(results.values())
            reverse = not args.ascending
            if args.sort_by == "savings":
                outcomes.sort(key=lambda o: o.annual_total_savings_usd, reverse=reverse)
            elif args.sort_by == "breakeven":
                outcomes.sort(key=lambda o: (o.break_even_years or float("inf")), reverse=reverse)
            elif args.sort_by == "risk":
                risk_order = {"low": 0, "medium": 1, "high": 2}
                outcomes.sort(key=lambda o: risk_order[o.quality_risk], reverse=reverse)

            # Generate output
            if args.format == "markdown":
                content = generate_comparison_table(outcomes)
            elif args.format == "json":
                content = generate_comparison_json(outcomes)
            else:
                content = "\n\n".join(render_report(outcome, "text") for outcome in outcomes)
            if args.output:
                args.output.write_text(content)
            else:
                print(content)
            return 0

        if args.command == "batch":
            batch_results = batch_process(
                args.directory, args.strategies, args.output, args.with_quantization
            )
            print(f"Processed {len(batch_results)} combinations")
            return 0

        if args.command == "sensitivity":
            scenario_name, scenario = _load_scenario_input(args.scenario)
            calculator = ROICalculator()
            calculator.register_scenario(scenario)
            calculator.register_strategy(STRATEGIES[args.strategy])
            sens_results = calculator.sensitivity_analysis(
                scenario_name, args.strategy, args.param, args.variations, args.step
            )

            if args.format == "csv" and args.output:
                export_to_csv((outcome for _, outcome in sens_results), args.output)
                print(f"Saved sensitivity results to {args.output}")
            elif args.format == "json":
                import json

                data = [
                    {
                        "variation_factor": factor,
                        "annual_savings_usd": outcome.annual_total_savings_usd,
                        "break_even_months": (
                            outcome.break_even_years * 12 if outcome.break_even_years else None
                        ),
                        "optimized_memory_gb": outcome.optimized_memory_gb,
                        "optimized_throughput": outcome.optimized_throughput_toks_per_sec,
                    }
                    for factor, outcome in sens_results
                ]
                content = json.dumps(data, indent=2)
                if args.output:
                    args.output.write_text(content)
                    print(f"Saved sensitivity results to {args.output}")
                else:
                    print(content)
            else:
                for factor, outcome in sens_results:
                    months = (
                        None if outcome.break_even_years is None else outcome.break_even_years * 12
                    )
                    break_even = f"{months:.1f} mo" if months is not None else "never"
                    print(
                        f"factor={factor:.2f} savings=${outcome.annual_total_savings_usd:,.2f} "
                        f"break_even={break_even} memory={outcome.optimized_memory_gb:.2f}GB"
                    )
            return 0

        if args.command == "monte-carlo":
            scenario_name, scenario = _load_scenario_input(args.scenario)
            calculator = ROICalculator()
            calculator.register_scenario(scenario)
            calculator.register_strategy(STRATEGIES[args.strategy])

            distributions = [
                ParameterDistribution(
                    param_name=p,
                    distribution=args.distribution,
                    std_dev=args.std_dev,
                    range_fraction=args.range_fraction,
                )
                for p in args.params
            ]

            result = calculator.monte_carlo_analysis(
                scenario_name,
                args.strategy,
                distributions,
                num_simulations=args.simulations,
                seed=args.seed,
            )

            if args.format == "json":
                import json

                mc_data: dict[str, object] = {
                    "scenario": result.scenario_name,
                    "strategy": result.strategy_name,
                    "simulations": result.num_simulations,
                    "savings": {
                        "mean": result.savings_mean,
                        "std": result.savings_std,
                        "p5": result.savings_p5,
                        "p25": result.savings_p25,
                        "median": result.savings_median,
                        "p75": result.savings_p75,
                        "p95": result.savings_p95,
                    },
                    "break_even": {
                        "mean": result.break_even_mean,
                        "median": result.break_even_median,
                    },
                    "probabilities": {
                        "positive_roi": result.probability_positive_roi,
                        "break_even_within_1yr": result.probability_break_even_within_1yr,
                        "break_even_within_2yr": result.probability_break_even_within_2yr,
                    },
                    "co2e_savings_mean_kg": result.co2e_savings_mean,
                    "memory_reduction_mean": result.memory_reduction_mean,
                }
                content = json.dumps(mc_data, indent=2)
                if args.output:
                    args.output.write_text(content)
                    print(f"Saved Monte Carlo results to {args.output}")
                else:
                    print(content)
            elif args.format == "csv":
                if args.output:
                    export_to_csv(result.all_outcomes, args.output)
                    print(f"Saved {len(result.all_outcomes)} outcomes to {args.output}")
                else:
                    print("CSV format requires --output file")
                    return 1
            else:
                print("\nMonte Carlo Uncertainty Analysis")
                print("=" * 50)
                print(f"Scenario: {result.scenario_name}")
                print(f"Strategy: {result.strategy_name}")
                print(f"Simulations: {result.num_simulations}")
                print(f"\nParameters varied: {', '.join(args.params)}")
                print(f"Distribution: {args.distribution}")
                print("\nAnnual Savings Distribution:")
                print(f"  Mean:   ${result.savings_mean:,.2f}")
                print(f"  StdDev: ${result.savings_std:,.2f}")
                print(f"  P5:     ${result.savings_p5:,.2f}")
                print(f"  P25:    ${result.savings_p25:,.2f}")
                print(f"  Median: ${result.savings_median:,.2f}")
                print(f"  P75:    ${result.savings_p75:,.2f}")
                print(f"  P95:    ${result.savings_p95:,.2f}")
                print("\nBreak-even Time:")
                if result.break_even_mean:
                    print(f"  Mean:   {result.break_even_mean:.2f} years")
                    print(f"  Median: {result.break_even_median:.2f} years")
                else:
                    print("  No break-even in most simulations")
                print("\nProbabilities:")
                print(f"  Positive ROI: {result.probability_positive_roi:.1%}")
                print(f"  Break-even <= 1yr: {result.probability_break_even_within_1yr:.1%}")
                print(f"  Break-even <= 2yr: {result.probability_break_even_within_2yr:.1%}")
                print("\nOther Metrics:")
                print(f"  Mean CO2e savings: {result.co2e_savings_mean:.1f} kg/year")
                print(f"  Mean memory reduction: {result.memory_reduction_mean:.1%}")

                if args.output:
                    args.output.write_text(
                        f"# Monte Carlo Analysis: {result.scenario_name}\n\n"
                        f"**Strategy:** {result.strategy_name}  \n"
                        f"**Simulations:** {result.num_simulations}  \n\n"
                        "## Annual Savings Distribution\n\n"
                        "| Metric | Value |\n"
                        "|--------|-------|\n"
                        f"| Mean | ${result.savings_mean:,.2f} |\n"
                        f"| P5 | ${result.savings_p5:,.2f} |\n"
                        f"| Median | ${result.savings_median:,.2f} |\n"
                        f"| P95 | ${result.savings_p95:,.2f} |\n\n"
                        "## Probabilities\n\n"
                        f"- **Positive ROI:** {result.probability_positive_roi:.1%}\n"
                        "- **Break-even <= 1 year:** "
                        f"{result.probability_break_even_within_1yr:.1%}\n"
                        "- **Break-even <= 2 years:** "
                        f"{result.probability_break_even_within_2yr:.1%}\n"
                    )
                    print(f"\nSaved report to {args.output}")
            return 0

        if args.command == "csv-to-markdown":
            markdown = csv_to_markdown(args.input, args.output)
            if args.output:
                print(f"Saved markdown report to {args.output}")
            else:
                print(markdown)
            return 0

        if args.command == "import-telemetry":
            # Parse field mapping if provided
            field_mapping = None
            if args.mapping:
                import json

                field_mapping = json.loads(args.mapping)

            # Get parser and parse telemetry
            parser_instance = get_parser(args.format, field_mapping)
            telemetry = parser_instance.parse_file(args.input)

            # Validate telemetry
            issues = validate_telemetry(telemetry)
            if issues:
                print("Telemetry validation issues:", file=sys.stderr)
                for issue in issues:
                    print(f"  - {issue}", file=sys.stderr)
                if any("must be" in i for i in issues):
                    return 1

            # Create scenario
            scenario = telemetry_to_scenario(
                telemetry,
                name=args.name,
                electricity_cost_per_kwh=args.electricity_cost,
                annual_hardware_cost_usd=args.hardware_cost,
                one_time_project_cost_usd=args.project_cost,
                requests_per_day=args.requests_per_day,
            )

            # Preview or save
            if args.preview or not args.output:
                print(f"\nScenario: {scenario.name}")
                print("=" * 50)
                print(f"Parameters: {scenario.parameters_b}B")
                print(f"Memory: {scenario.memory_gb:.1f} GB")
                print(f"Throughput: {scenario.throughput_toks_per_sec:.1f} tok/s")
                print(f"Power: {scenario.power_watts:.0f} W")
                print(f"Requests/day: {scenario.requests_per_day}")
                print(f"Tokens/request: {scenario.tokens_per_request}")
                print(f"Electricity cost: ${scenario.electricity_cost_per_kwh}/kWh")
                print(f"Annual hardware: ${scenario.annual_hardware_cost_usd:,.0f}")
                print(f"Project cost: ${scenario.one_time_project_cost_usd:,.0f}")
                print(f"\nSource: {telemetry.source}")
                if telemetry.raw_metrics:
                    print(f"\nRaw metrics available: {len(telemetry.raw_metrics)} fields")

            if args.output:
                import yaml

                # Convert dataclass to dict for YAML output
                scenario_dict = {
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
                with open(args.output, "w", encoding="utf-8") as f:
                    yaml.dump(scenario_dict, f, default_flow_style=False, sort_keys=False)
                print(f"\nSaved scenario to {args.output}")
            return 0

        if args.command == "import-experiment":
            # Validate args
            if not args.run_id and not args.experiment:
                print("Error: Either --run-id or --experiment must be provided", file=sys.stderr)
                return 1

            # Get tracker kwargs
            tracker_kwargs: dict[str, Any] = {}
            if args.entity:
                tracker_kwargs["entity"] = args.entity

            try:
                tracker = get_tracker(
                    args.tracker,
                    api_key=args.api_key,
                    host=args.host,
                    **tracker_kwargs,
                )

                if args.run_id:
                    run_kwargs = {}
                    if args.project:
                        run_kwargs["project"] = args.project
                    run_info = tracker.get_run(args.run_id, **run_kwargs)
                    runs = [run_info]
                else:
                    list_kwargs = {"limit": args.limit}
                    if args.entity:
                        list_kwargs["entity"] = args.entity
                    runs = tracker.list_runs(experiment=args.experiment, **list_kwargs)

                if not runs:
                    print("No runs found matching criteria")
                    return 0

                # Process runs
                for i, run in enumerate(runs):
                    name = args.name or f"{run.experiment}-{run.run_id}"
                    if len(runs) > 1:
                        name = f"{name}-{i + 1}"

                    scenario = run_to_scenario(
                        run,
                        name=name,
                        electricity_cost_per_kwh=args.electricity_cost,
                        annual_hardware_cost_usd=args.hardware_cost,
                        one_time_project_cost_usd=args.project_cost,
                        requests_per_day=args.requests_per_day,
                    )

                    # Preview
                    if args.preview or not args.output or len(runs) > 1:
                        print(f"\nScenario: {scenario.name}")
                        print("=" * 50)
                        print(f"Source: {run.tracker} / {run.experiment} / {run.run_id}")
                        if run.url:
                            print(f"URL: {run.url}")
                        print(f"Parameters: {scenario.parameters_b}B")
                        print(f"Memory: {scenario.memory_gb:.1f} GB")
                        print(f"Throughput: {scenario.throughput_toks_per_sec:.1f} tok/s")
                        print(f"Power: {scenario.power_watts:.0f} W")
                        print(f"Requests/day: {scenario.requests_per_day}")
                        print(f"Tokens/request: {scenario.tokens_per_request}")
                        if run.tags:
                            print(f"Tags: {', '.join(run.tags)}")

                    # Save if output specified and single run or last run
                    if args.output and len(runs) == 1:
                        import yaml

                        scenario_dict = {
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
                        with open(args.output, "w", encoding="utf-8") as f:
                            yaml.dump(scenario_dict, f, default_flow_style=False, sort_keys=False)
                        print(f"\nSaved scenario to {args.output}")

                if len(runs) > 1:
                    print(f"\nImported {len(runs)} scenarios from {args.tracker}")

                return 0

            except RuntimeError as e:
                print(f"Error: {e}", file=sys.stderr)
                print("Install the required package:", file=sys.stderr)
                if args.tracker == "wandb":
                    print("  pip install wandb", file=sys.stderr)
                elif args.tracker == "mlflow":
                    print("  pip install mlflow", file=sys.stderr)
                return 1

        if args.command == "list-carbon-presets":
            if args.region:
                # Show specific region
                intensity = get_carbon_intensity(args.region)
                preset = None
                for code, p in CARBON_PRESETS.items():
                    if code == args.region.upper() or p.region_name.lower() == args.region.lower():
                        preset = p
                        break

                if args.format == "json":
                    import json

                    region_data: dict[str, Any] = {
                        "region": args.region,
                        "carbon_intensity_kg_per_kwh": intensity,
                        "preset": {
                            "region_code": preset.region_code if preset else args.region.upper(),
                            "region_name": preset.region_name if preset else "Unknown",
                            "data_year": preset.data_year if preset else 2023,
                            "source": preset.source if preset else "Global average",
                            "notes": preset.notes if preset else "",
                        }
                        if preset
                        else None,
                    }
                    print(json.dumps(region_data, indent=2))
                else:
                    print(f"\nRegion: {args.region}")
                    print("=" * 50)
                    print(f"Carbon intensity: {intensity:.3f} kg CO2e/kWh")
                    if preset:
                        print(f"Region name: {preset.region_name}")
                        print(f"Data year: {preset.data_year}")
                        print(f"Source: {preset.source}")
                        if preset.notes:
                            print(f"Notes: {preset.notes}")
            else:
                # List all regions
                regions = list_regions()

                if args.format == "json":
                    import json

                    data_list: list[dict[str, Any]] = [
                        {
                            "region_code": code,
                            "region_name": CARBON_PRESETS[code].region_name,
                            "carbon_intensity_kg_per_kwh": CARBON_PRESETS[
                                code
                            ].carbon_intensity_kg_per_kwh,
                        }
                        for code in regions
                    ]
                    print(json.dumps(data_list, indent=2))
                else:
                    print("\nAvailable carbon intensity presets:")
                    print("=" * 70)
                    print(f"{'Code':<6} {'Region':<30} {'Intensity (kg/kWh)':<20}")
                    print("-" * 70)
                    for code in regions:
                        preset = CARBON_PRESETS[code]
                        print(
                            f"{code:<6} {preset.region_name:<30} "
                            f"{preset.carbon_intensity_kg_per_kwh:<20.3f}"
                        )
                    print("\nUse --region CODE for details")
                    print("Cloud regions (e.g., us-east-1) are also supported")
            return 0

        if args.command == "dashboard":
            try:
                from .dashboard import run_dashboard

                print(f"Starting Atropos dashboard at http://{args.host}:{args.port}")
                print("Press Ctrl+C to stop")
                run_dashboard(host=args.host, port=args.port, debug=args.debug)
                return 0
            except ImportError:
                print("Error: Dashboard dependencies not installed", file=sys.stderr)
                print("Install with: pip install dash plotly pandas", file=sys.stderr)
                return 1

        if args.command == "calibrate":
            # Load scenario
            scenario_name, scenario = _load_scenario_input(args.scenario)

            # Parse telemetry
            telemetry_parser = get_parser(args.parser) if args.parser else None
            if telemetry_parser:
                telemetry = telemetry_parser.parse(args.telemetry)
            else:
                # Auto-detect parser from file extension
                from .telemetry import PARSERS

                suffix = args.telemetry.suffix.lower()
                if suffix == ".json":
                    telemetry = PARSERS["vllm"]().parse_file(args.telemetry)
                elif suffix == ".csv":
                    telemetry = PARSERS["csv"]().parse_file(args.telemetry)
                else:
                    print(
                        f"Error: Cannot auto-detect parser for {suffix} files. "
                        "Use --parser.",
                        file=sys.stderr,
                    )
                    return 1

            # Validate telemetry
            issues = validate_telemetry(telemetry)
            if issues:
                print("Warning: Telemetry validation issues:", file=sys.stderr)
                for issue in issues:
                    print(f"  - {issue}", file=sys.stderr)

            # Run calibration
            calibration_result = calibrate_scenario(
                scenario, telemetry, tolerance_pct=args.tolerance
            )

            # Generate report
            report = generate_calibration_report(calibration_result, format=args.format)

            if args.output:
                args.output.write_text(report)
                print(f"Calibration report saved to {args.output}")
            else:
                print(report)

            return 0

        if args.command == "pipeline":
            # Load scenario and pipeline config
            scenario_name, scenario = _load_scenario_input(args.scenario)
            pipeline_config = PipelineConfig.from_yaml(args.config)
            strategy = STRATEGIES[args.strategy]
            grid_co2e = get_carbon_intensity(args.region) if args.region else 0.35

            print(f"Running pipeline: {pipeline_config.name}")
            print(f"Scenario: {scenario_name}")
            print(f"Strategy: {args.strategy}")
            if args.dry_run:
                print("Mode: DRY RUN (simulation only)")
            print()

            # Run pipeline
            pipeline_result = run_pipeline(
                config=pipeline_config,
                scenario=scenario,
                strategy=strategy,
                grid_co2e=grid_co2e,
                dry_run=args.dry_run,
            )

            # Print summary
            print(f"Pipeline status: {pipeline_result.final_status.name.lower()}")
            duration = pipeline_result.duration_seconds
            if duration:
                print(f"Duration: {duration:.1f}s")
            else:
                print("Duration: N/A")
            print()

            # Print stage results
            for stage in pipeline_result.stages:
                print(f"  {stage.stage.name.lower()}: {stage.status.name.lower()}")
                if stage.message:
                    print(f"    {stage.message}")

            # Save results if output specified
            if args.output:
                args.output.write_text(pipeline_result.to_json())
                print(f"\nResults saved to {args.output}")

            return_code = 0 if pipeline_result.final_status.name.lower() == "success" else 1
            return return_code

        if args.command == "validate-pipeline-config":
            try:
                pipeline_cfg = PipelineConfig.from_yaml(args.config)
                # Subconfigs are always set by __post_init__, but mypy doesn't know
                assert pipeline_cfg.thresholds is not None
                assert pipeline_cfg.pruning is not None
                assert pipeline_cfg.recovery is not None
                assert pipeline_cfg.validation is not None
                assert pipeline_cfg.deployment is not None

                print(f"Configuration valid: {pipeline_cfg.name}")
                print(f"  Auto-execute: {pipeline_cfg.auto_execute}")
                thresh = pipeline_cfg.thresholds
                print(f"  Thresholds: {thresh.max_break_even_months} months break-even, "
                      f"${thresh.min_annual_savings_usd:,.0f} min savings")
                fw = pipeline_cfg.pruning.framework
                sparsity = pipeline_cfg.pruning.target_sparsity
                print(f"  Pruning: {fw} at {sparsity:.0%} sparsity")
                recov = pipeline_cfg.recovery.enabled
                val_bm = pipeline_cfg.validation.quality_benchmark
                deploy = pipeline_cfg.deployment.auto_deploy
                print(f"  Recovery: {'enabled' if recov else 'disabled'}")
                print(f"  Validation: {val_bm} benchmark")
                print(f"  Deployment: {'auto' if deploy else 'manual'}")
                return 0
            except Exception as e:
                print(f"Configuration error: {e}", file=sys.stderr)
                return 1

        if args.command == "validate":
            # Load scenario and strategy
            scenario_name, scenario = _load_scenario_input(args.scenario)
            strategy = STRATEGIES[args.strategy]

            print(f"Validating Atropos projections for: {scenario_name}")
            print(f"Strategy: {args.strategy}")
            print(f"Device: {args.device}")
            if args.model:
                print(f"Model: {args.model}")
            print()

            # Run validation
            try:
                validation_result = run_validation(
                    scenario=scenario,
                    strategy=strategy,
                    model_name=args.model,
                    device=args.device,
                    pruning_method=args.pruning_method,
                )

                # Generate report
                if args.format == "json":
                    import json
                    report = json.dumps(validation_result.to_dict(), indent=2)
                else:
                    report = validation_result.to_markdown()

                if args.output:
                    args.output.write_text(report)
                    print(f"Validation report saved to {args.output}")
                else:
                    print(report)

                return 0
            except Exception as e:
                print(f"Validation failed: {e}", file=sys.stderr)
                return 1

        parser.error(f"Unsupported command: {args.command}")
        return 2
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"Error: Invalid input - {e}", file=sys.stderr)
        return 1
    except KeyError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
