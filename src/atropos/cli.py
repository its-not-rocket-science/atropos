"""Command-line interface for Atropos."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from .batch import batch_process
from .calculations import combine_strategies, estimate_outcome
from .core.calculator import ROICalculator
from .core.uncertainty import ParameterDistribution
from .io import csv_to_markdown, export_to_csv, load_scenario, render_report
from .models import DeploymentScenario
from .presets import QUANTIZATION_BONUS, SCENARIOS, STRATEGIES
from .reporting import generate_comparison_json, generate_comparison_table


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

    scenario_parser = subparsers.add_parser("scenario", help="Run a scenario from a YAML file.")
    scenario_parser.add_argument("path", type=Path)
    _add_strategy_args(scenario_parser)

    compare_parser = subparsers.add_parser("compare", help="Compare multiple strategies.")
    compare_parser.add_argument("scenario", help="Scenario name or path to YAML")
    compare_parser.add_argument(
        "--strategies", nargs="+", required=True, choices=sorted(STRATEGIES.keys())
    )
    compare_parser.add_argument("--with-quantization", action="store_true")
    compare_parser.add_argument(
        "--format", choices=["text", "markdown", "json"], default="text"
    )
    compare_parser.add_argument(
        "--sort-by", choices=["savings", "breakeven", "risk"], default="savings"
    )
    compare_parser.add_argument("--ascending", action="store_true", help="Sort in ascending order")
    compare_parser.add_argument("--output", "-o", type=Path)

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

    mc_parser = subparsers.add_parser(
        "monte-carlo", help="Run Monte Carlo uncertainty analysis."
    )
    mc_parser.add_argument("scenario", help="Scenario name or path to YAML")
    mc_parser.add_argument("--strategy", required=True, choices=sorted(STRATEGIES.keys()))
    mc_parser.add_argument(
        "--params",
        nargs="+",
        default=["memory_reduction_fraction", "throughput_improvement_fraction"],
        help=("Parameters to vary "
              "(default: memory_reduction_fraction throughput_improvement_fraction)"),
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
            print(render_report(estimate_outcome(scenario, strategy), args.report))
            return 0

        if args.command == "scenario":
            scenario = load_scenario(args.path)
            strategy = STRATEGIES[args.strategy]
            if args.with_quantization:
                strategy = combine_strategies(strategy, QUANTIZATION_BONUS)
            print(render_report(estimate_outcome(scenario, strategy), args.report))
            return 0

        if args.command == "compare":
            scenario_name, scenario = _load_scenario_input(args.scenario)
            calculator = ROICalculator()
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
                outcomes.sort(
                    key=lambda o: (o.break_even_years or float('inf')), reverse=reverse
                )
            elif args.sort_by == "risk":
                risk_order = {"low": 0, "medium": 1, "high": 2}
                outcomes.sort(key=lambda o: risk_order[o.quality_risk], reverse=reverse)

            # Generate output
            if args.format == "markdown":
                content = generate_comparison_table(outcomes)
            elif args.format == "json":
                content = generate_comparison_json(outcomes)
            else:
                content = "\n\n".join(
                    render_report(outcome, "text") for outcome in outcomes
                )
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

                data = {
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
                content = json.dumps(data, indent=2)
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
