"""Command-line interface for Atropos."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from .batch import batch_process
from .calculations import combine_strategies, estimate_outcome
from .core.calculator import ROICalculator
from .io import export_to_csv, load_scenario, render_report
from .models import DeploymentScenario
from .presets import QUANTIZATION_BONUS, SCENARIOS, STRATEGIES
from .reporting import generate_comparison_table


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
    compare_parser.add_argument("--format", choices=["text", "markdown"], default="text")
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
            if args.format == "markdown":
                content = generate_comparison_table(results.values())
            else:
                content = "\n\n".join(
                    render_report(outcome, "text") for outcome in results.values()
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
            if args.output:
                export_to_csv((outcome for _, outcome in sens_results), args.output)
                print(f"Saved sensitivity results to {args.output}")
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
