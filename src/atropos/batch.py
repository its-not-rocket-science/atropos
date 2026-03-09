"""Batch processing helpers for multiple scenarios."""

from __future__ import annotations

from pathlib import Path

from .core.calculator import ROICalculator
from .io import export_to_csv, load_scenario
from .models import OptimizationOutcome
from .presets import STRATEGIES


def batch_process(
    scenario_dir: str | Path,
    strategy_names: list[str],
    output_file: str | Path | None = None,
    with_quantization: bool = False,
) -> list[OptimizationOutcome]:
    """Process multiple scenarios with multiple strategies.

    Args:
        scenario_dir: Directory containing YAML scenario files.
        strategy_names: List of strategy names to apply.
        output_file: Optional path to write CSV results.
        with_quantization: Whether to apply quantization bonus.

    Returns:
        List of optimization outcomes.
    """
    calculator = ROICalculator()
    for name in strategy_names:
        if name not in STRATEGIES:
            raise KeyError(f"Strategy '{name}' not found")
        calculator.register_strategy(STRATEGIES[name])

    path = Path(scenario_dir)
    for yaml_file in sorted(path.glob("*.yaml")):
        calculator.register_scenario(load_scenario(yaml_file))

    results: list[OptimizationOutcome] = []
    for scenario_name in calculator.scenarios:
        for strategy_name in strategy_names:
            results.append(calculator.calculate(scenario_name, strategy_name, with_quantization))

    if output_file is not None:
        export_to_csv(results, output_file)
    return results
