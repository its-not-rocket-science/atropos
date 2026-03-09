"""Core ROI calculator with strategy management."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

from ..calculations import combine_strategies, estimate_outcome
from ..config import AtroposConfig
from ..models import DeploymentScenario, OptimizationOutcome, OptimizationStrategy
from ..presets import QUANTIZATION_BONUS
from .uncertainty import MonteCarloResult, ParameterDistribution, run_monte_carlo

if TYPE_CHECKING:
    pass


class ROICalculator:
    """Main calculator with strategy management and comparison capabilities."""

    def __init__(self, config: AtroposConfig | None = None):
        self.config = config or AtroposConfig()
        self.strategies: dict[str, OptimizationStrategy] = {}
        self.scenarios: dict[str, DeploymentScenario] = {}

    def register_strategy(self, strategy: OptimizationStrategy) -> None:
        """Register a strategy for use in calculations."""
        self.strategies[strategy.name] = strategy

    def register_scenario(self, scenario: DeploymentScenario) -> None:
        """Register a scenario for use in calculations."""
        self.scenarios[scenario.name] = scenario

    def calculate(
        self, scenario_name: str, strategy_name: str, with_quantization: bool = False
    ) -> OptimizationOutcome:
        """Calculate the outcome for a scenario and strategy.

        Args:
            scenario_name: Name of the registered scenario.
            strategy_name: Name of the registered strategy.
            with_quantization: Whether to apply quantization bonus.

        Returns:
            Optimization outcome with baseline and optimized metrics.

        Raises:
            KeyError: If scenario or strategy is not registered.
        """
        if scenario_name not in self.scenarios:
            raise KeyError(f"Scenario '{scenario_name}' not found")
        if strategy_name not in self.strategies:
            raise KeyError(f"Strategy '{strategy_name}' not found")

        scenario = self.scenarios[scenario_name]
        strategy = self.strategies[strategy_name]

        if with_quantization:
            strategy = combine_strategies(strategy, QUANTIZATION_BONUS)

        return estimate_outcome(
            scenario,
            strategy,
            grid_co2e_kg_per_kwh=self.config.grid_co2e_factor,
            hardware_savings_correlation=self.config.hardware_savings_correlation,
        )

    def compare_strategies(
        self, scenario_name: str, strategy_names: list[str], with_quantization: bool = False
    ) -> dict[str, OptimizationOutcome]:
        """Compare multiple strategies against a single scenario.

        Args:
            scenario_name: Name of the registered scenario.
            strategy_names: List of strategy names to compare.
            with_quantization: Whether to apply quantization bonus.

        Returns:
            Dictionary mapping strategy names to their outcomes.
        """
        return {
            name: self.calculate(scenario_name, name, with_quantization) for name in strategy_names
        }

    def sensitivity_analysis(
        self,
        scenario_name: str,
        strategy_name: str,
        param: str,
        variations: int = 5,
        step: float = 0.1,
    ) -> list[tuple[float, OptimizationOutcome]]:
        """Run sensitivity analysis by varying a strategy parameter.

        Args:
            scenario_name: Name of the registered scenario.
            strategy_name: Name of the registered strategy.
            param: Parameter name to vary.
            variations: Number of variations on each side of baseline.
            step: Step size as a fraction (0.1 = 10%).

        Returns:
            List of (factor, outcome) tuples sorted by factor.
        """
        scenario = self.scenarios[scenario_name]
        strategy = self.strategies[strategy_name]
        base_value = getattr(strategy, param)
        results: list[tuple[float, OptimizationOutcome]] = []

        for i in range(-variations, variations + 1):
            if i == 0:
                continue
            factor = 1.0 + (i * step)
            new_value = base_value * factor
            if param.endswith("_fraction") and not 0 <= new_value < 1.0:
                continue
            if param == "throughput_improvement_fraction" and new_value < 0:
                continue
            modified = dataclasses.replace(strategy, **{param: new_value})
            outcome = estimate_outcome(
                scenario,
                modified,
                grid_co2e_kg_per_kwh=self.config.grid_co2e_factor,
                hardware_savings_correlation=self.config.hardware_savings_correlation,
            )
            results.append((factor, outcome))
        return sorted(results, key=lambda x: x[0])

    def monte_carlo_analysis(
        self,
        scenario_name: str,
        strategy_name: str,
        distributions: list[ParameterDistribution],
        num_simulations: int = 1000,
        seed: int | None = None,
    ) -> MonteCarloResult:
        """Run Monte Carlo uncertainty analysis.

        Args:
            scenario_name: Name of the registered scenario.
            strategy_name: Name of the registered strategy.
            distributions: Parameter distributions to sample from.
            num_simulations: Number of simulation runs.
            seed: Random seed for reproducibility.

        Returns:
            MonteCarloResult with statistical summaries.

        Raises:
            KeyError: If scenario or strategy is not registered.
        """
        if scenario_name not in self.scenarios:
            raise KeyError(f"Scenario '{scenario_name}' not found")
        if strategy_name not in self.strategies:
            raise KeyError(f"Strategy '{strategy_name}' not found")

        scenario = self.scenarios[scenario_name]
        strategy = self.strategies[strategy_name]

        def estimator(scen: DeploymentScenario, strat: OptimizationStrategy) -> OptimizationOutcome:
            return estimate_outcome(
                scen,
                strat,
                grid_co2e_kg_per_kwh=self.config.grid_co2e_factor,
                hardware_savings_correlation=self.config.hardware_savings_correlation,
            )

        return run_monte_carlo(scenario, strategy, distributions, estimator, num_simulations, seed)
