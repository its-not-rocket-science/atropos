"""Monte Carlo uncertainty analysis for ROI estimation."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Callable

    from ..models import DeploymentScenario, OptimizationOutcome, OptimizationStrategy

DistributionType = Literal["normal", "uniform", "triangular"]


@dataclass(frozen=True)
class ParameterDistribution:
    """Distribution configuration for an uncertain parameter.

    Attributes:
        param_name: Name of the parameter to vary.
        distribution: Type of distribution (normal, uniform, triangular).
        std_dev: Standard deviation for normal distribution (as fraction of mean).
        range_fraction: Range for uniform/triangular (as +/- fraction of mean).
    """

    param_name: str
    distribution: DistributionType = "normal"
    std_dev: float = 0.1
    range_fraction: float = 0.2

    def sample(self, base_value: float) -> float:
        """Sample a value from the distribution around the base value."""
        if self.distribution == "normal":
            return random.gauss(base_value, base_value * self.std_dev)
        if self.distribution == "uniform":
            delta = base_value * self.range_fraction
            return random.uniform(base_value - delta, base_value + delta)
        if self.distribution == "triangular":
            delta = base_value * self.range_fraction
            return random.triangular(
                base_value - delta, base_value + delta, base_value
            )
        return base_value


@dataclass
class MonteCarloResult:
    """Results from Monte Carlo uncertainty analysis.

    Contains statistical summaries of multiple simulation runs.

    Attributes:
        scenario_name: Name of the scenario analyzed.
        strategy_name: Name of the strategy applied.
        num_simulations: Number of simulations run.
        distributions: Distributions used for each parameter.
        savings_mean: Mean annual savings across simulations.
        savings_std: Standard deviation of annual savings.
        savings_p5: 5th percentile of annual savings.
        savings_p25: 25th percentile of annual savings.
        savings_median: Median annual savings.
        savings_p75: 75th percentile of annual savings.
        savings_p95: 95th percentile of annual savings.
        break_even_mean: Mean break-even time in years.
        break_even_median: Median break-even time in years.
        probability_positive_roi: Probability that ROI is positive.
        probability_break_even_within_1yr: Probability of breaking even within 1 year.
        probability_break_even_within_2yr: Probability of breaking even within 2 years.
        co2e_savings_mean: Mean annual CO2e savings in kg.
        memory_reduction_mean: Mean memory reduction fraction.
        all_outcomes: List of all simulation outcomes (optional, for detailed export).
    """

    scenario_name: str
    strategy_name: str
    num_simulations: int
    distributions: list[ParameterDistribution] = field(repr=False)
    savings_mean: float
    savings_std: float
    savings_p5: float
    savings_p25: float
    savings_median: float
    savings_p75: float
    savings_p95: float
    break_even_mean: float | None
    break_even_median: float | None
    probability_positive_roi: float
    probability_break_even_within_1yr: float
    probability_break_even_within_2yr: float
    co2e_savings_mean: float
    memory_reduction_mean: float
    all_outcomes: list[OptimizationOutcome] = field(default_factory=list, repr=False)


def run_monte_carlo(
    scenario: DeploymentScenario,
    strategy: OptimizationStrategy,
    distributions: list[ParameterDistribution],
    estimator: Callable[[DeploymentScenario, OptimizationStrategy], OptimizationOutcome],
    num_simulations: int = 1000,
    seed: int | None = None,
) -> MonteCarloResult:
    """Run Monte Carlo simulation with parameter uncertainty.

    Args:
        scenario: Base deployment scenario.
        strategy: Base optimization strategy.
        distributions: Parameter distributions to sample from.
        num_simulations: Number of simulation runs.
        estimator: Function to estimate outcome from scenario and strategy.
        seed: Random seed for reproducibility.

    Returns:
        MonteCarloResult with statistical summaries.
    """
    if seed is not None:
        random.seed(seed)

    outcomes: list[OptimizationOutcome] = []

    # Map distribution names to actual distribution objects for quick lookup
    scenario_dists = {
        d.param_name: d for d in distributions if hasattr(scenario, d.param_name)
    }
    strategy_dists = {
        d.param_name: d for d in distributions if hasattr(strategy, d.param_name)
    }

    for _ in range(num_simulations):
        # Sample scenario parameters
        scenario_values = {
            k: scenario_dists[k].sample(v) if k in scenario_dists else v
            for k, v in scenario.__dict__.items()
            if not k.startswith("_")
        }
        sampled_scenario = type(scenario)(**scenario_values)

        # Sample strategy parameters
        strategy_values = {
            k: strategy_dists[k].sample(v) if k in strategy_dists else v
            for k, v in strategy.__dict__.items()
            if not k.startswith("_")
        }
        sampled_strategy = type(strategy)(**strategy_values)

        try:
            outcome = estimator(sampled_scenario, sampled_strategy)
            outcomes.append(outcome)
        except (ValueError, ArithmeticError):
            # Skip invalid samples (e.g., negative values, division by zero)
            continue

    if not outcomes:
        raise ValueError("No valid outcomes from Monte Carlo simulation")

    # Calculate statistics
    savings_values = sorted(o.annual_total_savings_usd for o in outcomes)
    break_even_values = [
        o.break_even_years for o in outcomes if o.break_even_years is not None
    ]
    co2e_values = [o.annual_co2e_savings_kg for o in outcomes]
    memory_values = [
        (o.baseline_memory_gb - o.optimized_memory_gb) / o.baseline_memory_gb
        for o in outcomes
    ]

    def percentile(values: list[float], p: float) -> float:
        """Calculate percentile from sorted values."""
        if not values:
            return 0.0
        k = (len(values) - 1) * p / 100.0
        f = int(k)
        c = f + 1 if f + 1 < len(values) else f
        return values[f] + (k - f) * (values[c] - values[f])

    savings_mean = sum(savings_values) / len(savings_values)
    savings_std = (
        sum((x - savings_mean) ** 2 for x in savings_values) / len(savings_values)
    ) ** 0.5

    positive_roi_count = sum(1 for s in savings_values if s > 0)
    break_even_1yr_count = sum(1 for b in break_even_values if b is not None and b <= 1)
    break_even_2yr_count = sum(1 for b in break_even_values if b is not None and b <= 2)

    return MonteCarloResult(
        scenario_name=scenario.name,
        strategy_name=strategy.name,
        num_simulations=len(outcomes),
        distributions=distributions,
        savings_mean=savings_mean,
        savings_std=savings_std,
        savings_p5=percentile(savings_values, 5),
        savings_p25=percentile(savings_values, 25),
        savings_median=percentile(savings_values, 50),
        savings_p75=percentile(savings_values, 75),
        savings_p95=percentile(savings_values, 95),
        break_even_mean=sum(break_even_values) / len(break_even_values)
        if break_even_values
        else None,
        break_even_median=percentile(sorted(break_even_values), 50)
        if break_even_values
        else None,
        probability_positive_roi=positive_roi_count / len(outcomes),
        probability_break_even_within_1yr=break_even_1yr_count / len(outcomes),
        probability_break_even_within_2yr=break_even_2yr_count / len(outcomes),
        co2e_savings_mean=sum(co2e_values) / len(co2e_values),
        memory_reduction_mean=sum(memory_values) / len(memory_values),
        all_outcomes=outcomes,
    )
