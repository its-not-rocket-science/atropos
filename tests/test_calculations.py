"""Tests for calculation functions."""

import pytest

from atropos.calculations import combine_strategies, estimate_outcome
from atropos.models import DeploymentScenario, OptimizationStrategy


def make_scenario() -> DeploymentScenario:
    """Create a test deployment scenario with default values."""
    return DeploymentScenario(
        name="test",
        parameters_b=10.0,
        memory_gb=8.0,
        throughput_toks_per_sec=20.0,
        power_watts=200.0,
        requests_per_day=1000,
        tokens_per_request=1000,
        electricity_cost_per_kwh=0.15,
        annual_hardware_cost_usd=1000.0,
        one_time_project_cost_usd=500.0,
    )


def test_estimate_outcome_reduces_energy_for_positive_optimization() -> None:
    """Test that positive optimization reduces energy and improves metrics."""
    strategy = OptimizationStrategy(
        name="test-strategy",
        parameter_reduction_fraction=0.2,
        memory_reduction_fraction=0.2,
        throughput_improvement_fraction=0.25,
        power_reduction_fraction=0.1,
        quality_risk="low",
    )

    outcome = estimate_outcome(make_scenario(), strategy)

    assert outcome.optimized_memory_gb < outcome.baseline_memory_gb
    assert outcome.optimized_throughput_toks_per_sec > outcome.baseline_throughput_toks_per_sec
    assert outcome.optimized_energy_wh_per_request < outcome.baseline_energy_wh_per_request
    assert outcome.annual_total_savings_usd > 0
    assert outcome.break_even_years is not None


def test_combine_strategies_increases_throughput_and_reduction() -> None:
    """Test that combining strategies multiplies throughput and reduction effects."""
    a = OptimizationStrategy("a", 0.1, 0.2, 0.1, 0.05, "low")
    b = OptimizationStrategy("b", 0.2, 0.1, 0.2, 0.05, "medium")
    combined = combine_strategies(a, b)
    assert combined.throughput_improvement_fraction > a.throughput_improvement_fraction
    assert combined.memory_reduction_fraction > max(
        a.memory_reduction_fraction, b.memory_reduction_fraction
    )
    assert combined.quality_risk == "medium"


def test_zero_improvement_strategy() -> None:
    """Test that zero improvement strategy leaves metrics unchanged."""
    strategy = OptimizationStrategy("no_change", 0.0, 0.0, 0.0, 0.0, "low")
    outcome = estimate_outcome(make_scenario(), strategy)
    assert outcome.optimized_memory_gb == outcome.baseline_memory_gb
    assert outcome.optimized_throughput_toks_per_sec == outcome.baseline_throughput_toks_per_sec
    assert outcome.annual_total_savings_usd == 0


def test_invalid_scenario_values_raise() -> None:
    """Test that invalid scenario values raise ValueError."""
    scenario = make_scenario()
    bad = DeploymentScenario(**{**scenario.__dict__, "power_watts": 0.0})
    strategy = OptimizationStrategy("s", 0.1, 0.1, 0.1, 0.1, "low")
    with pytest.raises(ValueError, match="power_watts must be positive"):
        estimate_outcome(bad, strategy)


def test_invalid_fraction_raises() -> None:
    """Test that invalid fraction values raise ValueError."""
    strategy = OptimizationStrategy("invalid", 1.5, 0.2, 0.25, 0.1, "low")
    with pytest.raises(ValueError):
        estimate_outcome(make_scenario(), strategy)
