"""Tests for Monte Carlo uncertainty analysis."""

import pytest

from atropos.calculations import estimate_outcome
from atropos.core.calculator import ROICalculator
from atropos.core.uncertainty import (
    MonteCarloResult,
    ParameterDistribution,
    run_monte_carlo,
)
from atropos.models import DeploymentScenario, OptimizationStrategy


@pytest.fixture
def sample_scenario():
    """Create a sample deployment scenario for testing."""
    return DeploymentScenario(
        name="test-scenario",
        parameters_b=7.0,
        memory_gb=14.0,
        throughput_toks_per_sec=40.0,
        power_watts=320.0,
        requests_per_day=50000,
        tokens_per_request=1200,
        electricity_cost_per_kwh=0.15,
        annual_hardware_cost_usd=24000.0,
        one_time_project_cost_usd=27000.0,
    )


@pytest.fixture
def sample_strategy():
    """Create a sample optimization strategy for testing."""
    return OptimizationStrategy(
        name="test-strategy",
        parameter_reduction_fraction=0.30,
        memory_reduction_fraction=0.22,
        throughput_improvement_fraction=0.20,
        power_reduction_fraction=0.14,
        quality_risk="medium",
    )


class TestParameterDistribution:
    """Tests for ParameterDistribution class."""

    def test_normal_distribution(self):
        """Test normal distribution sampling."""
        dist = ParameterDistribution(param_name="test", distribution="normal", std_dev=0.1)
        # With seed, should be reproducible
        values = [dist.sample(100.0) for _ in range(100)]
        # All values should be reasonably close to base value
        assert all(50 < v < 150 for v in values)
        # Mean should be close to base value
        mean = sum(values) / len(values)
        assert 90 < mean < 110

    def test_uniform_distribution(self):
        """Test uniform distribution sampling."""
        dist = ParameterDistribution(param_name="test", distribution="uniform", range_fraction=0.2)
        values = [dist.sample(100.0) for _ in range(100)]
        # All values should be within range
        assert all(80 <= v <= 120 for v in values)

    def test_triangular_distribution(self):
        """Test triangular distribution sampling."""
        dist = ParameterDistribution(
            param_name="test", distribution="triangular", range_fraction=0.2
        )
        values = [dist.sample(100.0) for _ in range(100)]
        # All values should be within range
        assert all(80 <= v <= 120 for v in values)


class TestRunMonteCarlo:
    """Tests for run_monte_carlo function."""

    def test_basic_simulation(self, sample_scenario, sample_strategy):
        """Test basic Monte Carlo simulation."""
        distributions = [
            ParameterDistribution(
                param_name="memory_reduction_fraction",
                distribution="normal",
                std_dev=0.05,
            )
        ]

        result = run_monte_carlo(
            scenario=sample_scenario,
            strategy=sample_strategy,
            distributions=distributions,
            estimator=estimate_outcome,
            num_simulations=100,
            seed=42,
        )

        assert isinstance(result, MonteCarloResult)
        assert result.scenario_name == "test-scenario"
        assert result.strategy_name == "test-strategy"
        assert result.num_simulations == 100
        assert len(result.all_outcomes) == 100

    def test_reproducibility_with_seed(self, sample_scenario, sample_strategy):
        """Test that same seed produces same results."""
        distributions = [
            ParameterDistribution(
                param_name="memory_reduction_fraction",
                distribution="uniform",
                range_fraction=0.1,
            )
        ]

        result1 = run_monte_carlo(
            scenario=sample_scenario,
            strategy=sample_strategy,
            distributions=distributions,
            estimator=estimate_outcome,
            num_simulations=50,
            seed=123,
        )

        result2 = run_monte_carlo(
            scenario=sample_scenario,
            strategy=sample_strategy,
            distributions=distributions,
            estimator=estimate_outcome,
            num_simulations=50,
            seed=123,
        )

        assert result1.savings_mean == result2.savings_mean
        assert result1.savings_median == result2.savings_median

    def test_statistical_percentiles(self, sample_scenario, sample_strategy):
        """Test that percentiles are ordered correctly."""
        distributions = [
            ParameterDistribution(
                param_name="throughput_improvement_fraction",
                distribution="uniform",
                range_fraction=0.3,
            )
        ]

        result = run_monte_carlo(
            scenario=sample_scenario,
            strategy=sample_strategy,
            distributions=distributions,
            estimator=estimate_outcome,
            num_simulations=200,
            seed=42,
        )

        # Percentiles should be ordered: p5 <= p25 <= median <= p75 <= p95
        assert result.savings_p5 <= result.savings_p25
        assert result.savings_p25 <= result.savings_median
        assert result.savings_median <= result.savings_p75
        assert result.savings_p75 <= result.savings_p95

    def test_multiple_parameters(self, sample_scenario, sample_strategy):
        """Test varying multiple parameters simultaneously."""
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

        result = run_monte_carlo(
            scenario=sample_scenario,
            strategy=sample_strategy,
            distributions=distributions,
            estimator=estimate_outcome,
            num_simulations=100,
            seed=42,
        )

        assert result.num_simulations == 100
        # Standard deviation should be larger with multiple varying params
        assert result.savings_std >= 0

    def test_scenario_parameter_variation(self, sample_scenario, sample_strategy):
        """Test varying scenario parameters."""
        distributions = [
            ParameterDistribution(
                param_name="requests_per_day",
                distribution="uniform",
                range_fraction=0.2,
            )
        ]

        result = run_monte_carlo(
            scenario=sample_scenario,
            strategy=sample_strategy,
            distributions=distributions,
            estimator=estimate_outcome,
            num_simulations=100,
            seed=42,
        )

        assert result.num_simulations == 100
        # All outcomes should have valid savings values
        assert all(
            isinstance(o.annual_total_savings_usd, (int, float)) for o in result.all_outcomes
        )

    def test_probability_calculations(self, sample_scenario, sample_strategy):
        """Test probability calculations."""
        distributions = [
            ParameterDistribution(
                param_name="memory_reduction_fraction",
                distribution="uniform",
                range_fraction=0.5,  # Wide range to get varied outcomes
            )
        ]

        result = run_monte_carlo(
            scenario=sample_scenario,
            strategy=sample_strategy,
            distributions=distributions,
            estimator=estimate_outcome,
            num_simulations=100,
            seed=42,
        )

        # Probabilities should be in valid range
        assert 0 <= result.probability_positive_roi <= 1
        assert 0 <= result.probability_break_even_within_1yr <= 1
        assert 0 <= result.probability_break_even_within_2yr <= 1

        # P(break_even <= 1yr) <= P(break_even <= 2yr)
        assert result.probability_break_even_within_1yr <= result.probability_break_even_within_2yr


class TestROICalculatorMonteCarlo:
    """Tests for Monte Carlo integration with ROICalculator."""

    def test_monte_carlo_analysis(self, sample_scenario, sample_strategy):
        """Test monte_carlo_analysis method."""
        calculator = ROICalculator()
        calculator.register_scenario(sample_scenario)
        calculator.register_strategy(sample_strategy)

        distributions = [
            ParameterDistribution(
                param_name="memory_reduction_fraction",
                distribution="normal",
                std_dev=0.1,
            )
        ]

        result = calculator.monte_carlo_analysis(
            scenario_name="test-scenario",
            strategy_name="test-strategy",
            distributions=distributions,
            num_simulations=50,
            seed=42,
        )

        assert isinstance(result, MonteCarloResult)
        assert result.scenario_name == "test-scenario"
        assert result.strategy_name == "test-strategy"

    def test_unregistered_scenario_raises(self, sample_strategy):
        """Test error for unregistered scenario."""
        calculator = ROICalculator()
        calculator.register_strategy(sample_strategy)

        with pytest.raises(KeyError, match="Scenario 'missing' not found"):
            calculator.monte_carlo_analysis(
                scenario_name="missing",
                strategy_name="test-strategy",
                distributions=[],
            )

    def test_unregistered_strategy_raises(self, sample_scenario):
        """Test error for unregistered strategy."""
        calculator = ROICalculator()
        calculator.register_scenario(sample_scenario)

        with pytest.raises(KeyError, match="Strategy 'missing' not found"):
            calculator.monte_carlo_analysis(
                scenario_name="test-scenario",
                strategy_name="missing",
                distributions=[],
            )
