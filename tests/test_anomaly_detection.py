"""Tests for cost anomaly detection."""

import pytest

from atropos.models import DeploymentScenario, OptimizationOutcome, OptimizationStrategy
from atropos.validation.anomaly_detection import (
    Anomaly,
    AnomalyDetectionResult,
    CostAnomalyDetector,
    detect_anomalies,
)


def make_scenario() -> DeploymentScenario:
    """Create a test deployment scenario."""
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


def make_strategy() -> OptimizationStrategy:
    """Create a test optimization strategy."""
    return OptimizationStrategy(
        name="test-strategy",
        parameter_reduction_fraction=0.2,
        memory_reduction_fraction=0.2,
        throughput_improvement_fraction=0.2,
        power_reduction_fraction=0.1,
        quality_risk="low",
    )


def make_outcome() -> OptimizationOutcome:
    """Create a test optimization outcome."""
    scenario = make_scenario()
    strategy = make_strategy()
    # Use the calculations module to compute a realistic outcome
    from atropos.calculations import estimate_outcome

    return estimate_outcome(scenario, strategy)


def test_anomaly_initialization() -> None:
    """Test that Anomaly class initializes correctly."""
    anomaly = Anomaly(
        metric="annual_savings_usd",
        value=10000.0,
        baseline_mean=5000.0,
        baseline_std=1000.0,
        z_score=5.0,
        threshold=3.0,
    )
    assert anomaly.metric == "annual_savings_usd"
    assert anomaly.value == 10000.0
    assert anomaly.z_score == 5.0
    assert anomaly.is_anomaly is True
    assert anomaly.description != ""


def test_anomaly_detection_result() -> None:
    """Test AnomalyDetectionResult initialization and serialization."""
    outcome = make_outcome()
    anomalies = [
        Anomaly(
            metric="annual_savings_usd",
            value=10000.0,
            baseline_mean=5000.0,
            baseline_std=1000.0,
            z_score=5.0,
        )
    ]
    result = AnomalyDetectionResult(
        outcome=outcome,
        anomalies=anomalies,
        threshold=3.0,
    )
    assert result.has_anomalies is True
    assert len(result.anomalies) == 1
    # Test dictionary serialization
    data = result.to_dict()
    assert data["scenario_name"] == outcome.scenario_name
    assert data["has_anomalies"] is True
    assert len(data["anomalies"]) == 1
    # Test markdown generation (should not crash)
    md = result.to_markdown()
    assert isinstance(md, str)
    assert "Anomaly Detection Report" in md


def test_cost_anomaly_detector_default_baselines() -> None:
    """Test that detector initializes with default baselines."""
    detector = CostAnomalyDetector()
    assert detector.threshold == 3.0
    assert "annual_savings_usd" in detector.baselines
    assert "mean" in detector.baselines["annual_savings_usd"]
    assert "std" in detector.baselines["annual_savings_usd"]


def test_detect_no_anomalies_with_normal_value() -> None:
    """Test that no anomalies are detected when values are within threshold."""
    detector = CostAnomalyDetector()
    # Create an outcome with values close to default baseline means
    outcome = make_outcome()
    # Override outcome values to match baseline means
    # This is a bit hacky, but we can directly modify the outcome dataclass fields
    # Since dataclasses are frozen, we need to replace
    import dataclasses

    outcome = dataclasses.replace(
        outcome,
        annual_total_savings_usd=detector.baselines["annual_savings_usd"]["mean"],
        break_even_years=detector.baselines["break_even_months"]["mean"] / 12,
        annual_co2e_savings_kg=detector.baselines["total_co2e_saved_kg"]["mean"],
    )
    result = detector.detect(outcome)
    assert result.has_anomalies is False
    assert len(result.anomalies) == 0


def test_detect_anomalies_with_extreme_value() -> None:
    """Test that anomalies are detected when values exceed threshold."""
    detector = CostAnomalyDetector(threshold=2.0)  # Lower threshold for easier detection
    # Create an outcome with extremely high savings (far from baseline mean)
    outcome = make_outcome()
    import dataclasses

    # Set savings to 10x baseline mean to guarantee anomaly
    baseline_mean = detector.baselines["annual_savings_usd"]["mean"]
    baseline_std = detector.baselines["annual_savings_usd"]["std"]
    extreme_value = baseline_mean + 10 * baseline_std
    outcome = dataclasses.replace(
        outcome,
        annual_total_savings_usd=extreme_value,
    )
    result = detector.detect(outcome)
    # Should have at least one anomaly (annual_savings_usd)
    assert result.has_anomalies is True
    assert len(result.anomalies) >= 1
    assert any(a.metric == "annual_savings_usd" for a in result.anomalies)


def test_baseline_computation_from_historical_data() -> None:
    """Test that detector computes baselines from historical outcomes."""
    # Create a few historical outcomes with known values
    outcomes = []
    for i in range(5):
        scenario = make_scenario()
        strategy = make_strategy()
        from atropos.calculations import estimate_outcome

        outcome = estimate_outcome(scenario, strategy)
        # Modify savings to create a known distribution
        import dataclasses

        outcome = dataclasses.replace(
            outcome,
            annual_total_savings_usd=1000.0 * (i + 1),  # 1000, 2000, 3000, 4000, 5000
        )
        outcomes.append(outcome)

    detector = CostAnomalyDetector(baseline_data=outcomes)
    # Check that baselines were computed (not defaults)
    # The mean should be 3000, std should be sqrt(2.5)*1000 ≈ 1581
    baseline = detector.baselines["annual_savings_usd"]
    assert abs(baseline["mean"] - 3000.0) < 0.1
    assert abs(baseline["std"] - 1581.13883) < 0.1


def test_detect_anomalies_convenience_function() -> None:
    """Test the top-level detect_anomalies convenience function."""
    outcome = make_outcome()
    result = detect_anomalies(outcome, threshold=3.0)
    assert isinstance(result, AnomalyDetectionResult)
    # Should not crash
    result.to_dict()
    result.to_markdown()


def test_anomaly_detector_with_empty_baseline_data() -> None:
    """Test that detector falls back to defaults when baseline_data is empty."""
    detector = CostAnomalyDetector(baseline_data=[])
    # Should still have default baselines
    assert "annual_savings_usd" in detector.baselines
    assert detector.baselines["annual_savings_usd"]["mean"] == 5000.0  # default mean


if __name__ == "__main__":
    pytest.main([__file__])
