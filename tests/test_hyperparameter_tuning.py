"""Tests for hyperparameter tuning functionality."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from atropos.models import DeploymentScenario
from atropos.tuning import (
    HyperparameterTuner,
    ModelArchitecture,
    ModelCharacteristics,
    TuningConstraints,
    TuningResult,
)


def make_scenario() -> DeploymentScenario:
    """Create a test deployment scenario."""
    return DeploymentScenario(
        name="test-scenario",
        parameters_b=7.0,  # 7B parameters
        memory_gb=14.0,
        throughput_toks_per_sec=25.0,
        power_watts=300.0,
        requests_per_day=50000,
        tokens_per_request=1500,
        electricity_cost_per_kwh=0.12,
        annual_hardware_cost_usd=5000.0,
        one_time_project_cost_usd=2000.0,
    )


def test_hyperparameter_tuner_initialization() -> None:
    """Test that tuner initializes with scenario."""
    scenario = make_scenario()
    tuner = HyperparameterTuner(scenario)

    assert tuner.scenario == scenario
    assert tuner.constraints is not None
    assert tuner.model_chars is not None
    assert tuner.model_chars.parameter_count_b == scenario.parameters_b


def test_hyperparameter_tuner_with_custom_model_characteristics() -> None:
    """Test tuner with custom model characteristics."""
    scenario = make_scenario()
    model_chars = ModelCharacteristics(
        parameter_count_b=13.0,
        architecture=ModelArchitecture.LLAMA,
        layer_types=["attention", "mlp", "layer_norm"],
        has_sparse_support=True,
    )

    tuner = HyperparameterTuner(scenario, model_characteristics=model_chars)

    assert tuner.model_chars == model_chars
    assert tuner.model_chars.architecture == ModelArchitecture.LLAMA
    assert tuner.model_chars.has_sparse_support is True


def test_hyperparameter_tuner_with_constraints() -> None:
    """Test tuner with deployment constraints."""
    scenario = make_scenario()
    constraints = TuningConstraints(
        max_memory_gb=10.0,
        min_throughput_toks_per_sec=30.0,
        max_latency_ms_per_request=100.0,
        max_power_watts=250.0,
        use_quantization=True,
    )

    tuner = HyperparameterTuner(scenario, constraints=constraints)

    assert tuner.constraints == constraints
    assert tuner.constraints.max_memory_gb == 10.0
    assert tuner.constraints.use_quantization is True


def test_infer_model_characteristics() -> None:
    """Test model characteristic inference."""
    scenario = make_scenario()
    tuner = HyperparameterTuner(scenario)
    chars = tuner._infer_model_characteristics()

    assert chars.parameter_count_b == scenario.parameters_b
    # 7B params should be classified as LLAMA (since >1.5 and <=13)
    assert chars.architecture == ModelArchitecture.LLAMA
    assert chars.layer_types  # Should have default layer types
    assert not chars.has_sparse_support  # Default assumption


def test_analyze_model() -> None:
    """Test model analysis."""
    scenario = make_scenario()
    tuner = HyperparameterTuner(scenario)
    analysis = tuner._analyze_model()

    assert "parameter_count_b" in analysis
    assert "architecture" in analysis
    assert "suitability_score" in analysis
    assert "is_known_architecture" in analysis
    assert "has_sparse_support" in analysis
    assert "layer_count_estimate" in analysis

    # 7B parameters should have suitability ~0.7 (7/10)
    assert 0.6 < analysis["suitability_score"] < 0.8
    assert analysis["is_known_architecture"] is True  # LLAMA is known


def test_determine_objectives_with_constraints() -> None:
    """Test objective determination with constraints."""
    scenario = make_scenario()
    constraints = TuningConstraints(
        max_memory_gb=10.0,  # Current is 14GB, target reduction ~28.6%
        min_throughput_toks_per_sec=30.0,  # Current is 25, target improvement 20%
    )
    tuner = HyperparameterTuner(scenario, constraints=constraints)

    model_analysis = tuner._analyze_model()
    objectives = tuner._determine_objectives(model_analysis)

    assert "memory_reduction" in objectives
    assert "throughput_improvement" in objectives
    assert objectives["memory_reduction"] == pytest.approx(0.286, rel=0.01)
    assert objectives["throughput_improvement"] == pytest.approx(0.2, rel=0.01)


def test_determine_objectives_without_constraints() -> None:
    """Test objective determination without constraints."""
    scenario = make_scenario()
    tuner = HyperparameterTuner(scenario)
    model_analysis = tuner._analyze_model()
    objectives = tuner._determine_objectives(model_analysis)

    # Should have balanced objectives based on model suitability
    assert "memory_reduction" in objectives
    assert "throughput_improvement" in objectives
    assert "power_reduction" in objectives
    assert all(0 <= v <= 1.0 for v in objectives.values())


def test_calculate_quality_risk() -> None:
    """Test quality risk calculation."""
    scenario = make_scenario()
    tuner = HyperparameterTuner(scenario)

    # Low parameter reduction -> low risk
    low_risk = tuner._calculate_quality_risk(0.1, 0.08, 0.05)
    assert 0 <= low_risk <= 0.3

    # High parameter reduction -> higher risk
    high_risk = tuner._calculate_quality_risk(0.4, 0.25, 0.35)
    assert high_risk > low_risk

    # Memory reduction lagging increases risk
    risk_lagging = tuner._calculate_quality_risk(0.3, 0.1, 0.1)  # mem_red < 0.7*param_red
    risk_proportional = tuner._calculate_quality_risk(0.3, 0.25, 0.1)
    assert risk_lagging > risk_proportional


def test_determine_risk_level() -> None:
    """Test risk level determination."""
    scenario = make_scenario()
    tuner = HyperparameterTuner(scenario)

    assert tuner._determine_risk_level(0.1, 0.08, 0.05) == "low"
    assert tuner._determine_risk_level(0.5, 0.4, 0.2) == "medium"  # risk_score = 0.3
    assert tuner._determine_risk_level(0.5, 0.3, 0.4) == "high"  # risk_score = 0.6


def test_enhance_framework_recommendation() -> None:
    """Test framework recommendation enhancement."""
    scenario = make_scenario()
    tuner = HyperparameterTuner(scenario)

    # Mock model characteristics for LLAMA
    tuner.model_chars = ModelCharacteristics(
        parameter_count_b=7.0,
        architecture=ModelArchitecture.LLAMA,
        layer_types=[],
        has_sparse_support=True,
    )

    from atropos.models import OptimizationStrategy

    strategy = OptimizationStrategy(
        name="test",
        parameter_reduction_fraction=0.25,
        memory_reduction_fraction=0.2,
        throughput_improvement_fraction=0.15,
        power_reduction_fraction=0.1,
        quality_risk="medium",
    )

    # LLAMA with sparse support should recommend wanda-patched
    framework = tuner._enhance_framework_recommendation("magnitude_pruning", strategy)
    assert framework == "wanda-patched"

    # Without sparse support should recommend wanda
    tuner.model_chars.has_sparse_support = False
    framework = tuner._enhance_framework_recommendation("magnitude_pruning", strategy)
    assert framework == "wanda"

    # GPT2 with high reduction should recommend sparsegpt
    tuner.model_chars.architecture = ModelArchitecture.GPT2
    tuner.model_chars.has_sparse_support = False
    # strategy already has parameter_reduction_fraction = 0.25 (>= 0.2)
    framework = tuner._enhance_framework_recommendation("magnitude_pruning", strategy)
    assert framework == "sparsegpt"

    # Prefer fast pruning
    tuner.model_chars.architecture = ModelArchitecture.UNKNOWN
    tuner.model_chars.has_sparse_support = False
    tuner.constraints.prefer_fast_pruning = True
    framework = tuner._enhance_framework_recommendation("slow_framework", strategy)
    assert framework == "wanda"  # Should switch to fast framework


def test_estimate_metrics() -> None:
    """Test metric estimation."""
    scenario = make_scenario()
    tuner = HyperparameterTuner(scenario)

    from atropos.models import OptimizationStrategy

    strategy = OptimizationStrategy(
        name="test",
        parameter_reduction_fraction=0.2,
        memory_reduction_fraction=0.25,
        throughput_improvement_fraction=0.3,
        power_reduction_fraction=0.15,
        quality_risk="low",
    )

    metrics = tuner._estimate_metrics(strategy)

    assert metrics["memory_gb"] == scenario.memory_gb * (1 - 0.25)
    assert metrics["throughput_toks_per_sec"] == scenario.throughput_toks_per_sec * (1 + 0.3)
    assert metrics["power_watts"] == scenario.power_watts * (1 - 0.15)
    # Latency should improve with throughput
    expected_latency = (scenario.tokens_per_request / metrics["throughput_toks_per_sec"]) * 1000
    assert metrics["latency_ms_per_request"] == pytest.approx(expected_latency)


def test_tune_with_heuristics() -> None:
    """Test tuning with heuristic optimization."""
    scenario = make_scenario()
    tuner = HyperparameterTuner(scenario)

    # Mock _optimize_parameters to use heuristics
    with patch.object(tuner, "_optimize_parameters") as mock_optimize:
        from atropos.models import OptimizationStrategy

        mock_strategy = OptimizationStrategy(
            name="test",
            parameter_reduction_fraction=0.2,
            memory_reduction_fraction=0.25,
            throughput_improvement_fraction=0.3,
            power_reduction_fraction=0.15,
            quality_risk="low",
        )
        mock_optimize.return_value = mock_strategy

        result = tuner.tune()

        assert isinstance(result, TuningResult)
        assert result.strategy == mock_strategy
        assert result.quality_risk == "low"
        assert "architecture" in result.tuning_metadata


@pytest.mark.skipif(
    __import__("importlib.util").util.find_spec("skopt") is None,
    reason="Requires scikit-optimize for Bayesian optimization",
)
def test_tune_with_bayesian() -> None:
    """Test tuning with Bayesian optimization (if available)."""
    scenario = make_scenario()
    tuner = HyperparameterTuner(scenario)

    # Mock import check to force Bayesian optimization
    with patch("importlib.util.find_spec", return_value=True):
        # Mock the actual Bayesian optimization to avoid long computation
        with patch.object(tuner, "_optimize_with_bayesian") as mock_bayesian:
            from atropos.models import OptimizationStrategy

            mock_strategy = OptimizationStrategy(
                name="test",
                parameter_reduction_fraction=0.18,
                memory_reduction_fraction=0.22,
                throughput_improvement_fraction=0.25,
                power_reduction_fraction=0.12,
                quality_risk="medium",
            )
            mock_bayesian.return_value = mock_strategy

            result = tuner.tune()

            assert isinstance(result, TuningResult)
            assert result.strategy == mock_strategy
            mock_bayesian.assert_called_once()


def test_tune_with_quantization() -> None:
    """Test tuning with quantization requested."""
    scenario = make_scenario()
    constraints = TuningConstraints(use_quantization=True)
    tuner = HyperparameterTuner(scenario, constraints=constraints)

    # Mock the optimization to return a simple strategy
    with patch.object(tuner, "_optimize_parameters") as mock_optimize:
        from atropos.models import OptimizationStrategy

        base_strategy = OptimizationStrategy(
            name="base",
            parameter_reduction_fraction=0.2,
            memory_reduction_fraction=0.25,
            throughput_improvement_fraction=0.3,
            power_reduction_fraction=0.15,
            quality_risk="low",
        )
        mock_optimize.return_value = base_strategy

        # Mock combine_strategies to verify quantization is applied
        with patch("atropos.calculations.combine_strategies") as mock_combine:
            mock_combine.return_value = base_strategy  # Same for simplicity
            result = tuner.tune()

            # Should have called combine_strategies with QUANTIZATION_BONUS
            mock_combine.assert_called_once()
            assert result.strategy == base_strategy


def test_tuning_result_structure() -> None:
    """Test TuningResult dataclass structure."""
    from atropos.models import OptimizationStrategy

    strategy = OptimizationStrategy(
        name="test",
        parameter_reduction_fraction=0.2,
        memory_reduction_fraction=0.25,
        throughput_improvement_fraction=0.3,
        power_reduction_fraction=0.15,
        quality_risk="low",
    )

    result = TuningResult(
        strategy=strategy,
        recommended_framework="wanda",
        quality_risk="low",
        expected_memory_gb=10.5,
        expected_throughput_toks_per_sec=32.5,
        expected_latency_ms_per_request=46.2,
        expected_power_watts=255.0,
        tuning_metadata={"test": "value"},
    )

    assert result.strategy == strategy
    assert result.recommended_framework == "wanda"
    assert result.expected_memory_gb == 10.5
    assert "test" in result.tuning_metadata
