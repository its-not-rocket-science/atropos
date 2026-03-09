"""Built-in scenario and strategy presets."""

from __future__ import annotations

from .models import DeploymentScenario, OptimizationStrategy

SCENARIOS: dict[str, DeploymentScenario] = {
    "frontier-assistant": DeploymentScenario(
        name="frontier-assistant",
        parameters_b=120.0,
        memory_gb=80.0,
        throughput_toks_per_sec=22.0,
        power_watts=900.0,
        requests_per_day=250000,
        tokens_per_request=1800,
        electricity_cost_per_kwh=0.12,
        annual_hardware_cost_usd=280000.0,
        one_time_project_cost_usd=350000.0,
    ),
    "medium-coder": DeploymentScenario(
        name="medium-coder",
        parameters_b=34.0,
        memory_gb=14.0,
        throughput_toks_per_sec=40.0,
        power_watts=320.0,
        requests_per_day=50000,
        tokens_per_request=1200,
        electricity_cost_per_kwh=0.15,
        annual_hardware_cost_usd=24000.0,
        one_time_project_cost_usd=27000.0,
    ),
    "edge-coder": DeploymentScenario(
        name="edge-coder",
        parameters_b=7.0,
        memory_gb=4.5,
        throughput_toks_per_sec=18.0,
        power_watts=75.0,
        requests_per_day=8000,
        tokens_per_request=600,
        electricity_cost_per_kwh=0.20,
        annual_hardware_cost_usd=2800.0,
        one_time_project_cost_usd=12000.0,
    ),
}

STRATEGIES: dict[str, OptimizationStrategy] = {
    "mild_pruning": OptimizationStrategy(
        name="mild_pruning",
        parameter_reduction_fraction=0.15,
        memory_reduction_fraction=0.10,
        throughput_improvement_fraction=0.08,
        power_reduction_fraction=0.06,
        quality_risk="low",
    ),
    "structured_pruning": OptimizationStrategy(
        name="structured_pruning",
        parameter_reduction_fraction=0.30,
        memory_reduction_fraction=0.22,
        throughput_improvement_fraction=0.20,
        power_reduction_fraction=0.14,
        quality_risk="medium",
    ),
    "hardware_aware_pruning": OptimizationStrategy(
        name="hardware_aware_pruning",
        parameter_reduction_fraction=0.35,
        memory_reduction_fraction=0.25,
        throughput_improvement_fraction=0.30,
        power_reduction_fraction=0.18,
        quality_risk="medium",
    ),
}

QUANTIZATION_BONUS = OptimizationStrategy(
    name="quantization_bonus",
    parameter_reduction_fraction=0.0,
    memory_reduction_fraction=0.3846153846,
    throughput_improvement_fraction=0.4666666667,
    power_reduction_fraction=0.09,
    quality_risk="low",
)
