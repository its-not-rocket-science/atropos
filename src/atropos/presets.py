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
        one_time_project_cost_usd=350000.0,
        gpu_tier="H100_80GB",
        gpu_count=2,
        batch_size=8,
        pricing_model="cloud",
        utilization=0.85,
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
        one_time_project_cost_usd=27000.0,
        gpu_tier="A100_40GB",
        gpu_count=1,
        batch_size=4,
        pricing_model="cloud",
        utilization=0.80,
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
        one_time_project_cost_usd=12000.0,
        gpu_tier="L4",
        gpu_count=1,
        batch_size=2,
        pricing_model="cloud",
        utilization=0.70,
    ),
}

STRATEGIES: dict[str, OptimizationStrategy] = {
    # Structured pruning (LLM-Pruner, Wanda with channel removal)
    # These achieve actual memory savings by removing parameters
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
    # Unstructured pruning (PyTorch magnitude-based)
    # Limited memory savings without sparse inference support
    "unstructured_mild": OptimizationStrategy(
        name="unstructured_mild",
        parameter_reduction_fraction=0.10,
        memory_reduction_fraction=0.03,  # ~3% actual from gpt2 testing
        throughput_improvement_fraction=0.02,
        power_reduction_fraction=0.01,
        quality_risk="low",
    ),
    "unstructured_moderate": OptimizationStrategy(
        name="unstructured_moderate",
        parameter_reduction_fraction=0.22,
        memory_reduction_fraction=0.07,  # ~7% actual from gpt2 testing
        throughput_improvement_fraction=0.04,
        power_reduction_fraction=0.02,
        quality_risk="medium",
    ),
    # Architecture-specific strategies (based on pruning exercise results)
    "opt_magnitude_pruning": OptimizationStrategy(
        name="opt_magnitude_pruning",
        parameter_reduction_fraction=0.22,
        memory_reduction_fraction=0.11,  # OPT achieved ~11% memory reduction
        throughput_improvement_fraction=0.08,
        power_reduction_fraction=0.05,
        quality_risk="medium",
    ),
    # Framework-specific pruning strategies (based on framework comparison)
    "magnitude_pruning": OptimizationStrategy(
        name="magnitude_pruning",
        parameter_reduction_fraction=0.10,
        memory_reduction_fraction=0.03,  # ~3% actual from GPT2 testing (unstructured)
        throughput_improvement_fraction=0.02,
        power_reduction_fraction=0.01,
        quality_risk="low",
    ),
    "wanda_pruning": OptimizationStrategy(
        name="wanda_pruning",
        parameter_reduction_fraction=0.10,
        memory_reduction_fraction=0.09,  # ~9% actual from GPT2 testing
        throughput_improvement_fraction=0.08,
        power_reduction_fraction=0.05,
        quality_risk="medium",
    ),
    "sparsegpt_pruning": OptimizationStrategy(
        name="sparsegpt_pruning",
        parameter_reduction_fraction=0.10,
        memory_reduction_fraction=0.095,  # ~9.5% actual from GPT2 testing
        throughput_improvement_fraction=0.10,
        power_reduction_fraction=0.06,
        quality_risk="low",
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
