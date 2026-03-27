"""Calculation engine for estimating optimization outcomes."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from .models import QualityRisk

from .models import DeploymentScenario, OptimizationOutcome, OptimizationStrategy

DEFAULT_GRID_CO2E_KG_PER_KWH = 0.35
DEFAULT_HARDWARE_SAVINGS_CORRELATION = 0.8
DEFAULT_BATCHING_EFFICIENCY = 0.85  # Sub-linear scaling for batching
DEFAULT_DATA_PARALLEL_SCALING_EFFICIENCY = 0.8  # Multi-GPU scaling efficiency for data parallelism


def _validate_fraction(value: float, name: str) -> None:
    """Validate that a value is a valid fraction in [0, 1)."""
    if not 0.0 <= value < 1.0:
        raise ValueError(f"{name} must be in the range [0, 1). Got {value}.")


def _validate_scenario(scenario: DeploymentScenario) -> None:
    """Validate scenario inputs for reasonable ranges."""
    if scenario.parameters_b <= 0:
        raise ValueError(f"parameters_b must be positive, got {scenario.parameters_b}")
    if scenario.memory_gb <= 0:
        raise ValueError(f"memory_gb must be positive, got {scenario.memory_gb}")
    if scenario.throughput_toks_per_sec <= 0:
        raise ValueError(
            f"throughput_toks_per_sec must be positive, got {scenario.throughput_toks_per_sec}"
        )
    if scenario.power_watts <= 0:
        raise ValueError(f"power_watts must be positive, got {scenario.power_watts}")
    if scenario.requests_per_day <= 0:
        raise ValueError(f"requests_per_day must be positive, got {scenario.requests_per_day}")
    if scenario.tokens_per_request <= 0:
        raise ValueError(f"tokens_per_request must be positive, got {scenario.tokens_per_request}")
    if scenario.electricity_cost_per_kwh <= 0:
        raise ValueError(
            f"electricity_cost_per_kwh must be positive, got {scenario.electricity_cost_per_kwh}"
        )
    if scenario.batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {scenario.batch_size}")
    if not 0.0 <= scenario.utilization <= 1.0:
        raise ValueError(f"utilization must be in [0, 1], got {scenario.utilization}")
    # Validate hardware cost source
    has_gpu_tier = scenario.gpu_tier is not None
    has_annual_cost = scenario.annual_hardware_cost_usd is not None
    if not has_gpu_tier and not has_annual_cost:
        raise ValueError(
            "Either gpu_tier or annual_hardware_cost_usd must be provided for cost modeling"
        )


def _compute_batching_throughput_multiplier(
    batch_size: int,
    efficiency: float = DEFAULT_BATCHING_EFFICIENCY,
) -> float:
    """Compute throughput multiplier from batching.

    Batching improves throughput sub-linearly due to memory bandwidth limits.
    Formula: multiplier = batch_size^efficiency

    Args:
        batch_size: Number of requests processed together.
        efficiency: Batching efficiency factor (0-1, higher = better scaling).

    Returns:
        Throughput multiplier relative to batch_size=1.
    """
    if batch_size <= 1:
        return 1.0
    result: float = batch_size**efficiency
    return result


def _compute_multi_gpu_throughput_scaling(
    gpu_count: int | None,
    parallel_strategy: str = "data",
    scaling_efficiency: float = DEFAULT_DATA_PARALLEL_SCALING_EFFICIENCY,
) -> float:
    """Compute throughput scaling factor for multi-GPU deployments.

    Throughput scales sub-linearly with GPU count due to communication overhead
    and memory bandwidth limitations.

    Args:
        gpu_count: Number of GPUs (None or 1 means single GPU, scaling = 1.0).
        parallel_strategy: Parallelization strategy ("data", "layer", "model").
            Only "data" parallelism is currently supported.
        scaling_efficiency: Scaling efficiency factor (0-1, higher = better scaling).

    Returns:
        Throughput multiplier relative to single GPU.
    """
    if gpu_count is None or gpu_count <= 1:
        return 1.0

    # After this point, gpu_count is guaranteed to be int > 1
    assert isinstance(gpu_count, int)

    if parallel_strategy == "data":
        # Data parallelism: throughput scales with GPU count but with efficiency loss
        # Formula: scaling = gpu_count^scaling_efficiency
        # scaling_efficiency=0.8 means 2 GPUs -> 2^0.8 ≈ 1.74x (87% efficiency)
        result: float = gpu_count**scaling_efficiency
        return result
    else:
        # Other parallel strategies not yet implemented
        # For now, assume same scaling as data parallelism
        result2: float = gpu_count**scaling_efficiency
        return result2


def _compute_hardware_cost(scenario: DeploymentScenario) -> float:
    """Compute annual hardware cost from GPU tier or fallback.

    Args:
        scenario: Deployment scenario with GPU tier or annual cost.

    Returns:
        Annual hardware cost in USD.
    """
    # Use explicit annual cost if GPU tier not specified
    if scenario.gpu_tier is None:
        return scenario.annual_hardware_cost_usd or 0.0

    # Import here to avoid circular imports
    from .hardware import estimate_gpu_count, get_gpu_tier

    tier = get_gpu_tier(scenario.gpu_tier)
    gpu_count = scenario.gpu_count or estimate_gpu_count(scenario.memory_gb, scenario.gpu_tier)

    return tier.annual_cost(
        gpu_count=gpu_count,
        utilization=scenario.utilization,
        pricing_model=scenario.pricing_model,
    )


def _compute_power_from_gpu_tier(scenario: DeploymentScenario) -> float:
    """Compute power draw from GPU tier if available.

    Args:
        scenario: Deployment scenario.

    Returns:
        Power in watts (falls back to scenario.power_watts if no GPU tier).
    """
    if scenario.gpu_tier is None:
        return scenario.power_watts

    from .hardware import estimate_gpu_count, get_gpu_tier

    tier = get_gpu_tier(scenario.gpu_tier)
    gpu_count = scenario.gpu_count or estimate_gpu_count(scenario.memory_gb, scenario.gpu_tier)

    # Power scales with utilization (idle power ~20% of max)
    idle_fraction = 0.2
    active_fraction = scenario.utilization
    power_multiplier = idle_fraction + (1 - idle_fraction) * active_fraction

    return tier.typical_power_w * gpu_count * power_multiplier


def combine_strategies(
    primary: OptimizationStrategy, secondary: OptimizationStrategy
) -> OptimizationStrategy:
    """Combine two strategies multiplicatively.

    Reductions are combined as 1 - (1-a)*(1-b) for multiplicative effect.
    Throughput improvements multiply: (1+a)*(1+b) - 1.
    Quality risk takes the maximum of the two strategies.
    """
    for fraction, label in (
        (primary.parameter_reduction_fraction, "primary.parameter_reduction_fraction"),
        (primary.memory_reduction_fraction, "primary.memory_reduction_fraction"),
        (primary.power_reduction_fraction, "primary.power_reduction_fraction"),
        (secondary.parameter_reduction_fraction, "secondary.parameter_reduction_fraction"),
        (secondary.memory_reduction_fraction, "secondary.memory_reduction_fraction"),
        (secondary.power_reduction_fraction, "secondary.power_reduction_fraction"),
    ):
        _validate_fraction(fraction, label)

    if primary.throughput_improvement_fraction < 0 or secondary.throughput_improvement_fraction < 0:
        raise ValueError("throughput_improvement_fraction must be >= 0.")

    parameter_reduction = 1 - (
        (1 - primary.parameter_reduction_fraction) * (1 - secondary.parameter_reduction_fraction)
    )
    memory_reduction = 1 - (
        (1 - primary.memory_reduction_fraction) * (1 - secondary.memory_reduction_fraction)
    )
    power_reduction = 1 - (
        (1 - primary.power_reduction_fraction) * (1 - secondary.power_reduction_fraction)
    )
    throughput_improvement = (1 + primary.throughput_improvement_fraction) * (
        1 + secondary.throughput_improvement_fraction
    ) - 1

    risk_rank = {"low": 1, "medium": 2, "high": 3}
    reverse_risk = {1: "low", 2: "medium", 3: "high"}
    quality_risk: QualityRisk = cast(
        "QualityRisk",
        reverse_risk[max(risk_rank[primary.quality_risk], risk_rank[secondary.quality_risk])],
    )

    return OptimizationStrategy(
        name=f"{primary.name} + {secondary.name}",
        parameter_reduction_fraction=parameter_reduction,
        memory_reduction_fraction=memory_reduction,
        throughput_improvement_fraction=throughput_improvement,
        power_reduction_fraction=power_reduction,
        quality_risk=quality_risk,
    )


def estimate_outcome(
    scenario: DeploymentScenario,
    strategy: OptimizationStrategy,
    grid_co2e_kg_per_kwh: float = DEFAULT_GRID_CO2E_KG_PER_KWH,
    hardware_savings_correlation: float = DEFAULT_HARDWARE_SAVINGS_CORRELATION,
) -> OptimizationOutcome:
    """Estimate outcome of applying a strategy to a deployment scenario.

    Calculates memory, throughput, energy, cost, and CO2e impacts.
    Supports GPU tier-based cost modeling and batching effects.
    Returns complete baseline vs optimized comparison.
    """
    _validate_scenario(scenario)
    _validate_fraction(strategy.parameter_reduction_fraction, "parameter_reduction_fraction")
    _validate_fraction(strategy.memory_reduction_fraction, "memory_reduction_fraction")
    _validate_fraction(strategy.power_reduction_fraction, "power_reduction_fraction")
    if strategy.throughput_improvement_fraction < 0:
        raise ValueError("throughput_improvement_fraction must be >= 0.")
    if not 0.0 <= hardware_savings_correlation <= 1.0:
        raise ValueError(
            "hardware_savings_correlation must be in the range [0, 1]. "
            f"Got {hardware_savings_correlation}."
        )

    baseline_memory_gb = scenario.memory_gb
    optimized_memory_gb = baseline_memory_gb * (1 - strategy.memory_reduction_fraction)

    # Apply batching effects to throughput
    batching_multiplier = _compute_batching_throughput_multiplier(scenario.batch_size)
    gpu_scaling_factor = _compute_multi_gpu_throughput_scaling(
        scenario.gpu_count, scenario.parallel_strategy
    )
    effective_baseline_throughput = (
        scenario.throughput_toks_per_sec * batching_multiplier * gpu_scaling_factor
    )
    baseline_throughput = effective_baseline_throughput
    optimized_throughput = baseline_throughput * (1 + strategy.throughput_improvement_fraction)

    baseline_latency_factor = 1.0
    optimized_latency_factor = baseline_throughput / optimized_throughput

    # Use GPU-tier based power if available
    baseline_power = _compute_power_from_gpu_tier(scenario)
    optimized_power = baseline_power * (1 - strategy.power_reduction_fraction)

    baseline_seconds_per_request = scenario.tokens_per_request / baseline_throughput
    optimized_seconds_per_request = scenario.tokens_per_request / optimized_throughput

    baseline_energy_wh_per_request = (baseline_power * baseline_seconds_per_request) / 3600
    optimized_energy_wh_per_request = (optimized_power * optimized_seconds_per_request) / 3600

    annual_requests = scenario.requests_per_day * 365

    baseline_annual_energy_kwh = baseline_energy_wh_per_request * annual_requests / 1000
    optimized_annual_energy_kwh = optimized_energy_wh_per_request * annual_requests / 1000

    baseline_annual_energy_cost = baseline_annual_energy_kwh * scenario.electricity_cost_per_kwh
    optimized_annual_energy_cost = optimized_annual_energy_kwh * scenario.electricity_cost_per_kwh

    # Compute hardware cost from GPU tier or fallback
    baseline_hardware_cost = _compute_hardware_cost(scenario)
    baseline_total_cost = baseline_annual_energy_cost + baseline_hardware_cost

    hardware_cost_savings_fraction = (
        strategy.memory_reduction_fraction * hardware_savings_correlation
    )
    optimized_hardware_cost = baseline_hardware_cost * (1 - hardware_cost_savings_fraction)
    optimized_total_cost = optimized_annual_energy_cost + optimized_hardware_cost

    annual_total_savings = baseline_total_cost - optimized_total_cost
    annual_energy_savings_kwh = baseline_annual_energy_kwh - optimized_annual_energy_kwh
    annual_co2e_savings_kg = annual_energy_savings_kwh * grid_co2e_kg_per_kwh

    break_even_years = None
    if annual_total_savings > 0:
        break_even_years = scenario.one_time_project_cost_usd / annual_total_savings

    return OptimizationOutcome(
        scenario_name=scenario.name,
        strategy_name=strategy.name,
        baseline_memory_gb=baseline_memory_gb,
        optimized_memory_gb=optimized_memory_gb,
        baseline_throughput_toks_per_sec=baseline_throughput,
        optimized_throughput_toks_per_sec=optimized_throughput,
        baseline_latency_factor=baseline_latency_factor,
        optimized_latency_factor=optimized_latency_factor,
        baseline_power_watts=baseline_power,
        optimized_power_watts=optimized_power,
        baseline_energy_wh_per_request=baseline_energy_wh_per_request,
        optimized_energy_wh_per_request=optimized_energy_wh_per_request,
        baseline_annual_energy_kwh=baseline_annual_energy_kwh,
        optimized_annual_energy_kwh=optimized_annual_energy_kwh,
        baseline_annual_energy_cost_usd=baseline_annual_energy_cost,
        optimized_annual_energy_cost_usd=optimized_annual_energy_cost,
        baseline_annual_total_cost_usd=baseline_total_cost,
        optimized_annual_total_cost_usd=optimized_total_cost,
        annual_total_savings_usd=annual_total_savings,
        annual_energy_savings_kwh=annual_energy_savings_kwh,
        annual_co2e_savings_kg=annual_co2e_savings_kg,
        break_even_years=break_even_years,
        quality_risk=strategy.quality_risk,
    )
