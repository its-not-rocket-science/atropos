"""Calculation engine for estimating optimization outcomes."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from .models import QualityRisk

from .models import DeploymentScenario, OptimizationOutcome, OptimizationStrategy

DEFAULT_GRID_CO2E_KG_PER_KWH = 0.35
DEFAULT_HARDWARE_SAVINGS_CORRELATION = 0.8


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
    if scenario.annual_hardware_cost_usd < 0:
        raise ValueError(
            "annual_hardware_cost_usd must be non-negative, got "
            f"{scenario.annual_hardware_cost_usd}"
        )
    if scenario.one_time_project_cost_usd < 0:
        raise ValueError(
            "one_time_project_cost_usd must be non-negative, "
            f"got {scenario.one_time_project_cost_usd}"
        )


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

    baseline_throughput = scenario.throughput_toks_per_sec
    optimized_throughput = baseline_throughput * (1 + strategy.throughput_improvement_fraction)

    baseline_latency_factor = 1.0
    optimized_latency_factor = baseline_throughput / optimized_throughput

    baseline_power = scenario.power_watts
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

    baseline_total_cost = baseline_annual_energy_cost + scenario.annual_hardware_cost_usd

    hardware_cost_savings_fraction = (
        strategy.memory_reduction_fraction * hardware_savings_correlation
    )
    optimized_hardware_cost = scenario.annual_hardware_cost_usd * (
        1 - hardware_cost_savings_fraction
    )
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
