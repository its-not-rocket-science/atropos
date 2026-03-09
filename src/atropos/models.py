"""Core domain models for Atropos."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .hardware import GPUType

QualityRisk = Literal["low", "medium", "high"]


@dataclass(frozen=True)
class DeploymentScenario:
    """Deployment scenario parameters for ROI estimation.

    Attributes:
        name: Scenario identifier.
        parameters_b: Model size in billions of parameters.
        memory_gb: Memory usage in gigabytes.
        throughput_toks_per_sec: Token throughput per second.
        power_watts: Power consumption in watts.
        requests_per_day: Number of requests per day.
        tokens_per_request: Average tokens per request.
        electricity_cost_per_kwh: Electricity cost per kWh.
        one_time_project_cost_usd: One-time optimization project cost in USD.
        gpu_tier: GPU hardware tier for cost modeling.
        gpu_count: Number of GPUs (auto-estimated if None).
        batch_size: Request batch size (1 = no batching).
        pricing_model: "cloud" or "reserved" pricing.
        utilization: GPU utilization factor (0-1).
        annual_hardware_cost_usd: Annual hardware cost (deprecated, use gpu_tier).
    """

    name: str
    parameters_b: float
    memory_gb: float
    throughput_toks_per_sec: float
    power_watts: float
    requests_per_day: int
    tokens_per_request: int
    electricity_cost_per_kwh: float
    one_time_project_cost_usd: float
    gpu_tier: GPUType | None = None
    gpu_count: int | None = None
    batch_size: int = 1
    pricing_model: Literal["cloud", "reserved"] = "cloud"
    utilization: float = 1.0
    annual_hardware_cost_usd: float | None = None


@dataclass(frozen=True)
class OptimizationStrategy:
    """Optimization strategy parameters for ROI estimation.

    Attributes:
        name: Strategy identifier.
        parameter_reduction_fraction: Fraction of parameters removed (0-1).
        memory_reduction_fraction: Fraction of memory saved (0-1).
        throughput_improvement_fraction: Fractional throughput improvement (0+).
        power_reduction_fraction: Fraction of power saved (0-1).
        quality_risk: Estimated quality risk level (low/medium/high).
    """

    name: str
    parameter_reduction_fraction: float
    memory_reduction_fraction: float
    throughput_improvement_fraction: float
    power_reduction_fraction: float
    quality_risk: QualityRisk


@dataclass(frozen=True)
class OptimizationOutcome:
    """Results of applying an optimization strategy to a deployment scenario.

    Contains both baseline and optimized metrics for comparison.

    Attributes:
        scenario_name: Name of the deployment scenario.
        strategy_name: Name of the optimization strategy applied.
        baseline_memory_gb: Memory usage before optimization (GB).
        optimized_memory_gb: Memory usage after optimization (GB).
        baseline_throughput_toks_per_sec: Throughput before optimization (tok/s).
        optimized_throughput_toks_per_sec: Throughput after optimization (tok/s).
        baseline_latency_factor: Latency factor before optimization (1.0).
        optimized_latency_factor: Latency factor after optimization.
        baseline_power_watts: Power draw before optimization (W).
        optimized_power_watts: Power draw after optimization (W).
        baseline_energy_wh_per_request: Energy per request before (Wh).
        optimized_energy_wh_per_request: Energy per request after (Wh).
        baseline_annual_energy_kwh: Annual energy before optimization (kWh).
        optimized_annual_energy_kwh: Annual energy after optimization (kWh).
        baseline_annual_energy_cost_usd: Annual energy cost before (USD).
        optimized_annual_energy_cost_usd: Annual energy cost after (USD).
        baseline_annual_total_cost_usd: Total annual cost before (USD).
        optimized_annual_total_cost_usd: Total annual cost after (USD).
        annual_total_savings_usd: Total annual savings (USD).
        annual_energy_savings_kwh: Annual energy savings (kWh).
        annual_co2e_savings_kg: Annual CO2e savings (kg).
        break_even_years: Years to break even on project cost, or None.
        quality_risk: Quality risk level of the optimization.
    """

    scenario_name: str
    strategy_name: str
    baseline_memory_gb: float
    optimized_memory_gb: float
    baseline_throughput_toks_per_sec: float
    optimized_throughput_toks_per_sec: float
    baseline_latency_factor: float
    optimized_latency_factor: float
    baseline_power_watts: float
    optimized_power_watts: float
    baseline_energy_wh_per_request: float
    optimized_energy_wh_per_request: float
    baseline_annual_energy_kwh: float
    optimized_annual_energy_kwh: float
    baseline_annual_energy_cost_usd: float
    optimized_annual_energy_cost_usd: float
    baseline_annual_total_cost_usd: float
    optimized_annual_total_cost_usd: float
    annual_total_savings_usd: float
    annual_energy_savings_kwh: float
    annual_co2e_savings_kg: float
    break_even_years: float | None
    quality_risk: QualityRisk
