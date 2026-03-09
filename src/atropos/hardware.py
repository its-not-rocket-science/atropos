"""Hardware profiles and GPU tier definitions for cost modeling.

Provides predefined GPU tiers with realistic pricing and performance characteristics.
Supports both cloud (hourly) and reserved/owned (annualized) cost models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

GPUType = Literal[
    "A100_40GB",
    "A100_80GB",
    "H100_80GB",
    "L4",
    "T4",
    "A10G",
    "RTX4090",
    "MI300X",
]

PricingModel = Literal["cloud", "reserved"]


@dataclass(frozen=True)
class GPUTier:
    """GPU hardware tier specification.

    Attributes:
        name: GPU type identifier (e.g., "A100_80GB").
        memory_gb: GPU memory capacity in GB.
        peak_tflops_fp16: Peak FP16 compute in TFLOPS.
        memory_bw_gbps: Memory bandwidth in GB/s.
        typical_power_w: Typical power draw under load in watts.
        cloud_cost_per_hour: On-demand cloud pricing per GPU per hour (USD).
        reserved_cost_per_hour: Reserved/committed pricing per hour (USD).
        release_year: Year the GPU was released (for depreciation modeling).
    """

    name: GPUType
    memory_gb: float
    peak_tflops_fp16: float
    memory_bw_gbps: float
    typical_power_w: float
    cloud_cost_per_hour: float
    reserved_cost_per_hour: float
    release_year: int

    def annual_cost(
        self,
        gpu_count: int = 1,
        utilization: float = 1.0,
        pricing_model: PricingModel = "cloud",
    ) -> float:
        """Calculate annual hardware cost.

        Args:
            gpu_count: Number of GPUs.
            utilization: Fraction of time GPUs are active (0-1).
            pricing_model: "cloud" for on-demand, "reserved" for committed.

        Returns:
            Annual cost in USD.
        """
        cost_per_hour = (
            self.cloud_cost_per_hour
            if pricing_model == "cloud"
            else self.reserved_cost_per_hour
        )
        hours_per_year = 365 * 24
        return cost_per_hour * gpu_count * hours_per_year * utilization


# GPU tier specifications (approximate market pricing as of 2024)
GPU_TIERS: dict[GPUType, GPUTier] = {
    "A100_40GB": GPUTier(
        name="A100_40GB",
        memory_gb=40.0,
        peak_tflops_fp16=312.0,
        memory_bw_gbps=1555.0,
        typical_power_w=400.0,
        cloud_cost_per_hour=2.50,
        reserved_cost_per_hour=1.75,
        release_year=2020,
    ),
    "A100_80GB": GPUTier(
        name="A100_80GB",
        memory_gb=80.0,
        peak_tflops_fp16=312.0,
        memory_bw_gbps=2039.0,
        typical_power_w=400.0,
        cloud_cost_per_hour=3.00,
        reserved_cost_per_hour=2.10,
        release_year=2020,
    ),
    "H100_80GB": GPUTier(
        name="H100_80GB",
        memory_gb=80.0,
        peak_tflops_fp16=989.0,
        memory_bw_gbps=3350.0,
        typical_power_w=700.0,
        cloud_cost_per_hour=5.00,
        reserved_cost_per_hour=3.50,
        release_year=2023,
    ),
    "L4": GPUTier(
        name="L4",
        memory_gb=24.0,
        peak_tflops_fp16=242.0,
        memory_bw_gbps=300.0,
        typical_power_w=72.0,
        cloud_cost_per_hour=0.80,
        reserved_cost_per_hour=0.56,
        release_year=2023,
    ),
    "T4": GPUTier(
        name="T4",
        memory_gb=16.0,
        peak_tflops_fp16=65.0,
        memory_bw_gbps=320.0,
        typical_power_w=70.0,
        cloud_cost_per_hour=0.50,
        reserved_cost_per_hour=0.35,
        release_year=2019,
    ),
    "A10G": GPUTier(
        name="A10G",
        memory_gb=24.0,
        peak_tflops_fp16=125.0,
        memory_bw_gbps=600.0,
        typical_power_w=150.0,
        cloud_cost_per_hour=1.20,
        reserved_cost_per_hour=0.84,
        release_year=2021,
    ),
    "RTX4090": GPUTier(
        name="RTX4090",
        memory_gb=24.0,
        peak_tflops_fp16=165.0,
        memory_bw_gbps=1008.0,
        typical_power_w=450.0,
        cloud_cost_per_hour=1.00,
        reserved_cost_per_hour=0.70,
        release_year=2022,
    ),
    "MI300X": GPUTier(
        name="MI300X",
        memory_gb=192.0,
        peak_tflops_fp16=1307.0,
        memory_bw_gbps=5300.0,
        typical_power_w=750.0,
        cloud_cost_per_hour=4.00,
        reserved_cost_per_hour=2.80,
        release_year=2024,
    ),
}


def get_gpu_tier(name: GPUType) -> GPUTier:
    """Get GPU tier by name.

    Args:
        name: GPU type identifier.

    Returns:
        GPUTier specification.

    Raises:
        KeyError: If GPU type not found.
    """
    if name not in GPU_TIERS:
        raise KeyError(f"Unknown GPU type: {name}. Available: {list(GPU_TIERS.keys())}")
    return GPU_TIERS[name]


def list_gpu_tiers() -> list[GPUType]:
    """List available GPU tier names."""
    return list(GPU_TIERS.keys())


def estimate_gpu_count(
    model_memory_gb: float,
    gpu_tier: GPUType,
    memory_overhead_factor: float = 1.2,
) -> int:
    """Estimate number of GPUs needed for a model.

    Args:
        model_memory_gb: Model memory requirement in GB.
        gpu_tier: GPU type to use.
        memory_overhead_factor: Multiplier for KV cache and overhead.

    Returns:
        Number of GPUs required (minimum 1).
    """
    tier = get_gpu_tier(gpu_tier)
    total_memory_needed = model_memory_gb * memory_overhead_factor
    gpu_count = int((total_memory_needed + tier.memory_gb - 1) // tier.memory_gb)
    return max(1, gpu_count)
