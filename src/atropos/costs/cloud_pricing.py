"""Cloud pricing models for provider-aware ROI and deployment analysis."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Literal, cast

from .providers import (
    fetch_aws_live_catalog,
    fetch_azure_live_catalog,
    fetch_gcp_live_catalog,
    get_aws_catalog,
    get_azure_catalog,
    get_gcp_catalog,
)

PurchaseOption = Literal["ondemand", "spot", "reserved"]


@dataclass(frozen=True)
class CloudCostRequest:
    """Request describing a cloud deployment workload."""

    provider: str
    instance_type: str
    purchase_option: PurchaseOption = "ondemand"
    region: str = "us-east-1"
    monthly_runtime_hours: float = 730.0
    monthly_inference_count: int = 0
    average_memory_gb: float = 0.0
    average_duration_seconds: float = 0.0
    monthly_storage_gb: float = 0.0
    monthly_data_transfer_gb: float = 0.0
    commitment_years: int = 0
    interruption_probability: float | None = None
    currency: str = "USD"


@dataclass(frozen=True)
class CloudCostEstimate:
    """Computed cloud cost with component breakdown."""

    provider: str
    instance_type: str
    purchase_option: PurchaseOption
    currency: str
    unit_compute_cost: float
    hourly_total_cost: float
    monthly_total_cost: float
    annual_total_cost: float
    compute_monthly_cost: float
    storage_monthly_cost: float
    network_monthly_cost: float
    serverless_request_monthly_cost: float
    serverless_duration_monthly_cost: float
    commitment_buyout_cost: float
    interruption_probability: float | None = None
    risk_warning: str | None = None
    source_timestamp: str | None = None


DEFAULT_RESERVED_DISCOUNTS = {1: 0.30, 3: 0.50}
DEFAULT_STORAGE_USD_PER_GB_MONTH = 0.023
DEFAULT_EGRESS_USD_PER_GB = 0.09
DEFAULT_SPOT_INTERRUPTION_PROBABILITY = 0.15
CACHE_MAX_AGE_DAYS = 30
LIVE_FETCH_COOLDOWN_HOURS = 6


class CloudPricingEngine:
    """Pricing engine backed by cached JSON catalogs with optional live refresh."""

    def __init__(self, data_dir: Path | None = None) -> None:
        self.data_dir = data_dir or Path("data")
        self.catalog = self._load_catalog()

    def _default_catalog(self) -> dict[str, Any]:
        return {
            "metadata": {
                "source": "atropos-static",
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "currency_base": "USD",
                "exchange_rate_date": date.today().isoformat(),
            },
            "exchange_rates": {
                "USD": 1.0,
                "EUR": 0.93,
                "GBP": 0.79,
                "JPY": 151.0,
                "INR": 83.0,
            },
            "providers": {
                "aws": get_aws_catalog(),
                "azure": get_azure_catalog(),
                "gcp": get_gcp_catalog(),
                "lambda-labs": {
                    "gpu": {
                        "a100-80gb": {
                            "ondemand_usd_per_hour": 2.49,
                            "spot_usd_per_hour": 1.69,
                        }
                    }
                },
                "runpod": {
                    "gpu": {
                        "a100-pcie-80gb": {
                            "ondemand_usd_per_hour": 2.79,
                            "spot_usd_per_hour": 1.15,
                        }
                    }
                },
                "vast.ai": {
                    "gpu": {
                        "a100-80gb-average": {
                            "ondemand_usd_per_hour": 1.95,
                            "spot_usd_per_hour": 0.99,
                        }
                    }
                },
                "together.ai": {
                    "serverless": {
                        "a100-inference": {
                            "usd_per_1k_tokens": 0.0008,
                            "usd_per_inference": 0.002,
                        }
                    }
                },
                "replicate": {
                    "serverless": {"l40s": {"usd_per_second": 0.00055, "usd_per_inference": 0.003}}
                },
                "banana": {
                    "serverless": {"a100": {"usd_per_second": 0.00062, "usd_per_inference": 0.0025}}
                },
                "modal": {
                    "serverless": {
                        "a100": {
                            "usd_per_second": 0.00064,
                            "usd_per_inference": 0.0018,
                            "usd_per_gb_second": 0.000005,
                        }
                    }
                },
            },
        }

    def _is_stale(self, stamp: str) -> bool:
        try:
            file_date = datetime.strptime(stamp, "%Y-%m-%d").date()
        except ValueError:
            return True
        return (date.today() - file_date).days > CACHE_MAX_AGE_DAYS

    def _load_catalog(self) -> dict[str, Any]:
        if not self.data_dir.exists():
            return self._default_catalog()

        files = sorted(self.data_dir.glob("cloud_pricing_*.json"), reverse=True)
        for file in files:
            stamp = file.stem.replace("cloud_pricing_", "")
            if self._is_stale(stamp):
                continue
            try:
                payload = json.loads(file.read_text())
                if isinstance(payload, dict):
                    return payload
            except json.JSONDecodeError:
                continue

        return self._default_catalog()

    def _last_live_fetch_at(self) -> datetime | None:
        raw = self.catalog.get("metadata", {}).get("live_fetched_at")
        if not raw:
            return None
        try:
            return datetime.fromisoformat(str(raw))
        except ValueError:
            return None

    def refresh_live_pricing(self, use_mock_api: bool = False) -> None:
        """Refresh selected provider catalogs.

        This entry point is CI-safe and API-key optional. To avoid provider rate-limit
        abuse we refuse to re-fetch if the previous refresh happened recently.
        """
        now = datetime.now(timezone.utc)
        last_fetch = self._last_live_fetch_at()
        if last_fetch is not None:
            age_hours = (now - last_fetch).total_seconds() / 3600.0
            if age_hours < LIVE_FETCH_COOLDOWN_HOURS:
                return

        catalog = self.catalog or self._default_catalog()
        providers = dict(catalog.get("providers", {}))
        providers["aws"] = fetch_aws_live_catalog(mock=use_mock_api)
        providers["azure"] = fetch_azure_live_catalog(mock=use_mock_api)
        providers["gcp"] = fetch_gcp_live_catalog(mock=use_mock_api)

        metadata = dict(catalog.get("metadata", {}))
        metadata["source"] = "atropos-live-mock" if use_mock_api else "atropos-live-fallback"
        metadata["updated_at"] = now.isoformat()
        metadata["live_fetched_at"] = now.isoformat()

        self.catalog = {**catalog, "metadata": metadata, "providers": providers}

    def list_providers(self) -> list[str]:
        return sorted(self.catalog.get("providers", {}).keys())

    def _get_currency_rate(self, currency: str) -> float:
        rates = self.catalog.get("exchange_rates", {})
        if currency not in rates:
            raise ValueError(f"Unsupported currency '{currency}'.")
        return float(rates[currency])

    def _find_instance_pricing(self, provider: str, instance_type: str) -> dict[str, Any]:
        provider_catalog = self.catalog.get("providers", {}).get(provider)
        if provider_catalog is None:
            raise KeyError(f"Unknown provider '{provider}'")

        for group_name, group_data in provider_catalog.items():
            if group_name.startswith("_"):
                continue
            if isinstance(group_data, dict) and instance_type in group_data:
                candidate = group_data[instance_type]
                if isinstance(candidate, dict):
                    return candidate

        raise KeyError(f"Unknown instance type '{instance_type}' for provider '{provider}'")

    def default_instance_type(self, provider: str) -> str:
        """Return first known instance type for provider."""
        provider_catalog = self.catalog.get("providers", {}).get(provider)
        if not isinstance(provider_catalog, dict):
            raise KeyError(f"Unknown provider '{provider}'")
        for group_name, group_data in provider_catalog.items():
            if group_name.startswith("_"):
                continue
            if isinstance(group_data, dict) and group_data:
                return str(next(iter(group_data.keys())))
        raise KeyError(f"No instance types available for provider '{provider}'")

    def _calc_reserved_buyout(self, unit_usd: float, request: CloudCostRequest) -> float:
        if request.purchase_option != "reserved" or request.commitment_years <= 0:
            return 0.0
        committed_hours = 365 * 24 * request.commitment_years
        used_hours = min(
            request.monthly_runtime_hours * 12 * request.commitment_years,
            committed_hours,
        )
        remaining_hours = max(committed_hours - used_hours, 0)
        return remaining_hours * unit_usd

    def estimate(self, request: CloudCostRequest) -> CloudCostEstimate:
        pricing = self._find_instance_pricing(request.provider, request.instance_type)

        if request.purchase_option == "spot":
            unit_usd = float(
                pricing.get("spot_usd_per_hour", pricing.get("ondemand_usd_per_hour", 0.0))
            )
            interruption_probability = (
                request.interruption_probability
                if request.interruption_probability is not None
                else DEFAULT_SPOT_INTERRUPTION_PROBABILITY
            )
            risk_warning = (
                "Spot/preemptible pricing may be interrupted; ensure autoscaling and checkpointing."
            )
        elif request.purchase_option == "reserved":
            on_demand = float(pricing.get("ondemand_usd_per_hour", 0.0))
            discount = DEFAULT_RESERVED_DISCOUNTS.get(request.commitment_years, 0.0)
            unit_usd = on_demand * (1.0 - discount)
            interruption_probability = None
            risk_warning = None
        else:
            unit_usd = float(pricing.get("ondemand_usd_per_hour", pricing.get("usd_per_hour", 0.0)))
            interruption_probability = None
            risk_warning = None

        if unit_usd == 0.0 and "usd_per_inference" in pricing:
            unit_usd = float(pricing["usd_per_inference"])

        compute_monthly = 0.0
        serverless_request_monthly = 0.0
        serverless_duration_monthly = 0.0

        if "usd_per_inference" in pricing:
            serverless_request_monthly = request.monthly_inference_count * float(
                pricing.get("usd_per_inference", 0.0)
            )
            per_second = float(pricing.get("usd_per_second", 0.0))
            duration_s = request.monthly_inference_count * request.average_duration_seconds
            serverless_duration_monthly = duration_s * per_second
            gb_second = float(pricing.get("usd_per_gb_second", 0.0))
            serverless_duration_monthly += (
                request.monthly_inference_count
                * request.average_duration_seconds
                * request.average_memory_gb
                * gb_second
            )
        else:
            compute_monthly = request.monthly_runtime_hours * unit_usd

        storage_monthly = request.monthly_storage_gb * DEFAULT_STORAGE_USD_PER_GB_MONTH
        network_monthly = request.monthly_data_transfer_gb * DEFAULT_EGRESS_USD_PER_GB
        buyout_cost = self._calc_reserved_buyout(unit_usd, request)

        monthly_total_usd = (
            compute_monthly
            + serverless_request_monthly
            + serverless_duration_monthly
            + storage_monthly
            + network_monthly
        )
        annual_total_usd = monthly_total_usd * 12.0

        rate = self._get_currency_rate(request.currency)

        def converted(value: float) -> float:
            return value * rate

        denominator = request.monthly_runtime_hours if request.monthly_runtime_hours > 0 else 730.0
        hourly = converted(monthly_total_usd / denominator)

        return CloudCostEstimate(
            provider=request.provider,
            instance_type=request.instance_type,
            purchase_option=request.purchase_option,
            currency=request.currency,
            unit_compute_cost=converted(unit_usd),
            hourly_total_cost=hourly,
            monthly_total_cost=converted(monthly_total_usd),
            annual_total_cost=converted(annual_total_usd),
            compute_monthly_cost=converted(compute_monthly),
            storage_monthly_cost=converted(storage_monthly),
            network_monthly_cost=converted(network_monthly),
            serverless_request_monthly_cost=converted(serverless_request_monthly),
            serverless_duration_monthly_cost=converted(serverless_duration_monthly),
            commitment_buyout_cost=converted(buyout_cost),
            interruption_probability=interruption_probability,
            risk_warning=risk_warning,
            source_timestamp=self.catalog.get("metadata", {}).get("updated_at"),
        )


def list_supported_providers(data_dir: Path | None = None) -> list[str]:
    """List providers available in the pricing catalog."""
    return CloudPricingEngine(data_dir=data_dir).list_providers()


def estimate_cloud_cost(
    request: CloudCostRequest,
    data_dir: Path | None = None,
) -> CloudCostEstimate:
    """Estimate cloud cost for a request."""
    return CloudPricingEngine(data_dir=data_dir).estimate(request)


def request_from_scenario_yaml(
    scenario_data: dict[str, Any],
    provider_override: str | None = None,
) -> CloudCostRequest:
    """Parse cloud request values from scenario YAML structure."""
    deployment = scenario_data.get("deployment", {})
    provider = provider_override or deployment.get("platform")
    if not provider:
        raise ValueError("Scenario must define deployment.platform or pass --provider.")

    purchase_option: PurchaseOption = "ondemand"
    raw_purchase_option = str(deployment.get("purchase_option", "ondemand"))
    if raw_purchase_option in {"ondemand", "spot", "reserved"}:
        purchase_option = cast(PurchaseOption, raw_purchase_option)

    return CloudCostRequest(
        provider=str(provider),
        instance_type=str(deployment.get("instance_type", "")),
        purchase_option=purchase_option,
        region=str(deployment.get("region", "us-east-1")),
        commitment_years=int(deployment.get("commitment_years", 0)),
        monthly_runtime_hours=float(scenario_data.get("monthly_runtime_hours", 730.0)),
        monthly_inference_count=int(scenario_data.get("monthly_inference_count", 0)),
        average_memory_gb=float(scenario_data.get("average_memory_gb", 0.0)),
        average_duration_seconds=float(scenario_data.get("average_duration_seconds", 0.0)),
        monthly_storage_gb=float(scenario_data.get("monthly_storage_gb", 0.0)),
        monthly_data_transfer_gb=float(scenario_data.get("monthly_data_transfer_gb", 0.0)),
        currency=str(scenario_data.get("currency", "USD")),
    )
