"""Cloud provider cost models and cached pricing catalog access."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any

from atropos.models import DeploymentConfig, DeploymentScenario

from .providers.aws import fetch_ec2_on_demand_samples
from .providers.azure import fetch_vm_prices
from .providers.gcp import fetch_catalog_snapshot

SUPPORTED_PROVIDERS = [
    "aws",
    "azure",
    "gcp",
    "lambda-labs",
    "runpod",
    "vast-ai",
    "together-ai",
    "replicate",
    "banana",
    "modal",
]


@dataclass(frozen=True)
class CloudCostEstimate:
    """Computed cloud costs by category."""

    provider: str
    instance_type: str
    purchase_option: str
    currency: str
    compute_cost: float
    storage_cost: float
    network_cost: float
    total_cost: float
    period: str
    interruption_probability: float | None = None
    interruption_risk_note: str | None = None
    commitment_buyout_usd: float = 0.0
    per_inference_cost: float | None = None


@dataclass(frozen=True)
class PricingEntry:
    provider: str
    instance_type: str
    region: str
    model: str
    price_usd: float
    interruption_probability: float | None = None


class CloudPricingEngine:
    """Reads cached catalog and estimates cloud deployment cost."""

    def __init__(self, data_dir: str | Path = "data") -> None:
        self.data_dir = Path(data_dir)

    def list_providers(self) -> list[str]:
        return list_supported_providers()

    def load_catalog(self, max_age_days: int = 30) -> dict[str, Any]:
        catalog_path = self._latest_catalog_path()
        if catalog_path is None:
            raise FileNotFoundError(
                "No cached cloud pricing catalog found. Expected data/cloud_pricing_YYYY-MM-DD.json"
            )
        catalog = json.loads(catalog_path.read_text())
        as_of = datetime.fromisoformat(str(catalog["as_of_date"]))
        age_days = (datetime.now(tz=timezone.utc) - as_of.replace(tzinfo=timezone.utc)).days
        if age_days > max_age_days:
            raise ValueError(
                f"Cached cloud pricing is {age_days} days old (>{max_age_days}); refresh required."
            )
        return catalog

    def estimate(
        self,
        scenario: DeploymentScenario,
        provider: str | None = None,
        period: str = "monthly",
    ) -> CloudCostEstimate:
        if scenario.deployment is None:
            raise ValueError("Scenario deployment section is required for cloud-pricing estimate.")
        deployment = scenario.deployment
        chosen_provider = provider or deployment.platform
        catalog = self.load_catalog()
        fx = catalog.get("fx_rates", {"USD": 1.0})

        price_info = _find_price_entry(catalog, chosen_provider, deployment)

        if price_info["model"] == "serverless":
            compute = _serverless_compute_cost(price_info, deployment)
            per_inf = compute / max(deployment.monthly_inferences or 1.0, 1.0)
            storage = deployment.storage_gb * price_info.get("storage_usd_per_gb_month", 0.0)
            network = deployment.data_egress_gb_per_month * price_info.get(
                "egress_usd_per_gb", 0.0
            )
            total = compute + storage + network
            estimate = CloudCostEstimate(
                provider=chosen_provider,
                instance_type=deployment.instance_type,
                purchase_option=deployment.purchase_option,
                currency=deployment.currency,
                compute_cost=_convert_currency(compute, deployment.currency, fx),
                storage_cost=_convert_currency(storage, deployment.currency, fx),
                network_cost=_convert_currency(network, deployment.currency, fx),
                total_cost=_convert_currency(total, deployment.currency, fx),
                period="monthly",
                per_inference_cost=_convert_currency(per_inf, deployment.currency, fx),
            )
            return estimate

        hourly = float(price_info["price_usd_per_hour"])
        purchase_option = deployment.purchase_option
        interruption_probability = price_info.get("interruption_probability")
        if deployment.interruption_probability is not None:
            interruption_probability = deployment.interruption_probability

        adjusted_hourly = hourly
        risk_note = None
        if purchase_option == "spot":
            if interruption_probability is None:
                interruption_probability = 0.08
            # Expected-value uplift to account for retries and occasional fallback.
            adjusted_hourly *= 1 + float(interruption_probability) * 0.5
            risk_note = (
                f"Spot/preemptible selected; interruption probability ~{interruption_probability:.0%}. "
                "Use checkpointing and autoscaling fallback."
            )
        elif purchase_option == "reserved":
            discount = 0.25 if deployment.commitment_years == 1 else 0.40
            adjusted_hourly *= 1 - discount

        monthly_hours = deployment.hours_per_month
        if period == "hourly":
            compute = adjusted_hourly
            storage = deployment.storage_gb * price_info.get("storage_usd_per_gb_month", 0.0) / 730.0
            network = deployment.data_egress_gb_per_month * price_info.get("egress_usd_per_gb", 0.0) / 730.0
        elif period == "annual":
            compute = adjusted_hourly * monthly_hours * 12
            storage = deployment.storage_gb * price_info.get("storage_usd_per_gb_month", 0.0) * 12
            network = deployment.data_egress_gb_per_month * price_info.get("egress_usd_per_gb", 0.0) * 12
        else:
            compute = adjusted_hourly * monthly_hours
            storage = deployment.storage_gb * price_info.get("storage_usd_per_gb_month", 0.0)
            network = deployment.data_egress_gb_per_month * price_info.get("egress_usd_per_gb", 0.0)

        commitment_buyout = 0.0
        if purchase_option == "reserved":
            months_remaining = max(deployment.commitment_years * 12 - 6, 0)
            commitment_buyout = adjusted_hourly * monthly_hours * months_remaining * 0.2

        total = compute + storage + network
        return CloudCostEstimate(
            provider=chosen_provider,
            instance_type=deployment.instance_type,
            purchase_option=purchase_option,
            currency=deployment.currency,
            compute_cost=_convert_currency(compute, deployment.currency, fx),
            storage_cost=_convert_currency(storage, deployment.currency, fx),
            network_cost=_convert_currency(network, deployment.currency, fx),
            total_cost=_convert_currency(total, deployment.currency, fx),
            period=period,
            interruption_probability=interruption_probability,
            interruption_risk_note=risk_note,
            commitment_buyout_usd=_convert_currency(commitment_buyout, deployment.currency, fx),
        )

    def compare(
        self,
        scenario: DeploymentScenario,
        providers: list[str],
        period: str = "monthly",
    ) -> list[CloudCostEstimate]:
        return [self.estimate(scenario, provider=p, period=period) for p in providers]

    def fetch_live_pricing(self) -> dict[str, dict[str, float]]:
        """Best-effort live pricing fetch (optional, API-free paths only)."""
        return {
            "aws": fetch_ec2_on_demand_samples(),
            "azure": fetch_vm_prices(),
            "gcp": fetch_catalog_snapshot(),
        }

    def _latest_catalog_path(self) -> Path | None:
        candidates = sorted(self.data_dir.glob("cloud_pricing_*.json"))
        if not candidates:
            return None
        return candidates[-1]


def list_supported_providers() -> list[str]:
    return SUPPORTED_PROVIDERS


def estimate_cloud_cost(
    scenario: DeploymentScenario,
    provider: str | None = None,
    period: str = "monthly",
    data_dir: str | Path = "data",
) -> CloudCostEstimate:
    return CloudPricingEngine(data_dir=data_dir).estimate(scenario, provider=provider, period=period)


def _find_price_entry(
    catalog: dict[str, Any], provider: str, deployment: DeploymentConfig
) -> dict[str, Any]:
    provider_data = catalog.get("providers", {}).get(provider)
    if provider_data is None:
        raise KeyError(f"Provider '{provider}' not found in pricing catalog.")

    entries = provider_data.get("instances", [])
    for entry in entries:
        if entry.get("instance_type") == deployment.instance_type and entry.get(
            "region", deployment.region
        ) == deployment.region:
            return entry

    raise KeyError(
        f"No pricing entry for provider={provider}, instance={deployment.instance_type}, "
        f"region={deployment.region}."
    )


def _serverless_compute_cost(price_info: dict[str, Any], deployment: DeploymentConfig) -> float:
    monthly_inferences = deployment.monthly_inferences or 0.0
    per_inf = price_info.get("usd_per_inference", 0.0) * monthly_inferences
    gbs = (
        (deployment.memory_per_inference_gb or 0.0)
        * (deployment.compute_seconds_per_inference or 0.0)
        * monthly_inferences
    )
    gb_seconds = gbs * price_info.get("usd_per_gb_second", 0.0)
    return per_inf + gb_seconds


def _convert_currency(amount_usd: float, target: str, fx_rates: dict[str, float]) -> float:
    if target == "USD":
        return amount_usd
    rate = fx_rates.get(target)
    if rate is None:
        return amount_usd
    return amount_usd * float(rate)


def find_serverless_break_even(
    reserved_monthly_cost_usd: float,
    per_inference_cost_usd: float,
) -> float | None:
    """Return monthly inference volume where serverless == reserved cost."""
    if per_inference_cost_usd <= 0:
        return None
    return math.ceil(reserved_monthly_cost_usd / per_inference_cost_usd)
