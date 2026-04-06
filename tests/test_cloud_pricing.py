"""Tests for cloud pricing models and provider comparisons."""

from __future__ import annotations

from pathlib import Path

import yaml

from atropos.costs.cloud_pricing import (
    CloudCostRequest,
    CloudPricingEngine,
    request_from_scenario_yaml,
)
from atropos.io import load_scenario


def test_cloud_engine_lists_major_providers() -> None:
    engine = CloudPricingEngine()
    providers = engine.list_providers()
    assert "aws" in providers
    assert "azure" in providers
    assert "gcp" in providers


def test_spot_pricing_emits_risk_warning() -> None:
    engine = CloudPricingEngine()
    estimate = engine.estimate(
        CloudCostRequest(
            provider="aws",
            instance_type="p4d.24xlarge",
            purchase_option="spot",
            monthly_runtime_hours=100.0,
        )
    )
    assert estimate.monthly_total_cost > 0
    assert estimate.interruption_probability is not None
    assert estimate.risk_warning is not None


def test_reserved_pricing_includes_buyout_estimate() -> None:
    engine = CloudPricingEngine()
    estimate = engine.estimate(
        CloudCostRequest(
            provider="azure",
            instance_type="Standard_NC24_A100_v2",
            purchase_option="reserved",
            commitment_years=1,
            monthly_runtime_hours=100,
        )
    )
    assert estimate.monthly_total_cost > 0
    assert estimate.commitment_buyout_cost > 0


def test_serverless_pricing_uses_inference_dimensions() -> None:
    engine = CloudPricingEngine()
    estimate = engine.estimate(
        CloudCostRequest(
            provider="modal",
            instance_type="a100",
            purchase_option="ondemand",
            monthly_inference_count=100_000,
            average_duration_seconds=1.2,
            average_memory_gb=16,
            monthly_storage_gb=10,
            monthly_data_transfer_gb=20,
        )
    )
    assert estimate.serverless_request_monthly_cost > 0
    assert estimate.serverless_duration_monthly_cost > 0
    assert estimate.compute_monthly_cost == 0


def test_refresh_live_pricing_accepts_mock_mode() -> None:
    engine = CloudPricingEngine()
    engine.refresh_live_pricing(use_mock_api=True)
    assert engine.catalog["metadata"]["source"] == "atropos-live-mock"


def test_load_scenario_allows_deployment_without_annual_hardware(tmp_path: Path) -> None:
    scenario_file = tmp_path / "cloud-scenario.yaml"
    scenario_file.write_text(
        yaml.safe_dump(
            {
                "name": "cloud-test",
                "parameters_b": 34.0,
                "memory_gb": 20.0,
                "throughput_toks_per_sec": 55.0,
                "power_watts": 350.0,
                "requests_per_day": 50000,
                "tokens_per_request": 1600,
                "electricity_cost_per_kwh": 0.15,
                "one_time_project_cost_usd": 27000,
                "deployment": {
                    "platform": "aws",
                    "instance_type": "p4d.24xlarge",
                    "purchase_option": "spot",
                    "region": "us-east-1",
                },
            }
        )
    )

    scenario = load_scenario(scenario_file)
    assert scenario.annual_hardware_cost_usd == 0.0


def test_request_from_scenario_yaml() -> None:
    request = request_from_scenario_yaml(
        {
            "deployment": {
                "platform": "azure",
                "instance_type": "Standard_NC24ads_A100_v4",
                "purchase_option": "reserved",
                "commitment_years": 1,
            },
            "currency": "EUR",
            "monthly_runtime_hours": 500,
            "monthly_data_transfer_gb": 100,
        }
    )
    assert request.provider == "azure"
    assert request.purchase_option == "reserved"
    assert request.currency == "EUR"
