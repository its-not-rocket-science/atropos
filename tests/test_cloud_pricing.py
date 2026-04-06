"""Tests for cloud pricing support."""

from __future__ import annotations

from pathlib import Path

from atropos.costs.cloud_pricing import CloudPricingEngine
from atropos.io import load_scenario


def _write_scenario(path: Path, deployment_block: str) -> None:
    path.write_text(
        f"""
name: cloud-test
parameters_b: 7
memory_gb: 20
throughput_toks_per_sec: 120
power_watts: 400
requests_per_day: 10000
tokens_per_request: 1024
electricity_cost_per_kwh: 0.12
annual_hardware_cost_usd: 10000
one_time_project_cost_usd: 20000
{deployment_block}
""".strip()
    )


def test_cloud_estimate_spot_has_warning(tmp_path: Path) -> None:
    scenario_path = tmp_path / "scenario.yaml"
    _write_scenario(
        scenario_path,
        """
deployment:
  platform: aws
  instance_type: p4d.24xlarge
  purchase_option: spot
  region: us-east-1
  storage_gb: 100
  data_egress_gb_per_month: 250
""",
    )
    scenario = load_scenario(scenario_path)
    engine = CloudPricingEngine(data_dir="data")
    estimate = engine.estimate(scenario, provider="aws")

    assert estimate.total_cost > 0
    assert estimate.interruption_risk_note is not None


def test_cloud_compare_p4d_vs_azure_a100(tmp_path: Path) -> None:
    scenario_path = tmp_path / "scenario.yaml"
    _write_scenario(
        scenario_path,
        """
deployment:
  platform: aws
  instance_type: p4d.24xlarge
  purchase_option: ondemand
  region: us-east-1
  storage_gb: 50
  data_egress_gb_per_month: 100
""",
    )

    scenario = load_scenario(scenario_path)
    engine = CloudPricingEngine(data_dir="data")
    aws_cost = engine.estimate(scenario, provider="aws")

    azure_scenario_path = tmp_path / "azure.yaml"
    _write_scenario(
        azure_scenario_path,
        """
deployment:
  platform: azure
  instance_type: nc_a100_v4
  purchase_option: ondemand
  region: eastus
  storage_gb: 50
  data_egress_gb_per_month: 100
""",
    )
    azure_cost = engine.estimate(load_scenario(azure_scenario_path), provider="azure")

    assert aws_cost.total_cost > azure_cost.total_cost


def test_serverless_per_inference_cost(tmp_path: Path) -> None:
    scenario_path = tmp_path / "serverless.yaml"
    _write_scenario(
        scenario_path,
        """
deployment:
  platform: replicate
  instance_type: a100
  purchase_option: ondemand
  region: global
  monthly_inferences: 100000
  memory_per_inference_gb: 3
  compute_seconds_per_inference: 1.2
""",
    )

    scenario = load_scenario(scenario_path)
    engine = CloudPricingEngine(data_dir="data")
    estimate = engine.estimate(scenario, provider="replicate")

    assert estimate.per_inference_cost is not None
    assert estimate.total_cost > 0
