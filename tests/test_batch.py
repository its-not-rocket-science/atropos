"""Tests for resilient batch processing behavior."""

from __future__ import annotations

import csv
import random
from pathlib import Path

from atropos.batch import batch_process
from atropos.utils.resilience import RetryPolicy, retry_call


def _write_scenario(path: Path, name: str) -> None:
    path.write_text(
        f"""
name: {name}
parameters_b: 7
memory_gb: 8
throughput_toks_per_sec: 20
power_watts: 180
requests_per_day: 10000
tokens_per_request: 1200
electricity_cost_per_kwh: 0.12
annual_hardware_cost_usd: 5000
one_time_project_cost_usd: 2000
""".strip()
        + "\n"
    )


def test_retry_call_retries_and_succeeds() -> None:
    attempts = {"count": 0}

    def flaky() -> str:
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise RuntimeError("temporary network issue")
        return "ok"

    result = retry_call(flaky, RetryPolicy(max_attempts=3, base_delay_seconds=0.0))
    assert result.value == "ok"
    assert result.retry_count == 2


def test_batch_partial_success_with_good_and_bad_scenario(tmp_path: Path) -> None:
    scenario_dir = tmp_path / "scenarios"
    scenario_dir.mkdir()
    _write_scenario(scenario_dir / "good.yaml", "good")
    # Missing required key annual_hardware_cost_usd -> config error
    (scenario_dir / "bad.yaml").write_text(
        """
name: bad
parameters_b: 7
memory_gb: 8
throughput_toks_per_sec: 20
power_watts: 180
requests_per_day: 10000
tokens_per_request: 1200
electricity_cost_per_kwh: 0.12
one_time_project_cost_usd: 2000
""".strip()
        + "\n"
    )

    output_csv = tmp_path / "batch.csv"
    error_log = tmp_path / "errors.json"

    report = batch_process(
        scenario_dir,
        ["mild_pruning"],
        output_csv,
        fail_fast=False,
        error_log=error_log,
        return_report=True,
    )

    assert report.successful == 1
    assert report.failed == 1
    assert output_csv.exists()
    assert error_log.exists()

    with output_csv.open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert {row["status"] for row in rows} == {"success", "failed"}


def test_stress_batch_random_failures(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    scenario_dir = tmp_path / "many"
    scenario_dir.mkdir()
    for idx in range(100):
        _write_scenario(scenario_dir / f"scenario_{idx}.yaml", f"scenario_{idx}")

    rng = random.Random(13)

    from atropos import batch as batch_module

    real_runner = batch_module._run_scenario_in_subprocess

    def fake_runner(scenario_path: str, strategy_name: str, with_quantization: bool):  # type: ignore[no-untyped-def]
        if rng.random() < 0.15:
            raise RuntimeError("transient network hiccup")
        return real_runner(scenario_path, strategy_name, with_quantization)

    monkeypatch.setattr("atropos.batch._run_scenario_in_subprocess", fake_runner)

    report = batch_process(
        scenario_dir,
        ["mild_pruning"],
        tmp_path / "stress.csv",
        fail_fast=False,
        max_errors=50,
        retry_attempts=1,
        return_report=True,
    )

    assert report.total_scenarios == 100
    assert report.successful + report.failed <= 100
    assert report.successful > 0
