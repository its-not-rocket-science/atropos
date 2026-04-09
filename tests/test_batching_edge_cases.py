"""Table-driven edge case tests for batch scheduling/accounting semantics."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from atropos.batch import batch_process


@dataclass(frozen=True)
class BatchCase:
    name: str
    scenario_count: int
    strategy_names: list[str]
    failures_before_success: int
    retry_attempts: int
    max_errors: int | None
    expected_successes: int
    expected_failures: int


def _write_scenarios(tmp_path: Path, count: int) -> Path:
    scenario_dir = tmp_path / "scenarios"
    scenario_dir.mkdir()
    for idx in range(count):
        # Content only needs to exist for file discovery in these monkeypatched tests.
        (scenario_dir / f"scenario_{idx}.yaml").write_text("name: scenario\n")
    return scenario_dir


def _fake_outcome(scenario_name: str, strategy_name: str) -> dict[str, object]:
    return {
        "scenario_name": scenario_name,
        "strategy_name": strategy_name,
        "baseline_memory_gb": 10.0,
        "optimized_memory_gb": 9.0,
        "baseline_throughput_toks_per_sec": 100.0,
        "optimized_throughput_toks_per_sec": 110.0,
        "baseline_latency_factor": 1.0,
        "optimized_latency_factor": 0.9,
        "baseline_power_watts": 250.0,
        "optimized_power_watts": 225.0,
        "baseline_energy_wh_per_request": 1.0,
        "optimized_energy_wh_per_request": 0.8,
        "baseline_annual_energy_kwh": 1000.0,
        "optimized_annual_energy_kwh": 850.0,
        "baseline_annual_energy_cost_usd": 120.0,
        "optimized_annual_energy_cost_usd": 102.0,
        "baseline_annual_total_cost_usd": 4000.0,
        "optimized_annual_total_cost_usd": 3600.0,
        "annual_total_savings_usd": 400.0,
        "annual_energy_savings_kwh": 150.0,
        "annual_co2e_savings_kg": 75.0,
        "break_even_years": 1.0,
        "quality_risk": "low",
    }


@pytest.mark.parametrize(
    "case",
    [
        BatchCase(
            name="mixed_group_sizes",
            scenario_count=3,
            strategy_names=["mild_pruning", "structured_pruning"],
            failures_before_success=0,
            retry_attempts=2,
            max_errors=None,
            expected_successes=6,
            expected_failures=0,
        ),
        BatchCase(
            name="impossible_packings_max_error_cap",
            scenario_count=4,
            strategy_names=["mild_pruning", "structured_pruning"],
            failures_before_success=999,
            retry_attempts=1,
            max_errors=2,
            expected_successes=0,
            expected_failures=2,
        ),
        BatchCase(
            name="min_allocation_oversubscription_retry_clamped",
            scenario_count=2,
            strategy_names=["mild_pruning"],
            failures_before_success=999,
            retry_attempts=0,
            max_errors=None,
            expected_successes=0,
            expected_failures=2,
        ),
    ],
    ids=lambda case: case.name,
)
def test_table_driven_batch_edge_cases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: BatchCase,
) -> None:
    scenario_dir = _write_scenarios(tmp_path, case.scenario_count)
    call_attempts: dict[tuple[str, str], int] = {}

    def fake_timeout(fn, timeout_seconds: int, **kwargs):  # type: ignore[no-untyped-def]
        del fn, timeout_seconds
        scenario_key = Path(kwargs["scenario_path"]).stem
        strategy = kwargs["strategy_name"]
        key = (scenario_key, strategy)
        call_attempts[key] = call_attempts.get(key, 0) + 1
        if call_attempts[key] <= case.failures_before_success:
            raise RuntimeError("transient network issue")
        return _fake_outcome(scenario_key, strategy)

    monkeypatch.setattr("atropos.batch.run_with_timeout", fake_timeout)

    report = batch_process(
        scenario_dir,
        case.strategy_names,
        tmp_path / "rows.csv",
        retry_attempts=case.retry_attempts,
        max_errors=case.max_errors,
        fail_fast=False,
        return_report=True,
    )

    assert report.successful == case.expected_successes
    assert report.failed == case.expected_failures


def test_buffered_partial_groups_checkpoint_flushes_tail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scenario_dir = _write_scenarios(tmp_path, 5)

    def fake_timeout(fn, timeout_seconds: int, **kwargs):  # type: ignore[no-untyped-def]
        del fn, timeout_seconds
        scenario_key = Path(kwargs["scenario_path"]).stem
        return _fake_outcome(scenario_key, kwargs["strategy_name"])

    monkeypatch.setattr("atropos.batch.run_with_timeout", fake_timeout)

    output_file = tmp_path / "batched.csv"
    batch_process(
        scenario_dir,
        ["mild_pruning"],
        output_file,
        checkpoint_every=3,
    )

    checkpoint_file = output_file.with_suffix(".csv.checkpoint.json")
    payload = json.loads(checkpoint_file.read_text())

    assert len(payload["rows"]) == 5
    assert all(row["status"] == "success" for row in payload["rows"])


def test_retry_accounting_on_failure_reports_consumed_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scenario_dir = _write_scenarios(tmp_path, 1)
    timeout_calls = {"count": 0}

    def always_fails(fn, timeout_seconds: int, **kwargs):  # type: ignore[no-untyped-def]
        del fn, timeout_seconds, kwargs
        timeout_calls["count"] += 1
        raise RuntimeError("transient backend timeout")

    monkeypatch.setattr("atropos.batch.run_with_timeout", always_fails)

    report = batch_process(
        scenario_dir,
        ["mild_pruning"],
        retry_attempts=3,
        return_report=True,
    )

    assert report.failed == 1
    assert timeout_calls["count"] == 3
    # retry_count reports consumed retries, so max_attempts=3 -> retry_count=2.
    assert report.failures[0].retry_count == 2
