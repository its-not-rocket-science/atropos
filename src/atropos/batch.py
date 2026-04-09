"""Batch processing helpers for multiple scenarios with resilience controls."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .core.calculator import ROICalculator
from .io import load_scenario
from .models import OptimizationOutcome
from .presets import STRATEGIES
from .utils.error_categories import ErrorCategory, categorize_error
from .utils.resilience import RetryPolicy, retry_call, run_with_timeout

DEFAULT_SCENARIO_TIMEOUT_SECONDS = 600
DEFAULT_RETRY_ATTEMPTS = 3
DEFAULT_CHECKPOINT_EVERY = 5


@dataclass(frozen=True)
class BatchFailure:
    """Represents a failed scenario/strategy processing step."""

    scenario: str
    strategy: str
    error: str
    category: ErrorCategory
    retry_count: int


@dataclass(frozen=True)
class BatchResultRow:
    """Represents one batch row (either successful or failed)."""

    scenario: str
    strategy: str
    status: str
    retry_count: int
    outcome: OptimizationOutcome | None = None
    error_message: str | None = None


@dataclass(frozen=True)
class BatchExecutionReport:
    """Batch report with partial-success metadata."""

    total_scenarios: int
    successful: int
    failed: int
    partial_success: int
    failures: list[BatchFailure]
    results: list[OptimizationOutcome]


def _run_scenario_in_subprocess(
    scenario_path: str,
    strategy_name: str,
    with_quantization: bool,
) -> dict[str, Any]:
    """Isolated scenario executor used in a subprocess."""
    calculator = ROICalculator()
    calculator.register_strategy(STRATEGIES[strategy_name])
    scenario = load_scenario(scenario_path)
    calculator.register_scenario(scenario)
    outcome = calculator.calculate(scenario.name, strategy_name, with_quantization)
    return asdict(outcome)


def _load_resume_pairs(resume_file: Path | None) -> set[tuple[str, str]]:
    """Load completed rows from a prior CSV output."""
    if resume_file is None or not resume_file.exists():
        return set()
    done: set[tuple[str, str]] = set()
    with resume_file.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            scenario = row.get("scenario", "")
            strategy = row.get("strategy", "")
            if scenario and strategy:
                done.add((scenario, strategy))
    return done


def _checkpoint_path(output_file: Path | None) -> Path | None:
    if output_file is None:
        return None
    return output_file.with_suffix(f"{output_file.suffix}.checkpoint.json")


def _save_checkpoint(
    path: Path | None, rows: list[BatchResultRow], failures: list[BatchFailure]
) -> None:
    if path is None:
        return
    payload = {
        "rows": [asdict(r) for r in rows],
        "failures": [asdict(f) for f in failures],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _write_batch_csv(rows: list[BatchResultRow], path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "scenario",
                "strategy",
                "memory_gb",
                "throughput_tok/s",
                "energy_wh_per_request",
                "annual_savings_usd",
                "break_even_months",
                "quality_risk",
                "co2e_savings_kg",
                "status",
                "error_message",
                "retry_count",
            ]
        )
        for row in rows:
            if row.outcome is None:
                writer.writerow(
                    [
                        row.scenario,
                        row.strategy,
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        row.status,
                        row.error_message or "",
                        row.retry_count,
                    ]
                )
                continue
            outcome = row.outcome
            months = None if outcome.break_even_years is None else outcome.break_even_years * 12
            writer.writerow(
                [
                    outcome.scenario_name,
                    outcome.strategy_name,
                    f"{outcome.optimized_memory_gb:.2f}",
                    f"{outcome.optimized_throughput_toks_per_sec:.2f}",
                    f"{outcome.optimized_energy_wh_per_request:.2f}",
                    f"{outcome.annual_total_savings_usd:.2f}",
                    f"{months:.2f}" if months is not None else "never",
                    outcome.quality_risk,
                    f"{outcome.annual_co2e_savings_kg:.2f}",
                    row.status,
                    row.error_message or "",
                    row.retry_count,
                ]
            )


def batch_process(
    scenario_dir: str | Path,
    strategy_names: list[str],
    output_file: str | Path | None = None,
    with_quantization: bool = False,
    *,
    fail_fast: bool = False,
    max_errors: int | None = None,
    error_log: str | Path | None = None,
    retry_attempts: int = DEFAULT_RETRY_ATTEMPTS,
    timeout_seconds: int = DEFAULT_SCENARIO_TIMEOUT_SECONDS,
    checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY,
    resume_file: str | Path | None = None,
    return_report: bool = False,
) -> list[OptimizationOutcome] | BatchExecutionReport:
    """Process multiple scenarios with multiple strategies.

    Args:
        scenario_dir: Directory containing YAML scenario files.
        strategy_names: List of strategy names to apply.
        output_file: Optional path to write CSV results.
        with_quantization: Whether to apply quantization bonus.

    Returns:
        List of optimization outcomes.
    """
    path = Path(scenario_dir)
    for name in strategy_names:
        if name not in STRATEGIES:
            raise KeyError(f"Strategy '{name}' not found")

    yaml_files = sorted(path.glob("*.yaml"))
    scenario_files = {yaml.stem: yaml for yaml in yaml_files}
    done_pairs = _load_resume_pairs(Path(resume_file) if resume_file else None)
    rows: list[BatchResultRow] = []
    failures: list[BatchFailure] = []
    results: list[OptimizationOutcome] = []

    retry_policy = RetryPolicy(max_attempts=max(1, retry_attempts))
    checkpoint_path = _checkpoint_path(Path(output_file) if output_file else None)

    for scenario_key, scenario_path in scenario_files.items():
        for strategy_name in strategy_names:
            row_key = (scenario_key, strategy_name)
            if row_key in done_pairs:
                continue

            attempts = 0
            attempt_calls = 0
            try:
                scenario_path_str = str(scenario_path)
                strategy_name_local = strategy_name

                def _attempt(
                    scenario_path_local: str = scenario_path_str,
                    strategy_name_local: str = strategy_name_local,
                ) -> dict[str, Any]:
                    nonlocal attempt_calls
                    attempt_calls += 1
                    return run_with_timeout(
                        _run_scenario_in_subprocess,
                        timeout_seconds=timeout_seconds,
                        scenario_path=scenario_path_local,
                        strategy_name=strategy_name_local,
                        with_quantization=with_quantization,
                    )

                retry_result = retry_call(_attempt, retry_policy=retry_policy)
                attempts = retry_result.retry_count
                outcome = OptimizationOutcome(**retry_result.value)
                results.append(outcome)
                rows.append(
                    BatchResultRow(
                        scenario=outcome.scenario_name,
                        strategy=outcome.strategy_name,
                        status="success",
                        retry_count=attempts,
                        outcome=outcome,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - batch must keep processing
                attempts = max(0, attempt_calls - 1)
                category = categorize_error(exc)
                failure = BatchFailure(
                    scenario=scenario_key,
                    strategy=strategy_name,
                    error=str(exc),
                    category=category,
                    retry_count=attempts,
                )
                failures.append(failure)
                rows.append(
                    BatchResultRow(
                        scenario=scenario_key,
                        strategy=strategy_name,
                        status="failed",
                        retry_count=attempts,
                        error_message=str(exc),
                    )
                )
                if fail_fast:
                    break
                if max_errors is not None and len(failures) >= max_errors:
                    break
            finally:
                # Best effort: clear framework GPU caches between runs.
                try:
                    import torch  # type: ignore[import-untyped]

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:  # noqa: BLE001 - optional cleanup
                    pass
                try:
                    import tensorflow as tf  # type: ignore[import-untyped]

                    tf.keras.backend.clear_session()
                except Exception:  # noqa: BLE001 - optional cleanup
                    pass

            if checkpoint_every > 0 and len(rows) % checkpoint_every == 0:
                _save_checkpoint(checkpoint_path, rows, failures)
                if output_file is not None:
                    _write_batch_csv(rows, Path(output_file))

        if fail_fast and failures:
            break
        if max_errors is not None and len(failures) >= max_errors:
            break

    if output_file is not None:
        _write_batch_csv(rows, Path(output_file))
    if error_log is not None:
        Path(error_log).write_text(
            json.dumps(
                [
                    {
                        "scenario": failure.scenario,
                        "strategy": failure.strategy,
                        "error": failure.error,
                        "category": failure.category,
                        "retry_count": failure.retry_count,
                    }
                    for failure in failures
                ],
                indent=2,
                sort_keys=True,
            )
        )
    _save_checkpoint(checkpoint_path, rows, failures)

    report = BatchExecutionReport(
        total_scenarios=len(scenario_files) * len(strategy_names),
        successful=len(results),
        failed=len(failures),
        partial_success=0 if not failures else (1 if results else 0),
        failures=failures,
        results=results,
    )
    if return_report:
        return report
    return results
