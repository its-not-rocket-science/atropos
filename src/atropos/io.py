"""Input/output utilities for scenarios and reports."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path

import yaml

from .models import DeploymentScenario, OptimizationOutcome

REQUIRED_SCENARIO_KEYS = {
    "name",
    "parameters_b",
    "memory_gb",
    "throughput_toks_per_sec",
    "power_watts",
    "requests_per_day",
    "tokens_per_request",
    "electricity_cost_per_kwh",
    "annual_hardware_cost_usd",
    "one_time_project_cost_usd",
}


def load_scenario(path: str | Path) -> DeploymentScenario:
    """Load a deployment scenario from a YAML file.

    Args:
        path: Path to the YAML file.

    Returns:
        DeploymentScenario parsed from the file.

    Raises:
        ValueError: If the file is missing required keys or has invalid format.
    """
    file_path = Path(path)
    data = yaml.safe_load(file_path.read_text())

    if not isinstance(data, dict):
        raise ValueError("Scenario file must contain a YAML mapping/object.")

    missing = REQUIRED_SCENARIO_KEYS - set(data.keys())
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"Scenario file is missing required keys: {missing_str}")

    return DeploymentScenario(
        name=str(data["name"]),
        parameters_b=float(data["parameters_b"]),
        memory_gb=float(data["memory_gb"]),
        throughput_toks_per_sec=float(data["throughput_toks_per_sec"]),
        power_watts=float(data["power_watts"]),
        requests_per_day=int(data["requests_per_day"]),
        tokens_per_request=int(data["tokens_per_request"]),
        electricity_cost_per_kwh=float(data["electricity_cost_per_kwh"]),
        annual_hardware_cost_usd=float(data["annual_hardware_cost_usd"]),
        one_time_project_cost_usd=float(data["one_time_project_cost_usd"]),
    )


def outcome_to_json(outcome: OptimizationOutcome) -> str:
    """Convert an outcome to a JSON string."""
    return json.dumps(asdict(outcome), indent=2, sort_keys=True)


def format_text_report(outcome: OptimizationOutcome) -> str:
    """Format an outcome as a human-readable text report."""
    months = None if outcome.break_even_years is None else outcome.break_even_years * 12
    break_even = "never" if months is None else f"{months:.2f} months"

    return (
        f"Scenario: {outcome.scenario_name}\n"
        f"Strategy: {outcome.strategy_name}\n\n"
        f"Model memory:      {outcome.baseline_memory_gb:.2f} GB -> "
        f"{outcome.optimized_memory_gb:.2f} GB\n"
        f"Throughput:        {outcome.baseline_throughput_toks_per_sec:.2f} tok/s -> "
        f"{outcome.optimized_throughput_toks_per_sec:.2f} tok/s\n"
        f"Latency factor:    {outcome.baseline_latency_factor:.2f}x -> "
        f"{outcome.optimized_latency_factor:.2f}x\n"
        f"Power draw:        {outcome.baseline_power_watts:.2f} W -> "
        f"{outcome.optimized_power_watts:.2f} W\n"
        f"Energy / request:  {outcome.baseline_energy_wh_per_request:.2f} Wh -> "
        f"{outcome.optimized_energy_wh_per_request:.2f} Wh\n"
        f"Annual cost:       ${outcome.baseline_annual_total_cost_usd:,.2f} -> "
        f"${outcome.optimized_annual_total_cost_usd:,.2f}\n"
        f"Annual savings:    ${outcome.annual_total_savings_usd:,.2f}\n"
        f"CO2e savings:      {outcome.annual_co2e_savings_kg:,.2f} kg/year\n"
        f"Break-even:        {break_even}\n"
        f"Quality risk:      {outcome.quality_risk}\n"
    )


def export_to_csv(outcomes: Iterable[OptimizationOutcome], path: str | Path) -> None:
    """Export outcomes to a CSV file.

    Args:
        outcomes: Iterable of optimization outcomes.
        path: Path to the output CSV file.
    """
    with Path(path).open("w", newline="") as f:
        writer = csv.writer(f)
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
            ]
        )
        for r in outcomes:
            months = None if r.break_even_years is None else r.break_even_years * 12
            writer.writerow(
                [
                    r.scenario_name,
                    r.strategy_name,
                    f"{r.optimized_memory_gb:.2f}",
                    f"{r.optimized_throughput_toks_per_sec:.2f}",
                    f"{r.optimized_energy_wh_per_request:.2f}",
                    f"{r.annual_total_savings_usd:.2f}",
                    f"{months:.2f}" if months is not None else "never",
                    r.quality_risk,
                    f"{r.annual_co2e_savings_kg:.2f}",
                ]
            )


def render_report(outcome: OptimizationOutcome, report_format: str) -> str:
    """Render an outcome in the specified format.

    Args:
        outcome: The optimization outcome to render.
        report_format: One of "json", "text", "markdown", or "html".

    Returns:
        The formatted report string.

    Raises:
        ValueError: If the report format is not supported.
    """
    if report_format == "json":
        return outcome_to_json(outcome)
    if report_format == "text":
        return format_text_report(outcome)
    if report_format == "markdown":
        from .reporting import generate_markdown_report

        return generate_markdown_report(outcome)
    if report_format == "html":
        from .reporting import generate_html_report

        return generate_html_report(outcome)
    raise ValueError(f"Unsupported report format: {report_format}")
