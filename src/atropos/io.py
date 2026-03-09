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


def csv_to_markdown(csv_path: str | Path, output_path: str | Path | None = None) -> str:
    """Convert a CSV file of outcomes to a markdown report.

    Args:
        csv_path: Path to the CSV file containing outcome data.
        output_path: Optional path to write the markdown output.
            If None, returns the markdown string.

    Returns:
        The formatted markdown report.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If the CSV format is invalid.
    """
    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    with csv_file.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return "# Atropos Batch Analysis Report\n\nNo data found in CSV."

    # Build markdown report
    lines = [
        "# Atropos Batch Analysis Report",
        "",
        f"Generated from: `{csv_file.name}`",
        "",
        "## Summary",
        "",
        f"- **Total scenarios analyzed**: {len(rows)}",
        "",
        "## Results by Scenario",
        "",
    ]

    # Group by scenario
    by_scenario: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        scenario = row.get("scenario", "unknown")
        if scenario not in by_scenario:
            by_scenario[scenario] = []
        by_scenario[scenario].append(row)

    for scenario_name, scenario_rows in sorted(by_scenario.items()):
        lines.append(f"### {scenario_name}")
        lines.append("")
        header = "| Strategy | Memory (GB) | Throughput (tok/s) | Annual Savings |"
        header += " Break-even | Risk |"
        lines.append(header)
        lines.append("|----------|-------------|-------------------|----------------|------------|------|")

        for row in scenario_rows:
            strategy = row.get("strategy", "unknown")
            memory = row.get("memory_gb", "N/A")
            throughput = row.get("throughput_tok/s", "N/A")
            savings_raw = row.get("annual_savings_usd", "N/A")
            be = row.get("break_even_months", "N/A")
            risk = row.get("quality_risk", "N/A")

            # Format currency
            savings_display = savings_raw
            if savings_raw != "N/A":
                try:
                    savings_display = f"${float(savings_raw):,.0f}"
                except ValueError:
                    pass

            lines.append(
                f"| {strategy} | {memory} | {throughput} | {savings_display} | {be} | {risk} |"
            )

        lines.append("")

    # Add aggregate statistics
    lines.append("## Aggregate Statistics")
    lines.append("")

    all_savings = []
    for row in rows:
        try:
            savings = float(row.get("annual_savings_usd", 0))
            all_savings.append(savings)
        except ValueError:
            pass

    if all_savings:
        total = sum(all_savings)
        avg = total / len(all_savings)
        max_savings = max(all_savings)
        min_savings = min(all_savings)

        lines.append(f"- **Total annual savings**: ${total:,.0f}")
        lines.append(f"- **Average per scenario**: ${avg:,.0f}")
        lines.append(f"- **Best case**: ${max_savings:,.0f}")
        lines.append(f"- **Worst case**: ${min_savings:,.0f}")
        lines.append("")

    markdown = "\n".join(lines)

    if output_path is not None:
        Path(output_path).write_text(markdown)

    return markdown


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
