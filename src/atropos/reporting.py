"""Rich reporting formats for optimization outcomes."""

from __future__ import annotations

from collections.abc import Iterable

from .models import OptimizationOutcome


def generate_markdown_report(outcome: OptimizationOutcome) -> str:
    """Generate a markdown report for an outcome."""
    months = None if outcome.break_even_years is None else outcome.break_even_years * 12
    break_even = "never" if months is None else f"{months:.1f} months"
    mem_reduction = (1 - outcome.optimized_memory_gb / outcome.baseline_memory_gb) * 100
    throughput_improvement = (
        (outcome.optimized_throughput_toks_per_sec / outcome.baseline_throughput_toks_per_sec) - 1
    ) * 100

    return f"""# ROI Analysis: {outcome.scenario_name}

## Optimization Strategy: {outcome.strategy_name}
Quality Risk: **{outcome.quality_risk.upper()}**

### Memory Footprint
- Baseline: {outcome.baseline_memory_gb:.1f} GB
- Optimized: {outcome.optimized_memory_gb:.1f} GB
- Reduction: **{mem_reduction:.1f}%**

### Performance
- Throughput: {outcome.baseline_throughput_toks_per_sec:.1f} →\
    {outcome.optimized_throughput_toks_per_sec:.1f} tok/s
- Improvement: **{throughput_improvement:.1f}%**
- Latency factor: {outcome.optimized_latency_factor:.2f}x baseline

### Energy & Cost
- Energy per request: {outcome.baseline_energy_wh_per_request:.1f} →\
    {outcome.optimized_energy_wh_per_request:.1f} Wh
- Annual energy cost: ${outcome.baseline_annual_energy_cost_usd:,.0f} →\
    ${outcome.optimized_annual_energy_cost_usd:,.0f}
- **Annual savings: ${outcome.annual_total_savings_usd:,.0f}**
- **Break-even: {break_even}**

### Environmental Impact
- CO₂e savings: {outcome.annual_co2e_savings_kg:,.0f} kg/year
"""


def generate_comparison_table(outcomes: Iterable[OptimizationOutcome]) -> str:
    """Generate a markdown comparison table for multiple outcomes."""
    rows = [
        "| Strategy | Memory (GB) | Throughput (tok/s) | Annual Savings | Break-even | Risk |",
        "|----------|-------------|-------------------|----------------|------------|------|",
    ]
    for o in outcomes:
        months = None if o.break_even_years is None else o.break_even_years * 12
        break_even = f"{months:.1f} mo" if months is not None else "never"
        rows.append(
            f"| {o.strategy_name} | {o.optimized_memory_gb:.1f} | "
            f"{o.optimized_throughput_toks_per_sec:.0f} | "
            f"${o.annual_total_savings_usd:,.0f} | {break_even} | "
            f"{o.quality_risk} |"
        )
    return "\n".join(rows)


def generate_html_report(outcome: OptimizationOutcome) -> str:
    """Generate an HTML report for an outcome."""
    months = None if outcome.break_even_years is None else outcome.break_even_years * 12
    break_even = "never" if months is None else f"{months:.1f} months"
    mem_reduction = (1 - outcome.optimized_memory_gb / outcome.baseline_memory_gb) * 100
    throughput_improvement = (
        (outcome.optimized_throughput_toks_per_sec / outcome.baseline_throughput_toks_per_sec) - 1
    ) * 100
    risk_colors = {"low": "green", "medium": "orange", "high": "red"}
    bl_mem = outcome.baseline_memory_gb
    opt_mem = outcome.optimized_memory_gb
    bl_thr = outcome.baseline_throughput_toks_per_sec
    opt_thr = outcome.optimized_throughput_toks_per_sec
    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset=\"utf-8\">
  <title>Atropos ROI Analysis: {outcome.scenario_name}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 40px; }}
    .container {{ max-width: 800px; margin: auto; }}
    .header {{ background: #f0f0f0; padding: 20px; border-radius: 5px; }}
    .metric {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd;
      border-radius: 5px; }}
    .improvement {{ color: green; font-weight: bold; }}
    .risk {{ color: {risk_colors[outcome.quality_risk]}; }}
  </style>
</head>
<body>
<div class=\"container\">
  <div class=\"header\">
    <h1>Atropos ROI Analysis</h1>
    <h2>{outcome.scenario_name} + {outcome.strategy_name}</h2>
    <p class=\"risk\">Quality Risk: <strong>{outcome.quality_risk.upper()}</strong></p>
  </div>
  <div class=\"metric\"><h3>Memory Footprint</h3>
    <p>Baseline: {bl_mem:.1f} GB → Optimized: {opt_mem:.1f} GB</p>
    <p class=\"improvement\">Reduction: {mem_reduction:.1f}%</p></div>
  <div class=\"metric\"><h3>Performance</h3>
    <p>Throughput: {bl_thr:.1f} → {opt_thr:.1f} tok/s</p>
    <p class=\"improvement\">Improvement: {throughput_improvement:.1f}%</p>
    <p>Latency factor: {outcome.optimized_latency_factor:.2f}x baseline</p></div>
  <div class=\"metric\"><h3>Financial Impact</h3>
    <p>Annual savings: <strong>${outcome.annual_total_savings_usd:,.0f}</strong></p>
    <p>Break-even: <strong>{break_even}</strong></p></div>
  <div class=\"metric\"><h3>Environmental Impact</h3>
    <p>CO₂e savings: {outcome.annual_co2e_savings_kg:,.0f} kg/year</p></div>
</div>
</body>
</html>"""
