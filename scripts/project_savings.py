#!/usr/bin/env python3
"""Run Atropos analysis to project savings for candidate models.

This script generates baseline projections for all 5 candidate models
before actual pruning is applied. Results are saved for later comparison
with actual pruned model performance.

Usage:
    python scripts/project_savings.py [--output results/projections.json]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# Model configuration with matching presets and strategies
# Note: Available presets are edge-coder, medium-coder, frontier-assistant
MODEL_CONFIGS = [
    {
        "model_id": "gpt2",
        "preset": "edge-coder",
        "params_b": 0.124,
        "strategies": ["mild_pruning", "structured_pruning"],
    },
    {
        "model_id": "gpt2-medium",
        "preset": "edge-coder",  # Use edge-coder for small models
        "params_b": 0.355,
        "strategies": ["mild_pruning", "structured_pruning"],
    },
    {
        "model_id": "gpt2-xl",
        "preset": "medium-coder",
        "params_b": 1.5,
        "strategies": ["mild_pruning", "structured_pruning"],
    },
    {
        "model_id": "facebook/opt-1.3b",
        "preset": "medium-coder",
        "params_b": 1.3,
        "strategies": ["mild_pruning", "structured_pruning"],
    },
    {
        "model_id": "EleutherAI/pythia-2.8b",
        "preset": "medium-coder",  # Use medium-coder (models up to 7B)
        "params_b": 2.8,
        "strategies": ["mild_pruning", "structured_pruning"],
    },
]


@dataclass
class ProjectionResult:
    """Result of Atropos projection for a model/strategy."""

    model_id: str
    preset: str
    strategy: str
    params_b: float
    status: str = "failed"
    projection_time_sec: float = 0.0

    # Projected savings
    memory_reduction_pct: float | None = None
    throughput_improvement_pct: float | None = None
    power_reduction_pct: float | None = None
    annual_savings_usd: float | None = None
    break_even_months: float | None = None
    quality_risk: str | None = None

    # Raw output
    raw_output: dict[str, Any] = field(default_factory=dict)
    error_message: str = ""
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ProjectionReport:
    """Complete report of all projections."""

    total_projections: int = 0
    successful: int = 0
    failed: int = 0
    results: list[ProjectionResult] = field(default_factory=list)
    start_time: str = ""
    end_time: str = ""

    @property
    def duration_sec(self) -> float:
        if self.start_time and self.end_time:
            start = datetime.fromisoformat(self.start_time)
            end = datetime.fromisoformat(self.end_time)
            return (end - start).total_seconds()
        return 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_projections": self.total_projections,
            "successful": self.successful,
            "failed": self.failed,
            "duration_sec": self.duration_sec,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "results": [r.to_dict() for r in self.results],
        }


def run_atropos_preset(
    preset: str,
    strategy: str,
    with_quantization: bool = False,
    timeout_sec: float = 60.0,
) -> ProjectionResult | None:
    """Run Atropos preset command and parse results."""
    import time

    start_time = time.time()
    result = ProjectionResult(
        model_id="",
        preset=preset,
        strategy=strategy,
        params_b=0.0,
        timestamp=datetime.now().isoformat(),
    )

    try:
        cmd = [
            "atropos-llm",
            "preset",
            preset,
            "--strategy",
            strategy,
            "--report",
            "json",
        ]

        if with_quantization:
            cmd.append("--with-quantization")

        proc_result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )

        result.projection_time_sec = time.time() - start_time

        if proc_result.returncode == 0:
            # Parse JSON output
            try:
                stdout = proc_result.stdout
                json_start = stdout.find("{")
                if json_start == -1:
                    raise json.JSONDecodeError("No JSON found", stdout, 0)

                data = json.loads(stdout[json_start:])
                result.raw_output = data

                # Extract key metrics (flat structure)
                # Memory reduction
                baseline_mem = data.get("baseline_memory_gb", 0)
                optimized_mem = data.get("optimized_memory_gb", 0)
                if baseline_mem > 0:
                    result.memory_reduction_pct = (
                        (baseline_mem - optimized_mem) / baseline_mem * 100
                    )

                # Throughput improvement
                baseline_thr = data.get("baseline_throughput_toks_per_sec", 0)
                optimized_thr = data.get("optimized_throughput_toks_per_sec", 0)
                if baseline_thr > 0:
                    result.throughput_improvement_pct = (
                        (optimized_thr - baseline_thr) / baseline_thr * 100
                    )

                # Power reduction
                baseline_pwr = data.get("baseline_power_watts", 0)
                optimized_pwr = data.get("optimized_power_watts", 0)
                if baseline_pwr > 0:
                    result.power_reduction_pct = (baseline_pwr - optimized_pwr) / baseline_pwr * 100

                # Financials
                baseline_cost = data.get("baseline_annual_total_cost_usd", 0)
                optimized_cost = data.get("optimized_annual_total_cost_usd", 0)
                result.annual_savings_usd = baseline_cost - optimized_cost

                break_even_years = data.get("break_even_years")
                if break_even_years is not None:
                    result.break_even_months = break_even_years * 12

                # Quality risk
                result.quality_risk = data.get("quality_risk")

                result.status = "success"

            except json.JSONDecodeError as e:
                result.error_message = f"JSON parse error: {e}"
        else:
            result.error_message = proc_result.stderr[:200]

    except subprocess.TimeoutExpired:
        result.error_message = f"Timeout after {timeout_sec}s"
        result.projection_time_sec = time.time() - start_time
    except Exception as e:
        result.error_message = str(e)
        result.projection_time_sec = time.time() - start_time

    return result


def run_all_projections(
    configs: list[dict[str, Any]] | None = None,
    with_quantization: bool = False,
) -> ProjectionReport:
    """Run projections for all model/strategy combinations."""
    configs = configs or MODEL_CONFIGS

    total = sum(len(c["strategies"]) for c in configs)
    report = ProjectionReport(
        total_projections=total,
        start_time=datetime.now().isoformat(),
    )

    print("=" * 70)
    print("Running Atropos Projections for Pruning Exercise")
    print("=" * 70)
    print(f"Models: {len(configs)}")
    print(f"Total projections: {total}")
    if with_quantization:
        print("Including quantization bonus")
    print()

    count = 0
    for config in configs:
        model_id = config["model_id"]
        preset = config["preset"]
        params_b = config["params_b"]

        print(f"\nModel: {model_id} ({params_b}B params)")
        print(f"Preset: {preset}")
        print("-" * 50)

        for strategy in config["strategies"]:
            count += 1
            print(f"  [{count}/{total}] Strategy: {strategy}...", end=" ", flush=True)

            result = run_atropos_preset(
                preset=preset,
                strategy=strategy,
                with_quantization=with_quantization,
            )

            if result:
                result.model_id = model_id
                result.params_b = params_b
                report.results.append(result)

                if result.status == "success":
                    report.successful += 1
                    print(f"OK ({result.projection_time_sec:.1f}s)")
                    if result.memory_reduction_pct:
                        print(f"      Memory: -{result.memory_reduction_pct:.1f}%")
                    if result.annual_savings_usd:
                        print(f"      Savings: ${result.annual_savings_usd:,.0f}/year")
                    if result.break_even_months:
                        print(f"      Break-even: {result.break_even_months:.1f} months")
                else:
                    report.failed += 1
                    print(f"FAIL - {result.error_message[:60]}")

    report.end_time = datetime.now().isoformat()
    return report


def print_summary(report: ProjectionReport) -> None:
    """Print summary of all projections."""
    print("\n" + "=" * 70)
    print("Projection Summary")
    print("=" * 70)
    print(f"Total: {report.total_projections}")
    print(f"Successful: {report.successful}")
    print(f"Failed: {report.failed}")
    print(f"Duration: {report.duration_sec:.1f}s")

    # Group by model
    print("\nProjections by Model:")
    print("-" * 70)

    by_model: dict[str, list[ProjectionResult]] = {}
    for r in report.results:
        by_model.setdefault(r.model_id, []).append(r)

    for model_id, results in by_model.items():
        print(f"\n{model_id}:")
        for r in results:
            status = "OK" if r.status == "success" else "FAIL"
            print(f"  {status:4s} {r.strategy:20s} ", end="")
            if r.memory_reduction_pct:
                print(f"mem={r.memory_reduction_pct:+.1f}% ", end="")
            if r.annual_savings_usd:
                print(f"save=${r.annual_savings_usd:,.0f}", end="")
            print()

    # Strategy comparison
    print("\n\nAggregated by Strategy:")
    print("-" * 70)

    by_strategy: dict[str, list[ProjectionResult]] = {}
    for r in report.results:
        by_strategy.setdefault(r.strategy, []).append(r)

    for strategy, results in by_strategy.items():
        successful = [r for r in results if r.status == "success"]
        if successful:
            avg_mem = sum(
                r.memory_reduction_pct for r in successful if r.memory_reduction_pct is not None
            ) / len(successful)
            avg_savings = sum(
                r.annual_savings_usd for r in successful if r.annual_savings_usd is not None
            ) / len(successful)

            print(f"\n{strategy}:")
            print(f"  Success rate: {len(successful)}/{len(results)}")
            print(f"  Avg memory reduction: {avg_mem:.1f}%")
            print(f"  Avg annual savings: ${avg_savings:,.0f}")


def generate_markdown_report(report: ProjectionReport, output_path: Path) -> None:
    """Generate a markdown report for documentation."""
    lines = [
        "# Atropos Pruning Projections Report",
        "",
        f"Generated: {report.end_time}",
        f"Duration: {report.duration_sec:.1f}s",
        "",
        "## Summary",
        "",
        f"- **Total projections:** {report.total_projections}",
        f"- **Successful:** {report.successful}",
        f"- **Failed:** {report.failed}",
        "",
        "## Projections by Model",
        "",
        "| Model | Params | Strategy | Memory Red. | Throughput | Annual Savings | Break-even |",
        "|-------|--------|----------|-------------|------------|----------------|------------|",
    ]

    for r in report.results:
        if r.status != "success":
            continue

        mem = f"{r.memory_reduction_pct:.1f}%" if r.memory_reduction_pct else "N/A"
        thr = f"+{r.throughput_improvement_pct:.1f}%" if r.throughput_improvement_pct else "N/A"
        savings = f"${r.annual_savings_usd:,.0f}" if r.annual_savings_usd else "N/A"
        be = f"{r.break_even_months:.1f}mo" if r.break_even_months else "N/A"

        lines.append(
            f"| {r.model_id} | {r.params_b}B | {r.strategy} | {mem} | {thr} | {savings} | {be} |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "These are *projected* savings before actual pruning is applied.",
            "Actual results will be compared against these projections.",
            "",
        ]
    )

    output_path.write_text("\n".join(lines))
    print(f"\nMarkdown report saved to: {output_path}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Run Atropos projections for pruning exercise")
    parser.add_argument(
        "--with-quantization",
        action="store_true",
        help="Include quantization bonus in projections",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("test_data/projections.json"),
        help="Output JSON file (default: test_data/projections.json)",
    )
    parser.add_argument(
        "--markdown",
        "-m",
        type=Path,
        default=Path("test_data/projections.md"),
        help="Output markdown report (default: test_data/projections.md)",
    )

    args = parser.parse_args()

    # Run projections
    report = run_all_projections(with_quantization=args.with_quantization)

    # Print summary
    print_summary(report)

    # Save JSON report
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report.to_dict(), indent=2))
    print(f"\nJSON report saved to: {args.output}")

    # Generate markdown report
    generate_markdown_report(report, args.markdown)

    if report.failed > 0:
        print(f"\n[WARNING] {report.failed} projection(s) failed")
        sys.exit(1)

    print("\n[OK] All projections completed!")


if __name__ == "__main__":
    main()
