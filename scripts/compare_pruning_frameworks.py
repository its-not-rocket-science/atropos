#!/usr/bin/env python3
"""Compare pruning frameworks: magnitude, wanda, sparsegpt.

This script runs all three pruning methods on the same models and compares
sparsity achieved, pruning time, and parameter reduction.

Usage:
    python scripts/compare_pruning_frameworks.py
    python scripts/compare_pruning_frameworks.py --models gpt2 gpt2-medium
    python scripts/compare_pruning_frameworks.py --sparsity 0.1 --device cpu
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.utils.prune as prune
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add src to path for pruning integration
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from atropos.pruning_integration import PruningResult as FrameworkPruningResult
from atropos.pruning_integration import get_pruning_framework

# Default models to test
DEFAULT_MODELS = [
    "gpt2",
]

# Default sparsity level
DEFAULT_SPARSITY = 0.1  # 10%

# Frameworks to compare
ALL_FRAMEWORKS = ["magnitude", "wanda-patched", "sparsegpt-patched"]


@dataclass
class PruningComparisonResult:
    """Result of pruning a single model with a single framework."""

    model: str
    framework: str
    status: str = "failed"  # success, failed, error
    target_sparsity: float = 0.0
    achieved_sparsity: float = 0.0
    original_params: int = 0
    pruned_params: int = 0
    pruning_time_sec: float = 0.0
    output_path: str = ""
    error_message: str = ""
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def parameter_reduction_fraction(self) -> float:
        """Calculate parameter reduction fraction."""
        if self.original_params == 0:
            return 0.0
        return (self.original_params - self.pruned_params) / self.original_params

    @property
    def sparsity_error(self) -> float:
        """Absolute difference between target and achieved sparsity."""
        return abs(self.target_sparsity - self.achieved_sparsity)


@dataclass
class ComparisonReport:
    """Complete comparison report."""

    total_tests: int = 0
    successful: int = 0
    failed: int = 0
    results: list[PruningComparisonResult] = field(default_factory=list)
    start_time: str = ""
    end_time: str = ""
    config: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_sec(self) -> float:
        if self.start_time and self.end_time:
            start = datetime.fromisoformat(self.start_time)
            end = datetime.fromisoformat(self.end_time)
            return (end - start).total_seconds()
        return 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_tests": self.total_tests,
            "successful": self.successful,
            "failed": self.failed,
            "duration_sec": self.duration_sec,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "config": self.config,
            "results": [r.to_dict() for r in self.results],
        }


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    """Count total and non-zero parameters in a model.

    Returns:
        Tuple of (total_params, non_zero_params)
    """
    total = 0
    non_zero = 0
    for param in model.parameters():
        total += param.numel()
        non_zero += int((param != 0).sum().item())
    return total, non_zero


def calculate_sparsity(model: torch.nn.Module) -> float:
    """Calculate the sparsity of a model."""
    total, non_zero = count_parameters(model)
    if total == 0:
        return 0.0
    return 1.0 - (non_zero / total)


def apply_magnitude_pruning(
    model: torch.nn.Module,
    target_sparsity: float,
) -> None:
    """Apply magnitude-based unstructured pruning to a model.

    Args:
        model: The model to prune
        target_sparsity: Target sparsity level (0-1)
    """
    # Collect all prune-able parameters
    parameters_to_prune = []
    for _name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            parameters_to_prune.append((module, "weight"))

    # Apply global unstructured pruning
    if parameters_to_prune:
        prune.global_unstructured(  # type: ignore
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=target_sparsity,
        )

        # Make pruning permanent
        for module, param_name in parameters_to_prune:
            prune.remove(module, param_name)  # type: ignore


def prune_magnitude(
    model_id: str,
    target_sparsity: float,
    output_dir: Path,
    device: str = "cpu",
) -> PruningComparisonResult:
    """Prune a model using magnitude-based pruning.

    Args:
        model_id: HuggingFace model ID
        target_sparsity: Target sparsity level
        output_dir: Directory to save pruned model
        device: Device to use

    Returns:
        PruningComparisonResult
    """
    result = PruningComparisonResult(
        model=model_id,
        framework="magnitude",
        target_sparsity=target_sparsity,
        timestamp=datetime.now().isoformat(),
    )

    try:
        print(f"    Loading {model_id}...", end=" ", flush=True)

        # Load model
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Count original parameters
        result.original_params = sum(p.numel() for p in model.parameters())

        print("[OK]")

        # Apply pruning
        print(
            f"    Applying magnitude pruning ({target_sparsity:.0%} sparsity)...",
            end=" ",
            flush=True,
        )
        start_time = time.time()

        apply_magnitude_pruning(model, target_sparsity)

        result.pruning_time_sec = time.time() - start_time
        result.achieved_sparsity = calculate_sparsity(model)

        print("[OK]")

        # Verify sparsity achieved
        print(f"      Target: {target_sparsity:.2%}, Achieved: {result.achieved_sparsity:.2%}")

        # Save pruned model
        model_output_dir = output_dir / f"{model_id.replace('/', '--')}_magnitude"
        model_output_dir.mkdir(parents=True, exist_ok=True)

        print(f"    Saving to {model_output_dir}...", end=" ", flush=True)
        model.save_pretrained(model_output_dir)
        tokenizer.save_pretrained(model_output_dir)
        print("[OK]")

        result.pruned_params = sum((p != 0).sum().item() for p in model.parameters())
        result.output_path = str(model_output_dir)
        result.status = "success"

        print(f"    [OK] Pruned: {result.original_params:,} -> {result.pruned_params:,} params")
        print(f"         Time: {result.pruning_time_sec:.1f}s")

        # Cleanup
        del model
        import gc

        gc.collect()

    except Exception as e:
        result.error_message = str(e)
        print(f"    [FAIL] {e}")

    return result


def prune_with_framework(
    framework_name: str,
    model_id: str,
    target_sparsity: float,
    output_dir: Path,
    nsamples: int = 1,
    seed: int = 0,
) -> PruningComparisonResult:
    """Prune a model using a pruning framework (wanda-patched or sparsegpt-patched).

    Args:
        framework_name: Framework name (wanda-patched, sparsegpt-patched)
        model_id: HuggingFace model ID
        target_sparsity: Target sparsity level
        output_dir: Directory to save pruned model
        nsamples: Number of calibration samples
        seed: Random seed

    Returns:
        PruningComparisonResult
    """
    result = PruningComparisonResult(
        model=model_id,
        framework=framework_name,
        target_sparsity=target_sparsity,
        timestamp=datetime.now().isoformat(),
    )

    try:
        print(f"    Loading framework {framework_name}...", end=" ", flush=True)
        framework = get_pruning_framework(framework_name)
        print("[OK]")

        # Create output path
        model_safe_name = model_id.replace("/", "--")
        framework_safe_name = framework_name.replace("-", "_")
        model_output_dir = output_dir / f"{model_safe_name}_{framework_safe_name}"
        model_output_dir.mkdir(parents=True, exist_ok=True)

        print(f"    Pruning {model_id} with {framework_name} ({target_sparsity:.0%} sparsity)...")
        print(f"    Output: {model_output_dir}")

        start_time = time.time()

        # Execute pruning
        framework_result: FrameworkPruningResult = framework.prune(
            model_name=model_id,
            output_path=model_output_dir,
            target_sparsity=target_sparsity,
            nsamples=nsamples,
            seed=seed,
        )

        result.pruning_time_sec = time.time() - start_time

        if framework_result.success:
            result.status = "success"
            result.original_params = framework_result.original_params or 0
            result.pruned_params = framework_result.pruned_params or 0
            result.achieved_sparsity = framework_result.sparsity_achieved or 0.0
            result.output_path = str(model_output_dir)

            print(f"    [OK] Pruned: {result.original_params:,} -> {result.pruned_params:,} params")
            print(f"         Sparsity achieved: {result.achieved_sparsity:.2%}")
            print(f"         Time: {result.pruning_time_sec:.1f}s")
        else:
            result.status = "failed"
            result.error_message = framework_result.error_message
            print(f"    [FAIL] Framework reported failure: {framework_result.error_message}")

    except Exception as e:
        result.error_message = str(e)
        print(f"    [FAIL] {e}")

    return result


def run_comparison(
    models: list[str],
    frameworks: list[str],
    target_sparsity: float,
    output_dir: Path,
    nsamples: int = 1,
    seed: int = 0,
    device: str = "cpu",
) -> ComparisonReport:
    """Run comprehensive pruning comparison.

    Args:
        models: List of model IDs to prune
        frameworks: List of framework names to test
        target_sparsity: Target sparsity level
        output_dir: Base output directory
        nsamples: Number of calibration samples for Wanda/SparseGPT
        seed: Random seed
        device: Device to use (currently only affects magnitude pruning)

    Returns:
        ComparisonReport with all results
    """
    total_tests = len(models) * len(frameworks)

    report = ComparisonReport(
        total_tests=total_tests,
        start_time=datetime.now().isoformat(),
        config={
            "models": models,
            "frameworks": frameworks,
            "target_sparsity": target_sparsity,
            "nsamples": nsamples,
            "seed": seed,
            "device": device,
        },
    )

    print("=" * 70)
    print("Pruning Framework Comparison")
    print("=" * 70)
    print(f"Models: {models}")
    print(f"Frameworks: {frameworks}")
    print(f"Target sparsity: {target_sparsity:.0%}")
    print(f"Output directory: {output_dir}")
    print(f"Total tests: {total_tests}")
    print("=" * 70)

    test_count = 0
    for model_id in models:
        for framework in frameworks:
            test_count += 1
            print(f"\n[{test_count}/{total_tests}] {model_id} - {framework}")
            print("-" * 50)

            # Create framework-specific output directory
            framework_output_dir = output_dir / framework / model_id.replace("/", "--")
            framework_output_dir.mkdir(parents=True, exist_ok=True)

            if framework == "magnitude":
                result = prune_magnitude(
                    model_id=model_id,
                    target_sparsity=target_sparsity,
                    output_dir=framework_output_dir,
                    device=device,
                )
            else:
                result = prune_with_framework(
                    framework_name=framework,
                    model_id=model_id,
                    target_sparsity=target_sparsity,
                    output_dir=framework_output_dir,
                    nsamples=nsamples,
                    seed=seed,
                )

            report.results.append(result)

            if result.status == "success":
                report.successful += 1
            else:
                report.failed += 1

    report.end_time = datetime.now().isoformat()
    return report


def print_summary(report: ComparisonReport) -> None:
    """Print comparison summary."""
    print("\n" + "=" * 70)
    print("Comparison Summary")
    print("=" * 70)
    print(f"Total tests: {report.total_tests}")
    print(f"Successful: {report.successful}")
    print(f"Failed: {report.failed}")
    print(f"Duration: {report.duration_sec:.1f}s")

    # Print results table
    print("\nResults:")
    print("-" * 70)
    print(
        f"{'Model':<20} {'Framework':<15} {'Target':<8} {'Achieved':<10} "
        f"{'Time (s)':<10} {'Status'}"
    )
    print("-" * 70)

    for result in report.results:
        target_str = f"{result.target_sparsity:.1%}"
        achieved_str = f"{result.achieved_sparsity:.1%}" if result.achieved_sparsity > 0 else "N/A"
        time_str = f"{result.pruning_time_sec:.1f}" if result.pruning_time_sec > 0 else "N/A"
        status = "OK" if result.status == "success" else "FAIL"

        print(
            f"{result.model:<20} {result.framework:<15} {target_str:<8} "
            f"{achieved_str:<10} {time_str:<10} {status}"
        )

    # Print framework comparison
    print("\nFramework Comparison (average across models):")
    print("-" * 70)
    print(f"{'Framework':<15} {'Avg Sparsity':<15} {'Avg Time (s)':<15} {'Success Rate'}")
    print("-" * 70)

    for framework in report.config.get("frameworks", []):
        framework_results = [
            r for r in report.results if r.framework == framework and r.status == "success"
        ]
        if not framework_results:
            continue

        avg_sparsity = sum(r.achieved_sparsity for r in framework_results) / len(framework_results)
        avg_time = sum(r.pruning_time_sec for r in framework_results) / len(framework_results)
        total_framework_runs = sum(1 for r in report.results if r.framework == framework)
        success_rate = (
            len(framework_results) / total_framework_runs if total_framework_runs > 0 else 0
        )

        print(f"{framework:<15} {avg_sparsity:.2%}{'':<5} {avg_time:<15.1f} {success_rate:.1%}")


def generate_markdown_report(report: ComparisonReport, output_path: Path) -> None:
    """Generate markdown comparison report."""
    lines = [
        "# Pruning Framework Comparison",
        "",
        f"Generated: {report.end_time}",
        f"Duration: {report.duration_sec:.1f}s",
        "",
        "## Configuration",
        "",
    ]

    # Configuration
    for key, value in report.config.items():
        if isinstance(value, list):
            lines.append(f"- **{key}**: {', '.join(value)}")
        else:
            lines.append(f"- **{key}**: {value}")

    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- **Total tests**: {report.total_tests}",
            f"- **Successful**: {report.successful}",
            f"- **Failed**: {report.failed}",
            "",
            "## Results",
            "",
            "| Model | Framework | Target Sparsity | Achieved Sparsity | Original Params | "
            "Pruned Params | Time (s) | Status |",
            "|-------|-----------|-----------------|-------------------|-----------------|"
            "---------------|----------|--------|",
        ]
    )

    for result in report.results:
        target_str = f"{result.target_sparsity:.1%}"
        achieved_str = f"{result.achieved_sparsity:.1%}" if result.achieved_sparsity > 0 else "N/A"
        orig_str = f"{result.original_params:,}" if result.original_params else "N/A"
        pruned_str = f"{result.pruned_params:,}" if result.pruned_params else "N/A"
        time_str = f"{result.pruning_time_sec:.1f}" if result.pruning_time_sec > 0 else "N/A"
        status = "✅" if result.status == "success" else "❌"

        lines.append(
            f"| {result.model} | {result.framework} | {target_str} | {achieved_str} | "
            f"{orig_str} | {pruned_str} | {time_str} | {status} |"
        )

    # Framework comparison table
    lines.extend(
        [
            "",
            "## Framework Comparison",
            "",
            "Average metrics across all successful tests:",
            "",
            "| Framework | Avg Sparsity Achieved | Avg Time (s) | Success Rate |",
            "|-----------|----------------------|--------------|--------------|",
        ]
    )

    for framework in report.config.get("frameworks", []):
        framework_results = [
            r for r in report.results if r.framework == framework and r.status == "success"
        ]
        total_framework_tests = sum(1 for r in report.results if r.framework == framework)

        if not framework_results:
            lines.append(f"| {framework} | N/A | N/A | 0% |")
            continue

        avg_sparsity = sum(r.achieved_sparsity for r in framework_results) / len(framework_results)
        avg_time = sum(r.pruning_time_sec for r in framework_results) / len(framework_results)
        success_rate = (
            len(framework_results) / total_framework_tests if total_framework_tests > 0 else 0
        )

        lines.append(f"| {framework} | {avg_sparsity:.2%} | {avg_time:.1f} | {success_rate:.1%} |")

    # Analysis
    lines.extend(
        [
            "",
            "## Analysis",
            "",
            "### Key Findings",
            "",
            "1. **Sparsity Accuracy**: How close each framework gets to the target sparsity.",
            "2. **Pruning Time**: Computational cost of each pruning method.",
            "3. **Success Rate**: Reliability across different model architectures.",
            "",
            "### Recommendations",
            "",
            "- **Magnitude pruning**: Fastest but may not achieve exact target sparsity "
            "due to global pruning.",
            "- **Wanda**: Good balance of accuracy and speed, "
            "uses first-order gradient information.",
            "- **SparseGPT**: Most accurate but computationally intensive, "
            "uses Hessian approximation.",
            "",
            "### Notes",
            "",
            "- All frameworks use unstructured pruning (weights set to zero).",
            "- Memory savings require sparse tensor formats or structured pruning.",
            "- Results may vary with different models, sparsity levels, and calibration data.",
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nMarkdown report saved to: {output_path}")


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Compare pruning frameworks")
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help=f"Models to prune (default: {DEFAULT_MODELS})",
    )
    parser.add_argument(
        "--frameworks",
        nargs="+",
        choices=["magnitude", "wanda-patched", "sparsegpt-patched", "all"],
        default=["all"],
        help="Frameworks to compare (default: all)",
    )
    parser.add_argument(
        "--sparsity",
        type=float,
        default=DEFAULT_SPARSITY,
        help=f"Target sparsity level (default: {DEFAULT_SPARSITY})",
    )
    parser.add_argument(
        "--nsamples",
        type=int,
        default=1,
        help="Number of calibration samples for Wanda/SparseGPT (default: 1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (default: 0)",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device for magnitude pruning (default: cpu)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("test_data/framework_comparison"),
        help="Output directory (default: test_data/framework_comparison)",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("test_data/framework_comparison.json"),
        help="JSON output path (default: test_data/framework_comparison.json)",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path("test_data/framework_comparison.md"),
        help="Markdown output path (default: test_data/framework_comparison.md)",
    )

    args = parser.parse_args()

    # Determine frameworks
    frameworks = args.frameworks
    if "all" in frameworks:
        frameworks = ALL_FRAMEWORKS

    # Run comparison
    report = run_comparison(
        models=args.models,
        frameworks=frameworks,
        target_sparsity=args.sparsity,
        output_dir=args.output_dir,
        nsamples=args.nsamples,
        seed=args.seed,
        device=args.device,
    )

    # Print summary
    print_summary(report)

    # Save JSON report
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    print(f"\nJSON report saved to: {args.json_output}")

    # Generate markdown report
    generate_markdown_report(report, args.markdown_output)

    if report.failed > 0:
        print(f"\n[WARNING] {report.failed} test(s) failed")
        return 1

    print("\n[OK] All tests completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
