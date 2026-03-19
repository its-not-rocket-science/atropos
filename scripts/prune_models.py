#!/usr/bin/env python3
"""Prune candidate models using PyTorch native pruning.

This script applies magnitude-based pruning to the downloaded models
to achieve target sparsity levels for the pruning exercise.

Pruning strategies:
- mild_pruning: ~10% sparsity (matches Atropos mild_pruning preset)
- structured_pruning: ~22% sparsity (matches Atropos structured_pruning preset)

Usage:
    python scripts/prune_models.py [--models gpt2] [--strategies mild_pruning]
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

# Model configurations with target sparsity
MODEL_CONFIGS = [
    {
        "model_id": "gpt2",
        "params_b": 0.124,
        "strategies": {
            "mild_pruning": 0.10,
            "structured_pruning": 0.22,
        },
    },
    {
        "model_id": "gpt2-medium",
        "params_b": 0.355,
        "strategies": {
            "mild_pruning": 0.10,
            "structured_pruning": 0.22,
        },
    },
    {
        "model_id": "gpt2-xl",
        "params_b": 1.5,
        "strategies": {
            "mild_pruning": 0.10,
            "structured_pruning": 0.22,
        },
    },
    {
        "model_id": "facebook/opt-1.3b",
        "params_b": 1.3,
        "strategies": {
            "mild_pruning": 0.10,
            "structured_pruning": 0.22,
        },
    },
    {
        "model_id": "EleutherAI/pythia-2.8b",
        "params_b": 2.8,
        "strategies": {
            "mild_pruning": 0.10,
            "structured_pruning": 0.22,
        },
    },
]


@dataclass
class PruningResult:
    """Result of pruning a model."""

    model_id: str
    strategy: str
    status: str = "failed"
    original_params: int = 0
    pruned_params: int = 0
    target_sparsity: float = 0.0
    actual_sparsity: float = 0.0
    pruning_time_sec: float = 0.0
    output_path: str = ""
    error_message: str = ""
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PruningReport:
    """Report of all pruning operations."""

    total_models: int = 0
    successful: int = 0
    failed: int = 0
    results: list[PruningResult] = field(default_factory=list)
    start_time: str = ""
    end_time: str = ""
    output_dir: str = ""

    @property
    def duration_sec(self) -> float:
        if self.start_time and self.end_time:
            start = datetime.fromisoformat(self.start_time)
            end = datetime.fromisoformat(self.end_time)
            return (end - start).total_seconds()
        return 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_models": self.total_models,
            "successful": self.successful,
            "failed": self.failed,
            "duration_sec": self.duration_sec,
            "output_dir": self.output_dir,
            "start_time": self.start_time,
            "end_time": self.end_time,
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
        non_zero += (param != 0).sum().item()  # type: ignore[assignment]
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
        prune.global_unstructured(  # type: ignore[no-untyped-call]
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=target_sparsity,
        )

        # Make pruning permanent
        for module, param_name in parameters_to_prune:
            prune.remove(module, param_name)  # type: ignore[no-untyped-call]


def prune_model(
    model_id: str,
    strategy: str,
    target_sparsity: float,
    output_dir: Path,
    cache_dir: Path | None = None,
    device: str = "cpu",
) -> PruningResult:
    """Prune a single model.

    Args:
        model_id: HuggingFace model ID
        strategy: Pruning strategy name
        target_sparsity: Target sparsity level
        output_dir: Directory to save pruned model
        cache_dir: Cache directory for models
        device: Device to use

    Returns:
        PruningResult with operation details
    """
    result = PruningResult(
        model_id=model_id,
        strategy=strategy,
        target_sparsity=target_sparsity,
        timestamp=datetime.now().isoformat(),
    )

    try:
        print(f"  Loading {model_id}...", end=" ", flush=True)

        # Load model
        load_kwargs: dict[str, Any] = {
            "torch_dtype": torch.float32,
        }
        cache_dir_str = str(cache_dir) if cache_dir else None
        if cache_dir_str:
            load_kwargs["cache_dir"] = cache_dir_str

        model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir=cache_dir_str,
        )

        print("[OK]")

        # Count original parameters
        result.original_params = sum(p.numel() for p in model.parameters())

        # Apply pruning
        print(f"  Applying {strategy} ({target_sparsity:.0%} sparsity)...", end=" ", flush=True)
        start_time = time.time()

        apply_magnitude_pruning(model, target_sparsity)

        result.pruning_time_sec = time.time() - start_time
        result.actual_sparsity = calculate_sparsity(model)

        print("[OK]")

        # Verify sparsity achieved
        print(f"    Target: {target_sparsity:.2%}, Achieved: {result.actual_sparsity:.2%}")

        # Save pruned model
        model_output_dir = output_dir / f"{model_id.replace('/', '--')}_{strategy}"
        model_output_dir.mkdir(parents=True, exist_ok=True)

        print(f"  Saving to {model_output_dir}...", end=" ", flush=True)
        model.save_pretrained(model_output_dir)
        tokenizer.save_pretrained(model_output_dir)
        print("[OK]")

        result.pruned_params = sum((p != 0).sum().item() for p in model.parameters())
        result.output_path = str(model_output_dir)
        result.status = "success"

        print(f"  [OK] Pruned: {result.original_params:,} -> {result.pruned_params:,} params")
        print(f"       Time: {result.pruning_time_sec:.1f}s")

        # Cleanup
        del model
        import gc

        gc.collect()

    except Exception as e:
        result.error_message = str(e)
        print(f"  [FAIL] {e}")

    return result


def prune_all_models(
    output_dir: Path,
    cache_dir: Path | None = None,
    configs: list[dict[str, Any]] | None = None,
    strategies: list[str] | None = None,
    device: str = "cpu",
) -> PruningReport:
    """Prune all configured models.

    Args:
        output_dir: Directory to save pruned models
        cache_dir: Cache directory for source models
        configs: Model configurations (default: MODEL_CONFIGS)
        strategies: Strategies to run (default: all)
        device: Device to use

    Returns:
        PruningReport with all results
    """
    configs = configs or MODEL_CONFIGS
    strategies = strategies or ["mild_pruning", "structured_pruning"]

    # Calculate total operations
    total = sum(len([s for s in c["strategies"].keys() if s in strategies]) for c in configs)  # type: ignore[misc, union-attr]

    report = PruningReport(
        total_models=total,
        start_time=datetime.now().isoformat(),
        output_dir=str(output_dir),
    )

    print("=" * 70)
    print("Pruning Candidate Models")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print(f"Cache directory: {cache_dir or 'default'}")
    print(f"Device: {device}")
    print(f"Models to prune: {total}")

    count = 0
    for config in configs:
        model_id = str(config["model_id"])  # type: ignore[index]

        for strategy_name, sparsity_raw in config["strategies"].items():  # type: ignore[union-attr]
            if strategy_name not in strategies:
                continue
            sparsity = float(sparsity_raw)

            count += 1
            print(f"\n[{count}/{total}] {model_id} - {strategy_name}")
            print("-" * 50)

            result = prune_model(
                model_id=model_id,
                strategy=strategy_name,
                target_sparsity=sparsity,
                output_dir=output_dir,
                cache_dir=cache_dir,
                device=device,
            )
            report.results.append(result)

            if result.status == "success":
                report.successful += 1
            else:
                report.failed += 1

    report.end_time = datetime.now().isoformat()
    return report


def print_summary(report: PruningReport) -> None:
    """Print summary of pruning operations."""
    print("\n" + "=" * 70)
    print("Pruning Summary")
    print("=" * 70)
    print(f"Total: {report.total_models}")
    print(f"Successful: {report.successful}")
    print(f"Failed: {report.failed}")
    print(f"Duration: {report.duration_sec:.1f}s")
    print(f"Output directory: {report.output_dir}")

    print("\nResults by Model:")
    print("-" * 70)

    # Group by model
    by_model: dict[str, list[PruningResult]] = {}
    for r in report.results:
        by_model.setdefault(r.model_id, []).append(r)

    for model_id, results in by_model.items():
        print(f"\n{model_id}:")
        for r in results:
            status = "OK" if r.status == "success" else "FAIL"
            print(f"  {status:4s} {r.strategy:20s} ", end="")
            if r.actual_sparsity > 0:
                print(f"sparsity={r.actual_sparsity:.1%} ", end="")
            if r.pruning_time_sec > 0:
                print(f"time={r.pruning_time_sec:.1f}s", end="")
            print()


def generate_markdown_report(report: PruningReport, output_path: Path) -> None:
    """Generate markdown report."""
    lines = [
        "# Pruning Exercise Results",
        "",
        f"Generated: {report.end_time}",
        f"Duration: {report.duration_sec:.1f}s",
        "",
        "## Summary",
        "",
        f"- **Total models pruned:** {report.total_models}",
        f"- **Successful:** {report.successful}",
        f"- **Failed:** {report.failed}",
        "",
        "## Pruned Models",
        "",
        "| Model | Strategy | Target Sparsity | Achieved Sparsity | Original Params | "
        "Pruned Params | Time |",
        "|-------|----------|-----------------|-------------------|-----------------|---------------|------|",
    ]

    for r in report.results:
        if r.status != "success":
            continue

        orig = f"{r.original_params:,}" if r.original_params else "N/A"
        pruned = f"{r.pruned_params:,}" if r.pruned_params else "N/A"
        time_str = f"{r.pruning_time_sec:.1f}s" if r.pruning_time_sec else "N/A"

        lines.append(
            f"| {r.model_id} | {r.strategy} | {r.target_sparsity:.0%} | "
            f"{r.actual_sparsity:.1%} | {orig} | {pruned} | {time_str} |"
        )

    lines.extend(
        [
            "",
            "## Comparison with Projections",
            "",
            "Compare these actual results with `test_data/projections.md`",
            "to validate Atropos projection accuracy.",
            "",
        ]
    )

    output_path.write_text("\n".join(lines))
    print(f"\nMarkdown report saved to: {output_path}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Prune candidate models for Atropos validation")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("test_data/pruned_models"),
        help="Directory to save pruned models (default: test_data/pruned_models)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("test_data/models"),
        help="Cache directory for source models (default: test_data/models)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Specific model IDs to prune (default: all)",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=["mild_pruning", "structured_pruning", "all"],
        default=["all"],
        help="Strategies to apply (default: all)",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device for pruning (default: cpu)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("test_data/pruning_results.md"),
        help="Output markdown report path",
    )

    args = parser.parse_args()

    # Filter configs if models specified
    configs = MODEL_CONFIGS
    if args.models:
        configs = [c for c in configs if c["model_id"] in args.models]
        if not configs:
            print(f"Error: No matching models found in {args.models}")
            sys.exit(1)

    # Determine strategies
    strategies = args.strategies
    if "all" in strategies:
        strategies = ["mild_pruning", "structured_pruning"]

    # Run pruning
    report = prune_all_models(
        output_dir=args.output_dir,
        cache_dir=args.cache_dir if args.cache_dir.exists() else None,
        configs=configs,
        strategies=strategies,
        device=args.device,
    )

    # Print summary
    print_summary(report)

    # Save JSON report
    json_path = args.output_dir.parent / "pruning_report.json"
    json_path.write_text(json.dumps(report.to_dict(), indent=2))
    print(f"\nJSON report saved to: {json_path}")

    # Generate markdown report
    generate_markdown_report(report, args.report)

    if report.failed > 0:
        print(f"\n[WARNING] {report.failed} pruning operation(s) failed")
        sys.exit(1)

    print("\n[OK] All models pruned successfully!")


if __name__ == "__main__":
    main()
