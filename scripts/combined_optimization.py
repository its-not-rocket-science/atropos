#!/usr/bin/env python3
"""Combine pruning and quantization optimizations and measure cumulative effects.

This script applies pruning and quantization in sequence to evaluate
combined memory reduction, speed improvement, and quality impact.

Pipeline:
1. Baseline model (original)
2. Pruned only
3. Quantized only
4. Pruned + Quantized (combined)

Usage:
    python scripts/combined_optimization.py [--models gpt2] [--strategies mild_pruning]
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
import torch.quantization as tq
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import from other scripts (add scripts directory to path)
sys.path.insert(0, str(Path(__file__).parent))
try:
    from prune_models import apply_magnitude_pruning, count_parameters  # type: ignore
    from quantize_models import apply_dynamic_quantization, get_model_size_mb  # type: ignore
except ImportError:
    print("WARNING: Could not import from prune_models or quantize_models")

    # Define fallback functions
    def apply_magnitude_pruning(model: torch.nn.Module, target_sparsity: float) -> None:
        parameters_to_prune = []
        for _name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                parameters_to_prune.append((module, "weight"))
        if parameters_to_prune:
            prune.global_unstructured(  # type: ignore[no-untyped-call]
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=target_sparsity,
            )
            for module, param_name in parameters_to_prune:
                prune.remove(module, param_name)  # type: ignore[no-untyped-call]

    def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
        total = 0
        non_zero = 0
        for param in model.parameters():
            total += param.numel()
            non_zero += (param != 0).sum().item()  # type: ignore[assignment]
        return total, non_zero

    def apply_dynamic_quantization(model: torch.nn.Module) -> torch.nn.Module:
        return tq.quantize_dynamic(  # type: ignore[attr-defined, no-any-return]
            model, {torch.nn.Linear}, dtype=torch.qint8
        )

    def get_model_size_mb(model: torch.nn.Module) -> float:
        """Calculate model size in MB (parameters only, approximate).

        Handles quantized tensors (qint8) which have element_size() == 1.
        """
        total_bytes = 0
        for p in model.parameters():
            total_bytes += p.numel() * p.element_size()
        for b in model.buffers():
            total_bytes += b.numel() * b.element_size()
        return total_bytes / (1024 * 1024)


# Model configurations
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
]

# Quantization methods
QUANTIZATION_METHODS = ["dynamic"]


@dataclass
class OptimizationResult:
    """Result of optimizing a model with a specific combination."""

    model_id: str
    strategy: str
    quantization_method: str
    combination: str  # "pruned_only", "quantized_only", "combined"
    status: str = "failed"

    # Model metrics
    original_params: int = 0
    optimized_params: int = 0
    original_size_mb: float = 0.0
    optimized_size_mb: float = 0.0
    memory_reduction_pct: float = 0.0

    # Performance metrics
    inference_time_ms_baseline: float = 0.0
    inference_time_ms_optimized: float = 0.0
    speedup_pct: float = 0.0

    # Quality metrics (to be filled by benchmark)
    perplexity_baseline: float = 0.0
    perplexity_optimized: float = 0.0
    perplexity_change_pct: float = 0.0
    completion_score_baseline: float = 0.0
    completion_score_optimized: float = 0.0
    completion_score_change_pct: float = 0.0

    optimization_time_sec: float = 0.0
    output_path: str = ""
    error_message: str = ""
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CombinedReport:
    """Report of all combined optimization experiments."""

    total_experiments: int = 0
    successful: int = 0
    failed: int = 0
    results: list[OptimizationResult] = field(default_factory=list)
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
            "total_experiments": self.total_experiments,
            "successful": self.successful,
            "failed": self.failed,
            "duration_sec": self.duration_sec,
            "output_dir": self.output_dir,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "results": [r.to_dict() for r in self.results],
        }


def measure_inference_time(
    model: torch.nn.Module, tokenizer: AutoTokenizer, device: str = "cpu"
) -> float:
    """Measure average inference time for a simple forward pass."""
    inputs = tokenizer("Hello, world!", return_tensors="pt").to(device)
    model.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model(**inputs)

    # Measure
    times = []
    with torch.no_grad():
        for _ in range(10):
            start = time.time()
            _ = model(**inputs)
            times.append((time.time() - start) * 1000)  # ms

    return sum(times) / len(times)


def calculate_perplexity(
    model: torch.nn.Module, tokenizer: AutoTokenizer, text: str, device: str = "cpu"
) -> float:
    """Calculate perplexity of a text under the model."""
    import math

    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)  # type: ignore[operator]
        loss = outputs.loss
        perplexity = math.exp(loss.item())

    return perplexity


def evaluate_completion(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompt: str,
    expected_keywords: list[str],
    device: str = "cpu",
    max_new_tokens: int = 50,
) -> float:
    """Evaluate a single completion prompt and return keyword score."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(  # type: ignore
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )

    completion = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated = completion[len(prompt) :]

    keyword_hits = sum(1 for kw in expected_keywords if kw in generated)
    score = keyword_hits / len(expected_keywords) if expected_keywords else 0
    return score


def measure_quality(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    device: str = "cpu",
) -> tuple[float, float]:
    """Measure average perplexity and completion score on a few prompts."""
    prompts = [
        {
            "text": 'def hello_world():\n    """Print hello world"""\n    ',
            "keywords": ["print", "hello"],
        },
        {
            "text": 'def add(a, b):\n    """Add two numbers"""\n    ',
            "keywords": ["return", "a", "b"],
        },
    ]

    perplexities = []
    completion_scores = []

    for prompt in prompts:
        ppl = calculate_perplexity(model, tokenizer, prompt["text"], device)  # type: ignore[arg-type]
        perplexities.append(ppl)

        score = evaluate_completion(model, tokenizer, prompt["text"], prompt["keywords"], device)  # type: ignore[arg-type]
        completion_scores.append(score)

    avg_perplexity = sum(perplexities) / len(perplexities)
    avg_score = sum(completion_scores) / len(completion_scores)
    return avg_perplexity, avg_score


def optimize_model(
    model_id: str,
    strategy: str,
    target_sparsity: float,
    quantization_method: str,
    combination: str,
    output_dir: Path,
    cache_dir: Path | None = None,
    device: str = "cpu",
) -> OptimizationResult:
    """Apply optimization combination to a model."""
    result = OptimizationResult(
        model_id=model_id,
        strategy=strategy,
        quantization_method=quantization_method,
        combination=combination,
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
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = model.to(device)
        model.eval()
        print("[OK]")

        # Baseline measurements
        result.original_params = sum(p.numel() for p in model.parameters())
        result.original_size_mb = get_model_size_mb(model)
        result.inference_time_ms_baseline = measure_inference_time(model, tokenizer, device)
        result.perplexity_baseline, result.completion_score_baseline = measure_quality(
            model, tokenizer, device
        )

        # Apply optimizations
        start_time = time.time()
        optimized_model = model

        if combination in ["pruned_only", "combined"]:
            print(
                f"  Applying {strategy} pruning ({target_sparsity:.0%} sparsity)...",
                end=" ",
                flush=True,
            )
            apply_magnitude_pruning(optimized_model, target_sparsity)
            print("[OK]")

        if combination in ["quantized_only", "combined"]:
            print(f"  Applying {quantization_method} quantization...", end=" ", flush=True)
            optimized_model = apply_dynamic_quantization(optimized_model)
            print("[OK]")

        result.optimization_time_sec = time.time() - start_time

        # Optimized measurements
        total, non_zero = count_parameters(optimized_model)
        result.optimized_params = non_zero
        result.optimized_size_mb = get_model_size_mb(optimized_model)
        result.memory_reduction_pct = (
            (result.original_size_mb - result.optimized_size_mb) / result.original_size_mb * 100
        )

        result.inference_time_ms_optimized = measure_inference_time(
            optimized_model, tokenizer, device
        )
        if result.inference_time_ms_baseline > 0:
            result.speedup_pct = (
                (result.inference_time_ms_baseline - result.inference_time_ms_optimized)
                / result.inference_time_ms_baseline
                * 100
            )

        result.perplexity_optimized, result.completion_score_optimized = measure_quality(
            optimized_model, tokenizer, device
        )

        if result.perplexity_baseline > 0:
            result.perplexity_change_pct = (
                (result.perplexity_optimized - result.perplexity_baseline)
                / result.perplexity_baseline
                * 100
            )
        if result.completion_score_baseline > 0:
            result.completion_score_change_pct = (
                (result.completion_score_optimized - result.completion_score_baseline)
                / result.completion_score_baseline
                * 100
            )

        # Save optimized model
        if combination == "pruned_only":
            suffix = f"{strategy}"
        elif combination == "quantized_only":
            suffix = f"{quantization_method}"
        else:
            suffix = f"{strategy}+{quantization_method}"

        model_output_dir = output_dir / f"{model_id.replace('/', '--')}_{suffix}"
        model_output_dir.mkdir(parents=True, exist_ok=True)

        print(f"  Saving to {model_output_dir}...", end=" ", flush=True)
        torch.save(
            optimized_model.state_dict(),
            model_output_dir / "pytorch_model.bin",
        )
        tokenizer.save_pretrained(model_output_dir)
        config = {
            "model_id": model_id,
            "strategy": strategy,
            "quantization_method": quantization_method,
            "combination": combination,
            "target_sparsity": target_sparsity,
            "optimized_at": datetime.now().isoformat(),
        }
        (model_output_dir / "optimization_config.json").write_text(json.dumps(config, indent=2))
        print("[OK]")

        result.output_path = str(model_output_dir)
        result.status = "success"

        print(
            f"  [OK] Size: {result.original_size_mb:.1f} MB -> "
            f"{result.optimized_size_mb:.1f} MB ({result.memory_reduction_pct:.1f}% reduction)"
        )
        print(
            f"       Inference: {result.inference_time_ms_baseline:.1f} ms -> "
            f"{result.inference_time_ms_optimized:.1f} ms ({result.speedup_pct:+.1f}% speedup)"
        )
        print(
            f"       Perplexity: {result.perplexity_baseline:.1f} -> "
            f"{result.perplexity_optimized:.1f} ({result.perplexity_change_pct:+.1f}%)"
        )
        print(
            f"       Completion score: {result.completion_score_baseline:.2f} -> "
            f"{result.completion_score_optimized:.2f} ({result.completion_score_change_pct:+.1f}%)"
        )
        print(f"       Time: {result.optimization_time_sec:.1f}s")

        # Cleanup
        del model, optimized_model
        import gc

        gc.collect()

    except Exception as e:
        result.error_message = str(e)
        print(f"  [FAIL] {e}")

    return result


def run_combined_optimization(
    output_dir: Path,
    cache_dir: Path | None = None,
    configs: list[dict[str, Any]] | None = None,
    strategies: list[str] | None = None,
    quantization_methods: list[str] | None = None,
    device: str = "cpu",
) -> CombinedReport:
    """Run all combined optimization experiments."""
    configs = configs or MODEL_CONFIGS
    strategies = strategies or ["mild_pruning"]
    quantization_methods = quantization_methods or ["dynamic"]

    # Build experiment list
    experiments: list[dict[str, Any]] = []
    for config in configs:
        model_id = str(config["model_id"])  # type: ignore[index]
        for strategy in strategies:
            target_sparsity = float(config["strategies"][strategy])  # type: ignore[index]
            for qmethod in quantization_methods:
                # Three combinations: pruned_only, quantized_only, combined
                experiments.append(
                    {
                        "model_id": model_id,
                        "strategy": strategy,
                        "target_sparsity": target_sparsity,
                        "quantization_method": qmethod,
                        "combination": "pruned_only",
                    }
                )
                experiments.append(
                    {
                        "model_id": model_id,
                        "strategy": strategy,
                        "target_sparsity": target_sparsity,
                        "quantization_method": qmethod,
                        "combination": "quantized_only",
                    }
                )
                experiments.append(
                    {
                        "model_id": model_id,
                        "strategy": strategy,
                        "target_sparsity": target_sparsity,
                        "quantization_method": qmethod,
                        "combination": "combined",
                    }
                )

    report = CombinedReport(
        total_experiments=len(experiments),
        start_time=datetime.now().isoformat(),
        output_dir=str(output_dir),
    )

    print("=" * 70)
    print("Combined Pruning + Quantization Study")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print(f"Cache directory: {cache_dir or 'default'}")
    print(f"Device: {device}")
    print(f"Experiments to run: {len(experiments)}")

    for i, exp in enumerate(experiments, 1):
        model_label = f"{exp['model_id']} - {exp['combination']}"
        if exp["combination"] != "quantized_only":
            model_label += f" ({exp['strategy']})"
        print(f"\n[{i}/{len(experiments)}] {model_label}")
        print("-" * 50)

        result = optimize_model(
            model_id=exp["model_id"],
            strategy=exp["strategy"],
            target_sparsity=exp["target_sparsity"],
            quantization_method=exp["quantization_method"],
            combination=exp["combination"],
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


def print_summary(report: CombinedReport) -> None:
    """Print summary of combined optimization experiments."""
    print("\n" + "=" * 70)
    print("Combined Optimization Summary")
    print("=" * 70)
    print(f"Total experiments: {report.total_experiments}")
    print(f"Successful: {report.successful}")
    print(f"Failed: {report.failed}")
    print(f"Duration: {report.duration_sec:.1f}s")
    print(f"Output directory: {report.output_dir}")

    print("\nResults by Model:")
    print("-" * 70)

    # Group by model
    by_model: dict[str, list[OptimizationResult]] = {}
    for r in report.results:
        by_model.setdefault(r.model_id, []).append(r)

    for model_id, results in by_model.items():
        print(f"\n{model_id}:")
        for r in results:
            status = "OK" if r.status == "success" else "FAIL"
            combo = r.combination
            if combo != "quantized_only":
                combo += f" ({r.strategy})"
            print(f"  {status:4s} {combo:30s} ", end="")
            if r.memory_reduction_pct > 0:
                print(f"size={r.memory_reduction_pct:.1f}% ", end="")
            if r.speedup_pct != 0:
                print(f"speed={r.speedup_pct:+.1f}% ", end="")
            if r.perplexity_change_pct != 0:
                print(f"ppl={r.perplexity_change_pct:+.1f}% ", end="")
            print()


def generate_markdown_report(report: CombinedReport, output_path: Path) -> None:
    """Generate markdown report."""
    lines = [
        "# Combined Pruning + Quantization Study",
        "",
        f"Generated: {report.end_time}",
        f"Duration: {report.duration_sec:.1f}s",
        "",
        "## Summary",
        "",
        f"- **Total experiments:** {report.total_experiments}",
        f"- **Successful:** {report.successful}",
        f"- **Failed:** {report.failed}",
        "",
        "## Results",
        "",
        "| Model | Combination | Strategy | Memory Reduction | Speedup | "
        "Perplexity Change | Score Change | Status |",
        "|-------|-------------|----------|------------------|---------|"
        "-------------------|--------------|--------|",
    ]

    for r in report.results:
        if r.status != "success":
            continue

        combo = r.combination
        strategy = r.strategy if r.combination != "quantized_only" else "N/A"
        lines.append(
            f"| {r.model_id} | {combo} | {strategy} | {r.memory_reduction_pct:.1f}% | "
            f"{r.speedup_pct:+.1f}% | {r.perplexity_change_pct:+.1f}% | "
            f"{r.completion_score_change_pct:+.1f}% | OK |"
        )

    # Add cumulative analysis
    lines.extend(
        [
            "",
            "## Cumulative Effects Analysis",
            "",
            "Comparing individual vs combined optimizations:",
            "",
            "| Model | Optimization | Memory Reduction | Speedup | Perplexity Change |",
            "|-------|--------------|------------------|---------|-------------------|",
        ]
    )

    # Group by model and compute cumulative
    by_model: dict[str, list[OptimizationResult]] = {}
    for r in report.results:
        if r.status == "success":
            by_model.setdefault(r.model_id, []).append(r)

    for model_id, results in by_model.items():
        # Find baseline? we don't have baseline in results.
        # We'll compute from first result's baseline metrics.
        # Instead, we'll just list each combination
        for r in results:
            combo = r.combination
            if combo == "quantized_only":
                label = "Quantization only"
            elif combo == "pruned_only":
                label = f"Pruning only ({r.strategy})"
            else:
                label = f"Combined ({r.strategy}+quantization)"
            lines.append(
                f"| {model_id} | {label} | {r.memory_reduction_pct:.1f}% | "
                f"{r.speedup_pct:+.1f}% | {r.perplexity_change_pct:+.1f}% |"
            )

    lines.extend(
        [
            "",
            "## Insights",
            "",
            "1. **Cumulative savings**: Combined optimization should provide "
            "multiplicative memory reduction and additive speed improvements.",
            "2. **Quality impact**: Check if combined optimization amplifies "
            "quality degradation beyond individual optimizations.",
            "3. **ROI implications**: Combined optimizations may accelerate "
            "break-even timelines in Atropos projections.",
            "",
        ]
    )

    output_path.write_text("\n".join(lines))
    print(f"\nMarkdown report saved to: {output_path}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Combine pruning and quantization optimizations")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("test_data/combined_optimization"),
        help="Directory to save optimized models (default: test_data/combined_optimization)",
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
        help="Specific model IDs to optimize (default: all)",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=["mild_pruning", "structured_pruning"],
        default=["mild_pruning"],
        help="Pruning strategies to apply (default: mild_pruning)",
    )
    parser.add_argument(
        "--quantization-methods",
        nargs="+",
        choices=["dynamic"],
        default=["dynamic"],
        help="Quantization methods to apply (default: dynamic)",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device for optimization (default: cpu)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("test_data/combined_optimization_report.md"),
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

    # Run combined optimization
    report = run_combined_optimization(
        output_dir=args.output_dir,
        cache_dir=args.cache_dir if args.cache_dir.exists() else None,
        configs=configs,
        strategies=args.strategies,
        quantization_methods=args.quantization_methods,
        device=args.device,
    )

    # Print summary
    print_summary(report)

    # Save JSON report
    json_path = args.output_dir.parent / "combined_optimization_report.json"
    json_path.write_text(json.dumps(report.to_dict(), indent=2))
    print(f"\nJSON report saved to: {json_path}")

    # Generate markdown report
    generate_markdown_report(report, args.report)

    if report.failed > 0:
        print(f"\n[WARNING] {report.failed} experiment(s) failed")
        sys.exit(1)

    print("\n[OK] All experiments completed!")


if __name__ == "__main__":
    main()
