#!/usr/bin/env python3
"""Benchmark quality of pruned models vs original.

This script evaluates model quality using simple code completion tasks
since full HumanEval requires significant setup. It measures:
1. Perplexity on code snippets
2. Completion quality for simple functions
3. Comparison between original and pruned models

Usage:
    python scripts/benchmark_quality.py [--models gpt2] [--pruned-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Simple code completion prompts for evaluation
EVAL_PROMPTS = [
    {
        "name": "hello_function",
        "prompt": 'def hello_world():\n    """Print hello world"""\n    ',
        "expected_keywords": ["print", "hello"],
    },
    {
        "name": "add_function",
        "prompt": 'def add(a, b):\n    """Add two numbers"""\n    ',
        "expected_keywords": ["return", "a", "b"],
    },
    {
        "name": "factorial",
        "prompt": 'def factorial(n):\n    """Calculate factorial"""\n    ',
        "expected_keywords": ["return", "n"],
    },
    {
        "name": "list_comprehension",
        "prompt": "# Create list of squares\nsquares = [",
        "expected_keywords": ["x", "**", "2", "for"],
    },
    {
        "name": "class_definition",
        "prompt": (
            'class Calculator:\n    """Simple calculator"""\n    \n'
            "    def __init__(self):\n        "
        ),
        "expected_keywords": ["self"],
    },
]

# Models to benchmark
BENCHMARK_MODELS = [
    {"model_id": "gpt2", "baseline": True},
    {"model_id": "gpt2-medium", "baseline": True},
    {"model_id": "facebook/opt-1.3b", "baseline": True},
]


@dataclass
class BenchmarkResult:
    """Result of benchmarking a single model."""

    model_id: str
    model_path: str
    is_pruned: bool = False
    strategy: str = ""
    status: str = "failed"

    # Metrics
    avg_perplexity: float | None = None
    completion_score: float | None = None
    inference_time_ms: float | None = None

    # Per-prompt results
    prompt_results: list[dict[str, Any]] = field(default_factory=list)

    error_message: str = ""
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkReport:
    """Complete benchmark report."""

    total_models: int = 0
    successful: int = 0
    failed: int = 0
    results: list[BenchmarkResult] = field(default_factory=list)
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
            "total_models": self.total_models,
            "successful": self.successful,
            "failed": self.failed,
            "duration_sec": self.duration_sec,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "results": [r.to_dict() for r in self.results],
        }


def calculate_perplexity(model, tokenizer, text: str, device: str = "cpu") -> float:
    """Calculate perplexity of a text under the model."""
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        perplexity = math.exp(loss.item())

    return perplexity


def evaluate_completion(
    model,
    tokenizer,
    prompt: str,
    expected_keywords: list[str],
    device: str = "cpu",
    max_new_tokens: int = 50,
) -> dict[str, Any]:
    """Evaluate a single completion prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Measure inference time
    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )
    inference_time = (time.time() - start) * 1000  # ms

    completion = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated = completion[len(prompt) :]

    # Check for expected keywords
    keyword_hits = sum(1 for kw in expected_keywords if kw in generated)
    score = keyword_hits / len(expected_keywords) if expected_keywords else 0

    return {
        "completion": generated[:100],  # Truncate for reporting
        "keyword_score": score,
        "inference_time_ms": inference_time,
    }


def benchmark_model(
    model_path: str,
    model_id: str,
    is_pruned: bool = False,
    strategy: str = "",
    cache_dir: Path | None = None,
    device: str = "cpu",
) -> BenchmarkResult:
    """Benchmark a single model."""
    result = BenchmarkResult(
        model_id=model_id,
        model_path=model_path,
        is_pruned=is_pruned,
        strategy=strategy,
        timestamp=datetime.now().isoformat(),
    )

    try:
        print("  Loading model...", end=" ", flush=True)

        # Load model
        load_kwargs = {"torch_dtype": torch.float32}
        if cache_dir and not is_pruned:
            load_kwargs["cache_dir"] = cache_dir

        model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            cache_dir=cache_dir if not is_pruned else None,
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = model.to(device)
        model.eval()

        print("[OK]")

        # Run evaluations
        perplexities = []
        completion_scores = []
        inference_times = []

        print(f"  Running {len(EVAL_PROMPTS)} evaluation prompts...")

        for i, prompt_data in enumerate(EVAL_PROMPTS, 1):
            print(f"    [{i}/{len(EVAL_PROMPTS)}] {prompt_data['name']}...", end=" ", flush=True)

            # Calculate perplexity on prompt
            ppl = calculate_perplexity(model, tokenizer, prompt_data["prompt"], device)
            perplexities.append(ppl)

            # Evaluate completion
            comp_result = evaluate_completion(
                model,
                tokenizer,
                prompt_data["prompt"],
                prompt_data["expected_keywords"],
                device,
            )

            completion_scores.append(comp_result["keyword_score"])
            inference_times.append(comp_result["inference_time_ms"])

            result.prompt_results.append(
                {
                    "name": prompt_data["name"],
                    "perplexity": ppl,
                    **comp_result,
                }
            )

            print(f"ppl={ppl:.1f}, score={comp_result['keyword_score']:.1f}")

        # Aggregate metrics
        result.avg_perplexity = sum(perplexities) / len(perplexities)
        result.completion_score = sum(completion_scores) / len(completion_scores)
        result.inference_time_ms = sum(inference_times) / len(inference_times)
        result.status = "success"

        print(f"  [OK] Avg perplexity: {result.avg_perplexity:.1f}")
        print(f"       Completion score: {result.completion_score:.2f}")

        # Cleanup
        del model
        import gc

        gc.collect()

    except Exception as e:
        result.error_message = str(e)
        print(f"  [FAIL] {e}")

    return result


def run_benchmarks(
    pruned_dir: Path,
    cache_dir: Path | None = None,
    models: list[dict[str, Any]] | None = None,
    device: str = "cpu",
) -> BenchmarkReport:
    """Run benchmarks on all models."""
    models = models or BENCHMARK_MODELS

    # Build list of models to benchmark (baseline + pruned variants)
    benchmark_configs = []

    for model_info in models:
        model_id = model_info["model_id"]

        # Add baseline
        benchmark_configs.append(
            {
                "path": model_id,
                "id": model_id,
                "is_pruned": False,
                "strategy": "baseline",
            }
        )

        # Add pruned variants if they exist
        for strategy in ["mild_pruning", "structured_pruning"]:
            pruned_path = pruned_dir / f"{model_id.replace('/', '--')}_{strategy}"
            if pruned_path.exists():
                benchmark_configs.append(
                    {
                        "path": str(pruned_path),
                        "id": model_id,
                        "is_pruned": True,
                        "strategy": strategy,
                    }
                )

    report = BenchmarkReport(
        total_models=len(benchmark_configs),
        start_time=datetime.now().isoformat(),
    )

    print("=" * 70)
    print("Benchmarking Model Quality")
    print("=" * 70)
    print(f"Models to benchmark: {len(benchmark_configs)}")
    print(f"Evaluation prompts: {len(EVAL_PROMPTS)}")
    print(f"Device: {device}")

    for i, config in enumerate(benchmark_configs, 1):
        model_label = f"{config['id']} ({config['strategy']})"
        print(f"\n[{i}/{len(benchmark_configs)}] {model_label}")
        print("-" * 50)

        result = benchmark_model(
            model_path=config["path"],
            model_id=config["id"],
            is_pruned=config["is_pruned"],
            strategy=config["strategy"],
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


def print_report(report: BenchmarkReport) -> None:
    """Print benchmark report."""
    print("\n" + "=" * 70)
    print("Benchmark Summary")
    print("=" * 70)
    print(f"Total: {report.total_models}")
    print(f"Successful: {report.successful}")
    print(f"Failed: {report.failed}")
    print(f"Duration: {report.duration_sec:.1f}s")

    # Group by model
    print("\nResults by Model:")
    print("-" * 70)
    print(f"{'Model':<30} {'Strategy':<15} {'Perplexity':<12} {'Score':<8} {'Status'}")
    print("-" * 70)

    by_model: dict[str, list[BenchmarkResult]] = {}
    for r in report.results:
        by_model.setdefault(r.model_id, []).append(r)

    for model_id, results in by_model.items():
        for r in results:
            status = "OK" if r.status == "success" else "FAIL"
            ppl = f"{r.avg_perplexity:.1f}" if r.avg_perplexity else "N/A"
            score = f"{r.completion_score:.2f}" if r.completion_score else "N/A"
            print(f"{model_id:<30} {r.strategy:<15} {ppl:<12} {score:<8} {status}")


def generate_markdown_report(report: BenchmarkReport, output_path: Path) -> None:
    """Generate markdown report."""
    lines = [
        "# Model Quality Benchmark Results",
        "",
        f"Generated: {report.end_time}",
        f"Duration: {report.duration_sec:.1f}s",
        "",
        "## Summary",
        "",
        f"- **Total models:** {report.total_models}",
        f"- **Successful:** {report.successful}",
        f"- **Failed:** {report.failed}",
        "",
        "## Results",
        "",
        "| Model | Strategy | Avg Perplexity | Completion Score | Inference (ms) | Status |",
        "|-------|----------|----------------|------------------|----------------|--------|",
    ]

    for r in report.results:
        if r.status != "success":
            continue

        ppl = f"{r.avg_perplexity:.1f}" if r.avg_perplexity else "N/A"
        score = f"{r.completion_score:.2f}" if r.completion_score else "N/A"
        time = f"{r.inference_time_ms:.1f}" if r.inference_time_ms else "N/A"

        lines.append(f"| {r.model_id} | {r.strategy} | {ppl} | {score} | {time} | OK |")

    # Quality degradation analysis
    lines.extend(
        [
            "",
            "## Quality Degradation Analysis",
            "",
            "Comparison of pruned models vs baseline:",
            "",
            "| Model | Strategy | Perplexity Change | Score Change |",
            "|-------|----------|-------------------|--------------|",
        ]
    )

    # Group by model and compare
    by_model: dict[str, list[BenchmarkResult]] = {}
    for r in report.results:
        by_model.setdefault(r.model_id, []).append(r)

    for model_id, results in by_model.items():
        baseline = next((r for r in results if r.strategy == "baseline"), None)
        if not baseline or not baseline.avg_perplexity:
            continue

        for r in results:
            if r.strategy == "baseline" or not r.avg_perplexity:
                continue

            ppl_change = (
                (r.avg_perplexity - baseline.avg_perplexity) / baseline.avg_perplexity * 100
            )

            if baseline.completion_score and r.completion_score:
                score_change = (r.completion_score - baseline.completion_score) * 100
            else:
                score_change = 0

            lines.append(
                f"| {model_id} | {r.strategy} | {ppl_change:+.1f}% | {score_change:+.1f}% |"
            )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Lower perplexity is better (model is more confident)",
            "- Completion score measures keyword presence in generated code",
            "- < 10% perplexity increase and < 5% score drop considered acceptable",
            "",
        ]
    )

    output_path.write_text("\n".join(lines))
    print(f"\nMarkdown report saved to: {output_path}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Benchmark quality of pruned models")
    parser.add_argument(
        "--pruned-dir",
        type=Path,
        default=Path("test_data/pruned_models"),
        help="Directory containing pruned models",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("test_data/models"),
        help="Cache directory for baseline models",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Specific model IDs to benchmark",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device for benchmarking",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("test_data/benchmark_report.json"),
        help="Output JSON path",
    )
    parser.add_argument(
        "--markdown",
        "-m",
        type=Path,
        default=Path("test_data/benchmark_report.md"),
        help="Output markdown path",
    )

    args = parser.parse_args()

    # Filter models if specified
    models = BENCHMARK_MODELS
    if args.models:
        models = [m for m in models if m["model_id"] in args.models]
        if not models:
            print(f"Error: No matching models in {args.models}")
            sys.exit(1)

    # Run benchmarks
    report = run_benchmarks(
        pruned_dir=args.pruned_dir,
        cache_dir=args.cache_dir if args.cache_dir.exists() else None,
        models=models,
        device=args.device,
    )

    # Print report
    print_report(report)

    # Save JSON
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report.to_dict(), indent=2))
    print(f"\nJSON report saved to: {args.output}")

    # Generate markdown
    generate_markdown_report(report, args.markdown)

    if report.failed > 0:
        print(f"\n[WARNING] {report.failed} benchmark(s) failed")
        sys.exit(1)

    print("\n[OK] All benchmarks completed!")


if __name__ == "__main__":
    main()
