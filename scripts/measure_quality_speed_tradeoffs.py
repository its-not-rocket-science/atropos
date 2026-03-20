#!/usr/bin/env python3
"""Measure quality/speed trade-offs for pruning frameworks.

This script benchmarks pruned models from different frameworks (magnitude, wanda, sparsegpt)
to measure quality degradation vs speed improvement trade-offs.

Usage:
    python scripts/measure_quality_speed_tradeoffs.py
    python scripts/measure_quality_speed_tradeoffs.py --models gpt2 --frameworks all
    python scripts/measure_quality_speed_tradeoffs.py --sparsity 0.1 --device cpu
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

# Add src to path for importing atropos
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import functions from existing scripts
try:
    from benchmark_quality import (
        EVAL_PROMPTS,
        calculate_perplexity,
        evaluate_completion,
    )
    from benchmark_quality import (
        BenchmarkResult as QualityResult,
    )

except ImportError:
    # Define locally if import fails
    EVAL_PROMPTS = []

    def calculate_perplexity(
        model: PreTrainedModel, tokenizer: PreTrainedTokenizer, text: str, device: str = "cpu"
    ) -> float:
        """Calculate perplexity of a text under the model."""
        encodings = tokenizer(text, return_tensors="pt")
        input_ids = encodings.input_ids.to(device)

        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
            perplexity = torch.exp(loss).item()

        return perplexity

    def evaluate_completion(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        prompt: str,
        expected_keywords: list[str],
        device: str = "cpu",
        max_new_tokens: int = 50,
    ) -> dict[str, Any]:
        """Evaluate a single completion prompt."""
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # Measure inference time
        if device == "cuda" and torch.cuda.is_available():
            start_event = torch.cuda.Event(enable_timing=True)  # type: ignore
            end_event = torch.cuda.Event(enable_timing=True)  # type: ignore
            start_event.record()  # type: ignore
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id,
                )
            end_event.record()  # type: ignore
            torch.cuda.synchronize()
            inference_time_ms = start_event.elapsed_time(end_event)  # type: ignore
        else:
            import time

            start_time = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id,
                )
            inference_time_ms = (time.time() - start_time) * 1000

        completion = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated = completion[len(prompt) :]

        # Check for expected keywords
        keyword_hits = sum(1 for kw in expected_keywords if kw in generated)
        score = keyword_hits / len(expected_keywords) if expected_keywords else 0

        return {
            "completion": generated[:100],  # Truncate for reporting
            "keyword_score": score,
            "inference_time_ms": inference_time_ms,
        }

    @dataclass
    class QualityResult:  # type: ignore[no-redef]
        """Result of quality benchmarking a single model."""

        model_id: str
        model_path: str
        is_pruned: bool = False
        strategy: str = ""
        status: str = "failed"
        avg_perplexity: float | None = None
        completion_score: float | None = None
        inference_time_ms: float | None = None
        prompt_results: list[dict[str, Any]] = field(default_factory=list)
        error_message: str = ""
        timestamp: str = ""

        def to_dict(self) -> dict[str, Any]:
            return asdict(self)


# Default models to test
DEFAULT_MODELS = [
    "gpt2",
]

# Default frameworks to compare
ALL_FRAMEWORKS = ["magnitude", "wanda-patched", "sparsegpt-patched"]

# Default sparsity level
DEFAULT_SPARSITY = 0.1  # 10%

# Default pruning output directory
DEFAULT_PRUNED_DIR = Path("test_data/framework_comparison")

# Default cache directory for baseline models
DEFAULT_CACHE_DIR = Path("test_data/models")


@dataclass
class TradeOffResult:
    """Result of quality/speed trade-off analysis for a single model-framework pair."""

    model: str
    framework: str
    status: str = "failed"  # success, failed

    # Pruning metrics (from framework comparison)
    target_sparsity: float = 0.0
    achieved_sparsity: float = 0.0
    original_params: int = 0
    pruned_params: int = 0
    pruning_time_sec: float = 0.0
    pruning_output_path: str = ""

    # Quality metrics (from benchmarking)
    baseline_perplexity: float | None = None
    pruned_perplexity: float | None = None
    perplexity_change_pct: float | None = None

    baseline_completion_score: float | None = None
    pruned_completion_score: float | None = None
    completion_score_change_pct: float | None = None

    baseline_inference_time_ms: float | None = None
    pruned_inference_time_ms: float | None = None
    inference_speedup_pct: float | None = None

    # Trade-off metrics
    quality_speed_ratio: float | None = None  # (quality degradation) / (speed improvement)

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
    def quality_degradation_pct(self) -> float | None:
        """Calculate overall quality degradation percentage."""
        if self.perplexity_change_pct is None:
            return None
        # Positive perplexity change means degradation
        # Negative completion score change means degradation
        perplexity_weight = 0.7  # Weight perplexity more
        completion_weight = 0.3

        perplexity_degradation = max(0, self.perplexity_change_pct)
        if self.completion_score_change_pct is not None:
            completion_degradation = max(0, -self.completion_score_change_pct)
        else:
            completion_degradation = 0.0

        return (
            perplexity_degradation * perplexity_weight + completion_degradation * completion_weight
        )


@dataclass
class TradeOffReport:
    """Complete trade-off analysis report."""

    total_tests: int = 0
    successful: int = 0
    failed: int = 0
    results: list[TradeOffResult] = field(default_factory=list)
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


def find_pruned_models(
    model_id: str,
    framework: str,
    pruned_dir: Path = DEFAULT_PRUNED_DIR,
    target_sparsity: float = DEFAULT_SPARSITY,
) -> Path | None:
    """Find pruned model directory for given model and framework.

    Args:
        model_id: HuggingFace model ID
        framework: Pruning framework name
        pruned_dir: Base directory containing pruned models
        target_sparsity: Target sparsity level (used to find appropriate model)

    Returns:
        Path to pruned model directory, or None if not found
    """
    # Convert model ID to safe directory name
    model_safe = model_id.replace("/", "--")

    # Framework-specific directory structure
    # Framework comparison script saves to: {pruned_dir}/{framework}/{model_safe}_{framework_safe}
    # where framework_safe = framework.replace("-", "_")
    framework_safe = framework.replace("-", "_")
    possible_paths = [
        pruned_dir / framework / model_safe / f"{model_safe}_{framework_safe}",
        pruned_dir / framework / f"{model_safe}_{framework_safe}",
        pruned_dir / f"{model_safe}_{framework_safe}",
    ]

    # Legacy directory structure (from earlier pruning exercises)
    # test_data/pruned_models/{model_safe}_{legacy_suffix}
    legacy_suffix_map = {
        "sparsegpt-patched": "sparsegpt",
        "wanda-patched": "wanda",
        "magnitude": "magnitude",
    }
    legacy_suffix = legacy_suffix_map.get(framework, framework.replace("-", "_"))
    possible_paths.append(pruned_dir.parent / "pruned_models" / f"{model_safe}_{legacy_suffix}")

    for path in possible_paths:
        if path.exists() and (path / "config.json").exists():
            return path

    return None


def benchmark_model_quality(
    model_path: str,
    model_id: str,
    framework: str,
    is_pruned: bool = False,
    device: str = "cpu",
    cache_dir: Path | None = None,
) -> QualityResult:
    """Benchmark quality and speed of a single model.

    This is adapted from benchmark_quality.py but returns a QualityResult.
    """
    result = QualityResult(
        model_id=model_id,
        model_path=model_path,
        is_pruned=is_pruned,
        strategy=framework,
        timestamp=datetime.now().isoformat(),
    )

    try:
        print(f"  Loading {model_id} ({framework})...", end=" ", flush=True)

        # Load model
        load_kwargs: dict[str, Any] = {"torch_dtype": torch.float32}
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
                prompt_data.get("expected_keywords", []),
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
        print(f"       Avg inference time: {result.inference_time_ms:.1f}ms")

        # Cleanup
        del model
        import gc

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception as e:
        result.error_message = str(e)
        print(f"  [FAIL] {e}")

    return result


def load_pruning_metrics(
    model_id: str,
    framework: str,
    pruned_dir: Path = DEFAULT_PRUNED_DIR,
) -> dict[str, Any] | None:
    """Load pruning metrics from framework comparison JSON if available.

    Args:
        model_id: HuggingFace model ID
        framework: Pruning framework name
        pruned_dir: Base directory containing pruned models

    Returns:
        Dictionary with pruning metrics, or None if not found
    """
    # Look for framework comparison JSON file
    json_paths = [
        pruned_dir / "framework_comparison.json",
        pruned_dir.parent / "framework_comparison.json",
        pruned_dir / f"{framework}_comparison.json",
    ]

    for json_path in json_paths:
        if json_path.exists():
            try:
                with open(json_path) as f:
                    data = json.load(f)

                # Find matching result
                for result in data.get("results", []):
                    if result.get("model") == model_id and result.get("framework") == framework:
                        return cast(dict[str, Any], result)
            except (json.JSONDecodeError, KeyError):
                continue

    return None


def run_tradeoff_analysis(
    model_id: str,
    framework: str,
    pruned_dir: Path = DEFAULT_PRUNED_DIR,
    cache_dir: Path | None = None,
    device: str = "cpu",
) -> TradeOffResult:
    """Run quality/speed trade-off analysis for a single model-framework pair.

    Args:
        model_id: HuggingFace model ID
        framework: Pruning framework name
        pruned_dir: Directory containing pruned models
        cache_dir: Cache directory for baseline models
        device: Device for benchmarking

    Returns:
        TradeOffResult with analysis
    """
    result = TradeOffResult(
        model=model_id,
        framework=framework,
        timestamp=datetime.now().isoformat(),
    )

    try:
        print(f"\nAnalyzing {model_id} - {framework}")
        print("-" * 50)

        # 1. Find pruned model
        pruned_model_path = find_pruned_models(model_id, framework, pruned_dir)
        if not pruned_model_path:
            result.error_message = f"Pruned model not found for {model_id} - {framework}"
            result.status = "failed"
            print(f"  [FAIL] {result.error_message}")
            return result

        result.pruning_output_path = str(pruned_model_path)

        # 2. Load pruning metrics
        pruning_metrics = load_pruning_metrics(model_id, framework, pruned_dir)
        if pruning_metrics:
            result.target_sparsity = pruning_metrics.get("target_sparsity", 0.0)
            result.achieved_sparsity = pruning_metrics.get("achieved_sparsity", 0.0)
            result.original_params = pruning_metrics.get("original_params", 0)
            result.pruned_params = pruning_metrics.get("pruned_params", 0)
            result.pruning_time_sec = pruning_metrics.get("pruning_time_sec", 0.0)
            print("  Pruning metrics loaded from JSON")

        # 3. Benchmark baseline model
        print(f"  Benchmarking baseline {model_id}...")
        baseline_result = benchmark_model_quality(
            model_path=model_id,
            model_id=model_id,
            framework="baseline",
            is_pruned=False,
            device=device,
            cache_dir=cache_dir,
        )

        if baseline_result.status != "success":
            result.error_message = f"Baseline benchmarking failed: {baseline_result.error_message}"
            result.status = "failed"
            print(f"  [FAIL] {result.error_message}")
            return result

        result.baseline_perplexity = baseline_result.avg_perplexity
        result.baseline_completion_score = baseline_result.completion_score
        result.baseline_inference_time_ms = baseline_result.inference_time_ms

        # 4. Benchmark pruned model
        print(f"  Benchmarking pruned {model_id} ({framework})...")
        pruned_result = benchmark_model_quality(
            model_path=str(pruned_model_path),
            model_id=model_id,
            framework=framework,
            is_pruned=True,
            device=device,
            cache_dir=cache_dir,
        )

        if pruned_result.status != "success":
            error_msg = f"Pruned model benchmarking failed: {pruned_result.error_message}"
            result.error_message = error_msg
            result.status = "failed"
            print(f"  [FAIL] {result.error_message}")
            return result

        result.pruned_perplexity = pruned_result.avg_perplexity
        result.pruned_completion_score = pruned_result.completion_score
        result.pruned_inference_time_ms = pruned_result.inference_time_ms

        # 5. Calculate changes
        if (
            result.baseline_perplexity
            and result.pruned_perplexity
            and result.baseline_perplexity > 0
        ):
            result.perplexity_change_pct = (
                (result.pruned_perplexity - result.baseline_perplexity)
                / result.baseline_perplexity
                * 100
            )

        if (
            result.baseline_completion_score is not None
            and result.pruned_completion_score is not None
        ):
            result.completion_score_change_pct = (
                result.pruned_completion_score - result.baseline_completion_score
            ) * 100

        if (
            result.baseline_inference_time_ms
            and result.pruned_inference_time_ms
            and result.baseline_inference_time_ms > 0
        ):
            result.inference_speedup_pct = (
                (result.baseline_inference_time_ms - result.pruned_inference_time_ms)
                / result.baseline_inference_time_ms
                * 100
            )

        # 6. Calculate quality-speed trade-off ratio
        # Negative ratio means quality degradation per unit speed improvement
        if (
            result.perplexity_change_pct is not None
            and result.inference_speedup_pct is not None
            and result.inference_speedup_pct > 0
        ):
            # Quality degradation (positive is bad) per speed improvement (positive is good)
            result.quality_speed_ratio = result.perplexity_change_pct / result.inference_speedup_pct

        result.status = "success"
        print("  [OK] Analysis complete")

    except Exception as e:
        result.error_message = str(e)
        result.status = "failed"
        print(f"  [FAIL] {e}")

    return result


def run_tradeoff_comparison(
    models: list[str],
    frameworks: list[str],
    pruned_dir: Path = DEFAULT_PRUNED_DIR,
    cache_dir: Path | None = None,
    device: str = "cpu",
) -> TradeOffReport:
    """Run quality/speed trade-off analysis across multiple models and frameworks.

    Args:
        models: List of model IDs to analyze
        frameworks: List of framework names to test
        pruned_dir: Directory containing pruned models
        cache_dir: Cache directory for baseline models
        device: Device for benchmarking

    Returns:
        TradeOffReport with all results
    """
    total_tests = len(models) * len(frameworks)

    report = TradeOffReport(
        total_tests=total_tests,
        start_time=datetime.now().isoformat(),
        config={
            "models": models,
            "frameworks": frameworks,
            "pruned_dir": str(pruned_dir),
            "cache_dir": str(cache_dir) if cache_dir else None,
            "device": device,
        },
    )

    print("=" * 70)
    print("Quality/Speed Trade-off Analysis")
    print("=" * 70)
    print(f"Models: {models}")
    print(f"Frameworks: {frameworks}")
    print(f"Pruned models directory: {pruned_dir}")
    print(f"Device: {device}")
    print(f"Total tests: {total_tests}")
    print("=" * 70)

    test_count = 0
    for model_id in models:
        for framework in frameworks:
            test_count += 1
            print(f"\n[{test_count}/{total_tests}] {model_id} - {framework}")
            print("-" * 50)

            result = run_tradeoff_analysis(
                model_id=model_id,
                framework=framework,
                pruned_dir=pruned_dir,
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


def print_tradeoff_summary(report: TradeOffReport) -> None:
    """Print trade-off analysis summary."""
    print("\n" + "=" * 70)
    print("Trade-off Analysis Summary")
    print("=" * 70)
    print(f"Total tests: {report.total_tests}")
    print(f"Successful: {report.successful}")
    print(f"Failed: {report.failed}")
    print(f"Duration: {report.duration_sec:.1f}s")

    # Print results table
    print("\nResults:")
    print("-" * 70)
    print(
        f"{'Model':<15} {'Framework':<15} {'Sparsity':<10} "
        f"{'PPL Change%':<12} {'Score Change%':<12} {'Speedup%':<10} {'Q/S Ratio':<12} {'Status'}"
    )
    print("-" * 70)

    for result in report.results:
        if result.status != "success":
            continue

        sparsity = f"{result.achieved_sparsity:.1%}"
        if result.perplexity_change_pct is not None:
            ppl_change = f"{result.perplexity_change_pct:+.1f}%"
        else:
            ppl_change = "N/A"
        if result.completion_score_change_pct is not None:
            score_change = f"{result.completion_score_change_pct:+.1f}%"
        else:
            score_change = "N/A"
        if result.inference_speedup_pct is not None:
            speedup = f"{result.inference_speedup_pct:+.1f}%"
        else:
            speedup = "N/A"
        if result.quality_speed_ratio is not None:
            qs_ratio = f"{result.quality_speed_ratio:+.2f}"
        else:
            qs_ratio = "N/A"
        status = "OK"

        print(
            f"{result.model:<15} {result.framework:<15} {sparsity:<10} "
            f"{ppl_change:<10} {score_change:<10} {speedup:<10} {qs_ratio:<12} {status}"
        )

    # Print framework comparison
    print("\nFramework Comparison (average across models):")
    print("-" * 70)
    print(
        f"{'Framework':<15} {'Sparsity':<10} {'PPL Change%':<12} "
        f"{'Speedup%':<10} {'Q/S Ratio':<12} {'Success Rate'}"
    )
    print("-" * 70)

    for framework in report.config.get("frameworks", []):
        framework_results = [
            r for r in report.results if r.framework == framework and r.status == "success"
        ]
        if not framework_results:
            continue

        avg_sparsity = sum(r.achieved_sparsity for r in framework_results) / len(framework_results)
        total_ppl_change = sum(r.perplexity_change_pct or 0 for r in framework_results)
        avg_ppl_change = total_ppl_change / len(framework_results)
        total_speedup = sum(r.inference_speedup_pct or 0 for r in framework_results)
        avg_speedup = total_speedup / len(framework_results)
        total_qs_ratio = sum(r.quality_speed_ratio or 0 for r in framework_results)
        avg_qs_ratio = total_qs_ratio / len(framework_results)
        success_rate = len(framework_results) / report.total_tests * 100

        print(
            f"{framework:<15} {avg_sparsity:.1%}  "
            f"{avg_ppl_change:+.1f}%  "
            f"{avg_speedup:+.1f}%  "
            f"{avg_qs_ratio:+.2f}  "
            f"{success_rate:.0f}%"
        )


def generate_markdown_report(report: TradeOffReport, output_path: Path) -> None:
    """Generate markdown trade-off report."""
    lines = [
        "# Quality/Speed Trade-off Analysis",
        "",
        f"Generated: {report.end_time}",
        f"Duration: {report.duration_sec:.1f}s",
        "",
        "## Summary",
        "",
        f"- **Total tests:** {report.total_tests}",
        f"- **Successful:** {report.successful}",
        f"- **Failed:** {report.failed}",
        "",
        "## Configuration",
        "",
    ]

    for key, value in report.config.items():
        lines.append(f"- **{key}:** {value}")

    lines.extend(
        [
            "",
            "## Results",
            "",
            "| Model | Framework | Sparsity | Perplexity Change% | Score Change% | Speedup% |"
            " Q/S Ratio | Status |",
            "|-------|-----------|----------|-------------------|--------------|----------|"
            "-----------|--------|",
        ]
    )

    for result in report.results:
        if result.status != "success":
            continue

        sparsity = f"{result.achieved_sparsity:.1%}"
        if result.perplexity_change_pct is not None:
            ppl_change = f"{result.perplexity_change_pct:+.1f}%"
        else:
            ppl_change = "N/A"
        if result.completion_score_change_pct is not None:
            score_change = f"{result.completion_score_change_pct:+.1f}%"
        else:
            score_change = "N/A"
        if result.inference_speedup_pct is not None:
            speedup = f"{result.inference_speedup_pct:+.1f}%"
        else:
            speedup = "N/A"
        if result.quality_speed_ratio is not None:
            qs_ratio = f"{result.quality_speed_ratio:+.2f}"
        else:
            qs_ratio = "N/A"

        lines.append(
            f"| {result.model} | {result.framework} | {sparsity} | {ppl_change} | {score_change} |"
            f" {speedup} | {qs_ratio} | OK |"
        )

    # Trade-off analysis
    lines.extend(
        [
            "",
            "## Trade-off Analysis",
            "",
            "### Key Metrics",
            "",
            "- **Perplexity Change%**: Positive = degradation, Negative = improvement",
            "- **Score Change%**: Positive = improvement, Negative = degradation",
            "- **Speedup%**: Positive = faster inference, Negative = slower",
            "- **Q/S Ratio**: Quality degradation per unit speed improvement",
            "  (lower/negative is better)",
            "",
            "### Recommendations",
            "",
            "Based on Q/S Ratio:",
            "- **Q/S Ratio < -1**: Good trade-off",
            "  (significant speed improvement with minimal quality loss)",
            "- **-1 < Q/S Ratio < 0**: Acceptable trade-off",
            "- **Q/S Ratio > 0**: Poor trade-off (quality degrades faster than speed improves)",
            "",
            "## Notes",
            "",
            "- Quality metrics measured on code completion prompts",
            "- Speed measured as inference time for 50-token generation",
            "- Results may vary with different prompts and generation parameters",
            "- Always validate with task-specific benchmarks before production deployment",
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nMarkdown report saved to: {output_path}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Measure quality/speed trade-offs for pruning frameworks"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help=f"Model IDs to analyze (default: {DEFAULT_MODELS})",
    )
    parser.add_argument(
        "--frameworks",
        nargs="+",
        choices=["magnitude", "wanda-patched", "sparsegpt-patched", "all"],
        default=["all"],
        help="Frameworks to analyze (default: all)",
    )
    parser.add_argument(
        "--pruned-dir",
        type=Path,
        default=DEFAULT_PRUNED_DIR,
        help=f"Directory containing pruned models (default: {DEFAULT_PRUNED_DIR})",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help=f"Cache directory for baseline models (default: {DEFAULT_CACHE_DIR})",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device for benchmarking (default: cpu)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("test_data/tradeoff_analysis.json"),
        help="Output JSON path (default: test_data/tradeoff_analysis.json)",
    )
    parser.add_argument(
        "--markdown",
        "-m",
        type=Path,
        default=Path("test_data/tradeoff_analysis.md"),
        help="Output markdown path (default: test_data/tradeoff_analysis.md)",
    )
    parser.add_argument(
        "--skip-if-exists",
        action="store_true",
        help="Skip analysis if output files already exist",
    )

    args = parser.parse_args()

    # Handle "all" frameworks
    if "all" in args.frameworks:
        args.frameworks = ALL_FRAMEWORKS

    # Check if we should skip
    if args.skip_if_exists and args.output.exists() and args.markdown.exists():
        print(f"Output files already exist: {args.output}, {args.markdown}")
        print("Use --skip-if-exists to skip, or remove files to re-run.")
        return

    # Run trade-off analysis
    report = run_tradeoff_comparison(
        models=args.models,
        frameworks=args.frameworks,
        pruned_dir=args.pruned_dir,
        cache_dir=args.cache_dir if args.cache_dir.exists() else None,
        device=args.device,
    )

    # Print summary
    print_tradeoff_summary(report)

    # Save JSON
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    print(f"\nJSON report saved to: {args.output}")

    # Generate markdown
    generate_markdown_report(report, args.markdown)

    if report.failed > 0:
        print(f"\n[WARNING] {report.failed} test(s) failed")
        sys.exit(1)

    print("\n[OK] Trade-off analysis completed!")


if __name__ == "__main__":
    main()
