#!/usr/bin/env python3
"""Quantize models using PyTorch dynamic quantization (INT8).

This script applies INT8 dynamic quantization to linear layers of models
to reduce memory footprint and potentially improve inference speed.

Quantization methods:
- dynamic: Quantize weights to int8, activations quantized on-the-fly
- static: Requires calibration dataset (not implemented)

Usage:
    python scripts/quantize_models.py [--models gpt2] [--methods dynamic]
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
import torch.quantization as tq
from transformers import AutoModelForCausalLM, AutoTokenizer

# Model configurations
MODEL_CONFIGS = [
    {
        "model_id": "gpt2",
        "params_b": 0.124,
    },
    {
        "model_id": "gpt2-medium",
        "params_b": 0.355,
    },
    {
        "model_id": "gpt2-xl",
        "params_b": 1.5,
    },
    {
        "model_id": "facebook/opt-1.3b",
        "params_b": 1.3,
    },
    {
        "model_id": "EleutherAI/pythia-2.8b",
        "params_b": 2.8,
    },
]

# Quantization methods
QUANTIZATION_METHODS = {
    "dynamic": tq.quantize_dynamic,  # type: ignore[attr-defined]
}


@dataclass
class QuantizationResult:
    """Result of quantizing a model."""

    model_id: str
    method: str
    status: str = "failed"
    original_params: int = 0
    original_size_mb: float = 0.0
    quantized_size_mb: float = 0.0
    memory_reduction_pct: float = 0.0
    inference_time_ms_before: float = 0.0
    inference_time_ms_after: float = 0.0
    speedup_pct: float = 0.0
    quantization_time_sec: float = 0.0
    output_path: str = ""
    error_message: str = ""
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class QuantizationReport:
    """Report of all quantization operations."""

    total_models: int = 0
    successful: int = 0
    failed: int = 0
    results: list[QuantizationResult] = field(default_factory=list)
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


def get_model_size_mb(model: torch.nn.Module) -> float:
    """Calculate model size in MB (parameters only, approximate).

    Handles quantized tensors (qint8) which have element_size() == 1.
    """
    total_bytes = 0
    for p in model.parameters():
        # For quantized tensors, element_size() returns correct size
        total_bytes += p.numel() * p.element_size()
    for b in model.buffers():
        total_bytes += b.numel() * b.element_size()
    return total_bytes / (1024 * 1024)


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


def apply_dynamic_quantization(model: torch.nn.Module) -> torch.nn.Module:
    """Apply dynamic quantization to linear layers of a model."""
    # Quantize all linear layers
    quantized_model = tq.quantize_dynamic(  # type: ignore[attr-defined]
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    return quantized_model  # type: ignore[no-any-return]


def quantize_model(
    model_id: str,
    method: str,
    output_dir: Path,
    cache_dir: Path | None = None,
    device: str = "cpu",
) -> QuantizationResult:
    """Quantize a single model.

    Args:
        model_id: HuggingFace model ID
        method: Quantization method name
        output_dir: Directory to save quantized model
        cache_dir: Cache directory for source model
        device: Device to use

    Returns:
        QuantizationResult with operation details
    """
    result = QuantizationResult(
        model_id=model_id,
        method=method,
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

        # Count original parameters
        result.original_params = sum(p.numel() for p in model.parameters())
        result.original_size_mb = get_model_size_mb(model)

        # Measure baseline inference time
        print("  Measuring baseline inference...", end=" ", flush=True)
        result.inference_time_ms_before = measure_inference_time(model, tokenizer, device)
        print(f"[OK] ({result.inference_time_ms_before:.1f} ms)")

        # Apply quantization
        print(f"  Applying {method} quantization...", end=" ", flush=True)
        start_time = time.time()

        if method == "dynamic":
            quantized_model = apply_dynamic_quantization(model)
        else:
            raise ValueError(f"Unknown quantization method: {method}")

        result.quantization_time_sec = time.time() - start_time
        print("[OK]")

        # Measure quantized model size and inference time
        print("  Measuring quantized model...", end=" ", flush=True)
        result.quantized_size_mb = get_model_size_mb(quantized_model)
        result.memory_reduction_pct = (
            (result.original_size_mb - result.quantized_size_mb) / result.original_size_mb * 100
        )

        result.inference_time_ms_after = measure_inference_time(quantized_model, tokenizer, device)
        if result.inference_time_ms_before > 0:
            result.speedup_pct = (
                (result.inference_time_ms_before - result.inference_time_ms_after)
                / result.inference_time_ms_before
                * 100
            )
        print("[OK]")

        # Save quantized model
        model_output_dir = output_dir / f"{model_id.replace('/', '--')}_{method}"
        model_output_dir.mkdir(parents=True, exist_ok=True)

        print(f"  Saving to {model_output_dir}...", end=" ", flush=True)
        # Save state dict and config
        torch.save(
            quantized_model.state_dict(),
            model_output_dir / "pytorch_model.bin",
        )
        # Save tokenizer and config
        tokenizer.save_pretrained(model_output_dir)
        # Save quantization config
        quant_config = {
            "quantization_method": method,
            "original_model": model_id,
            "quantized_at": datetime.now().isoformat(),
        }
        (model_output_dir / "quantization_config.json").write_text(
            json.dumps(quant_config, indent=2)
        )
        print("[OK]")

        result.output_path = str(model_output_dir)
        result.status = "success"

        print(
            f"  [OK] Size: {result.original_size_mb:.1f} MB -> {result.quantized_size_mb:.1f} MB "
            f"({result.memory_reduction_pct:.1f}% reduction)"
        )
        print(
            f"       Inference: {result.inference_time_ms_before:.1f} ms -> "
            f"{result.inference_time_ms_after:.1f} ms ({result.speedup_pct:+.1f}% speedup)"
        )
        print(f"       Time: {result.quantization_time_sec:.1f}s")

        # Cleanup
        del model, quantized_model
        import gc

        gc.collect()

    except Exception as e:
        result.error_message = str(e)
        print(f"  [FAIL] {e}")

    return result


def quantize_all_models(
    output_dir: Path,
    cache_dir: Path | None = None,
    configs: list[dict[str, Any]] | None = None,
    methods: list[str] | None = None,
    device: str = "cpu",
) -> QuantizationReport:
    """Quantize all configured models.

    Args:
        output_dir: Directory to save quantized models
        cache_dir: Cache directory for source models
        configs: Model configurations (default: MODEL_CONFIGS)
        methods: Quantization methods to apply (default: ["dynamic"])
        device: Device to use

    Returns:
        QuantizationReport with all results
    """
    configs = configs or MODEL_CONFIGS
    methods = methods or ["dynamic"]

    # Calculate total operations
    total = len(configs) * len(methods)

    report = QuantizationReport(
        total_models=total,
        start_time=datetime.now().isoformat(),
        output_dir=str(output_dir),
    )

    print("=" * 70)
    print("Quantizing Models")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print(f"Cache directory: {cache_dir or 'default'}")
    print(f"Device: {device}")
    print(f"Models to quantize: {total}")
    print(f"Methods: {', '.join(methods)}")

    count = 0
    for config in configs:
        model_id = str(config["model_id"])

        for method in methods:
            count += 1
            print(f"\n[{count}/{total}] {model_id} - {method}")
            print("-" * 50)

            result = quantize_model(
                model_id=model_id,
                method=method,
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


def print_summary(report: QuantizationReport) -> None:
    """Print summary of quantization operations."""
    print("\n" + "=" * 70)
    print("Quantization Summary")
    print("=" * 70)
    print(f"Total: {report.total_models}")
    print(f"Successful: {report.successful}")
    print(f"Failed: {report.failed}")
    print(f"Duration: {report.duration_sec:.1f}s")
    print(f"Output directory: {report.output_dir}")

    print("\nResults by Model:")
    print("-" * 70)

    # Group by model
    by_model: dict[str, list[QuantizationResult]] = {}
    for r in report.results:
        by_model.setdefault(r.model_id, []).append(r)

    for model_id, results in by_model.items():
        print(f"\n{model_id}:")
        for r in results:
            status = "OK" if r.status == "success" else "FAIL"
            print(f"  {status:4s} {r.method:15s} ", end="")
            if r.memory_reduction_pct > 0:
                print(f"size={r.memory_reduction_pct:.1f}% ", end="")
            if r.speedup_pct != 0:
                print(f"speed={r.speedup_pct:+.1f}% ", end="")
            if r.quantization_time_sec > 0:
                print(f"time={r.quantization_time_sec:.1f}s", end="")
            print()


def generate_markdown_report(report: QuantizationReport, output_path: Path) -> None:
    """Generate markdown report."""
    lines = [
        "# Quantization Results",
        "",
        f"Generated: {report.end_time}",
        f"Duration: {report.duration_sec:.1f}s",
        "",
        "## Summary",
        "",
        f"- **Total models quantized:** {report.total_models}",
        f"- **Successful:** {report.successful}",
        f"- **Failed:** {report.failed}",
        "",
        "## Quantized Models",
        "",
        "| Model | Method | Original Size (MB) | Quantized Size (MB) | Memory Reduction | "
        "Inference Before (ms) | Inference After (ms) | Speedup | Time (s) |",
        "|-------|--------|-------------------|---------------------|------------------|"
        "------------------------|---------------------|---------|----------|",
    ]

    for r in report.results:
        if r.status != "success":
            continue

        row = (
            f"| {r.model_id} | {r.method} | {r.original_size_mb:.1f} | "
            f"{r.quantized_size_mb:.1f} | {r.memory_reduction_pct:.1f}% | "
            f"{r.inference_time_ms_before:.1f} | {r.inference_time_ms_after:.1f} | "
            f"{r.speedup_pct:+.1f}% | {r.quantization_time_sec:.1f} |"
        )
        lines.append(row)

    lines.extend(
        [
            "",
            "## Comparison with Atropos Projections",
            "",
            "Compare these actual results with Atropos quantization bonus preset "
            "(memory_reduction_fraction=0.3846, throughput_improvement_fraction=0.4667).",
            "",
        ]
    )

    output_path.write_text("\n".join(lines))
    print(f"\nMarkdown report saved to: {output_path}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Quantize models for Atropos quantization study")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("test_data/quantized_models"),
        help="Directory to save quantized models (default: test_data/quantized_models)",
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
        help="Specific model IDs to quantize (default: all)",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["dynamic"],
        default=["dynamic"],
        help="Quantization methods to apply (default: dynamic)",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device for quantization (default: cpu)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("test_data/quantization_results.md"),
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

    # Run quantization
    report = quantize_all_models(
        output_dir=args.output_dir,
        cache_dir=args.cache_dir if args.cache_dir.exists() else None,
        configs=configs,
        methods=args.methods,
        device=args.device,
    )

    # Print summary
    print_summary(report)

    # Save JSON report
    json_path = args.output_dir.parent / "quantization_report.json"
    json_path.write_text(json.dumps(report.to_dict(), indent=2))
    print(f"\nJSON report saved to: {json_path}")

    # Generate markdown report
    generate_markdown_report(report, args.report)

    if report.failed > 0:
        print(f"\n[WARNING] {report.failed} quantization operation(s) failed")
        sys.exit(1)

    print("\n[OK] All models quantized successfully!")


if __name__ == "__main__":
    main()
