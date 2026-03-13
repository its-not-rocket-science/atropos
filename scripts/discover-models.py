#!/usr/bin/env python3
"""Discover and test available models for Atropos validation.

This script helps you:
1. Find available models on HuggingFace Hub
2. Test which models can be loaded on your system
3. Run Atropos validation against working models
4. Generate a compatibility report

Usage:
    python discover-models.py --list          # List available models
    python discover-models.py --test          # Test model loading
    python discover-models.py --validate      # Run Atropos validation
    python discover-models.py --full          # Do all of the above
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Default test models organized by size
TEST_MODELS = {
    "tiny": [
        "gpt2",  # 124M
        "facebook/opt-125m",  # 125M
        "EleutherAI/pythia-160m",  # 160M
    ],
    "small": [
        "gpt2-medium",  # 355M
        "facebook/opt-350m",  # 350M
        "EleutherAI/pythia-410m",  # 410M
    ],
    "medium": [
        "gpt2-large",  # 774M
        "bigscience/bloom-560m",  # 560M
    ],
    "large": [
        "gpt2-xl",  # 1.5B
        "TinyLlama/TinyLlama-1.1B-v1.0",  # 1.1B
    ],
}

ALL_MODELS = [m for models in TEST_MODELS.values() for m in models]


@dataclass
class ModelInfo:
    """Information about a model."""

    name: str
    params_b: float | None = None
    loadable: bool = False
    error: str | None = None
    memory_gb: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def check_dependencies() -> bool:
    """Check if required dependencies are installed."""
    try:
        import torch
        import transformers

        print(f"✓ PyTorch {torch.__version__} installed")
        print(f"✓ Transformers {transformers.__version__} installed")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
        return True
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("\nInstall with: pip install torch transformers")
        return False


def estimate_params(model_name: str) -> float | None:
    """Estimate parameter count from model name."""
    name_lower = model_name.lower()

    # Try to extract from name patterns
    import re

    # Patterns like "125m", "1.1b", "7b"
    patterns = [
        (r"(\d+\.?\d*)m", 1e6),  # 125m -> 125M
        (r"(\d+\.?\d*)b", 1e9),  # 1.1b -> 1.1B
    ]

    for pattern, multiplier in patterns:
        match = re.search(pattern, name_lower)
        if match:
            try:
                return float(match.group(1)) * multiplier / 1e9
            except ValueError:
                continue

    # Known models
    known = {
        "gpt2": 0.124,
        "gpt2-medium": 0.355,
        "gpt2-large": 0.774,
        "gpt2-xl": 1.5,
    }

    for key, value in known.items():
        if key in name_lower:
            return value

    return None


def test_model_loading(model_name: str, device: str = "cpu") -> ModelInfo:
    """Test if a model can be loaded."""
    info = ModelInfo(name=model_name, params_b=estimate_params(model_name))

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"  Loading {model_name}...", end=" ", flush=True)

        # Try to load tokenizer
        _tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Try to load model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="auto" if device == "cuda" else None,
        )

        if device == "cpu":
            model = model.cpu()

        # Count parameters
        param_count = sum(p.numel() for p in model.parameters())
        info.params_b = param_count / 1e9

        # Estimate memory
        info.memory_gb = param_count * 4 / (1024**3)  # float32

        info.loadable = True
        print("✓")

        # Cleanup
        del model
        import gc

        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    except Exception as e:
        info.error = str(e)
        print(f"✗ ({e})")

    return info


def run_atropos_validation(
    model_name: str, scenario: str = "edge-coder", device: str = "cpu"
) -> dict[str, Any] | None:
    """Run Atropos validation against a model."""
    print("  Running Atropos validation...", end=" ", flush=True)

    try:
        result = subprocess.run(
            [
                "atropos",
                "validate",
                scenario,
                "--model",
                model_name,
                "--device",
                device,
                "--format",
                "json",
            ],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )

        if result.returncode == 0:
            print("✓")
            return json.loads(result.stdout)
        else:
            print(f"✗ ({result.stderr[:100]})")
            return None

    except subprocess.TimeoutExpired:
        print("✗ (timeout)")
        return None
    except Exception as e:
        print(f"✗ ({e})")
        return None


def list_models():
    """List recommended models."""
    print("Recommended Models for Atropos Testing")
    print("=" * 60)

    for category, models in TEST_MODELS.items():
        print(f"\n{category.upper()} Models:")
        for model in models:
            params = estimate_params(model)
            params_str = f"{params:.2f}B" if params else "unknown"
            print(f"  - {model:40s} ({params_str})")

    print("\nTo test a specific model:")
    print("  atropos validate edge-coder --model gpt2")


def test_models(models: list[str], device: str = "cpu") -> list[ModelInfo]:
    """Test loading multiple models."""
    print(f"\nTesting {len(models)} models on {device}...")
    print("=" * 60)

    results = []
    for model in models:
        info = test_model_loading(model, device)
        results.append(info)

    return results


def validate_models(models: list[str], device: str = "cpu") -> dict[str, Any]:
    """Run Atropos validation on models."""
    print("\nRunning Atropos validation...")
    print("=" * 60)

    results = {}
    for model in models:
        result = run_atropos_validation(model, device=device)
        if result:
            results[model] = result

    return results


def generate_report(
    test_results: list[ModelInfo],
    validation_results: dict[str, Any],
    output: Path | None = None,
):
    """Generate a compatibility report."""
    report_lines = [
        "# Atropos Model Compatibility Report",
        "",
        "## System Information",
        "",
    ]

    # System info
    try:
        import torch
        import transformers

        report_lines.extend(
            [
                f"- PyTorch: {torch.__version__}",
                f"- Transformers: {transformers.__version__}",
                f"- CUDA Available: {torch.cuda.is_available()}",
            ]
        )
    except ImportError:
        report_lines.append("- Dependencies not installed")

    report_lines.extend(["", "## Model Loading Tests", ""])

    # Test results
    loadable = [r for r in test_results if r.loadable]
    failed = [r for r in test_results if not r.loadable]

    report_lines.append(f"**Loadable:** {len(loadable)}/{len(test_results)}")
    report_lines.append("")

    if loadable:
        report_lines.append("| Model | Params | Memory (GB) |")
        report_lines.append("|-------|--------|-------------|")
        for r in loadable:
            params = f"{r.params_b:.2f}B" if r.params_b else "?"
            mem = f"{r.memory_gb:.2f}" if r.memory_gb else "?"
            report_lines.append(f"| {r.name} | {params} | {mem} |")

    if failed:
        report_lines.extend(["", "### Failed to Load", ""])
        for r in failed:
            report_lines.append(f"- **{r.name}**: {r.error}")

    # Validation results
    if validation_results:
        report_lines.extend(["", "## Atropos Validation Results", ""])
        report_lines.append("| Model | Memory Var | Throughput Var | Accuracy |")
        report_lines.append("|-------|-----------|----------------|----------|")

        for model, result in validation_results.items():
            comparisons = {c["name"]: c for c in result.get("comparisons", [])}

            mem_var = comparisons.get("Memory", {}).get("variance_pct", "N/A")
            thr_var = comparisons.get("Throughput", {}).get("variance_pct", "N/A")
            acc = result.get("savings_accuracy", "N/A")

            mem_str = f"{mem_var:+.1f}%" if isinstance(mem_var, (int, float)) else mem_var
            thr_str = f"{thr_var:+.1f}%" if isinstance(thr_var, (int, float)) else thr_var
            acc_str = f"{acc:.1f}%" if isinstance(acc, (int, float)) else acc

            report_lines.append(f"| {model} | {mem_str} | {thr_str} | {acc_str} |")

    report_text = "\n".join(report_lines)

    if output:
        output.write_text(report_text)
        print(f"\nReport saved to: {output}")
    else:
        print("\n" + report_text)


def main():
    parser = argparse.ArgumentParser(description="Discover and test models for Atropos validation")
    parser.add_argument("--list", action="store_true", help="List recommended models")
    parser.add_argument("--test", action="store_true", help="Test model loading")
    parser.add_argument("--validate", action="store_true", help="Run Atropos validation")
    parser.add_argument("--full", action="store_true", help="Run complete workflow")
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Specific models to test (default: tiny + small)",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device to test on",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output file for report",
    )

    args = parser.parse_args()

    # Default action if none specified
    if not any([args.list, args.test, args.validate, args.full]):
        args.list = True

    if args.full:
        args.list = args.test = args.validate = True

    # List models
    if args.list:
        list_models()

    # Check dependencies
    if args.test or args.validate:
        if not check_dependencies():
            sys.exit(1)

    # Determine models to test
    models = args.models
    if not models:
        # Default to tiny + small models
        models = TEST_MODELS["tiny"] + TEST_MODELS["small"]

    # Test loading
    test_results = []
    if args.test:
        test_results = test_models(models, args.device)

    # Run validation
    validation_results = {}
    if args.validate:
        # Only validate models that passed loading test
        if test_results:
            loadable_models = [r.name for r in test_results if r.loadable]
        else:
            loadable_models = models

        if loadable_models:
            validation_results = validate_models(loadable_models, args.device)
        else:
            print("No loadable models to validate")

    # Generate report
    if args.test or args.validate:
        generate_report(test_results, validation_results, args.output)


if __name__ == "__main__":
    main()
