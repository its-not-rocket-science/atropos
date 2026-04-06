"""Automated model testing suite for Atropos.

Tests models from HuggingFace Hub for compatibility with Atropos validation.
Generates a catalog of working models with their specifications.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml  # type: ignore[import-untyped]

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass
class ModelTestResult:
    """Result of testing a single model."""

    model_id: str
    status: str  # "success", "failed", "skipped"
    params_b: float | None = None
    memory_gb: float | None = None
    load_time_sec: float | None = None
    inference_time_ms: float | None = None
    error_message: str = ""
    test_timestamp: str = ""
    device: str = "cpu"
    tags: list[str] = field(default_factory=list)
    architecture: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TestSuiteResult:
    """Results from running the full test suite."""

    total_models: int = 0
    successful: int = 0
    failed: int = 0
    skipped: int = 0
    results: list[ModelTestResult] = field(default_factory=list)
    start_time: str = ""
    end_time: str = ""
    test_device: str = ""

    @property
    def duration_sec(self) -> float:
        """Calculate test duration."""
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
            "skipped": self.skipped,
            "duration_sec": self.duration_sec,
            "test_device": self.test_device,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "results": [r.to_dict() for r in self.results],
        }


def estimate_params_from_name(model_id: str) -> float | None:
    """Estimate parameter count from model name."""
    import re

    name_lower = model_id.lower()

    # Patterns like "125m", "1.1b", "7b"
    patterns = [
        (r"(\d+\.?\d*)b", 1e9),
        (r"(\d+\.?\d*)m", 1e6),
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


def test_model(
    model_id: str,
    device: str = "cpu",
    timeout_sec: float = 300.0,
    skip_if_params_b: float | None = None,
) -> ModelTestResult:
    """Test a single model from HuggingFace Hub.

    Args:
        model_id: HuggingFace model ID (e.g., "gpt2").
        device: Device to test on ("cpu" or "cuda").
        timeout_sec: Timeout for model loading.
        skip_if_params_b: Skip if estimated params > this value.

    Returns:
        ModelTestResult with test status and metrics.
    """
    print(f"  Testing {model_id}...", end=" ", flush=True)

    # Estimate parameters first (quick check)
    params_b = estimate_params_from_name(model_id)
    if skip_if_params_b and params_b and params_b > skip_if_params_b:
        print("SKIP (too large)")
        return ModelTestResult(
            model_id=model_id,
            status="skipped",
            params_b=params_b,
            error_message=f"Estimated {params_b:.2f}B params > {skip_if_params_b}B limit",
            test_timestamp=datetime.now().isoformat(),
        )

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("FAIL (dependencies)")
        return ModelTestResult(
            model_id=model_id,
            status="failed",
            error_message="torch/transformers not installed",
            test_timestamp=datetime.now().isoformat(),
        )

    load_start = time.time()
    try:
        # Test tokenizer loading
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Test model loading with timeout
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            device_map="auto" if device == "cuda" else None,
            low_cpu_mem_usage=True,
        )

        if device == "cpu":
            model = model.cpu()

        load_time = time.time() - load_start

        # Count parameters
        param_count = sum(p.numel() for p in model.parameters())
        params_b = param_count / 1e9
        memory_gb = param_count * 4 / (1024**3)  # float32 estimate

        # Test inference
        test_input = "Hello, world!"
        inputs = tokenizer(test_input, return_tensors="pt")
        if device == "cuda":
            inputs = inputs.to(device)

        with torch.no_grad():
            inference_start = time.time()
            _ = model(**inputs)
            inference_time = (time.time() - inference_start) * 1000

        # Cleanup
        del model
        import gc

        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

        print(f"OK ({params_b:.2f}B, {load_time:.1f}s)")

        return ModelTestResult(
            model_id=model_id,
            status="success",
            params_b=params_b,
            memory_gb=memory_gb,
            load_time_sec=load_time,
            inference_time_ms=inference_time,
            test_timestamp=datetime.now().isoformat(),
            device=device,
        )

    except torch.cuda.OutOfMemoryError:
        print("FAIL (OOM)")
        return ModelTestResult(
            model_id=model_id,
            status="failed",
            params_b=params_b,
            error_message="CUDA out of memory",
            test_timestamp=datetime.now().isoformat(),
            device=device,
        )
    except Exception as e:
        print(f"FAIL ({str(e)[:50]})")
        return ModelTestResult(
            model_id=model_id,
            status="failed",
            params_b=params_b,
            error_message=str(e)[:200],
            test_timestamp=datetime.now().isoformat(),
            device=device,
        )


def run_test_suite(
    models: Sequence[str],
    device: str = "cpu",
    max_params_b: float | None = 3.0,
    output_path: Path | None = None,
) -> TestSuiteResult:
    """Run tests on multiple models.

    Args:
        models: List of HuggingFace model IDs to test.
        device: Device to test on.
        max_params_b: Skip models with more than this many parameters.
        output_path: Optional path to save JSON results.

    Returns:
        TestSuiteResult with all test results.
    """
    result = TestSuiteResult(
        total_models=len(models),
        test_device=device,
        start_time=datetime.now().isoformat(),
    )

    print("\nRunning model test suite")
    print(f"Device: {device}")
    print(f"Models to test: {len(models)}")
    print(f"Max params: {max_params_b or 'unlimited'}B")
    print("=" * 60)

    for i, model_id in enumerate(models, 1):
        print(f"[{i}/{len(models)}]", end=" ")
        test_result = test_model(
            model_id,
            device=device,
            skip_if_params_b=max_params_b,
        )
        result.results.append(test_result)

        if test_result.status == "success":
            result.successful += 1
        elif test_result.status == "failed":
            result.failed += 1
        else:
            result.skipped += 1

    result.end_time = datetime.now().isoformat()

    # Print summary
    print("\n" + "=" * 60)
    print("Test Suite Summary")
    print(f"Total: {result.total_models}")
    print(f"Successful: {result.successful} [OK]")
    print(f"Failed: {result.failed} [FAIL]")
    print(f"Skipped: {result.skipped} [SKIP]")
    print(f"Duration: {result.duration_sec:.1f}s")

    # Save results
    if output_path:
        output_path.write_text(json.dumps(result.to_dict(), indent=2))
        print(f"\nResults saved to: {output_path}")

    return result


def generate_catalog(
    test_results: TestSuiteResult,
    output_path: Path,
    min_success_rate: float = 0.0,
) -> None:
    """Generate models-catalog.yaml from test results.

    Args:
        test_results: Results from test suite.
        output_path: Path to write YAML catalog.
        min_success_rate: Only include if success rate >= this.
    """
    successful = [r for r in test_results.results if r.status == "success"]

    if test_results.total_models > 0:
        success_rate = len(successful) / test_results.total_models
        if success_rate < min_success_rate:
            print(f"Success rate {success_rate:.1%} below minimum {min_success_rate:.1%}")
            return

    # Group by size tier
    edge_models = [r for r in successful if r.params_b and r.params_b <= 1]
    medium_models = [r for r in successful if r.params_b and 1 < r.params_b <= 7]
    large_models = [r for r in successful if r.params_b and r.params_b > 7]

    catalog = {
        "catalog_metadata": {
            "generated_at": test_results.end_time,
            "test_device": test_results.test_device,
            "total_tested": test_results.total_models,
            "successful": test_results.successful,
            "success_rate": f"{test_results.successful / test_results.total_models:.1%}",
        },
        "models": {
            "edge": [
                {
                    "model_id": r.model_id,
                    "params_b": r.params_b,
                    "memory_gb": r.memory_gb,
                    "inference_ms": r.inference_time_ms,
                }
                for r in sorted(edge_models, key=lambda x: x.params_b or 0)
            ],
            "medium": [
                {
                    "model_id": r.model_id,
                    "params_b": r.params_b,
                    "memory_gb": r.memory_gb,
                    "inference_ms": r.inference_time_ms,
                }
                for r in sorted(medium_models, key=lambda x: x.params_b or 0)
            ],
            "large": [
                {
                    "model_id": r.model_id,
                    "params_b": r.params_b,
                    "memory_gb": r.memory_gb,
                    "inference_ms": r.inference_time_ms,
                }
                for r in sorted(large_models, key=lambda x: x.params_b or 0)
            ],
        },
    }

    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(catalog, f, default_flow_style=False, sort_keys=False)

    print(f"Catalog saved to: {output_path}")


def get_recommended_test_models() -> list[str]:
    """Get curated list of recommended models to test.

    Returns:
        List of HuggingFace model IDs suitable for testing.
    """
    return [
        # Edge models (< 1B)
        "gpt2",  # 124M
        "facebook/opt-125m",
        "EleutherAI/pythia-160m",
        "gpt2-medium",  # 355M
        "facebook/opt-350m",
        "EleutherAI/pythia-410m",
        "bigscience/bloom-560m",
        "gpt2-large",  # 774M
        # Medium models (1-7B)
        "gpt2-xl",  # 1.5B
        "TinyLlama/TinyLlama-1.1B-v1.0",
        "bigscience/bloom-1b7",
        "facebook/opt-1.3b",
        "EleutherAI/pythia-2.8b",
        "openlm-research/open_llama_3b",
        "stabilityai/stablelm-3b-4e1t",
        # Code-specific
        "microsoft/codegpt-small",
        "Salesforce/codet5-small",
        "t5-small",
    ]


def main() -> None:
    """CLI entry point for model testing."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Test HuggingFace models for Atropos compatibility"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Specific model IDs to test (default: recommended list)",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device to test on",
    )
    parser.add_argument(
        "--max-params",
        type=float,
        default=3.0,
        help="Maximum parameter count in billions (default: 3)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("model-test-results.json"),
        help="Output JSON file for test results",
    )
    parser.add_argument(
        "--catalog",
        "-c",
        type=Path,
        help="Generate YAML catalog file",
    )

    args = parser.parse_args()

    models = args.models or get_recommended_test_models()

    results = run_test_suite(
        models=models,
        device=args.device,
        max_params_b=args.max_params,
        output_path=args.output,
    )

    if args.catalog:
        generate_catalog(results, args.catalog)


if __name__ == "__main__":
    main()
