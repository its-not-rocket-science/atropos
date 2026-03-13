#!/usr/bin/env python3
"""Download and cache pruning candidate models for Atropos testing.

This script downloads 5 candidate models for the pruning exercise:
- gpt2 (124M) - Fast baseline
- gpt2-medium (355M) - Realistic edge scenario
- gpt2-xl (1.5B) - Pre-LLaMA scale
- facebook/opt-1.3b (1.3B) - Open-source alternative
- EleutherAI/pythia-2.8b (2.8B) - Research-grade, near 3B boundary

Usage:
    python scripts/download_test_models.py [--test-data-dir PATH]
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

# Candidate models for pruning exercise
CANDIDATE_MODELS = [
    {"id": "gpt2", "params": 0.124, "license": "MIT"},
    {"id": "gpt2-medium", "params": 0.355, "license": "MIT"},
    {"id": "gpt2-xl", "params": 1.5, "license": "MIT"},
    {"id": "facebook/opt-1.3b", "params": 1.3, "license": "Apache-2.0"},
    {"id": "EleutherAI/pythia-2.8b", "params": 2.8, "license": "Apache-2.0"},
]


@dataclass
class DownloadResult:
    """Result of downloading a model."""

    model_id: str
    status: str  # "success", "failed", "skipped"
    params_b: float
    license: str
    download_time_sec: float = 0.0
    disk_size_mb: float | None = None
    cache_path: str | None = None
    error_message: str = ""
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DownloadReport:
    """Report of all download operations."""

    total_models: int = 0
    successful: int = 0
    failed: int = 0
    skipped: int = 0
    results: list[DownloadResult] = field(default_factory=list)
    start_time: str = ""
    end_time: str = ""
    test_data_dir: str = ""

    @property
    def duration_sec(self) -> float:
        """Calculate total duration."""
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
            "test_data_dir": self.test_data_dir,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "results": [r.to_dict() for r in self.results],
        }


def check_dependencies() -> bool:
    """Check if required dependencies are installed."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: F401

        print(f"[OK] PyTorch {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        return True
    except ImportError as e:
        print(f"[FAIL] Missing dependency: {e}")
        print("\nInstall with: pip install torch transformers")
        return False


def get_cache_size(cache_path: Path) -> float:
    """Calculate directory size in MB."""
    if not cache_path.exists():
        return 0.0

    total_size = 0
    for file in cache_path.rglob("*"):
        if file.is_file():
            total_size += file.stat().st_size

    return total_size / (1024 * 1024)  # Convert to MB


def download_model(
    model_info: dict[str, Any],
    cache_dir: Path,
    device: str = "cpu",
) -> DownloadResult:
    """Download a single model to the cache directory.

    Args:
        model_info: Dictionary with model id, params, license
        cache_dir: Directory to cache models
        device: Device to load on for testing

    Returns:
        DownloadResult with status and metadata
    """
    model_id = model_info["id"]
    params_b = model_info["params"]
    license = model_info["license"]

    print(f"\n  Downloading {model_id} ({params_b}B params)...")
    start_time = time.time()

    result = DownloadResult(
        model_id=model_id,
        status="failed",
        params_b=params_b,
        license=license,
        timestamp=datetime.now().isoformat(),
    )

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Download tokenizer
        print("    Tokenizer...", end=" ", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir=cache_dir,
        )
        print("[OK]")

        # Download model
        print("    Model...", end=" ", flush=True)
        load_kwargs = {
            "cache_dir": cache_dir,
            "torch_dtype": torch.float32,
        }
        if device == "cuda":
            load_kwargs["device_map"] = "auto"
            load_kwargs["low_cpu_mem_usage"] = True

        model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        print("[OK]")

        # Test inference
        print("    Testing inference...", end=" ", flush=True)
        test_input = "def hello_world():"
        inputs = tokenizer(test_input, return_tensors="pt")
        if device == "cuda":
            inputs = inputs.to(device)

        with torch.no_grad():
            _ = model(**inputs)
        print("[OK]")

        # Calculate metrics
        download_time = time.time() - start_time

        # Find cache path
        model_slug = model_id.replace("/", "--")
        cache_path = cache_dir / f"models--{model_slug}"

        disk_size = get_cache_size(cache_path) if cache_path.exists() else None

        result.status = "success"
        result.download_time_sec = download_time
        result.disk_size_mb = disk_size
        result.cache_path = str(cache_path) if cache_path.exists() else None

        print(f"    [OK] Success ({download_time:.1f}s, {disk_size:.1f} MB)")

        # Cleanup memory
        del model
        import gc

        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    except Exception as e:
        result.error_message = str(e)
        print(f"    [FAIL] Failed: {e}")

    return result


def download_all_models(
    test_data_dir: Path,
    models: list[dict[str, Any]] | None = None,
    device: str = "cpu",
) -> DownloadReport:
    """Download all candidate models.

    Args:
        test_data_dir: Directory to store test data
        models: List of models to download (default: CANDIDATE_MODELS)
        device: Device for testing

    Returns:
        DownloadReport with all results
    """
    models = models or CANDIDATE_MODELS

    report = DownloadReport(
        total_models=len(models),
        start_time=datetime.now().isoformat(),
        test_data_dir=str(test_data_dir),
    )

    # Create cache directory
    cache_dir = test_data_dir / "models"
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Downloading Candidate Models for Pruning Exercise")
    print("=" * 60)
    print(f"Cache directory: {cache_dir}")
    print(f"Device: {device}")
    print(f"Models to download: {len(models)}")

    for i, model_info in enumerate(models, 1):
        print(f"\n[{i}/{len(models)}]", end="")
        result = download_model(model_info, cache_dir, device)
        report.results.append(result)

        if result.status == "success":
            report.successful += 1
        elif result.status == "skipped":
            report.skipped += 1
        else:
            report.failed += 1

    report.end_time = datetime.now().isoformat()

    return report


def print_summary(report: DownloadReport) -> None:
    """Print a summary of the download operation."""
    print("\n" + "=" * 60)
    print("Download Summary")
    print("=" * 60)
    print(f"Total models: {report.total_models}")
    print(f"Successful: {report.successful} [OK]")
    print(f"Failed: {report.failed} [FAIL]")
    print(f"Skipped: {report.skipped} [SKIP]")
    print(f"Duration: {report.duration_sec:.1f}s")
    print(f"Test data directory: {report.test_data_dir}")

    print("\nModel Details:")
    print("-" * 60)
    print(f"{'Model':<30} {'Status':<10} {'Size':<12} {'Time':<10}")
    print("-" * 60)

    for r in report.results:
        size_str = f"{r.disk_size_mb:.1f} MB" if r.disk_size_mb else "N/A"
        time_str = f"{r.download_time_sec:.1f}s" if r.download_time_sec else "N/A"
        icon = "[OK]" if r.status == "success" else "[FAIL]" if r.status == "failed" else "[SKIP]"
        print(f"{r.model_id:<30} {icon:<6} {r.status:<8} {size_str:<12} {time_str:<10}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Download candidate models for Atropos pruning exercise"
    )
    parser.add_argument(
        "--test-data-dir",
        type=Path,
        default=Path("test_data"),
        help="Directory to store downloaded models (default: test_data)",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device for testing model loading",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output JSON file for report",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Specific model IDs to download (default: all candidates)",
    )

    args = parser.parse_args()

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    # Filter models if specified
    models = None
    if args.models:
        models = [m for m in CANDIDATE_MODELS if m["id"] in args.models]
        if not models:
            print(f"Error: No matching models found in {args.models}")
            sys.exit(1)

    # Download models
    report = download_all_models(
        test_data_dir=args.test_data_dir,
        models=models,
        device=args.device,
    )

    # Print summary
    print_summary(report)

    # Save report
    output_path = args.output or args.test_data_dir / "download_report.json"
    output_path = Path(output_path)
    output_path.write_text(json.dumps(report.to_dict(), indent=2))
    print(f"\nReport saved to: {output_path}")

    # Exit with error code if any downloads failed
    if report.failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
