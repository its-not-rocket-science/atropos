#!/usr/bin/env python3
"""Prune models using advanced frameworks (Wanda, SparseGPT).

This script runs Wanda and SparseGPT pruning on candidate models for comparison.

Usage:
    python scripts/prune_advanced_frameworks.py
    python scripts/prune_advanced_frameworks.py --models gpt2 gpt2-medium
    python scripts/prune_advanced_frameworks.py --sparsity 0.5 --device cuda
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Default models to prune
DEFAULT_MODELS = [
    "gpt2",
    "gpt2-medium",
]

# Sparsity levels to test
SPARSITY_LEVELS = [0.3, 0.5]

# Methods to test
METHODS = ["wanda", "sparsegpt"]


def run_pruning(
    model: str,
    method: str,
    sparsity: float,
    test_data_dir: Path,
    device: str,
) -> dict[str, Any]:
    """Run pruning with specified method.

    Args:
        model: Model name
        method: Pruning method (wanda or sparsegpt)
        sparsity: Target sparsity ratio
        test_data_dir: Test data directory
        device: Device to use

    Returns:
        Result dictionary
    """
    script_name = f"prune_{method}.py"
    script_path = Path(__file__).parent / script_name

    cmd = [
        sys.executable,
        str(script_path),
        "--model",
        model,
        "--sparsity",
        str(sparsity),
        "--device",
        device,
        "--test-data-dir",
        str(test_data_dir),
    ]

    print(f"\nRunning: {' '.join(cmd)}")
    print("-" * 50)

    start_time = datetime.now()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200,  # 2 hour timeout
        )

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        success = result.returncode == 0

        return {
            "model": model,
            "method": method,
            "sparsity_target": sparsity,
            "status": "success" if success else "failed",
            "duration_sec": duration,
            "stdout": result.stdout[-500:] if len(result.stdout) > 500 else result.stdout,
            "stderr": result.stderr[-500:] if len(result.stderr) > 500 else result.stderr,
            "timestamp": start_time.isoformat(),
        }

    except subprocess.TimeoutExpired:
        return {
            "model": model,
            "method": method,
            "sparsity_target": sparsity,
            "status": "timeout",
            "duration_sec": 7200,
            "error": "Pruning timed out after 2 hours",
            "timestamp": start_time.isoformat(),
        }
    except Exception as e:
        return {
            "model": model,
            "method": method,
            "sparsity_target": sparsity,
            "status": "error",
            "error": str(e),
            "timestamp": start_time.isoformat(),
        }


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Prune models using Wanda and SparseGPT")
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help=f"Models to prune (default: {DEFAULT_MODELS})",
    )
    parser.add_argument(
        "--sparsity-levels",
        nargs="+",
        type=float,
        default=SPARSITY_LEVELS,
        help=f"Sparsity levels (default: {SPARSITY_LEVELS})",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["wanda", "sparsegpt", "both"],
        default=["both"],
        help="Methods to run (default: both)",
    )
    parser.add_argument(
        "--device",
        default="cuda" if __import__("torch").cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--test-data-dir",
        type=Path,
        default=Path("test_data"),
        help="Test data directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("test_data/advanced_pruning_report.json"),
        help="Output report path",
    )

    args = parser.parse_args()

    # Determine which methods to run
    methods = []
    if "both" in args.methods:
        methods = ["wanda", "sparsegpt"]
    else:
        methods = args.methods

    print("=" * 70)
    print("Advanced Pruning Frameworks - Batch Pruning")
    print("=" * 70)
    print(f"Models: {args.models}")
    print(f"Methods: {methods}")
    print(f"Sparsity levels: {args.sparsity_levels}")
    print(f"Device: {args.device}")
    print(f"Test data: {args.test_data_dir}")
    print("=" * 70)

    # Check if external/wanda exists
    wanda_path = Path(__file__).parent.parent / "external" / "wanda"
    if not wanda_path.exists():
        print(f"[FAIL] Wanda repo not found at {wanda_path}")
        print("Clone with: git clone --depth 1 https://github.com/locuslab/wanda.git")
        print("  external/wanda")
        return 1

    results = []
    total = len(args.models) * len(methods) * len(args.sparsity_levels)
    completed = 0

    for model in args.models:
        for method in methods:
            for sparsity in args.sparsity_levels:
                completed += 1
                print(f"\n[{completed}/{total}] {model} - {method} - {sparsity:.0%} sparsity")

                result = run_pruning(
                    model=model,
                    method=method,
                    sparsity=sparsity,
                    test_data_dir=args.test_data_dir,
                    device=args.device,
                )
                results.append(result)

                status = "✓" if result["status"] == "success" else "✗"
                print(f"{status} {result['status']} ({result.get('duration_sec', 0):.1f}s)")

    # Generate summary
    successful = sum(1 for r in results if r["status"] == "success")
    failed = len(results) - successful

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Total: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")

    # Save report
    report = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total": len(results),
            "successful": successful,
            "failed": failed,
        },
        "config": {
            "models": args.models,
            "methods": methods,
            "sparsity_levels": args.sparsity_levels,
            "device": args.device,
        },
        "results": results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nReport saved to: {args.output}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
