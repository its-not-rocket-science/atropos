#!/usr/bin/env python3
"""Prune models using Wanda (Pruning by Weights AND Activations).

Wanda pruning considers both weight magnitudes and input activation norms
for per-output pruning decisions. This often achieves better results than
magnitude-based pruning alone.

Reference: "A Simple and Effective Pruning Approach for Large Language Models"
https://arxiv.org/abs/2306.11695

Usage:
    python scripts/prune_wanda.py --model gpt2 --sparsity 0.3
    python scripts/prune_wanda.py --model gpt2-medium --sparsity 0.5 --save output/
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Add external/wanda to path
sys.path.insert(0, str(Path(__file__).parent.parent / "external" / "wanda"))

try:
    import numpy as np
    import torch

    # Import both pruning modules
    from lib.prune import check_sparsity as check_sparsity_base  # noqa: F401
    from lib.prune import prune_wanda as prune_wanda_base  # noqa: F401
    from lib.prune_opt import check_sparsity as check_sparsity_opt  # noqa: F401
    from lib.prune_opt import prune_wanda as prune_wanda_opt  # noqa: F401
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Import patched pruning functions
    sys.path.insert(0, str(Path(__file__).parent))
    from patched_prune import (
        check_sparsity_patched,
        prepare_calibration_input_patched,  # noqa: F401
        prune_wanda_patched,
    )
except ImportError as e:
    print(f"[FAIL] Required dependency not found: {e}")
    print("Install with: pip install torch transformers accelerate")
    sys.exit(1)


def get_model_path(test_data_dir: Path, model_name: str) -> Path:
    """Find model path handling HF cache structure."""
    # Direct path
    direct_path = test_data_dir / "models" / model_name.replace("/", "--")
    if direct_path.exists() and (direct_path / "config.json").exists():
        return direct_path

    # HF cache structure
    cache_path = test_data_dir / "models" / f"models--{model_name.replace('/', '--')}"
    if cache_path.exists():
        # Find snapshot directory
        snapshots = cache_path / "snapshots"
        if snapshots.exists():
            for snapshot in snapshots.iterdir():
                if snapshot.is_dir():
                    return snapshot

    raise FileNotFoundError(f"Model not found for {model_name} in {test_data_dir}/models")


def prune_model_wanda(
    model_name: str,
    model_path: Path,
    sparsity_ratio: float,
    output_path: Path,
    nsamples: int = 128,
    seed: int = 0,
    device: str = "cuda",
) -> dict[str, Any]:
    """Prune a model using Wanda method.

    Args:
        model_name: Original model identifier
        model_path: Path to local model directory
        sparsity_ratio: Target sparsity (0-1)
        output_path: Where to save pruned model
        nsamples: Number of calibration samples
        seed: Random seed
        device: Device to use

    Returns:
        Result dictionary with pruning metrics
    """
    import traceback

    # Use patched pruning functions that handle architecture detection
    check_sparsity_fn = check_sparsity_patched
    prune_wanda_fn = prune_wanda_patched

    start_time = datetime.now()
    result = {
        "model_id": model_name,
        "method": "wanda",
        "sparsity_target": sparsity_ratio,
        "sparsity_actual": 0.0,
        "status": "failed",
        "error": "",
        "duration_sec": 0.0,
        "output_path": str(output_path),
        "timestamp": start_time.isoformat(),
    }

    try:
        print(f"  Loading model from {model_path}...")

        # Set seeds
        np.random.seed(seed)
        torch.random.manual_seed(seed)

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
            device_map="auto" if device == "cuda" else None,
        )
        model.eval()
        if not hasattr(model, "hf_device_map"):
            model.hf_device_map = {}

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), use_fast=False)
        tokenizer.pad_token = tokenizer.eos_token

        # Get device
        if device == "cuda" and torch.cuda.is_available():
            device_obj = torch.device("cuda:0")
        else:
            device_obj = torch.device("cpu")
            model = model.to(device_obj)

        # Set seqlen attribute required by Wanda
        if hasattr(model.config, "max_position_embeddings"):
            model.seqlen = model.config.max_position_embeddings
        elif hasattr(model.config, "n_positions"):
            model.seqlen = model.config.n_positions
        else:
            model.seqlen = 2048  # default
        print(f"  Using device: {device_obj}")

        # Create args namespace for Wanda
        args = argparse.Namespace(
            model=str(model_path),
            seed=seed,
            nsamples=nsamples,
            sparsity_ratio=sparsity_ratio,
            sparsity_type="unstructured",
            use_variant=False,
            save=str(output_path.parent / "logs"),
            save_model=str(output_path),
            cache_dir=str(model_path),
        )

        print(f"  Pruning with Wanda (sparsity={sparsity_ratio:.2f})...")

        # Run Wanda pruning
        prune_wanda_fn(args, model, tokenizer, device_obj, prune_n=0, prune_m=0)

        # Check actual sparsity
        actual_sparsity = check_sparsity_fn(model)
        print(f"  Actual sparsity: {actual_sparsity:.4f}")

        # Save pruned model
        output_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        result["sparsity_actual"] = float(actual_sparsity)
        result["status"] = "success"
        result["duration_sec"] = duration

        print(f"  [OK] Saved to {output_path}")
        print(f"  Duration: {duration:.1f}s")

        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    except Exception as e:
        result["error"] = str(e)
        print(f"  [FAIL] {e}")
        traceback.print_exc()

    return result


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Prune models using Wanda method")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name (e.g., gpt2, gpt2-medium)",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        help="Path to model (default: test_data/models/{model})",
    )
    parser.add_argument(
        "--sparsity",
        type=float,
        default=0.3,
        help="Target sparsity ratio (default: 0.3)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("test_data/pruned_models"),
        help="Output directory (default: test_data/pruned_models)",
    )
    parser.add_argument(
        "--nsamples",
        type=int,
        default=128,
        help="Number of calibration samples (default: 128)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (default: 0)",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (default: cuda if available)",
    )
    parser.add_argument(
        "--test-data-dir",
        type=Path,
        default=Path("test_data"),
        help="Test data directory",
    )

    args = parser.parse_args()

    # Determine model path
    if args.model_path:
        model_path = args.model_path
    else:
        try:
            model_path = get_model_path(args.test_data_dir, args.model)
        except FileNotFoundError as e:
            print(f"[FAIL] {e}")
            return 1

    # Determine output path
    output_path = args.output / f"{args.model.replace('/', '--')}_wanda"

    print("=" * 70)
    print("Wanda Pruning")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Model path: {model_path}")
    print(f"Target sparsity: {args.sparsity:.2f}")
    print(f"Output: {output_path}")
    print(f"Device: {args.device}")
    print("-" * 70)

    result = prune_model_wanda(
        model_name=args.model,
        model_path=model_path,
        sparsity_ratio=args.sparsity,
        output_path=output_path,
        nsamples=args.nsamples,
        seed=args.seed,
        device=args.device,
    )

    # Save report
    report_path = args.test_data_dir / "wanda_report.json"
    reports = []
    if report_path.exists():
        with open(report_path, encoding="utf-8") as f:
            reports = json.load(f)

    reports.append(result)

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2)

    print(f"\nReport saved to: {report_path}")

    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
