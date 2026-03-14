import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from patched_prune import check_sparsity_patched, prune_wanda_patched
from transformers import AutoModelForCausalLM, AutoTokenizer


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

        # Run Wanda pruning using patched function
        prune_wanda_patched(args, model, tokenizer, device_obj, prune_n=0, prune_m=0)

        # Check actual sparsity
        actual_sparsity = check_sparsity_patched(model)
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

    return result
