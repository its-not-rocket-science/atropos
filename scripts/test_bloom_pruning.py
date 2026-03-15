#!/usr/bin/env python3
"""Test BLOOM pruning with patched functions."""

import sys
from pathlib import Path

# Add external/wanda to path
sys.path.insert(0, str(Path(__file__).parent.parent / "external" / "wanda"))

import torch
from patched_prune import (
    check_sparsity_patched,
    prune_sparsegpt_patched,
    prune_wanda_patched,
    get_model_architecture,
)
from transformers import AutoModelForCausalLM, AutoTokenizer


def test_bloom(model_id: str = "bigscience/bloom-560m", device: torch.device = None):
    """Test pruning on BLOOM model."""
    print(f"\n{'=' * 60}")
    print(f"Testing BLOOM: {model_id}")
    print(f"{'=' * 60}")

    try:
        # Download model on the fly
        print("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Detect architecture
        arch = get_model_architecture(model)
        print(f"Detected architecture: {arch}")

        # Create args
        import argparse
        args = argparse.Namespace(
            model=model_id,
            seed=0,
            nsamples=2,  # minimal calibration samples
            sparsity_ratio=0.1,
            sparsity_type="unstructured",
            use_variant=False,
            save="test_output",
            save_model="test_output",
            cache_dir=None,
        )

        if device is None:
            device = torch.device("cpu")
        print(f"Using device: {device}")
        model.to(device)

        # Test Wanda pruning
        print("\n--- Testing Wanda pruning ---")
        sparsity_before = check_sparsity_patched(model)
        print(f"Sparsity before: {sparsity_before:.6f}")

        print("Starting Wanda pruning...")
        prune_wanda_patched(args, model, tokenizer, device, prune_n=0, prune_m=0)
        print("Wanda pruning completed.")

        sparsity_after = check_sparsity_patched(model)
        print(f"Sparsity after: {sparsity_after:.6f}")
        print(f"Target sparsity: {args.sparsity_ratio:.2f}")
        print(f"Difference: {abs(sparsity_after - args.sparsity_ratio):.6f}")

        if sparsity_after > 0.05:
            print("[OK] Wanda pruning applied successfully")
            return True
        else:
            print("[FAIL] Wanda pruning failed to achieve significant sparsity")
            return False

    except Exception as e:
        print(f"[FAIL] Error testing BLOOM: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    device = torch.device("cpu")
    success = test_bloom(device=device)
    sys.exit(0 if success else 1)