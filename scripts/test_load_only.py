#!/usr/bin/env python3
"""Test model loading and architecture detection only."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "external" / "wanda"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def test_model_load(model_id: str):
    print(f"\n{'='*60}")
    print(f"Testing: {model_id}")
    print(f"{'='*60}")

    try:
        print("1. Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        print("   OK")

        print("2. Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        print("   OK")

        print("3. Importing patched_prune...")
        import patched_prune
        print("   OK")

        print("4. Detecting architecture...")
        arch = patched_prune.get_model_architecture(model)
        print(f"   Architecture: {arch}")

        print("5. Checking sparsity...")
        sparsity = patched_prune.check_sparsity_patched(model)
        print(f"   Sparsity: {sparsity:.6f}")

        return True
    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Test with a small model first
    print("Testing with gpt2 (small)...")
    success = test_model_load("gpt2")

    if success:
        print("\nTesting with opt-125m...")
        test_model_load("facebook/opt-125m")

    print("\nDone.")