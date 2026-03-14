#!/usr/bin/env python3
"""Test architecture detection in patched_prune."""

import sys
from pathlib import Path

# Add external/wanda to path
sys.path.insert(0, str(Path(__file__).parent / "external" / "wanda"))

import torch
from transformers import AutoModelForCausalLM

from scripts.patched_prune import adapt_model_for_pruning, get_model_architecture


def test_model(model_name: str, expected_arch: str):
    print(f"\n--- Testing {model_name} ---")
    try:
        model_path = Path(f"test_data/models/models--{model_name.replace('/', '--')}")
        # Find snapshot
        snapshots = model_path / "snapshots"
        if not snapshots.exists():
            print(f"  No snapshot for {model_name}, skipping")
            return
        snapshot = next(snapshots.iterdir())
        model = AutoModelForCausalLM.from_pretrained(
            str(snapshot),
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        arch = get_model_architecture(model)
        print(f"  Detected arch: {arch} (expected: {expected_arch})")
        assert arch == expected_arch, f"Arch mismatch: {arch} != {expected_arch}"
        # Test adapt_model_for_pruning
        adapted, original = adapt_model_for_pruning(model, arch)
        print(
            f"  Adapted successfully, model.model.layers exists: {hasattr(adapted.model, 'layers')}"
        )
        # Restore
        from scripts.patched_prune import restore_model_attrs

        restore_model_attrs(adapted, original)
        print("  OK Pass")
    except Exception as e:
        print(f"  FAIL: {e}")


if __name__ == "__main__":
    # Test GPT2
    test_model("gpt2", "gpt2")
    # Test OPT
    test_model("facebook/opt-1.3b", "opt")
    # Test Pythia (GPT-NeoX)
    test_model("EleutherAI/pythia-2.8b", "gpt_neox")
    print("\nAll tests completed.")
