#!/usr/bin/env python3
"""Integration test for SparseGPTPatchedFramework."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import tempfile

from atropos.pruning_integration import PruningResult, get_pruning_framework


def test_framework_availability():
    """Test that the framework can be instantiated."""
    print("Testing SparseGPTPatchedFramework availability...")
    try:
        framework = get_pruning_framework("sparsegpt-patched")
        print(f"[OK] Framework instantiated: {framework.name}")
        print(f"  Description: {framework.description}")
        return True
    except Exception as e:
        print(f"[FAIL] Failed to instantiate framework: {e}")
        return False


def test_prune_gpt2_small():
    """Test pruning on GPT2 small (quick test)."""
    print("\nTesting pruning on GPT2 small...")

    # Create a temporary directory for output
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "pruned_gpt2"

        try:
            framework = get_pruning_framework("sparsegpt-patched")

            # Use a small target sparsity to keep test fast
            target_sparsity = 0.1  # 10%

            print(f"Pruning 'gpt2' with target sparsity {target_sparsity:.1%}")
            print(f"Output will be saved to: {output_path}")

            result: PruningResult = framework.prune(
                model_name="gpt2",
                output_path=output_path,
                target_sparsity=target_sparsity,
                nsamples=1,  # minimal calibration samples
                seed=0,
            )

            print(f"Pruning result success: {result.success}")
            if result.success:
                print(f"  Original params: {result.original_params}")
                print(f"  Pruned params: {result.pruned_params}")
                print(f"  Sparsity achieved: {result.sparsity_achieved:.4f}")
                print(f"  Parameter reduction: {result.parameter_reduction_fraction:.2%}")
                print(f"  Output path exists: {output_path.exists()}")
                return True
            else:
                print(f"  Error: {result.error_message}")
                return False

        except Exception as e:
            print(f"[FAIL] Pruning test failed: {e}")
            import traceback

            traceback.print_exc()
            return False


if __name__ == "__main__":
    print("=" * 60)
    print("SparseGPTPatchedFramework Integration Test")
    print("=" * 60)

    # Test 1: Framework availability
    avail_ok = test_framework_availability()
    if not avail_ok:
        print("Framework availability test failed. Exiting.")
        sys.exit(1)

    # Test 2: Actual pruning (optional, can be skipped if slow)
    # Check if we should run pruning test (might take minutes)
    if len(sys.argv) > 1 and sys.argv[1] == "--run-pruning":
        prune_ok = test_prune_gpt2_small()
        if not prune_ok:
            print("Pruning test failed.")
            sys.exit(1)
        else:
            print("\n[OK] All tests passed!")
    else:
        print("\nSkipping actual pruning test (use --run-pruning to enable).")
        print("Framework integration appears functional.")
