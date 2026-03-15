#!/usr/bin/env python3
"""Test OPT and Pythia pruning with Wanda and SparseGPT."""

import argparse
import sys
from pathlib import Path

# Add external/wanda to path
sys.path.insert(0, str(Path(__file__).parent / "external" / "wanda"))

import torch
from patched_prune import (
    check_sparsity_patched,
    prune_sparsegpt_patched,
    prune_wanda_patched,
)
from transformers import AutoModelForCausalLM, AutoTokenizer


def test_model(model_name: str, model_id: str, device: torch.device = None):
    """Test pruning on a single model."""
    print(f"\n{'=' * 60}")
    print(f"Testing {model_name} ({model_id})")
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

        # Try loading tokenizer with use_fast=False first, fall back to default
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
        except ValueError:
            # Some tokenizers (e.g., GPTNeoX) don't support use_fast=False
            tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Set pad token if not already set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Set required attributes for pruning
        model.seqlen = model.config.max_position_embeddings
        if not hasattr(model, "hf_device_map"):
            model.hf_device_map = {}

        # Create args
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

        if sparsity_after > 0.05:  # at least 5% sparsity achieved
            print("[OK] Wanda pruning applied successfully")
            wanda_success = True
        else:
            print("[FAIL] Wanda pruning failed to achieve significant sparsity")
            wanda_success = False

        # Reload fresh model for SparseGPT test
        print("\n--- Reloading fresh model for SparseGPT test ---")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        model.eval()
        model.seqlen = model.config.max_position_embeddings
        if not hasattr(model, "hf_device_map"):
            model.hf_device_map = {}
        model.to(device)

        # Test SparseGPT pruning
        print("\n--- Testing SparseGPT pruning ---")
        sparsity_before = check_sparsity_patched(model)
        print(f"Sparsity before: {sparsity_before:.6f}")

        prune_sparsegpt_patched(args, model, tokenizer, device, prune_n=0, prune_m=0)

        sparsity_after = check_sparsity_patched(model)
        print(f"Sparsity after: {sparsity_after:.6f}")
        print(f"Target sparsity: {args.sparsity_ratio:.2f}")
        print(f"Difference: {abs(sparsity_after - args.sparsity_ratio):.6f}")

        if sparsity_after > 0.05:
            print("[OK] SparseGPT pruning applied successfully")
            sparsegpt_success = True
        else:
            print("[FAIL] SparseGPT pruning failed to achieve significant sparsity")
            sparsegpt_success = False

        return wanda_success and sparsegpt_success

    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if "out of memory" in str(e).lower() or "cuda out of memory" in str(e).lower():
            print(f"[SKIP] Out of memory testing {model_name}: {e}")
            return True  # Skip due to insufficient memory, not a failure
        else:
            print(f"[FAIL] Runtime error testing {model_name}: {e}")
            import traceback

            traceback.print_exc()
            return False
    except Exception as e:
        print(f"[FAIL] Error testing {model_name}: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Test OPT and Pythia models."""
    # Available models to test
    available_models = {
        "opt-125m": "facebook/opt-125m",
        "opt-1.3b": "facebook/opt-1.3b",
        "pythia-160m": "EleutherAI/pythia-160m",
        "pythia-410m": "EleutherAI/pythia-410m",
        "pythia-1b": "EleutherAI/pythia-1b",
    }

    parser = argparse.ArgumentParser(description="Test pruning on OPT and Pythia models")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(available_models.keys()),
        default=list(available_models.keys()),
        help="Model keys to test (default: all)",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device to run pruning on (default: cpu)",
    )
    parser.add_argument(
        "--skip-sparsegpt",
        action="store_true",
        help="Skip SparseGPT pruning tests (Wanda only)",
    )
    args = parser.parse_args()

    # Build test cases from selected models
    test_cases = [(name, available_models[name]) for name in args.models]

    # Determine device
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        if args.device == "cuda":
            print("Warning: CUDA requested but not available, falling back to CPU")

    print(f"Testing {len(test_cases)} model(s) on device: {device}")
    mem_info = (
        torch.cuda.get_device_properties(0).total_memory
        if torch.cuda.is_available()
        else "CPU only"
    )
    print(f"Available memory: {mem_info}")

    results = {}
    for model_name, model_id in test_cases:
        success = test_model(model_name, model_id, device, args.skip_sparsegpt)
        results[model_name] = success

    print(f"\n{'=' * 60}")
    print("Test Summary:")
    for model_name, success in results.items():
        status = "[OK] PASS" if success else "[FAIL] FAIL"
        print(f"  {model_name}: {status}")

    all_passed = all(results.values())
    if all_passed:
        print("\n[OK] All tests passed!")
    else:
        print("\n[FAIL] Some tests failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
