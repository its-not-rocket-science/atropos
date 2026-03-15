#!/usr/bin/env python3
"""Test BLOOM pruning with mocked dataloader."""

import sys
from pathlib import Path

# Add external/wanda to path
sys.path.insert(0, str(Path(__file__).parent.parent / "external" / "wanda"))

import torch
import random
import lib.data

# Mock get_loaders to avoid dataset download
original_get_loaders = lib.data.get_loaders

def mock_get_loaders(name, nsamples=128, seed=0, seqlen=2048, tokenizer=None):
    print(f"[MOCK] get_loaders called with name={name}, nsamples={nsamples}, seqlen={seqlen}")
    # Create dummy dataloader with random token IDs
    random.seed(seed)
    trainloader = []
    vocab_size = tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else 50257
    for _ in range(nsamples):
        # Generate random token IDs
        inp = torch.randint(0, vocab_size, (1, seqlen), dtype=torch.long)
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    # Create dummy testenc (just a tensor)
    testenc_input_ids = torch.randint(0, vocab_size, (1, seqlen * 2), dtype=torch.long)
    # Wrap in a simple object with input_ids attribute
    class DummyTestEnc:
        def __init__(self, input_ids):
            self.input_ids = input_ids
    testenc = DummyTestEnc(testenc_input_ids)
    return trainloader, testenc

lib.data.get_loaders = mock_get_loaders
print("Patched lib.data.get_loaders")

# Now import patched_prune (which imports lib.prune)
from patched_prune import (
    check_sparsity_patched,
    prune_wanda_patched,
    get_model_architecture,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "bigscience/bloom-560m"
print(f"Loading {model_id}...")
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

device = torch.device("cpu")
print(f"Using device: {device}")
model.to(device)

# Test sparsity check
print("\n--- Testing sparsity check ---")
sparsity_before = check_sparsity_patched(model)
print(f"Sparsity before: {sparsity_before:.6f}")

# Test Wanda pruning
print("\n--- Testing Wanda pruning ---")
try:
    prune_wanda_patched(args, model, tokenizer, device, prune_n=0, prune_m=0)
    print("Wanda pruning completed.")
except Exception as e:
    print(f"[FAIL] Error during pruning: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

sparsity_after = check_sparsity_patched(model)
print(f"Sparsity after: {sparsity_after:.6f}")
print(f"Target sparsity: {args.sparsity_ratio:.2f}")
print(f"Difference: {abs(sparsity_after - args.sparsity_ratio):.6f}")

if sparsity_after > 0.05:
    print("[OK] Wanda pruning applied successfully")
else:
    print("[FAIL] Wanda pruning failed to achieve significant sparsity")
    sys.exit(1)

# Restore original get_loaders
lib.data.get_loaders = original_get_loaders
print("Restored original get_loaders")

print("\nTest completed successfully.")