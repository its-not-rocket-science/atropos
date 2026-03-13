#!/usr/bin/env python3
"""Test GPT2 pruning with SparseGPT."""

import argparse
import sys
from pathlib import Path

# Add external/wanda to path
sys.path.insert(0, str(Path(__file__).parent / "external" / "wanda"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.patched_prune import check_sparsity_patched, prune_sparsegpt_patched


def test():
    model_path = Path(
        "test_data/models/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e"
    )

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    # Set required attributes
    model.seqlen = model.config.max_position_embeddings
    if not hasattr(model, "hf_device_map"):
        model.hf_device_map = {}

    # Create args
    args = argparse.Namespace(
        model=str(model_path),
        seed=0,
        nsamples=2,  # minimal
        sparsity_ratio=0.1,
        sparsity_type="unstructured",
        use_variant=False,
        save="test_output",
        save_model="test_output",
        cache_dir=str(model_path),
    )

    device = torch.device("cpu")

    print("Checking initial sparsity...")
    sparsity_before = check_sparsity_patched(model)
    print(f"Sparsity before: {sparsity_before:.6f}")

    print("Pruning...")
    prune_sparsegpt_patched(args, model, tokenizer, device, prune_n=0, prune_m=0)

    print("Checking sparsity after...")
    sparsity_after = check_sparsity_patched(model)
    print(f"Sparsity after: {sparsity_after:.6f}")
    print(f"Target sparsity: {args.sparsity_ratio:.2f}")
    print(f"Difference: {abs(sparsity_after - args.sparsity_ratio):.6f}")

    if sparsity_after > 0.05:  # at least 5% sparsity achieved
        print("SUCCESS: Pruning applied")
    else:
        print("WARNING: Low sparsity achieved")


if __name__ == "__main__":
    test()