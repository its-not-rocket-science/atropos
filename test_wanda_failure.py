#!/usr/bin/env python3
"""Test where Wanda pruning fails with GPT2."""

import sys
from pathlib import Path

import torch
from lib.prune import prune_wanda
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add external/wanda to path
sys.path.insert(0, str(Path(__file__).parent / "external" / "wanda"))


def test_gpt2():
    print("Testing GPT2 with Wanda...")

    model_path = Path(
        "test_data/models/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e"
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.eval()

    # Set required attributes
    model.seqlen = model.config.max_position_embeddings
    if not hasattr(model, "hf_device_map"):
        model.hf_device_map = {}

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    # Create args
    import argparse

    args = argparse.Namespace(
        model=str(model_path),
        seed=0,
        nsamples=2,
        sparsity_ratio=0.3,
        sparsity_type="unstructured",
        use_variant=False,
        save="test_output",
        save_model="test_output",
        cache_dir=str(model_path),
    )

    device = torch.device("cpu")

    try:
        print("Calling prune_wanda...")
        prune_wanda(args, model, tokenizer, device, prune_n=0, prune_m=0)
        print("Success!")
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_gpt2()
