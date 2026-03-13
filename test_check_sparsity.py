#!/usr/bin/env python3
"""Test check_sparsity for GPT2."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "external" / "wanda"))
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

import torch
from patched_prune import (
    adapt_model_for_pruning,
    check_sparsity_patched,
    get_model_architecture,
    restore_model_attrs,
)
from transformers import AutoModelForCausalLM


def test_gpt2():
    print("Testing GPT2...")
    model_path = Path(
        "test_data/models/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e"
    )

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.eval()
    if not hasattr(model, "hf_device_map"):
        model.hf_device_map = {}

    # Set seqlen
    if hasattr(model.config, "max_position_embeddings"):
        model.seqlen = model.config.max_position_embeddings

    arch = get_model_architecture(model)
    print(f"Architecture: {arch}")

    # Adapt model
    model, original = adapt_model_for_pruning(model, arch)
    try:
        # Get layers
        layers = model.model.layers
        print(f"Number of layers: {len(layers)}")

        # Import find_layers
        from lib.prune import find_layers

        for i, layer in enumerate(layers):
            subset = find_layers(layer)
            print(f"Layer {i}: {len(subset)} linear layers")
            for name, module in subset.items():
                print(f"  {name}: {module.weight.shape}")

        # Call check_sparsity
        print("\nCalling check_sparsity_patched...")
        sparsity = check_sparsity_patched(model)
        print(f"Sparsity: {sparsity}")
    finally:
        restore_model_attrs(model, original)


if __name__ == "__main__":
    test_gpt2()
