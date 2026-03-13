#!/usr/bin/env python3
"""Debug GPT2 layer structure."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "external" / "wanda"))

import torch
import torch.nn as nn
from lib.prune import find_layers
from transformers import AutoModelForCausalLM


def inspect_layer(layer, depth=0, prefix=""):
    indent = "  " * depth
    print(f"{indent}{prefix}{type(layer).__name__}")
    if hasattr(layer, "_modules"):
        for name, child in layer._modules.items():
            if child is not None:
                inspect_layer(child, depth + 1, f"{name}.")


def test_gpt2():
    print("Inspecting GPT2 layers...")
    model_path = Path(
        "test_data/models/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e"
    )

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.eval()

    # Get first block
    block = model.transformer.h[0]
    print("First GPT2Block structure:")
    inspect_layer(block)

    # Try find_layers
    print("\nfind_layers result:")
    subset = find_layers(block)
    print(f"Found {len(subset)} linear layers")
    for name, module in subset.items():
        print(f"  {name}: {module}")

    # Manually search for nn.Linear and Conv1D
    print("\nManual search for nn.Linear and Conv1D:")
    for name, module in block.named_modules():
        if isinstance(module, nn.Linear):
            print(f"  Linear {name}: {module.weight.shape}")
        # Check for Conv1D
        if type(module).__name__ == "Conv1D":
            print(f"  Conv1D {name}: {module.weight.shape}")
            print(f"    full type: {type(module)}")
            print(f"    module class: {module.__class__.__module__}.{module.__class__.__name__}")

    # Try to import Conv1D
    try:
        from transformers.modeling_utils import Conv1D

        print(f"\nConv1D class imported: {Conv1D}")
    except ImportError:
        print("\nCould not import Conv1D from transformers.modeling_utils")
        # Try other import
        try:
            from transformers.pytorch_utils import Conv1D

            print(f"Conv1D from pytorch_utils: {Conv1D}")
        except ImportError:
            print("Could not import Conv1D from pytorch_utils")


if __name__ == "__main__":
    test_gpt2()
