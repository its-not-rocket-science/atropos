#!/usr/bin/env python3
"""Inspect GPT-J layer structure."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "external" / "wanda"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "EleutherAI/gpt-j-6B"
print(f"Loading {model_id}...")
try:
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="cpu",
    )
    model.eval()
except Exception as e:
    print(f"Failed to load: {e}")
    sys.exit(1)

print(f"Model type: {type(model).__name__}")
print(f"Model config type: {model.config.model_type}")

# Get layer container
if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
    layers = model.transformer.h
    print(f"Number of layers: {len(layers)}")
    if len(layers) > 0:
        layer = layers[0]
        print(f"First layer type: {type(layer)}")
        print("Attributes:")
        for attr in dir(layer):
            if not attr.startswith('_'):
                val = getattr(layer, attr)
                if not callable(val):
                    print(f"  {attr}: {type(val)}")
                else:
                    print(f"  {attr}: callable")
        # Check for rotary_emb
        if hasattr(layer, 'attention'):
            print("\nLayer.attention attributes:")
            for attr in dir(layer.attention):
                if not attr.startswith('_'):
                    print(f"  {attr}")
            if hasattr(layer.attention, 'rotary_emb'):
                print("Found rotary_emb in attention")
        elif hasattr(layer, 'attn'):
            print("\nLayer.attn attributes:")
            for attr in dir(layer.attn):
                if not attr.startswith('_'):
                    print(f"  {attr}")
            if hasattr(layer.attn, 'rotary_emb'):
                print("Found rotary_emb in attn")
        else:
            print("No attention or attn attribute")
else:
    print("No transformer.h found")

print("\nDone.")