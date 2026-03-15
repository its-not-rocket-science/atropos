#!/usr/bin/env python3
"""Inspect BLOOM model structure."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "external" / "wanda"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "bigscience/bloom-560m"
print(f"Loading {model_id}...")
try:
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.eval()
except Exception as e:
    print(f"Failed to load: {e}")
    sys.exit(1)

print(f"Model type: {type(model).__name__}")
print(f"Model config type: {model.config.model_type}")
print(f"Attributes:")
for attr in dir(model):
    if not attr.startswith('_'):
        print(f"  {attr}")

# Check for transformer.h
if hasattr(model, 'transformer'):
    print("\nModel has transformer attribute")
    if hasattr(model.transformer, 'h'):
        print("Model.transformer.h exists (likely layer container)")
        print(f"Type: {type(model.transformer.h)}")
        print(f"Length: {len(model.transformer.h)}")
        # Inspect first layer
        if len(model.transformer.h) > 0:
            layer = model.transformer.h[0]
            print(f"First layer type: {type(layer)}")
            for attr in dir(layer):
                if not attr.startswith('_'):
                    print(f"    {attr}")
    else:
        print("Model.transformer.h not found")
        for sub in dir(model.transformer):
            if not sub.startswith('_'):
                print(f"  transformer.{sub}")
else:
    print("No transformer attribute")

# Check for model.model.layers (LLaMA style)
if hasattr(model, 'model'):
    print("\nModel has model attribute")
    if hasattr(model.model, 'layers'):
        print("Model.model.layers exists")
        print(f"Length: {len(model.model.layers)}")
    else:
        for sub in dir(model.model):
            if not sub.startswith('_'):
                print(f"  model.{sub}")

# Check for position_ids usage
print("\nChecking config for position embeddings:")
if hasattr(model.config, 'use_position_embeddings'):
    print(f"  use_position_embeddings: {model.config.use_position_embeddings}")
if hasattr(model.config, 'position_embedding_type'):
    print(f"  position_embedding_type: {model.config.position_embedding_type}")

# Check if model has seqlen attribute
if hasattr(model, 'seqlen'):
    print(f"Model.seqlen: {model.seqlen}")
else:
    print("Model.seqlen not set")

print("\nDone.")