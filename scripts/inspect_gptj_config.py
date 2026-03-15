#!/usr/bin/env python3
"""Inspect GPT-J config and structure."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "external" / "wanda"))

from transformers import AutoConfig

model_id = "EleutherAI/gpt-j-6B"
print(f"Loading config for {model_id}...")
config = AutoConfig.from_pretrained(model_id)
print(f"Config type: {config.model_type}")
print(f"Config attributes:")
for attr in dir(config):
    if not attr.startswith('_'):
        val = getattr(config, attr)
        if not callable(val):
            print(f"  {attr}: {val}")

# Try to infer layer container from known patterns
print("\nChecking known layer container patterns:")
print("Has max_position_embeddings?", hasattr(config, 'max_position_embeddings'))
print("Has n_positions?", hasattr(config, 'n_positions'))
print("Has seq_length?", hasattr(config, 'seq_length'))
print("Has rotary?", hasattr(config, 'rotary'))
print("Has rotary_dim?", hasattr(config, 'rotary_dim'))

# Load a tiny model with torch_dtype and low memory, but only one layer?
# Not now.

print("\nDone.")