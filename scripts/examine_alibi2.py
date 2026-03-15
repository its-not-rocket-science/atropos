#!/usr/bin/env python3
"""Examine BLOOM's build_alibi_tensor function."""

import transformers
import os
import inspect

bloom_path = os.path.join(os.path.dirname(transformers.__file__), "models", "bloom", "modeling_bloom.py")
print(f"Bloom path: {bloom_path}")

if os.path.exists(bloom_path):
    with open(bloom_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if 'def build_alibi_tensor' in line:
                print(f"Found at line {i+1}")
                # Print next 60 lines
                for j in range(i, min(i+60, len(lines))):
                    print(lines[j].rstrip())
                break

# Also try to import and inspect
try:
    from transformers.models.bloom.modeling_bloom import build_alibi_tensor
    print("\n=== Import successful ===")
    print("Signature:", inspect.signature(build_alibi_tensor))
    # Try to call with dummy inputs to see expected shapes
    import torch
    attention_mask = torch.ones(1, 1, 10, 10)  # 4D causal mask
    print(f"4D mask shape: {attention_mask.shape}")
    try:
        out = build_alibi_tensor(attention_mask, num_heads=8, dtype=torch.float32)
        print(f"Output shape with 4D mask: {out.shape}")
    except Exception as e:
        print(f"Error with 4D mask: {e}")
    # Try 2D mask
    attention_mask_2d = torch.ones(1, 10)
    try:
        out2 = build_alibi_tensor(attention_mask_2d, num_heads=8, dtype=torch.float32)
        print(f"Output shape with 2D mask: {out2.shape}")
    except Exception as e:
        print(f"Error with 2D mask: {e}")
except ImportError as e:
    print(f"Import error: {e}")