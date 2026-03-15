#!/usr/bin/env python3
"""Examine BLOOM's build_alibi_tensor function."""

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "external" / "wanda"))

try:
    from transformers.models.bloom.modeling_bloom import build_alibi_tensor
    import inspect
    print("Function signature:")
    print(inspect.signature(build_alibi_tensor))
    print("\nSource:")
    print(inspect.getsource(build_alibi_tensor))
except ImportError as e:
    print(f"Import error: {e}")
    # Try to find file
    import transformers
    import os
    bloom_path = os.path.join(os.path.dirname(transformers.__file__), "models", "bloom", "modeling_bloom.py")
    print(f"Bloom path: {bloom_path}")
    if os.path.exists(bloom_path):
        with open(bloom_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if 'def build_alibi_tensor' in line:
                    print(f"Found at line {i+1}")
                    for j in range(i, min(i+50, len(lines))):
                        print(lines[j].rstrip())
                    break