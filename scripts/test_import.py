#!/usr/bin/env python3
"""Test imports step by step."""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "external" / "wanda"))
print("1. Importing torch")
import torch
print("   OK")
print("2. Importing transformers")
import transformers
print("   OK")
print("3. Importing lib.data")
try:
    import lib.data
    print("   OK")
except Exception as e:
    print(f"   ERROR: {e}")
print("4. Importing patched_prune")
try:
    import patched_prune
    print("   OK")
except Exception as e:
    print(f"   ERROR: {e}")
print("5. Loading small model")
from transformers import AutoModelForCausalLM, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
print("   Tokenizer OK")
model = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=torch.float16, low_cpu_mem_usage=True)
print("   Model OK")
print("All imports successful.")