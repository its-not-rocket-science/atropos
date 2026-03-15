#!/usr/bin/env python3
"""Debug where the hang occurs."""

import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent.parent / "external" / "wanda"))

print("Step 1: Import transformers")
import transformers
print("  OK")

print("Step 2: Import lib.data")
import lib.data
print("  OK")

print("Step 3: Patch get_loaders")
original_get_loaders = lib.data.get_loaders
def mock(*args, **kwargs):
    print("Mock get_loaders called")
    return [], None
lib.data.get_loaders = mock
print("  OK")

print("Step 4: Import patched_prune")
start = time.time()
try:
    import patched_prune
    print(f"  OK (took {time.time() - start:.2f}s)")
except Exception as e:
    print(f"  ERROR: {e}")

print("Step 5: Load model")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
model_id = "bigscience/bloom-560m"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)
print("  OK")

print("All steps completed.")