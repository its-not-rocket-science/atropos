#!/usr/bin/env python3
"""Debug GPT2 pruning."""

import sys
from pathlib import Path

# Add external/wanda to path
sys.path.insert(0, str(Path(__file__).parent.parent / "external" / "wanda"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from patched_prune import check_sparsity_patched, get_model_architecture

model_name = "gpt2"
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True,
)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

# Detect architecture
arch = get_model_architecture(model)
print(f"Detected architecture: {arch}")

# Check sparsity
sparsity = check_sparsity_patched(model)
print(f"Sparsity: {sparsity}")

print("Test passed.")