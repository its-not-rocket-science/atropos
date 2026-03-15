#!/usr/bin/env python3
"""Load BLOOM model to ensure it's cached."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "external" / "wanda"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "bigscience/bloom-560m"
print(f"Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
print(f"Tokenizer loaded.")

print(f"Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)
print(f"Model loaded.")
print(f"Model type: {type(model).__name__}")
print(f"Model config type: {model.config.model_type}")
print(f"Model device: {next(model.parameters()).device}")

# Test forward
input_text = "Hello world"
inputs = tokenizer(input_text, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
print(f"Forward pass successful.")
print(f"Output logits shape: {outputs.logits.shape}")

print("Done.")