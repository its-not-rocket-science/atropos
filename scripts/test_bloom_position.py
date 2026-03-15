#!/usr/bin/env python3
"""Test if BLOOM accepts position_ids."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "external" / "wanda"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "bigscience/bloom-560m"
print(f"Loading {model_id}...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True,
)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# Get a sample input
input_text = "Hello world"
inputs = tokenizer(input_text, return_tensors="pt")
input_ids = inputs.input_ids
attention_mask = inputs.attention_mask

# Try forward with position_ids
print("Testing forward with position_ids...")
try:
    position_ids = torch.arange(input_ids.shape[1], dtype=torch.long).unsqueeze(0)
    outputs = model(input_ids, attention_mask=attention_mask, position_ids=position_ids)
    print("Success: position_ids accepted")
except TypeError as e:
    print(f"Error: {e}")
    print("position_ids not accepted")

# Try forward without position_ids
print("\nTesting forward without position_ids...")
try:
    outputs = model(input_ids, attention_mask=attention_mask)
    print("Success: forward works without position_ids")
except Exception as e:
    print(f"Error: {e}")

# Test layer forward
print("\nTesting first layer forward...")
layer = model.transformer.h[0]
layer_input = torch.randn(1, input_ids.shape[1], model.config.hidden_size)
try:
    out = layer(layer_input, attention_mask=attention_mask, position_ids=position_ids)
    print("Layer accepts position_ids")
except Exception as e:
    print(f"Layer error with position_ids: {e}")
    try:
        out = layer(layer_input, attention_mask=attention_mask)
        print("Layer works without position_ids")
    except Exception as e2:
        print(f"Layer error without position_ids: {e2}")

print("\nDone.")