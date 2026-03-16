#!/usr/bin/env python3
"""Test loading GPT2."""

import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add external/wanda to path
sys.path.insert(0, str(Path(__file__).parent.parent / "external" / "wanda"))

MODEL_NAME = "gpt2"
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)
print("Model loaded.")
model.eval()
print("Model eval.")

# Try loading tokenizer with use_fast=False first, fall back to default
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
except ValueError:
    # Some tokenizers (e.g., GPTNeoX) don't support use_fast=False
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print("Tokenizer loaded.")

# Set pad token if not already set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print("Set pad token.")

# Set required attributes for pruning
model.seqlen = model.config.max_position_embeddings
if not hasattr(model, "hf_device_map"):
    model.hf_device_map = {}
print("Attributes set.")

print("Success.")
