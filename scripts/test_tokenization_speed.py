#!/usr/bin/env python3
"""Test tokenization speed."""

import sys
from pathlib import Path
import time

# Add external/wanda to path
sys.path.insert(0, str(Path(__file__).parent.parent / "external" / "wanda"))

from datasets import load_dataset
from transformers import AutoTokenizer

# Load wikitext2
print("Loading wikitext2 dataset...")
traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
print(f"Train samples: {len(traindata)}, Test samples: {len(testdata)}")

# Get first 10 samples
train_text = " ".join(traindata["text"][:10])
print(f"First 10 samples length: {len(train_text)} chars")

# Test GPT2 tokenizer slow
print("\nTesting GPT2 tokenizer (use_fast=False)...")
start = time.time()
tokenizer_slow = AutoTokenizer.from_pretrained("gpt2", use_fast=False)
tokenizer_slow.pad_token = tokenizer_slow.eos_token
tokenize_start = time.time()
enc_slow = tokenizer_slow(train_text, return_tensors="pt")
print(f"Tokenizer load: {tokenize_start - start:.2f}s, Tokenization: {time.time() - tokenize_start:.2f}s")
print(f"Encoded shape: {enc_slow.input_ids.shape}")

# Test GPT2 tokenizer fast
print("\nTesting GPT2 tokenizer (use_fast=True)...")
start = time.time()
tokenizer_fast = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
tokenizer_fast.pad_token = tokenizer_fast.eos_token
tokenize_start = time.time()
enc_fast = tokenizer_fast(train_text, return_tensors="pt")
print(f"Tokenizer load: {tokenize_start - start:.2f}s, Tokenization: {time.time() - tokenize_start:.2f}s")
print(f"Encoded shape: {enc_fast.input_ids.shape}")

# Test OPT tokenizer
print("\nTesting OPT-125m tokenizer (use_fast=False)...")
start = time.time()
tokenizer_opt = AutoTokenizer.from_pretrained("facebook/opt-125m", use_fast=False)
tokenizer_opt.pad_token = tokenizer_opt.eos_token
tokenize_start = time.time()
enc_opt = tokenizer_opt(train_text, return_tensors="pt")
print(f"Tokenizer load: {tokenize_start - start:.2f}s, Tokenization: {time.time() - tokenize_start:.2f}s")
print(f"Encoded shape: {enc_opt.input_ids.shape}")