#!/usr/bin/env python3
"""Test original get_wikitext2 with GPT2 tokenizer."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "external" / "wanda"))

import time
from transformers import AutoTokenizer
import lib.data

print("Loading GPT2 tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

print("Testing original get_wikitext2 with seqlen=1024, nsamples=2...")
start = time.time()
try:
    trainloader, testenc = lib.data.get_wikitext2(2, 0, 1024, tokenizer)
    elapsed = time.time() - start
    print(f"Success! Took {elapsed:.2f} seconds")
    print(f"Got {len(trainloader)} samples")
    print(f"Sample shape: {trainloader[0][0].shape}")
except Exception as e:
    elapsed = time.time() - start
    print(f"Error after {elapsed:.2f} seconds: {e}")
    import traceback
    traceback.print_exc()

print("\nTesting with seqlen=512...")
start = time.time()
try:
    trainloader, testenc = lib.data.get_wikitext2(2, 0, 512, tokenizer)
    elapsed = time.time() - start
    print(f"Success! Took {elapsed:.2f} seconds")
    print(f"Got {len(trainloader)} samples")
except Exception as e:
    elapsed = time.time() - start
    print(f"Error after {elapsed:.2f} seconds: {e}")