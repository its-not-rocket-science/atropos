#!/usr/bin/env python3
"""Test tokenizer speed."""

import sys
from pathlib import Path
import time

# Add external/wanda to path
sys.path.insert(0, str(Path(__file__).parent.parent / "external" / "wanda"))

from lib.data import get_wikitext2
from transformers import AutoTokenizer

model_name = "gpt2"
print("Loading tokenizer with use_fast=False...")
start = time.time()
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
print(f"Tokenizer loaded in {time.time() - start:.2f}s")
tokenizer.pad_token = tokenizer.eos_token

print("Loading tokenizer with use_fast=True...")
start = time.time()
tokenizer_fast = AutoTokenizer.from_pretrained(model_name, use_fast=True)
print(f"Tokenizer loaded in {time.time() - start:.2f}s")
tokenizer_fast.pad_token = tokenizer_fast.eos_token

print("Testing get_wikitext2 with fast tokenizer...")
start = time.time()
try:
    trainloader, testenc = get_wikitext2(nsamples=1, seed=0, seqlen=1024, tokenizer=tokenizer_fast)
    print(f"get_wikitext2 succeeded in {time.time() - start:.2f}s")
except Exception as e:
    print(f"Error with fast tokenizer: {e}")
    import traceback
    traceback.print_exc()

print("Testing get_wikitext2 with slow tokenizer...")
start = time.time()
try:
    trainloader, testenc = get_wikitext2(nsamples=1, seed=0, seqlen=1024, tokenizer=tokenizer)
    print(f"get_wikitext2 succeeded in {time.time() - start:.2f}s")
except Exception as e:
    print(f"Error with slow tokenizer: {e}")
    import traceback
    traceback.print_exc()