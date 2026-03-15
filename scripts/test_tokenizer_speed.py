#!/usr/bin/env python3
"""Test tokenizer speed for GPT2."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "external" / "wanda"))

import time
from transformers import AutoTokenizer
from patched_prune import get_wikitext2_patched

print("Loading GPT2 tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

print("Testing get_wikitext2_patched with nsamples=2, seqlen=1024...")
start = time.time()
try:
    trainloader, testenc = get_wikitext2_patched(2, 0, 1024, tokenizer)
    elapsed = time.time() - start
    print(f"Success! Took {elapsed:.2f} seconds")
    print(f"Got {len(trainloader)} samples")
except Exception as e:
    elapsed = time.time() - start
    print(f"Error after {elapsed:.2f} seconds: {e}")
    import traceback
    traceback.print_exc()