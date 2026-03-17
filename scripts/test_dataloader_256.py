#!/usr/bin/env python3
"""Test dataloader with seqlen=256."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "external" / "wanda"))

import time

from patched_prune import get_wikitext2_patched
from transformers import AutoTokenizer

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

print("Calling get_wikitext2_patched with seqlen=256...")
start = time.time()
try:
    loader, _ = get_wikitext2_patched(2, 0, 256, tokenizer)
    print(f"Success in {time.time() - start:.2f}s")
    print(f"Loader length: {len(loader)}")
    print(f"Sample shape: {loader[0][0].shape}")
except Exception as e:
    print(f"Error: {e}")
    import traceback

    traceback.print_exc()
