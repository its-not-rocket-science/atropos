#!/usr/bin/env python3
"""Test patched dataloader."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "external" / "wanda"))

from patched_prune import get_wikitext2_patched
from transformers import AutoTokenizer

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

print("Calling get_wikitext2_patched...")
try:
    loader, _ = get_wikitext2_patched(2, 0, 1024, tokenizer)
    print(f"Success! Loader length: {len(loader)}")
except Exception as e:
    print(f"Error: {e}")
    import traceback

    traceback.print_exc()
