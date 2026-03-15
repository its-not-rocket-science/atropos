#!/usr/bin/env python3
"""Test dataloader."""

import sys
from pathlib import Path

# Add external/wanda to path
sys.path.insert(0, str(Path(__file__).parent.parent / "external" / "wanda"))

from lib.data import get_loaders
from transformers import AutoTokenizer

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

print("Getting dataloader...")
try:
    dataloader, _ = get_loaders(
        "wikitext2",
        nsamples=1,
        seed=0,
        seqlen=1024,
        tokenizer=tokenizer,
    )
    print("Success! Dataloader obtained.")
    # Try to get one batch
    for batch in dataloader:
        print(f"Batch shape: {batch[0].shape}")
        break
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()