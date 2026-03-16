#!/usr/bin/env python3
import sys
sys.path.insert(0, 'external/wanda')
from transformers import AutoTokenizer
from patched_prune import get_wikitext2_patched
import time

print("Starting test with seqlen=128")
tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

start = time.time()
try:
    loader, _ = get_wikitext2_patched(2, 0, 128, tokenizer)
    print(f"Success in {time.time() - start:.2f}s")
    print(f"Loader length: {len(loader)}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()