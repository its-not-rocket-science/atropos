#!/usr/bin/env python3
"""Check document lengths in wikitext2."""

from datasets import load_dataset
from transformers import AutoTokenizer
import sys

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

print("Loading dataset...")
traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
print(f"Total documents: {len(traindata)}")

seqlen = 512
max_docs = 500
min_char_length = seqlen * 4

print(f"Checking first {max_docs} documents for >= {seqlen} tokens...")
suitable = []
for i in range(min(max_docs, len(traindata))):
    if i % 100 == 0:
        print(f"Processed {i}...")
        sys.stdout.flush()
    text = traindata[i]["text"]
    if not text or text.strip() == "":
        continue
    if len(text) < min_char_length:
        continue
    try:
        enc = tokenizer(text, padding=False, truncation=False, return_tensors="pt")
        if enc.input_ids.shape[1] >= seqlen:
            suitable.append((i, enc.input_ids.shape[1]))
            print(f"Found doc {i}: {enc.input_ids.shape[1]} tokens")
            if len(suitable) >= 10:
                break
    except Exception as e:
        print(f"Warning doc {i}: {e}")
        continue

print(f"\nFound {len(suitable)} suitable documents:")
for idx, tok_len in suitable:
    print(f"  Doc {idx}: {tok_len} tokens")

if len(suitable) == 0:
    print("No suitable documents. Checking token lengths of first few documents...")
    for i in range(min(10, len(traindata))):
        text = traindata[i]["text"]
        enc = tokenizer(text, padding=False, truncation=False, return_tensors="pt")
        print(f"Doc {i}: {enc.input_ids.shape[1]} tokens")