#!/usr/bin/env python3
"""Download wikitext2 dataset."""

from datasets import load_dataset
import os

print("Loading wikitext2 dataset...")
# This will download and cache
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
print(f"Dataset loaded: {dataset}")
print("Test split size:", len(dataset["test"]))
print("Dataset cached.")