import sys
sys.stdout.flush()
from datasets import load_dataset
print("Loading dataset...")
sys.stdout.flush()
traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
print(f"Loaded {len(traindata)} documents")
sys.stdout.flush()
text = traindata[0]["text"]
print(f"First document length: {len(text)} chars")
sys.stdout.flush()
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
print("Tokenizing...")
sys.stdout.flush()
import time
start = time.time()
enc = tokenizer(text, padding=False, truncation=False, return_tensors="pt")
print(f"Tokenized in {time.time()-start:.2f}s, shape {enc.input_ids.shape}")
sys.stdout.flush()