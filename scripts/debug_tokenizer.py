import sys
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import logging
logging.basicConfig(level=logging.DEBUG)
import time
print("Starting tokenizer load...")
sys.stdout.flush()
start = time.time()
from transformers import AutoTokenizer
print("AutoTokenizer imported")
sys.stdout.flush()
try:
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    print(f"Tokenizer loaded in {time.time()-start:.2f}s")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()