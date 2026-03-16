import sys
sys.path.insert(0, 'external/wanda')
from patched_prune import get_wikitext2_patched
from transformers import AutoTokenizer
import time

print("Testing dataloader with seqlen=1024")
tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

start = time.time()
try:
    loader, _ = get_wikitext2_patched(2, 0, 1024, tokenizer)
    print(f"Success in {time.time() - start:.2f}s")
    print(f"Loader length: {len(loader)}")
    print(f"Sample shape: {loader[0][0].shape}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()