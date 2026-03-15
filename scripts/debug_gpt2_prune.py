#!/usr/bin/env python3
"""Debug GPT2 pruning step by step."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "external" / "wanda"))

import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import patched_prune

print("1. Loading model and tokenizer...")
model_id = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)
model.eval()
print("   OK")

print("2. Detecting architecture...")
arch = patched_prune.get_model_architecture(model)
print(f"   Architecture: {arch}")

print("3. Creating args...")
args = argparse.Namespace(
    model=model_id,
    seed=0,
    nsamples=2,
    sparsity_ratio=0.1,
    sparsity_type="unstructured",
    use_variant=False,
    save="test_output",
    save_model="test_output",
    cache_dir=None,
)

device = torch.device("cpu")
model.to(device)

print("4. Checking sparsity before pruning...")
sparsity_before = patched_prune.check_sparsity_patched(model)
print(f"   Sparsity before: {sparsity_before:.6f}")

print("5. Testing get_wikitext2_patched directly...")
from patched_prune import get_wikitext2_patched
print("   Calling get_wikitext2_patched with nsamples=2...")
try:
    trainloader, testenc = get_wikitext2_patched(2, 0, model.seqlen, tokenizer)
    print(f"   Success: got {len(trainloader)} samples")
except Exception as e:
    print(f"   ERROR in get_wikitext2_patched: {e}")
    import traceback
    traceback.print_exc()

print("6. Testing prune_wanda_gpt2 directly (with monkey-patched get_loaders)...")
# Monkey-patch get_loaders to use our patched version
import lib.data
original_get_loaders = lib.data.get_loaders
def mock_get_loaders(name, nsamples, seed, seqlen, tokenizer):
    print(f"   Mock get_loaders called with name={name}")
    if name == "wikitext2":
        return get_wikitext2_patched(nsamples, seed, seqlen, tokenizer)
    else:
        return original_get_loaders(name, nsamples, seed, seqlen, tokenizer)
lib.data.get_loaders = mock_get_loaders

print("   Calling prune_wanda_gpt2...")
try:
    patched_prune.prune_wanda_gpt2(args, model, tokenizer, device, prune_n=0, prune_m=0)
    print("   prune_wanda_gpt2 completed")
except Exception as e:
    print(f"   ERROR in prune_wanda_gpt2: {e}")
    import traceback
    traceback.print_exc()

print("7. Checking sparsity after pruning...")
sparsity_after = patched_prune.check_sparsity_patched(model)
print(f"   Sparsity after: {sparsity_after:.6f}")

print("\nDebug complete.")