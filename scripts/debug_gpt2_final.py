#!/usr/bin/env python3
"""Debug GPT2 pruning with proper adaptation."""

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

print("2. Adapting model for pruning...")
arch = patched_prune.get_model_architecture(model)
model, original = patched_prune.adapt_model_for_pruning(model, arch)
print(f"   Architecture: {arch}")
print(f"   Model.seqlen: {model.seqlen}")
print(f"   Original attrs: {list(original.keys())}")

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

print("5. Calling prune_wanda_gpt2 directly...")
try:
    patched_prune.prune_wanda_gpt2(args, model, tokenizer, device, prune_n=0, prune_m=0)
    print("   prune_wanda_gpt2 completed successfully")
except Exception as e:
    print(f"   ERROR in prune_wanda_gpt2: {e}")
    import traceback
    traceback.print_exc()

print("6. Checking sparsity after pruning...")
sparsity_after = patched_prune.check_sparsity_patched(model)
print(f"   Sparsity after: {sparsity_after:.6f}")

print("7. Restoring model attributes...")
patched_prune.restore_model_attrs(model, original)
print("   OK")

print("\nDebug complete.")