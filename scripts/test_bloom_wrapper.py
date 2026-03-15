#!/usr/bin/env python3
"""Test BLOOM layer wrapper with dummy data."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "external" / "wanda"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from patched_prune import wrap_bloom_layer_forwards, unwrap_bloom_layer_forwards

model_id = "bigscience/bloom-560m"
print(f"Loading {model_id}...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)
model.eval()

# Get layers
layers = model.transformer.h
print(f"Number of layers: {len(layers)}")
layer = layers[0]
print(f"Layer type: {type(layer)}")

# Wrap forward
original_forwards = wrap_bloom_layer_forwards([layer])
print("Wrapped layer forward")

# Create dummy inputs
batch_size = 1
seq_len = 16
hidden_size = model.config.hidden_size
hidden_states = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float16)
# Causal mask shape (batch, 1, seq_len, seq_len)
attention_mask = torch.tril(torch.ones(batch_size, 1, seq_len, seq_len, dtype=torch.float16))
print(f"Hidden states shape: {hidden_states.shape}")
print(f"Attention mask shape: {attention_mask.shape}")
print(f"Attention mask dim: {attention_mask.dim()}")

# Test forward with position_ids (should be ignored)
position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
print(f"Position ids shape: {position_ids.shape}")

try:
    with torch.no_grad():
        output = layer(hidden_states, attention_mask=attention_mask, position_ids=position_ids)
    print(f"Forward succeeded! Output shape: {output.shape}")
except Exception as e:
    print(f"Forward failed: {e}")
    import traceback
    traceback.print_exc()

# Restore
unwrap_bloom_layer_forwards([layer], original_forwards)
print("Restored original forward")

# Test original forward (without position_ids)
try:
    with torch.no_grad():
        output2 = layer(hidden_states, attention_mask=attention_mask)
    print(f"Original forward succeeded! Output shape: {output2.shape}")
except Exception as e:
    print(f"Original forward failed: {e}")
    import traceback
    traceback.print_exc()

print("Done.")