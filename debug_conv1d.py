#!/usr/bin/env python3
"""Debug Conv1D layer shapes."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "external" / "wanda"))

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from transformers.pytorch_utils import Conv1D


def test_conv1d():
    print("Testing Conv1D...")
    model_path = Path(
        "test_data/models/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e"
    )

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.eval()

    # Get first Conv1D layer
    conv1d = model.transformer.h[0].attn.c_attn
    print(f"Conv1D layer: {conv1d}")
    print(f"  weight shape: {conv1d.weight.shape}")
    print(f"  bias shape: {conv1d.bias.shape}")
    print(f"  isinstance(conv1d, Conv1D): {isinstance(conv1d, Conv1D)}")
    print(f"  isinstance(conv1d, nn.Linear): {isinstance(conv1d, nn.Linear)}")
    print(f"  type(conv1d): {type(conv1d)}")
    print(f"  type(conv1d).__bases__: {type(conv1d).__bases__}")

    # Check forward signature
    import inspect

    sig = inspect.signature(conv1d.forward)
    print(f"  forward signature: {sig}")

    # Test forward pass with dummy input
    batch_size = 2
    seq_len = 10
    hidden_size = 768
    dummy_input = torch.randn(batch_size, seq_len, hidden_size)
    print(f"\nInput shape: {dummy_input.shape}")

    with torch.no_grad():
        output = conv1d(dummy_input)
        print(f"Output shape: {output.shape}")
        print("Expected output shape based on weight[768, 2304]:")
        print("  weight[in_features=768, out_features=2304]")
        print("  input [batch, seq_len, 768] -> output [batch, seq_len, 2304]")

    # Check what WrappedGPT expects
    print("\nWrappedGPT analysis:")
    print(f"  weight.shape[0] = {conv1d.weight.shape[0]} (rows)")
    print(f"  weight.shape[1] = {conv1d.weight.shape[1]} (columns)")
    print(f"  WrappedGPT.scaler_row shape = columns = {conv1d.weight.shape[1]}")

    # Simulate what happens in add_batch
    print("\nSimulating add_batch:")
    # Assuming inp is output from previous layer (input to this Conv1D)
    # Shape: [batch, seq_len, hidden] or [batch, hidden]
    inp = dummy_input  # [2, 10, 768]
    print(f"  inp shape: {inp.shape}")
    if len(inp.shape) == 2:
        inp = inp.unsqueeze(0)
        print(f"  unsqueezed: {inp.shape}")
    tmp = inp.shape[0]  # 2
    print(f"  tmp (batch): {tmp}")
    # Not isinstance(conv1d, nn.Linear) -> skip transpose/reshape
    inp = inp.type(torch.float32)
    print(f"  inp after float32: {inp.shape}")
    # torch.norm(inp, p=2, dim=1) on shape [2, 10, 768] -> dim=1 is seq_len dimension
    # Result shape: [2, 768]
    norm = torch.norm(inp, p=2, dim=1)
    print(f"  norm shape: {norm.shape}")
    print(f"  Expected scaler_row shape: {conv1d.weight.shape[1]} = 2304")
    print("  Mismatch!")

    # What if we treat Conv1D like Linear?
    print("\nIf we treat Conv1D like Linear:")
    inp_linear = inp.reshape((-1, inp.shape[-1]))  # [20, 768]
    inp_linear = inp_linear.t()  # [768, 20]
    print(f"  reshape -> {inp_linear.shape}")
    print(f"  transpose -> {inp_linear.shape}")
    norm_linear = torch.norm(inp_linear, p=2, dim=1)  # [768]
    print(f"  norm shape: {norm_linear.shape}")
    print("  Still mismatch with scaler_row (2304)!")

    # Actually, for Linear weight shape [out_features, in_features]
    # For Conv1D weight shape [in_features, out_features] (transposed)
    # So maybe we need to treat Conv1D differently.


if __name__ == "__main__":
    test_conv1d()
