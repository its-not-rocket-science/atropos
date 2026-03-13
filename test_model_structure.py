#!/usr/bin/env python3
"""Test model structure to understand architecture differences."""

import torch
from transformers import AutoModelForCausalLM


def test_model_structure(model_name, model_path):
    print(f"\n=== Testing {model_name} ===")

    try:
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        model.eval()

        print(f"Model type: {type(model).__name__}")
        print(f"Model config: {model.config.model_type}")

        # Check for common attributes
        attrs_to_check = [
            "model",
            "transformer",
            "decoder",
            "layers",
            "h",
            "encoder",
            "model.layers",
            "transformer.h",
            "decoder.layers",
        ]

        for attr in attrs_to_check:
            try:
                parts = attr.split(".")
                obj = model
                for part in parts:
                    obj = getattr(obj, part)
                print(f"  Found: {attr} (type: {type(obj).__name__})")
                if hasattr(obj, "__len__"):
                    print(f"    Length: {len(obj)}")
            except AttributeError:
                pass

        # Check seqlen attributes
        seqlen_attrs = ["max_position_embeddings", "n_positions", "seq_len"]
        for attr in seqlen_attrs:
            if hasattr(model.config, attr):
                print(f"  Config.{attr}: {getattr(model.config, attr)}")

        # Check hf_device_map
        if hasattr(model, "hf_device_map"):
            print(f"  hf_device_map: {model.hf_device_map}")
        else:
            print("  hf_device_map: Not set")

    except Exception as e:
        print(f"  Error: {e}")


if __name__ == "__main__":
    from pathlib import Path

    test_data_dir = Path("test_data")

    # Test GPT2
    gpt2_path = (
        test_data_dir
        / "models"
        / "models--gpt2"
        / "snapshots"
        / "607a30d783dfa663caf39e06633721c8d4cfcd7e"
    )
    if gpt2_path.exists():
        test_model_structure("gpt2", str(gpt2_path))

    # Test OPT
    opt_path = (
        test_data_dir
        / "models"
        / "models--facebook--opt-1.3b"
        / "snapshots"
        / "3f5c25d0bc631cb57ac65913f76e22c2dfb61d62"
    )
    if opt_path.exists():
        test_model_structure("facebook/opt-1.3b", str(opt_path))
