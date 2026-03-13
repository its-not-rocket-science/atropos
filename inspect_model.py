#!/usr/bin/env python3
"""Inspect model architecture."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "external" / "wanda"))

import torch
from transformers import AutoModelForCausalLM


def inspect(model_name: str):
    print(f"\n=== {model_name} ===")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        print(f"Model type: {type(model).__name__}")
        print(f"Config model_type: {model.config.model_type}")
        # Check layers
        if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            print(f"GPT2-style layers: model.transformer.h length {len(model.transformer.h)}")
        if hasattr(model, "model"):
            print("Has model attribute")
            if hasattr(model.model, "layers"):
                print(f"  model.model.layers length {len(model.model.layers)}")
            if hasattr(model.model, "decoder"):
                print("  model.model.decoder exists")
                if hasattr(model.model.decoder, "layers"):
                    print(f"    decoder.layers length {len(model.model.decoder.layers)}")
        # Check seqlen-related config
        if hasattr(model.config, "max_position_embeddings"):
            print(f"max_position_embeddings: {model.config.max_position_embeddings}")
        if hasattr(model.config, "n_positions"):
            print(f"n_positions: {model.config.n_positions}")
        if hasattr(model.config, "seq_length"):
            print(f"seq_length: {model.config.seq_length}")
        # Check forward signature
        import inspect

        sig = inspect.signature(model.forward)
        params = list(sig.parameters.keys())
        print(f"Forward params: {params}")
        # Cleanup
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    models = ["gpt2", "facebook/opt-1.3b", "EleutherAI/pythia-2.8b"]
    for m in models:
        inspect(m)
