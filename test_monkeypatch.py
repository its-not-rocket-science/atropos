#!/usr/bin/env python3
"""Test monkey-patching approach for Wanda with GPT2."""

import sys
from pathlib import Path

from lib.data import get_loaders

# Add external/wanda to path
sys.path.insert(0, str(Path(__file__).parent / "external" / "wanda"))

import torch
from lib.prune import prepare_calibration_input as original_prepare_calibration_input
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_model_layers(model):
    """Get layer container based on architecture."""
    model_type = type(model).__name__
    config_type = model.config.model_type

    print(f"Model type: {model_type}, Config type: {config_type}")

    # GPT2
    if model_type == "GPT2LMHeadModel" or config_type == "gpt2":
        if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            return model.transformer.h
        else:
            # Try to find layers
            for name, _module in model.named_modules():
                if "h." in name or ".h" in name:
                    # This is a hack - need to get the actual container
                    pass

    # OPT
    elif model_type == "OPTForCausalLM" or config_type == "opt":
        if (
            hasattr(model, "model")
            and hasattr(model.model, "decoder")
            and hasattr(model.model.decoder, "layers")
        ):
            return model.model.decoder.layers

    # LLaMA (default)
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers

    # Try to find any layer container
    for attr in ["layers", "h", "blocks", "encoder", "decoder"]:
        if hasattr(model, attr):
            container = getattr(model, attr)
            if hasattr(container, "__len__") and len(container) > 0:
                return container
        # Also check model.attr
        if hasattr(model, "model") and hasattr(model.model, attr):
            container = getattr(model.model, attr)
            if hasattr(container, "__len__") and len(container) > 0:
                return container

    raise ValueError(f"Cannot find layer container for model type {model_type}")


def patched_prepare_calibration_input(model, dataloader, device):
    """Patched version that works with different architectures."""

    # Get the actual layer container
    layers = get_model_layers(model)

    # Monkey-patch model.model.layers temporarily
    if not hasattr(model, "model"):
        model.model = type("obj", (object,), {})()
    original_layers = getattr(model.model, "layers", None)
    model.model.layers = layers

    try:
        # Call original function
        result = original_prepare_calibration_input(model, dataloader, device)
        return result
    finally:
        # Restore original
        if original_layers is None:
            delattr(model.model, "layers")
        else:
            model.model.layers = original_layers


def test_gpt2_with_patch():
    print("Testing GPT2 with monkey-patched Wanda...")

    model_path = Path(
        "test_data/models/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e"
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.eval()

    # Set required attributes
    model.seqlen = model.config.max_position_embeddings
    if not hasattr(model, "hf_device_map"):
        model.hf_device_map = {}

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token


    device = torch.device("cpu")

    try:
        print("Testing patched prepare_calibration_input...")

        # First test getting layers

        layers = get_model_layers(model)
        print(f"Found layers: {type(layers)}, length: {len(layers)}")

        # Test patched function
        print("Testing calibration...")
        dataloader, _ = get_loaders(
            "wikitext2", nsamples=2, seed=0, seqlen=model.seqlen, tokenizer=tokenizer
        )

        result = patched_prepare_calibration_input(model, dataloader, device)
        print(f"Calibration succeeded: {len(result)} results")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_gpt2_with_patch()
