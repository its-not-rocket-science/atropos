#!/usr/bin/env python3
"""Debug division by zero in Wanda pruning for GPT2."""

import argparse
import sys
import traceback
from pathlib import Path

# Add external/wanda to path
sys.path.insert(0, str(Path(__file__).parent / "external" / "wanda"))

import torch

# Import patched pruning functions
sys.path.insert(0, str(Path(__file__).parent / "scripts"))
try:
    from patched_prune import (
        adapt_model_for_pruning,
        get_model_architecture,
        prune_wanda_patched,
    )
except ImportError as e:
    print(f"[FAIL] Cannot import patched_prune: {e}")
    sys.exit(1)

from transformers import AutoModelForCausalLM, AutoTokenizer


def debug_gpt2():
    print("Loading GPT2 model...")
    model_path = Path(
        "test_data/models/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e"
    )
    if not model_path.exists():
        print(f"Model path not found: {model_path}")
        return

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    # Set required attributes
    model.seqlen = model.config.max_position_embeddings
    if not hasattr(model, "hf_device_map"):
        model.hf_device_map = {}

    # Create args namespace
    args = argparse.Namespace(
        model=str(model_path),
        seed=0,
        nsamples=2,  # minimal calibration samples
        sparsity_ratio=0.1,
        sparsity_type="unstructured",
        use_variant=False,
        save="test_output",
        save_model="test_output",
        cache_dir=str(model_path),
    )

    device = torch.device("cpu")

    # Patch find_layers to include Conv1D for GPT2
    try:
        import lib.prune as prune_module
        from transformers.pytorch_utils import Conv1D

        original_find_layers = prune_module.find_layers

        def patched_find_layers(module, layers=None, name=""):
            if layers is None:
                layers = [torch.nn.Linear]
            layers = list(layers) + [Conv1D]
            return original_find_layers(module, layers, name)

        prune_module.find_layers = patched_find_layers
        print("Patched find_layers to include Conv1D")
    except Exception as e:
        print(f"Warning: Could not patch find_layers: {e}")

    # Wrap layer forwards to ignore position_ids
    from patched_prune import unwrap_layer_forwards, wrap_layer_forwards

    arch = get_model_architecture(model)
    model, original_attrs = adapt_model_for_pruning(model, arch)
    layers = model.model.layers
    original_forwards = wrap_layer_forwards(layers)
    print(f"Wrapped {len(original_forwards)} layer forwards")

    # Try to run prepare_calibration_input patched to see scaler_row values
    print("\nAttempting to run calibration...")
    try:
        # We'll manually call prepare_calibration_input_patched and inspect scaler_row
        import lib.prune as prune_module
        from patched_prune import prepare_calibration_input_patched

        # Temporarily replace prepare_calibration_input
        original_prepare = prune_module.prepare_calibration_input

        def patched_prepare(m, d, dev):
            return prepare_calibration_input_patched(m, d, dev, arch)

        prune_module.prepare_calibration_input = patched_prepare

        # Get dataloader
        from lib.data import get_loaders

        dataloader, _ = get_loaders(
            "wikitext2",
            nsamples=args.nsamples,
            seed=args.seed,
            seqlen=model.seqlen,
            tokenizer=tokenizer,
        )

        # Run calibration and capture intermediate values
        # We'll monkey-patch WrappedGPT.add_batch to print scaler_row
        import lib.layerwrapper as lw

        original_add_batch = lw.WrappedGPT.add_batch

        def debug_add_batch(self, inp, out):
            print(f"  add_batch: nsamples={self.nsamples}, inp shape={inp.shape}")
            result = original_add_batch(self, inp, out)
            print(
                f"    scaler_row min={self.scaler_row.min().item():.6f}, "
                f"max={self.scaler_row.max().item():.6f}, "
                f"mean={self.scaler_row.mean().item():.6f}"
            )
            return result

        lw.WrappedGPT.add_batch = debug_add_batch

        # Run prepare_calibration_input
        inps, outs, attention_mask, position_ids = prune_module.prepare_calibration_input(
            model, dataloader, device
        )
        print(f"Calibration completed: inps shape {inps.shape}")

        # Restore
        lw.WrappedGPT.add_batch = original_add_batch
        prune_module.prepare_calibration_input = original_prepare

    except Exception as e:
        print(f"Error during calibration: {e}")
        traceback.print_exc()
        # Restore patches
        try:
            prune_module.prepare_calibration_input = original_prepare
        except Exception:
            pass

    # Now attempt pruning
    print("\nAttempting prune_wanda_patched...")
    try:
        prune_wanda_patched(args, model, tokenizer, device, prune_n=0, prune_m=0)
        print("Pruning succeeded!")
    except ZeroDivisionError as e:
        print(f"ZeroDivisionError caught: {e}")
        traceback.print_exc()
        # Additional debugging: maybe scaler_row is zero?
        # Let's inspect wrapped_layers if possible
    except Exception as e:
        print(f"Other error: {e}")
        traceback.print_exc()
    finally:
        # Restore
        unwrap_layer_forwards(layers, original_forwards)
        from patched_prune import restore_model_attrs

        restore_model_attrs(model, original_attrs)
        if "original_find_layers" in locals():
            prune_module.find_layers = original_find_layers


if __name__ == "__main__":
    debug_gpt2()
