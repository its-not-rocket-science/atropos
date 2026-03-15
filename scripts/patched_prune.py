#!/usr/bin/env python3
"""Patched pruning functions to handle multiple architectures.

This module provides architecture-aware wrappers around the original
Wanda and SparseGPT pruning functions from external/wanda/lib.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Add external/wanda to path
sys.path.insert(0, str(Path(__file__).parent.parent / "external" / "wanda"))

import torch
import torch.nn as nn
from lib.prune import (
    check_sparsity as original_check_sparsity,
)
from lib.prune import (
    prune_sparsegpt as original_prune_sparsegpt,
)

# Import original modules
from lib.prune import (
    prune_wanda as original_prune_wanda,
)
from lib.prune_opt import (
    check_sparsity as original_check_sparsity_opt,
)
from lib.prune_opt import (
    prepare_calibration_input as original_prepare_calibration_input_opt,
)
from lib.prune_opt import (
    prune_sparsegpt as original_prune_sparsegpt_opt,
)
from lib.prune_opt import (
    prune_wanda as original_prune_wanda_opt,
)


def patch_find_layers_for_gpt2():
    """Monkey-patch find_layers to include Conv1D layers for GPT2."""
    try:
        # Import find_layers from lib.prune
        import lib.prune as prune_module
        from transformers.pytorch_utils import Conv1D

        original_find_layers = prune_module.find_layers

        def patched_find_layers(module, layers=None, name=""):
            # Add Conv1D to layers list
            if layers is None:
                layers = [nn.Linear]
            layers = list(layers) + [Conv1D]
            return original_find_layers(module, layers, name)

        prune_module.find_layers = patched_find_layers
        return original_find_layers
    except ImportError:
        # Conv1D not available, keep original
        return None


def unpatch_find_layers(original_find_layers):
    """Restore original find_layers."""
    if original_find_layers is not None:
        import lib.prune as prune_module

        prune_module.find_layers = original_find_layers


def patch_wrapped_gpt_for_conv1d():
    """Monkey-patch WrappedGPT.__init__ and add_batch to handle Conv1D correctly."""
    try:
        import lib.layerwrapper as lw
        from transformers.pytorch_utils import Conv1D

        original_init = lw.WrappedGPT.__init__
        original_add_batch = lw.WrappedGPT.add_batch

        def patched_init(self, layer, layer_id=0, layer_name="none"):
            # For Conv1D, weight shape is (input_dim, output_dim)
            # Swap rows and columns to match Linear convention (output, input)
            if isinstance(layer, Conv1D):
                # layer.weight.shape = (input, output)
                # We want columns = input_dim, rows = output_dim
                # But original init expects weight.shape[1] = input
                # So we can pass a fake layer with swapped weight?
                # Instead we'll monkey-patch the layer's weight shape by swapping axes in init
                # Let's compute correct rows and columns
                input_dim, output_dim = layer.weight.shape
                # Set rows = output_dim, columns = input_dim
                self.layer = layer
                self.dev = layer.weight.device
                self.rows = output_dim
                self.columns = input_dim
                self.scaler_row = torch.zeros((self.columns), device=self.dev)
                self.nsamples = 0
                self.layer_id = layer_id
                self.layer_name = layer_name
            else:
                # Call original init
                original_init(self, layer, layer_id, layer_name)

        # Keep add_batch same as original (since columns now correct)
        # However we still need to treat Conv1D as Linear in add_batch
        def patched_add_batch(self, inp, out):
            if len(inp.shape) == 2:
                inp = inp.unsqueeze(0)
            tmp = inp.shape[0]
            # Treat Conv1D same as Linear (but Conv1D already handled in __init__)
            if isinstance(self.layer, (torch.nn.Linear, Conv1D)):
                if len(inp.shape) == 3:
                    inp = inp.reshape((-1, inp.shape[-1]))
                inp = inp.t()

            self.scaler_row *= self.nsamples / (self.nsamples + tmp)
            self.nsamples += tmp

            inp = inp.type(torch.float32)
            self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2 / self.nsamples

        lw.WrappedGPT.__init__ = patched_init
        lw.WrappedGPT.add_batch = patched_add_batch
        return (original_add_batch, original_init)
    except ImportError:
        # Conv1D not available, keep original
        return None


def unpatch_wrapped_gpt(original_pair):
    """Restore original add_batch and __init__."""
    if original_pair is None:
        return
    import lib.layerwrapper as lw

    if isinstance(original_pair, tuple):
        original_add_batch, original_init = original_pair
        lw.WrappedGPT.add_batch = original_add_batch
        lw.WrappedGPT.__init__ = original_init
    else:
        # backward compatibility
        lw.WrappedGPT.add_batch = original_pair


def get_model_architecture(model: torch.nn.Module) -> str:
    """Detect model architecture and return a string identifier."""
    model_type = type(model).__name__
    config_type = model.config.model_type if hasattr(model.config, "model_type") else ""

    # GPT2 family
    if model_type == "GPT2LMHeadModel" or config_type == "gpt2":
        return "gpt2"
    # OPT family
    elif model_type == "OPTForCausalLM" or config_type == "opt":
        return "opt"
    # LLaMA family (including LLaMA, Llama2, etc.)
    elif "Llama" in model_type or config_type == "llama":
        return "llama"
    # GPT-NeoX family (Pythia)
    elif model_type == "GPTNeoXForCausalLM" or config_type == "gpt_neox":
        return "gpt_neox"
    # Default to base (assume LLaMA-like)
    else:
        return "base"


def get_layer_container(model: torch.nn.Module, arch: str) -> torch.nn.Module:
    """Return the layer container for the given architecture."""
    if arch == "gpt2":
        # GPT2: model.transformer.h
        if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            return model.transformer.h
        else:
            raise AttributeError("GPT2 model missing transformer.h")
    elif arch == "opt":
        # OPT: model.model.decoder.layers
        if (
            hasattr(model, "model")
            and hasattr(model.model, "decoder")
            and hasattr(model.model.decoder, "layers")
        ):
            return model.model.decoder.layers
        else:
            raise AttributeError("OPT model missing model.model.decoder.layers")
    elif arch == "llama" or arch == "base":
        # LLaMA: model.model.layers
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return model.model.layers
        else:
            raise AttributeError(f"Model missing model.model.layers (arch={arch})")
    elif arch == "gpt_neox":
        # GPT-NeoX: model.gpt_neox.layers
        if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
            return model.gpt_neox.layers
        else:
            raise AttributeError("GPT-NeoX model missing gpt_neox.layers")
    else:
        raise ValueError(f"Unsupported architecture: {arch}")


def adapt_model_for_pruning(
    model: torch.nn.Module,
    arch: str | None = None,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Adapt a model for pruning by setting required attributes.

    Returns:
        (adapted_model, original_attrs) where original_attrs is a dict of
        original attribute values that were changed (for restoration).
    """
    if arch is None:
        arch = get_model_architecture(model)

    original = {}

    # Ensure seqlen is set
    if not hasattr(model, "seqlen"):
        if hasattr(model.config, "max_position_embeddings"):
            model.seqlen = model.config.max_position_embeddings
        elif hasattr(model.config, "n_positions"):
            model.seqlen = model.config.n_positions
        elif hasattr(model.config, "seq_length"):
            model.seqlen = model.config.seq_length
        else:
            model.seqlen = 2048  # default
        original["seqlen"] = None  # didn't exist

    # Ensure hf_device_map exists
    if not hasattr(model, "hf_device_map"):
        model.hf_device_map = {}
        original["hf_device_map"] = None

    # Create model.model.layers if needed (for base/llama pruning functions)
    # We'll monkey-patch model.model.layers to point to the actual layer container
    if arch != "opt":
        # For non-OPT architectures, base pruning functions expect model.model.layers
        if not hasattr(model, "model"):
            model.model = type("obj", (object,), {})()
            original["model"] = None
        elif "model" not in original:
            original["model"] = model.model

        layers = get_layer_container(model, arch)
        original_layers = getattr(model.model, "layers", None)
        if original_layers is not None and "model.layers" not in original:
            original["model.layers"] = original_layers
        model.model.layers = layers

    # For OPT architecture, we'll use the OPT-specific pruning functions
    # They expect model.model.decoder.layers (already exists)
    # Nothing to do.

    return model, original


def restore_model_attrs(model: torch.nn.Module, original_attrs: dict[str, Any]) -> None:
    """Restore original model attributes after pruning."""
    for attr, value in original_attrs.items():
        if value is None:
            # Attribute was added, delete it
            if attr == "seqlen":
                delattr(model, "seqlen")
            elif attr == "hf_device_map":
                delattr(model, "hf_device_map")
            elif attr == "model":
                delattr(model, "model")
            elif attr == "model.layers":
                delattr(model.model, "layers")
        else:
            # Attribute was replaced, restore it
            if attr == "model":
                model.model = value
            elif attr == "model.layers":
                model.model.layers = value
            else:
                setattr(model, attr, value)


def prepare_calibration_input_patched(
    model: torch.nn.Module,
    dataloader: Any,
    device: torch.device,
    arch: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Patched version of prepare_calibration_input that handles missing position_ids.

    Returns (inps, outs, attention_mask, position_ids) where position_ids may be None.
    """
    if arch is None:
        arch = get_model_architecture(model)

    # Use OPT version for OPT models (doesn't use position_ids)
    if arch == "opt":
        return original_prepare_calibration_input_opt(model, dataloader, device)

    # For other architectures, we need to handle missing position_ids
    # We'll monkey-patch the Catcher class inside the original function
    # Instead, we'll implement a simplified version that works for GPT2 and LLaMA
    # Copying the original but making position_ids optional.

    use_cache = model.config.use_cache
    model.config.use_cache = False

    layers = model.model.layers  # Already patched by adapt_model_for_pruning

    if "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((128, model.seqlen, model.config.hidden_size), dtype=dtype, device=device)
    inps.requires_grad = False
    cache = {"i": 0, "attention_mask": None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            # position_ids may not be present
            if "position_ids" in kwargs:
                cache["position_ids"] = kwargs["position_ids"]
            raise ValueError

    # Temporarily replace first layer with Catcher
    original_first_layer = layers[0]
    layers[0] = Catcher(layers[0])

    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass

    # Restore first layer
    layers[0] = original_first_layer

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]

    if position_ids is None and arch == "gpt_neox":
        # generate position_ids for rotary embeddings
        batch_size = inps.shape[0]
        position_ids = torch.arange(model.seqlen, device=inps.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

    model.config.use_cache = use_cache

    return inps, outs, attention_mask, position_ids


def wrap_layer_forwards(layers):
    """Wrap forward methods of layers to ignore unexpected keyword arguments."""
    original_forwards = []
    for _i, layer in enumerate(layers):
        original_forward = layer.forward
        # Create a closure that captures original_forward per layer
        layer.forward = lambda *args, orig=original_forward, **kwargs: (
            kwargs.pop("position_ids", None),
            orig(*args, **kwargs),
        )[1]
        original_forwards.append(original_forward)
    return original_forwards


def unwrap_layer_forwards(layers, original_forwards):
    """Restore original forward methods."""
    for layer, original in zip(layers, original_forwards, strict=True):
        layer.forward = original


def wrap_gpt_neox_layer_forwards(layers):
    """Wrap forward methods of GPT-NeoX layers to compute position_embeddings if missing."""
    original_forwards = []
    for layer in layers:
        original_forward = layer.forward
        # Check if attention has rotary_emb
        if hasattr(layer.attention, "rotary_emb"):
            rotary_emb = layer.attention.rotary_emb
        else:
            # fallback: maybe model has rotary_emb, but we can't access easily
            # skip wrapping
            original_forwards.append(original_forward)
            continue

        # Capture rotary_emb in closure
        def make_wrapped(orig, rot):
            def wrapped(*args, **kwargs):
                if (
                    kwargs.get("position_embeddings") is None
                    and kwargs.get("position_ids") is not None
                ):
                    position_ids = kwargs["position_ids"]
                    # rotary_emb.forward expects x (dummy) and position_ids
                    dummy = torch.zeros(1, dtype=rot.inv_freq.dtype, device=position_ids.device)
                    cos, sin = rot.forward(dummy, position_ids)
                    kwargs["position_embeddings"] = (cos, sin)
                return orig(*args, **kwargs)

            return wrapped

        layer.forward = make_wrapped(original_forward, rotary_emb)
        original_forwards.append(original_forward)
    return original_forwards


def unwrap_gpt_neox_layer_forwards(layers, original_forwards):
    """Restore original forward methods."""
    for layer, original in zip(layers, original_forwards, strict=True):
        layer.forward = original


def prune_wanda_patched(
    args: Any,
    model: torch.nn.Module,
    tokenizer: Any,
    device: torch.device | None = None,
    prune_n: int = 0,
    prune_m: int = 0,
) -> None:
    """Architecture-aware wrapper for prune_wanda."""
    if device is None:
        device = torch.device("cuda:0")
    arch = get_model_architecture(model)

    # Adapt model attributes
    model, original = adapt_model_for_pruning(model, arch)

    try:
        if arch == "opt":
            return original_prune_wanda_opt(args, model, tokenizer, device, prune_n, prune_m)
        else:
            # For non-OPT, we need to use patched prepare_calibration_input
            # We'll temporarily replace the function in the module
            import lib.prune as prune_module

            original_prepare = prune_module.prepare_calibration_input
            prune_module.prepare_calibration_input = lambda m, d, dev: (
                prepare_calibration_input_patched(m, d, dev, arch)
            )
            layers = model.model.layers
            # Additionally, for GPT2 we need to wrap layer forwards to ignore position_ids
            # and patch find_layers to include Conv1D
            if arch == "gpt2":
                original_forwards = wrap_layer_forwards(layers)
                original_find_layers = patch_find_layers_for_gpt2()
                original_wrapped_pair = patch_wrapped_gpt_for_conv1d()
            elif arch == "gpt_neox":
                original_forwards = wrap_gpt_neox_layer_forwards(layers)
                original_find_layers = None
                original_wrapped_pair = None
            else:
                original_forwards = None
                original_find_layers = None
                original_wrapped_pair = None
            try:
                if arch == "gpt2":
                    return prune_wanda_gpt2(args, model, tokenizer, device, prune_n, prune_m)
                else:
                    return original_prune_wanda(args, model, tokenizer, device, prune_n, prune_m)
            finally:
                prune_module.prepare_calibration_input = original_prepare
                if arch == "gpt2":
                    unwrap_layer_forwards(layers, original_forwards)
                elif arch == "gpt_neox":
                    unwrap_gpt_neox_layer_forwards(layers, original_forwards)
                if original_find_layers is not None:
                    unpatch_find_layers(original_find_layers)
                if original_wrapped_pair is not None:
                    unpatch_wrapped_gpt(original_wrapped_pair)
    finally:
        restore_model_attrs(model, original)


def prune_sparsegpt_patched(
    args: Any,
    model: torch.nn.Module,
    tokenizer: Any,
    device: torch.device | None = None,
    prune_n: int = 0,
    prune_m: int = 0,
) -> None:
    """Architecture-aware wrapper for prune_sparsegpt."""
    if device is None:
        device = torch.device("cuda:0")
    arch = get_model_architecture(model)

    model, original = adapt_model_for_pruning(model, arch)
    if arch != "opt":
        layers = model.model.layers

    try:
        if arch == "opt":
            return original_prune_sparsegpt_opt(args, model, tokenizer, device, prune_n, prune_m)
        elif arch == "gpt2":
            # GPT2 requires custom SparseGPT pruning due to Conv1D layers and missing position_ids
            original_forwards = wrap_layer_forwards(layers)
            original_find_layers = patch_find_layers_for_gpt2()
            try:
                return prune_sparsegpt_gpt2(args, model, tokenizer, device, prune_n, prune_m)
            finally:
                unwrap_layer_forwards(layers, original_forwards)
                unpatch_find_layers(original_find_layers)
        elif arch == "gpt_neox":
            # GPT-NeoX needs position_embeddings computed
            original_forwards = wrap_gpt_neox_layer_forwards(layers)
            import lib.prune as prune_module

            original_prepare = prune_module.prepare_calibration_input
            prune_module.prepare_calibration_input = lambda m, d, dev: (
                prepare_calibration_input_patched(m, d, dev, arch)
            )
            try:
                return original_prune_sparsegpt(args, model, tokenizer, device, prune_n, prune_m)
            finally:
                prune_module.prepare_calibration_input = original_prepare
                unwrap_gpt_neox_layer_forwards(layers, original_forwards)
        else:
            # For other architectures (LLaMA), SparseGPT uses "
            "prepare_calibration_input internally"
            import lib.prune as prune_module

            original_prepare = prune_module.prepare_calibration_input
            prune_module.prepare_calibration_input = lambda m, d, dev: (
                prepare_calibration_input_patched(m, d, dev, arch)
            )
            try:
                return original_prune_sparsegpt(args, model, tokenizer, device, prune_n, prune_m)
            finally:
                prune_module.prepare_calibration_input = original_prepare
    finally:
        restore_model_attrs(model, original)


def check_sparsity_patched(model: torch.nn.Module) -> float:
    """Architecture-aware wrapper for check_sparsity."""
    arch = get_model_architecture(model)

    model, original = adapt_model_for_pruning(model, arch)

    try:
        if arch == "opt":
            return original_check_sparsity_opt(model)
        else:
            # For GPT2, patch find_layers to include Conv1D
            if arch == "gpt2":
                original_find_layers = patch_find_layers_for_gpt2()
            else:
                original_find_layers = None
            try:
                return original_check_sparsity(model)
            finally:
                if original_find_layers is not None:
                    unpatch_find_layers(original_find_layers)
    finally:
        restore_model_attrs(model, original)


def prune_wanda_gpt2(args, model, tokenizer, device=None, prune_n=0, prune_m=0):
    """Wanda pruning for GPT2 with Conv1D support."""
    print("[INFO] Using custom GPT2 pruning with Conv1D support")
    # Import inside function to avoid circular imports
    import torch
    from lib.data import get_loaders
    from lib.layerwrapper import WrappedGPT
    from lib.prune import find_layers, prepare_calibration_input
    from transformers.pytorch_utils import Conv1D

    if device is None:
        device = torch.device("cuda:0")

    use_cache = model.config.use_cache
    model.config.use_cache = False

    print("loading calibration data")
    dataloader, _ = get_loaders(
        "wikitext2",
        nsamples=args.nsamples,
        seed=args.seed,
        seqlen=model.seqlen,
        tokenizer=tokenizer,
    )
    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(
            model, dataloader, device
        )

    layers = model.model.layers
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = (
                inps.to(dev),
                outs.to(dev),
                attention_mask.to(dev),
                position_ids.to(dev),
            )

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out, wl=wrapped_layers, n=name):  # noqa: B023
                wl[n].add_batch(inp[0].data, out.data)

            return tmp  # noqa: B023

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(
                    inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids
                )[0]
        for h in handles:
            h.remove()

        for name in subset:
            print(f"pruning layer {i} name {name}")
            layer_module = subset[name]
            weight = layer_module.weight.data
            is_conv1d = isinstance(layer_module, Conv1D)

            # For Conv1D, transpose weight to (output, input) for metric calculation
            if is_conv1d:
                weight_t = weight.t()  # (output, input)
            else:
                weight_t = weight  # (output, input) already for Linear

            # scaler_row corresponds to input dimension (columns in WrappedGPT)
            scaler = wrapped_layers[name].scaler_row
            w_metric = torch.abs(weight_t) * torch.sqrt(scaler.reshape((1, -1)))

            w_mask = torch.zeros_like(w_metric) == 1
            if prune_n != 0:
                # structured n:m sparsity
                for ii in range(w_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = w_metric[:, ii : (ii + prune_m)].float()
                        w_mask.scatter_(
                            1, ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True
                        )
            else:
                sort_res = torch.sort(w_metric, dim=-1, stable=True)

                if args.use_variant:
                    # wanda variant
                    tmp_metric = torch.cumsum(sort_res[0], dim=1)
                    sum_before = w_metric.sum(dim=1)

                    alpha = 0.4
                    alpha_hist = [0.0, 0.8]
                    from lib.prune import return_given_alpha

                    w_mask, cur_sparsity = return_given_alpha(
                        alpha, sort_res, w_metric, tmp_metric, sum_before
                    )
                    while (torch.abs(cur_sparsity - args.sparsity_ratio) > 0.001) and (
                        alpha_hist[1] - alpha_hist[0] >= 0.001
                    ):
                        if cur_sparsity > args.sparsity_ratio:
                            alpha_new = (alpha + alpha_hist[0]) / 2.0
                            alpha_hist[1] = alpha
                        else:
                            alpha_new = (alpha + alpha_hist[1]) / 2.0
                            alpha_hist[0] = alpha

                        alpha = alpha_new
                        w_mask, cur_sparsity = return_given_alpha(
                            alpha, sort_res, w_metric, tmp_metric, sum_before
                        )
                    print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                else:
                    # unstructured pruning
                    indices = sort_res[1][:, : int(w_metric.shape[1] * args.sparsity_ratio)]
                    w_mask.scatter_(1, indices, True)

            # If Conv1D, transpose mask back to (input, output)
            if is_conv1d:
                w_mask = w_mask.t()  # (input, output)

            # Apply mask to original weight
            weight[w_mask] = 0

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(
                    inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids
                )[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def prune_sparsegpt_gpt2(args, model, tokenizer, device=None, prune_n=0, prune_m=0):
    """SparseGPT pruning for GPT2 with Conv1D support and position_ids handling."""
    print("[INFO] Using custom GPT2 SparseGPT pruning")
    # Import inside function to avoid circular imports
    import torch

    if device is None:
        device = torch.device("cuda:0")
    from lib.data import get_loaders
    from lib.prune import find_layers
    from lib.sparsegpt import SparseGPT

    # Monkey-patch SparseGPT.fasterprune to guard CUDA calls
    original_fasterprune = SparseGPT.fasterprune

    def patched_fasterprune(self, sparsity, prune_n=0, prune_m=0, blocksize=128, percdamp=0.01):
        # Temporarily replace torch.cuda.synchronize and torch.cuda.empty_cache with safe versions
        import torch.cuda

        original_sync = torch.cuda.synchronize
        original_empty = torch.cuda.empty_cache

        def safe_sync():
            if torch.cuda.is_available():
                original_sync()

        def safe_empty():
            if torch.cuda.is_available():
                original_empty()

        try:
            torch.cuda.synchronize = safe_sync
            torch.cuda.empty_cache = safe_empty
            return original_fasterprune(self, sparsity, prune_n, prune_m, blocksize, percdamp)
        finally:
            torch.cuda.synchronize = original_sync
            torch.cuda.empty_cache = original_empty

    SparseGPT.fasterprune = patched_fasterprune
    patched = True

    use_cache = model.config.use_cache
    model.config.use_cache = False

    print("loading calibration data")
    dataloader, _ = get_loaders(
        "wikitext2",
        nsamples=args.nsamples,
        seed=args.seed,
        seqlen=model.seqlen,
        tokenizer=tokenizer,
    )
    print("dataset loading complete")

    layers = model.model.layers
    if "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=device
    )
    cache = {"i": 0, "attention_mask": None, "position_ids": None}

    # Modified Catcher that handles missing position_ids
    import torch.nn as nn

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            # position_ids may not be present
            if "position_ids" in kwargs:
                cache["position_ids"] = kwargs["position_ids"]
            else:
                cache["position_ids"] = None
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass
    layers[0] = layers[0].module
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]

    print("Ready.")

    for i in range(len(layers)):
        layer = layers[i]
        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            print(f"layer {i} device {dev}")
            inps, outs, attention_mask, position_ids = (
                inps.to(dev),
                outs.to(dev),
                attention_mask.to(dev),
                position_ids.to(dev),
            )

        subset = find_layers(layer)

        gpts = {}
        for name in subset:
            gpts[name] = SparseGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out, g=gpts, n=name):  # noqa: B023
                g[n].add_batch(inp[0].data, out.data)

            return tmp  # noqa: B023

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            # Forward with position_ids only if not None (layer forward wrapper will strip it)
            if position_ids is not None:
                outs[j] = layer(
                    inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids
                )[0]
            else:
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print("Pruning ...")
            gpts[name].fasterprune(
                args.sparsity_ratio, prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128
            )
            gpts[name].free()

        for j in range(args.nsamples):
            if position_ids is not None:
                outs[j] = layer(
                    inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids
                )[0]
            else:
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

        layers[i] = layer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        inps, outs = outs, inps

    # Restore original fasterprune
    if patched:
        SparseGPT.fasterprune = original_fasterprune

    model.config.use_cache = use_cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
