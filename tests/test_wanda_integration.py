"""Integration test for Wanda pruning with GPT2."""

import os
import sys
from pathlib import Path

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add external/wanda and scripts to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "external" / "wanda"))
sys.path.insert(0, str(project_root / "scripts"))


@pytest.mark.integration
@pytest.mark.skipif(os.getenv("CI") == "true", reason="Requires external dependencies")
def test_wanda_pruning_gpt2(tmp_path: Path):
    """Test Wanda pruning with GPT2 model."""
    from patched_prune import prune_wanda_patched

    model_path = (
        project_root
        / "test_data/models/models--gpt2/snapshots"
        / "607a30d783dfa663caf39e06633721c8d4cfcd7e"
    )

    if not model_path.exists():
        pytest.skip(f"Model path not found: {model_path}")

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

    # Create args
    import argparse

    args = argparse.Namespace(
        model=str(model_path),
        seed=0,
        nsamples=2,
        sparsity_ratio=0.3,
        sparsity_type="unstructured",
        use_variant=False,
        save=str(tmp_path / "test_output"),
        save_model=str(tmp_path / "test_output"),
        cache_dir=str(model_path),
    )

    device = torch.device("cpu")

    # Should not raise exception
    prune_wanda_patched(args, model, tokenizer, device, prune_n=0, prune_m=0)
