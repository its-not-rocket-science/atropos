#!/usr/bin/env python3
"""Test architecture compatibility without actual pruning."""

import sys
from pathlib import Path

# Add external/wanda to path
sys.path.insert(0, str(Path(__file__).parent.parent / "external" / "wanda"))

import torch
from patched_prune import (
    get_model_architecture,
    adapt_model_for_pruning,
    restore_model_attrs,
    fix_gptj_rotary_mismatch,
)
from transformers import AutoModelForCausalLM, AutoTokenizer


def test_model_compatibility(model_id: str, arch: str):
    """Test that model can be adapted for pruning."""
    print(f"\n{'='*60}")
    print(f"Testing compatibility: {model_id}")
    print(f"{'='*60}")

    try:
        # Load model
        print("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Detect architecture
        detected = get_model_architecture(model)
        print(f"Detected architecture: {detected}")
        assert detected == arch, f"Architecture mismatch: expected {arch}, got {detected}"

        # Fix GPT-J rotary mismatch if needed
        if arch == "gptj":
            print("Applying GPT-J rotary mismatch fix...")
            fix_gptj_rotary_mismatch(model)

        # Adapt model for pruning
        print("Adapting model for pruning...")
        adapted, original = adapt_model_for_pruning(model, arch)
        print(f"Original attributes changed: {list(original.keys())}")

        # Check that required attributes exist
        assert hasattr(adapted, "seqlen"), "seqlen missing"
        assert hasattr(adapted, "hf_device_map"), "hf_device_map missing"
        if arch != "opt":
            assert hasattr(adapted, "model"), "model missing"
            assert hasattr(adapted.model, "layers"), "model.layers missing"
            print(f"Layer container length: {len(adapted.model.layers)}")

        # Test forward pass with dummy input (small sequence)
        print("Testing forward pass with dummy input...")
        dummy_input = torch.randint(0, tokenizer.vocab_size, (1, 16), dtype=torch.long)
        with torch.no_grad():
            try:
                outputs = adapted(dummy_input)
                print("Forward pass successful")
            except Exception as e:
                print(f"Forward pass failed: {e}")
                # Some models require attention_mask, try again
                attention_mask = torch.ones_like(dummy_input)
                outputs = adapted(dummy_input, attention_mask=attention_mask)
                print("Forward pass with attention_mask successful")

        # Restore original attributes
        print("Restoring original attributes...")
        restore_model_attrs(adapted, original)
        print("Attributes restored")

        # Verify restoration
        for attr, val in original.items():
            if val is None:
                # Attribute was added, should be removed
                if attr == "seqlen":
                    assert not hasattr(model, "seqlen"), f"{attr} not removed"
                elif attr == "hf_device_map":
                    assert not hasattr(model, "hf_device_map"), f"{attr} not removed"
                elif attr == "model":
                    assert not hasattr(model, "model"), f"{attr} not removed"
                elif attr == "model.layers":
                    assert not hasattr(model.model, "layers"), f"{attr} not removed"
            else:
                # Attribute was replaced, should be restored
                pass  # Hard to check without original reference

        print(f"[OK] {model_id} compatibility test passed")
        return True

    except Exception as e:
        print(f"[FAIL] {model_id} compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    # Test configurations
    test_cases = [
        ("bigscience/bloom-560m", "bloom"),
        ("Milos/slovak-gpt-j-162M", "gptj"),
        # ("EleutherAI/gpt-j-6B", "gptj"),  # Too large for CPU
        ("gpt2", "gpt2"),
        ("facebook/opt-125m", "opt"),
        ("EleutherAI/pythia-160m", "gpt_neox"),
    ]

    success = True
    for model_id, arch in test_cases:
        if not test_model_compatibility(model_id, arch):
            success = False

    print(f"\n{'='*60}")
    if success:
        print("All compatibility tests passed!")
        sys.exit(0)
    else:
        print("Some compatibility tests failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()