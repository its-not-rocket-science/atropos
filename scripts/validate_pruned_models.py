"""Validate pruned models maintain performance vs original models.

This script compares original and pruned models side-by-side to verify
that pruning hasn't degraded performance beyond acceptable tolerance.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Add src to path for importing atropos
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError as e:
    print(f"[FAIL] Required dependency not found: {e}")
    print("Install with: pip install torch transformers")
    sys.exit(1)


# Test prompts for generation comparison
TEST_PROMPTS = [
    "def factorial(n):",
    "def fibonacci(n):",
    "class Calculator:",
    "import os",
    "def quicksort(arr):",
]

# Validation thresholds
PERPLEXITY_TOLERANCE = 0.20  # 20% increase allowed
GENERATION_SIMILARITY_THRESHOLD = 0.70  # 70% token overlap expected
MAX_ACCEPTABLE_PERPLEXITY = 50.0  # Hard cap for very small models


def calculate_perplexity(model, tokenizer, text: str, device: str) -> float:
    """Calculate perplexity of model on given text."""
    encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    input_ids = encodings.input_ids.to(device)

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        perplexity = torch.exp(loss).item()

    return perplexity


def generate_text(model, tokenizer, prompt: str, device: str, max_length: int = 50) -> str:
    """Generate text from model given a prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            do_sample=False,  # Deterministic for comparison
            pad_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def token_similarity(text1: str, text2: str) -> float:
    """Calculate token-level similarity between two texts."""
    tokens1 = set(text1.lower().split())
    tokens2 = set(text2.lower().split())

    if not tokens1 and not tokens2:
        return 1.0
    if not tokens1 or not tokens2:
        return 0.0

    intersection = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)

    return intersection / union if union > 0 else 0.0


def validate_model(
    model_name: str,
    strategy: str,
    test_data_dir: Path,
    device: str,
) -> dict[str, Any]:
    """Validate a single pruned model against its original."""
    print(f"\n[VALIDATING] {model_name} - {strategy}")

    original_path = test_data_dir / "models" / model_name.replace("/", "--")
    pruned_path = test_data_dir / "pruned_models" / f"{model_name.replace('/', '--')}_{strategy}"

    result = {
        "model": model_name,
        "strategy": strategy,
        "original_path": str(original_path),
        "pruned_path": str(pruned_path),
        "passed": False,
        "metrics": {},
        "errors": [],
    }

    # Check if both models exist
    if not original_path.exists():
        result["errors"].append(f"Original model not found: {original_path}")
        print("  [FAIL] Original model not found")
        return result

    if not pruned_path.exists():
        result["errors"].append(f"Pruned model not found: {pruned_path}")
        print("  [FAIL] Pruned model not found")
        return result

    try:
        # Load tokenizers (use original tokenizer for both)
        print("  Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(original_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load models
        print("  Loading original model...")
        original_model = AutoModelForCausalLM.from_pretrained(original_path)
        original_model.to(device)
        original_model.eval()

        print("  Loading pruned model...")
        pruned_model = AutoModelForCausalLM.from_pretrained(pruned_path)
        pruned_model.to(device)
        pruned_model.eval()

        # Test 1: Perplexity comparison
        print("  Testing perplexity...")
        test_text = " ".join(TEST_PROMPTS)

        original_ppl = calculate_perplexity(original_model, tokenizer, test_text, device)
        pruned_ppl = calculate_perplexity(pruned_model, tokenizer, test_text, device)

        if original_ppl > 0:
            ppl_increase = (pruned_ppl - original_ppl) / original_ppl
        else:
            ppl_increase = float('inf')

        result["metrics"]["original_perplexity"] = round(original_ppl, 2)
        result["metrics"]["pruned_perplexity"] = round(pruned_ppl, 2)
        result["metrics"]["perplexity_increase_pct"] = round(ppl_increase * 100, 1)

        ppl_passed = (
            ppl_increase <= PERPLEXITY_TOLERANCE
            and pruned_ppl <= MAX_ACCEPTABLE_PERPLEXITY
        )

        status = "[PASS]" if ppl_passed else "[FAIL]"
        ppl_pct = ppl_increase * 100
        print(f"    {status} Perplexity: {original_ppl:.2f} -> {pruned_ppl:.2f} ({ppl_pct:+.1f}%)")

        # Test 2: Generation comparison
        print("  Testing generation quality...")
        similarities = []

        for i, prompt in enumerate(TEST_PROMPTS):
            original_out = generate_text(original_model, tokenizer, prompt, device)
            pruned_out = generate_text(pruned_model, tokenizer, prompt, device)

            sim = token_similarity(original_out, pruned_out)
            similarities.append(sim)

            status = "[PASS]" if sim >= GENERATION_SIMILARITY_THRESHOLD else "[WARN]"
            print(f"    {status} Prompt {i+1}: {sim*100:.1f}% similarity")

        avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
        result["metrics"]["generation_similarities"] = [round(s, 3) for s in similarities]
        result["metrics"]["avg_generation_similarity"] = round(avg_similarity, 3)

        gen_passed = avg_similarity >= GENERATION_SIMILARITY_THRESHOLD

        # Overall pass/fail
        result["passed"] = ppl_passed and gen_passed
        result["perplexity_passed"] = ppl_passed
        result["generation_passed"] = gen_passed

        # Cleanup
        del original_model
        del pruned_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        status = "[PASS]" if result["passed"] else "[FAIL]"
        print(f"  {status} Overall validation: {'PASSED' if result['passed'] else 'FAILED'}")

    except Exception as e:
        error_msg = str(e)
        result["errors"].append(error_msg)
        print(f"  [ERROR] {error_msg}")

    return result


def generate_markdown_report(results: list[dict], output_path: Path) -> None:
    """Generate markdown validation report."""
    passed = sum(1 for r in results if r.get("passed"))
    total = len(results)

    lines = [
        "# Pruned Model Validation Report",
        "",
        f"**Date:** {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Summary:** {passed}/{total} models passed validation",
        "",
        "## Validation Criteria",
        "",
        f"- **Perplexity increase tolerance:** {PERPLEXITY_TOLERANCE*100:.0f}%",
        f"- **Max acceptable perplexity:** {MAX_ACCEPTABLE_PERPLEXITY}",
        f"- **Generation similarity threshold:** {GENERATION_SIMILARITY_THRESHOLD*100:.0f}%",
        "",
        "## Results Summary",
        "",
        "| Model | Strategy | Perplexity | Change | Gen Sim | Status |",
        "|-------|----------|------------|--------|---------|--------|",
    ]

    for r in results:
        m = r.get("metrics", {})
        status = "✅ PASS" if r.get("passed") else "❌ FAIL"
        ppl = m.get("pruned_perplexity", "N/A")
        ppl_change = m.get("perplexity_increase_pct", "N/A")
        gen_sim = m.get("avg_generation_similarity", "N/A")

        if isinstance(gen_sim, (int, float)):
            gen_sim = f"{gen_sim*100:.1f}%"
        if isinstance(ppl_change, (int, float)):
            ppl_change = f"{ppl_change:+.1f}%"

        lines.append(
            f"| {r['model']} | {r['strategy']} | {ppl} | {ppl_change} | "
            f"{gen_sim} | {status} |"
        )

    lines.extend([
        "",
        "## Detailed Results",
        "",
    ])

    for r in results:
        status = "✅ PASSED" if r.get("passed") else "❌ FAILED"
        lines.extend([
            f"### {r['model']} - {r['strategy']}",
            "",
            f"**Status:** {status}",
            "",
            "**Metrics:**",
            f"- Original perplexity: {r.get('metrics', {}).get('original_perplexity', 'N/A')}",
            f"- Pruned perplexity: {r.get('metrics', {}).get('pruned_perplexity', 'N/A')}",
            f"- Perplexity increase: {r.get('metrics', {}).get('perplexity_increase_pct', 'N/A')}%",
            "- Avg generation similarity: "
            f"{r.get('metrics', {}).get('avg_generation_similarity', 'N/A')}",
            "",
        ])

        if r.get("errors"):
            lines.extend([
                "**Errors:**",
                *[f"- {e}" for e in r["errors"]],
                "",
            ])

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[OK] Markdown report written to: {output_path}")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate pruned models against originals"
    )
    parser.add_argument(
        "--test-data-dir",
        type=Path,
        default=Path("test_data"),
        help="Directory containing models and pruned_models",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Specific models to validate (default: all with pruned variants)",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["mild_pruning", "structured_pruning"],
        help="Strategies to validate",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to run on (cuda/cpu/auto)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("test_data/validation_report.json"),
        help="Output JSON path",
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=Path("test_data/validation_report.md"),
        help="Output markdown path",
    )

    args = parser.parse_args()

    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"Using device: {device}")
    print(f"Test data directory: {args.test_data_dir}")

    # Find models to validate
    if args.models:
        models = args.models
    else:
        # Auto-discover from pruned_models directory
        pruned_dir = args.test_data_dir / "pruned_models"
        if not pruned_dir.exists():
            print(f"[FAIL] Pruned models directory not found: {pruned_dir}")
            return 1

        models = set()
        for p in pruned_dir.iterdir():
            if p.is_dir():
                # Extract model name from directory name
                name = p.name
                for strategy in args.strategies:
                    if name.endswith(f"_{strategy}"):
                        model_name = name[: -len(f"_{strategy}")].replace("--", "/")
                        models.add(model_name)
                        break

        models = sorted(models)

    if not models:
        print("[FAIL] No models found to validate")
        return 1

    print(f"Found {len(models)} models to validate")

    # Run validation
    all_results = []

    for model in models:
        for strategy in args.strategies:
            result = validate_model(model, strategy, args.test_data_dir, device)
            all_results.append(result)

    # Save JSON report
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "total": len(all_results),
                "passed": sum(1 for r in all_results if r.get("passed")),
                "failed": sum(1 for r in all_results if not r.get("passed")),
            },
            "criteria": {
                "perplexity_tolerance_pct": PERPLEXITY_TOLERANCE * 100,
                "max_perplexity": MAX_ACCEPTABLE_PERPLEXITY,
                "generation_similarity_threshold": GENERATION_SIMILARITY_THRESHOLD,
            },
            "results": all_results,
        }, f, indent=2)

    print(f"\n[OK] JSON report written to: {args.output}")

    # Generate markdown report
    generate_markdown_report(all_results, args.markdown)

    # Summary
    passed = sum(1 for r in all_results if r.get("passed"))
    total = len(all_results)
    print(f"\n{'='*50}")
    print(f"Validation complete: {passed}/{total} passed")
    print(f"{'='*50}")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
