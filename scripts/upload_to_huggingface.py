#!/usr/bin/env python3
"""Upload pruned models to HuggingFace Hub.

This script uploads the pruned models from the pruning exercise
to HuggingFace Hub for public access.

Requirements:
    pip install huggingface-hub
    huggingface-cli login

Or set environment variable:
    export HF_TOKEN=your_token_here

Usage:
    python scripts/upload_to_huggingface.py --org your-org-name

Note:
    You need a HuggingFace account and token to upload models.
    Get your token at: https://huggingface.co/settings/tokens
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class UploadResult:
    """Result of uploading a model."""

    model_id: str
    strategy: str
    local_path: str
    repo_id: str
    status: str = "failed"
    url: str = ""
    error_message: str = ""
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class UploadReport:
    """Report of all upload operations."""

    total_models: int = 0
    successful: int = 0
    failed: int = 0
    results: list[UploadResult] = field(default_factory=list)
    start_time: str = ""
    end_time: str = ""
    org_name: str = ""

    @property
    def duration_sec(self) -> float:
        if self.start_time and self.end_time:
            start = datetime.fromisoformat(self.start_time)
            end = datetime.fromisoformat(self.end_time)
            return (end - start).total_seconds()
        return 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_models": self.total_models,
            "successful": self.successful,
            "failed": self.failed,
            "duration_sec": self.duration_sec,
            "org_name": self.org_name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "results": [r.to_dict() for r in self.results],
        }


def check_hf_cli() -> str | None:
    """Check if huggingface-cli is available and user is logged in.

    Returns:
        Username if logged in, None otherwise
    """
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        user = api.whoami()
        print(f"[OK] Logged in as: {user['name']}")
        return user["name"]
    except ImportError:
        print("[FAIL] huggingface-hub not installed")
        print("Install with: pip install huggingface-hub")
        return None
    except Exception as e:
        print(f"[FAIL] Not logged in: {e}")
        print("Login with: huggingface-cli login")
        return None


def check_validation_report(test_data_dir: Path, force: bool = False) -> tuple[set[str], set[str]]:
    """Check which models have passed/failed validation.

    Args:
        test_data_dir: Path to test_data directory
        force: If True, skip validation check

    Returns:
        Tuple of (passing models set, failing models set)
        Model identifiers are "model_name--strategy"
    """
    if force:
        return set(), set()

    validation_path = test_data_dir / "validation_report.json"

    if not validation_path.exists():
        print("[WARNING] Validation report not found.")
        print("          Run: python scripts/validate_pruned_models.py")
        print("          Use --force to upload without validation")
        return set(), set()

    try:
        with open(validation_path, encoding="utf-8") as f:
            report = json.load(f)

        passing = set()
        failing = set()

        for result in report.get("results", []):
            model_key = f"{result['model']}--{result['strategy']}"
            if result.get("passed", False):
                passing.add(model_key)
            else:
                failing.add(model_key)

        if failing:
            print(f"[INFO] {len(failing)} model(s) failed validation (will skip):")
            for m in failing:
                print(f"       - {m}")

        print(f"[OK] {len(passing)} model(s) passed validation")
        return passing, failing

    except Exception as e:
        print(f"[WARNING] Could not read validation report: {e}")
        return set(), set()


def upload_model(
    local_path: Path,
    model_id: str,
    strategy: str,
    org_name: str | None = None,
    username: str | None = None,
    private: bool = False,
) -> UploadResult:
    """Upload a single model to HuggingFace Hub.

    Args:
        local_path: Path to local model directory
        model_id: Original model ID (e.g., "gpt2")
        strategy: Pruning strategy (e.g., "mild_pruning")
        org_name: Organization name (optional)
        username: HF username for personal repos (optional)
        private: Whether to create private repo

    Returns:
        UploadResult with operation details
    """
    import os

    from huggingface_hub import HfApi

    # Get token from environment
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN environment variable not set")

    # Initialize API with token
    api = HfApi(token=token)

    # Build repo ID first - must include namespace
    base_name = model_id.replace("/", "-")
    repo_name = f"{base_name}-{strategy}"
    if org_name:
        repo_id = f"{org_name}/{repo_name}"
    elif username:
        repo_id = f"{username}/{repo_name}"
    else:
        raise ValueError("Either org_name or username must be provided")

    result = UploadResult(
        model_id=model_id,
        strategy=strategy,
        local_path=str(local_path),
        repo_id=repo_id,
        timestamp=datetime.now().isoformat(),
    )

    try:
        print(f"  Creating repo: {result.repo_id}...", end=" ", flush=True)

        # Create repository using API
        api.create_repo(
            repo_id=result.repo_id,
            private=private,
            exist_ok=True,
        )
        print("[OK]")

        # Create model card
        card_content = generate_model_card(model_id, strategy, repo_name)
        card_path = local_path / "README.md"
        card_path.write_text(card_content)

        # Upload model files using API
        print("  Uploading files...", end=" ", flush=True)
        api.upload_folder(
            folder_path=str(local_path),
            repo_id=result.repo_id,
            repo_type="model",
        )
        print("[OK]")

        result.status = "success"
        result.url = f"https://huggingface.co/{result.repo_id}"
        print(f"  [OK] Uploaded to: {result.url}")

    except Exception as e:
        result.error_message = str(e)
        print(f"  [FAIL] {e}")

    return result


def generate_model_card(
    model_id: str,
    strategy: str,
    repo_name: str,
) -> str:
    """Generate a HuggingFace model card."""
    return f"""---
language: en
license: mit  # Adjust based on original model
tags:
  - pruning
  - sparse
  - atropos
  - code-generation
---

# {repo_name}

This is a pruned version of [{model_id}](https://huggingface.co/{model_id})
generated as part of the Atropos pruning exercise.

## Pruning Details

- **Strategy:** {strategy}
- **Method:** PyTorch magnitude-based unstructured pruning
- **Generated by:** Atropos pruning exercise scripts

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{repo_name}")
tokenizer = AutoTokenizer.from_pretrained("{repo_name}")
```

## About Atropos

[Atropos](https://github.com/its-not-rocket-science/atropos) estimates ROI
for pruning and quantization optimizations in LLM deployments.

This model was pruned to validate Atropos projections against actual results.

## Citation

If you use this model, please cite the original model and Atropos:

```bibtex
@software{{atropos,
  title = {{Atropos: ROI Estimation for LLM Pruning}},
  year = {{2026}},
}}
```
"""


def upload_all_models(
    pruned_dir: Path,
    org_name: str | None = None,
    username: str | None = None,
    private: bool = False,
    passing_models: set[str] | None = None,
) -> UploadReport:
    """Upload pruned models to HuggingFace.

    Args:
        pruned_dir: Directory containing pruned models
        org_name: Organization name (optional)
        username: HF username for personal repos (optional)
        private: Whether to create private repos
        passing_models: Set of "model--strategy" keys that passed validation

    Returns:
        UploadReport with all results
    """
    # Find all pruned model directories
    all_model_dirs = sorted([d for d in pruned_dir.iterdir() if d.is_dir()])

    # Filter to only passing models if validation data provided
    model_dirs = []
    skipped = []
    for d in all_model_dirs:
        if passing_models is None:
            model_dirs.append(d)
        else:
            # Parse model key from directory name
            dir_name = d.name
            for s in ["mild_pruning", "structured_pruning"]:
                if dir_name.endswith(f"_{s}"):
                    model_name = dir_name[: -len(f"_{s}")].replace("--", "/")
                    key = f"{model_name}--{s}"
                    if key in passing_models:
                        model_dirs.append(d)
                    else:
                        skipped.append(dir_name)
                    break

    report = UploadReport(
        total_models=len(model_dirs),
        start_time=datetime.now().isoformat(),
        org_name=org_name or "personal",
    )

    print("=" * 70)
    print("Uploading Pruned Models to HuggingFace Hub")
    print("=" * 70)
    if passing_models is not None:
        print(f"Total pruned models: {len(all_model_dirs)}")
        print(f"Passed validation: {len(model_dirs)}")
        print(f"Skipped (failed validation): {len(skipped)}")
    print(f"Models to upload: {len(model_dirs)}")
    print(f"Organization: {org_name or 'personal account'}")
    print(f"Visibility: {'private' if private else 'public'}")

    for i, model_dir in enumerate(model_dirs, 1):
        # Parse model ID and strategy from directory name
        # Format: model--name_strategy or model_name_strategy
        dir_name = model_dir.name

        # Try to extract strategy
        strategy = None
        for s in ["mild_pruning", "structured_pruning"]:
            if dir_name.endswith(f"_{s}"):
                strategy = s
                base_name = dir_name[: -len(f"_{s}")]
                break

        if not strategy:
            print(f"\n[{i}/{len(model_dirs)}] Skipping {dir_name} - unknown strategy")
            continue

        # Convert base_name back to model ID
        model_id = base_name.replace("--", "/")

        print(f"\n[{i}/{len(model_dirs)}] {model_id} - {strategy}")
        print("-" * 50)

        result = upload_model(
            local_path=model_dir,
            model_id=model_id,
            strategy=strategy,
            org_name=org_name,
            username=username,
            private=private,
        )
        report.results.append(result)

        if result.status == "success":
            report.successful += 1
        else:
            report.failed += 1

    report.end_time = datetime.now().isoformat()
    return report


def print_report(report: UploadReport) -> None:
    """Print upload report."""
    print("\n" + "=" * 70)
    print("Upload Summary")
    print("=" * 70)
    print(f"Total: {report.total_models}")
    print(f"Successful: {report.successful}")
    print(f"Failed: {report.failed}")
    print(f"Duration: {report.duration_sec:.1f}s")
    print(f"Organization: {report.org_name}")

    print("\nUploaded Models:")
    print("-" * 70)
    for r in report.results:
        status = "OK" if r.status == "success" else "FAIL"
        print(f"  [{status}] {r.model_id} ({r.strategy})")
        if r.url:
            print(f"       {r.url}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Upload pruned models to HuggingFace Hub")
    parser.add_argument(
        "--pruned-dir",
        type=Path,
        default=Path("test_data/pruned_models"),
        help="Directory containing pruned models (default: test_data/pruned_models)",
    )
    parser.add_argument(
        "--org",
        type=str,
        default=None,
        help="Organization name (default: personal account)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create private repositories",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("test_data/upload_report.json"),
        help="Output report path (default: test_data/upload_report.json)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Upload without validation (use with caution)",
    )

    args = parser.parse_args()

    # Check HF CLI is available and get username
    username = check_hf_cli()
    if not username:
        sys.exit(1)

    # Check validation status
    test_data_dir = args.pruned_dir.parent
    passing, failing = check_validation_report(test_data_dir, args.force)

    if args.force:
        passing_models = None  # Upload all
    else:
        if not passing and not failing:
            print("\n[FAIL] No validation data found. Run validation first or use --force.")
            sys.exit(1)
        passing_models = passing

    # Check pruned directory exists
    if not args.pruned_dir.exists():
        print(f"Error: Pruned directory not found: {args.pruned_dir}")
        sys.exit(1)

    # Upload models
    report = upload_all_models(
        pruned_dir=args.pruned_dir,
        org_name=args.org,
        username=username,
        private=args.private,
        passing_models=passing_models,
    )

    # Print report
    print_report(report)

    # Save report
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report.to_dict(), indent=2))
    print(f"\nReport saved to: {args.output}")

    if report.failed > 0:
        print(f"\n[WARNING] {report.failed} upload(s) failed")
        sys.exit(1)

    print("\n[OK] All models uploaded successfully!")


if __name__ == "__main__":
    main()
