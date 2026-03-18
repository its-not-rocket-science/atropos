#!/usr/bin/env python3
"""Setup script for Wanda pruning framework dependencies.

This script installs the specific dependencies required by the external/wanda
submodule to run pruning experiments. These dependencies may conflict with
Atropos' main dependencies, so consider using a separate environment.
"""

import subprocess
import sys

# Dependencies from external/wanda/INSTALL.md
WANDA_DEPS = [
    "torch==1.10.1",
    "torchvision==0.11.2",
    "torchaudio==0.10.1",
    "transformers==4.28.0",
    "datasets==2.11.0",
    "wandb",
    "sentencepiece",
    "accelerate==0.18.0",
]


def main() -> None:
    """Install Wanda dependencies."""
    print("Wanda pruning framework dependency installer")
    print("=" * 50)
    print()
    print("NOTE: These dependencies may conflict with Atropos' main dependencies.")
    print("Consider using a separate environment for pruning experiments.")
    print()

    # Check if we're in a virtual environment
    if not hasattr(sys, "real_prefix") and sys.prefix == sys.base_prefix:
        print("WARNING: Not running in a virtual environment.")
        print("It's recommended to create a separate environment:")
        print("  conda create -n wanda python=3.9")
        print("  conda activate wanda")
        print()

    response = input("Proceed with installation? [y/N]: ").strip().lower()
    if response not in ("y", "yes"):
        print("Installation cancelled.")
        return

    print(f"Installing {len(WANDA_DEPS)} packages...")

    # Install each package
    for dep in WANDA_DEPS:
        print(f"Installing {dep}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
        except subprocess.CalledProcessError as e:
            print(f"Failed to install {dep}: {e}")
            print("You may need to install CUDA versions of PyTorch.")
            print("See external/wanda/INSTALL.md for details.")
            sys.exit(1)

    print()
    print("Installation complete.")
    print("You can now run pruning scripts from the scripts/ directory.")
    print("Example: python scripts/prune_wanda.py --model gpt2 --sparsity 0.3")


if __name__ == "__main__":
    main()
