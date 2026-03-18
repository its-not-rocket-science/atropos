#!/usr/bin/env python3
"""Health check for CI environment.

Verifies that critical imports work and external dependencies are accessible.
"""

import sys
from pathlib import Path


def check_imports() -> None:
    """Verify critical imports work."""
    try:
        import atropos
        import atropos.calculations
        import atropos.config
        import atropos.models

        # Mark imports as used to satisfy linter
        _ = atropos.calculations
        _ = atropos.config
        _ = atropos.models
        print("[OK] Atropos core imports successful")
    except ImportError as e:
        print(f"[FAIL] Failed to import Atropos module: {e}")
        sys.exit(1)


def check_external_wanda() -> None:
    """Check that external/wanda submodule is accessible."""
    wanda_path = Path(__file__).parent.parent / "external" / "wanda"
    if not wanda_path.exists():
        print("[WARN] external/wanda directory not found (submodule may not be initialized)")
        return

    # Check for critical wanda files
    lib_path = wanda_path / "lib"
    if not lib_path.exists():
        print("[WARN] external/wanda/lib directory not found")
        return

    print("[OK] external/wanda submodule accessible")


def check_python_version() -> None:
    """Verify Python version compatibility."""
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")
    if version.major == 3 and version.minor >= 10:
        print("[OK] Python version compatible (>=3.10)")
    else:
        print("[FAIL] Python version too old (requires >=3.10)")
        sys.exit(1)


def main() -> None:
    """Run all health checks."""
    print("Running CI health checks...")
    print("=" * 40)

    check_python_version()
    print()

    check_imports()
    print()

    check_external_wanda()
    print()

    print("=" * 40)
    print("Health check completed successfully")


if __name__ == "__main__":
    main()
