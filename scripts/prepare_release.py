#!/usr/bin/env python3
"""Release preparation script for Atropos.

Validates version consistency, runs tests and linting, and generates a release checklist.
"""

import datetime
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent


def get_version_from_pyproject() -> str:
    """Extract version from pyproject.toml."""
    content = (ROOT / "pyproject.toml").read_text()
    match = re.search(r'version\s*=\s*"([^"]+)"', content)
    if not match:
        raise RuntimeError("Could not find version in pyproject.toml")
    return match.group(1)


def get_version_from_init() -> str:
    """Extract version from src/atropos/__init__.py."""
    content = (ROOT / "src" / "atropos" / "__init__.py").read_text()
    match = re.search(r'__version__\s*=\s*"([^"]+)"', content)
    if not match:
        raise RuntimeError("Could not find __version__ in src/atropos/__init__.py")
    return match.group(1)


def check_version_consistency() -> bool:
    """Verify version matches across all locations."""
    pyproject_version = get_version_from_pyproject()
    init_version = get_version_from_init()

    print(f"pyproject.toml version: {pyproject_version}")
    print(f"src/atropos/__init__.py version: {init_version}")

    if pyproject_version != init_version:
        print("ERROR: Version mismatch!", file=sys.stderr)
        return False

    # Validate semantic version format
    if not re.match(r"^\d+\.\d+\.\d+$", pyproject_version):
        print(f"WARNING: Version {pyproject_version} doesn't follow semver x.y.z format")

    return True


def run_command(cmd: list[str], cwd: Path = ROOT) -> bool:
    """Run a shell command and return success status."""
    print(f"\n>>> {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"STDOUT:\n{result.stdout}")
            print(f"STDERR:\n{result.stderr}")
            print(f"Command failed with exit code {result.returncode}")
            return False
        print(result.stdout)
        return True
    except Exception as e:
        print(f"Error running command: {e}")
        return False


def run_tests() -> bool:
    """Run full test suite."""
    print("\n=== Running test suite ===")
    return run_command(
        [
            "pytest",
            "tests/",
            "-v",
            "--cov=atropos",
            "--cov-report=term-missing",
            "-m",
            "not integration",
        ]
    )


def run_linting() -> bool:
    """Run linting and formatting checks."""
    print("\n=== Running linting checks ===")
    success1 = run_command(["ruff", "check", "src/", "tests/"])
    success2 = run_command(["ruff", "format", "--check", "src/", "tests/"])
    return success1 and success2


def run_typecheck() -> bool:
    """Run type checking."""
    print("\n=== Running type checking ===")
    return run_command(["mypy", "src"])


def build_package() -> bool:
    """Build distribution packages."""
    print("\n=== Building package ===")
    # Clean build artifacts first
    for dir_name in ["build", "dist"]:
        dir_path = ROOT / dir_name
        if dir_path.exists():
            shutil.rmtree(dir_path)
    for egg_info in ROOT.glob("*.egg-info"):
        if egg_info.exists():
            shutil.rmtree(egg_info)
    # Remove pycache directories
    for pycache in ROOT.rglob("__pycache__"):
        shutil.rmtree(pycache, ignore_errors=True)
    for pyc in ROOT.rglob("*.pyc"):
        pyc.unlink(missing_ok=True)
    # Build
    return run_command(["python", "-m", "build"])


def generate_changelog_entry(version: str) -> None:
    """Generate a template for the changelog entry."""
    print(f"\n=== Changelog entry for version {version} ===")
    print(f"\n## [{version}] - {datetime.date.today().isoformat()}")
    print("\n### Added")
    print("\n### Changed")
    print("\n### Fixed")
    print("\n---")
    print("Add this to CHANGELOG.md under [Unreleased] section.")


def print_checklist(version: str) -> None:
    """Print release checklist."""
    print(f"\n{'=' * 60}")
    print(f"RELEASE CHECKLIST FOR ATROPOS v{version}")
    print(f"{'=' * 60}")
    print("\n[OK]  Version consistency verified")
    print("[OK]  Tests passed")
    print("[OK]  Linting passed")
    print("[OK]  Type checking passed")
    print("[OK]  Package builds successfully")
    print("\n[PACKAGE]  Release steps:")
    print("1. Update CHANGELOG.md with changes for this version")
    print("2. Commit changes: git commit -am 'Prepare release v{version}'")
    print("3. Create tag: git tag -a v{version} -m 'Release v{version}'")
    print("4. Push tag: git push origin v{version}")
    print("5. Monitor GitHub Actions release workflow")
    print("6. Verify package uploaded to PyPI")
    print("7. Create GitHub release from tag")
    print(f"\n[TEST]  Test installation: pip install atropos=={version}")
    print(f"\n{'=' * 60}")


def main() -> int:
    """Main release preparation routine."""
    print("Atropos Release Preparation")
    print("=" * 60)

    # Step 1: Version consistency
    if not check_version_consistency():
        return 1

    version = get_version_from_pyproject()

    # Step 2: Run validation
    success = True
    success = run_tests() and success
    success = run_linting() and success
    success = run_typecheck() and success
    success = build_package() and success

    if not success:
        print("\n[FAILED] Validation failed. Please fix issues before releasing.")
        return 1

    # Step 3: Generate guidance
    generate_changelog_entry(version)
    print_checklist(version)

    print("\n[SUCCESS] Release preparation completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
