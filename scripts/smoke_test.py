#!/usr/bin/env python3
"""Smoke test for Atropos package installation.

Verifies basic functionality after installation from PyPI or local build.
"""

import importlib
import subprocess
import sys


def test_imports() -> bool:
    """Test that core modules can be imported."""
    print("Testing imports...")
    modules = [
        "atropos",
        "atropos.calculations",
        "atropos.models",
        "atropos.cli",
        "atropos.presets",
    ]
    success = True
    for module in modules:
        try:
            importlib.import_module(module)
            print(f"  [OK] {module}")
        except ImportError as e:
            print(f"  [FAIL] {module}: {e}")
            success = False
    return success


def test_version() -> bool:
    """Check that version is accessible."""
    print("\nTesting version...")
    try:
        import atropos

        print(f"  Version: {atropos.__version__}")
        return True
    except AttributeError as e:
        print(f"  [FAIL] Could not get version: {e}")
        return False


def test_cli_help() -> bool:
    """Test that CLI help works."""
    print("\nTesting CLI help...")
    try:
        result = subprocess.run(
            ["atropos-llm", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            print("  [OK] atropos-llm --help")
            return True
        else:
            print(f"  [FAIL] atropos-llm --help failed: {result.stderr}")
            return False
    except FileNotFoundError:
        print("  [FAIL] 'atropos-llm' command not found")
        return False
    except subprocess.TimeoutExpired:
        print("  [FAIL] Command timed out")
        return False


def test_list_presets() -> bool:
    """Test list-presets command."""
    print("\nTesting list-presets...")
    try:
        result = subprocess.run(
            ["atropos-llm", "list-presets"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0 and "mild_pruning" in result.stdout:
            print("  [OK] atropos-llm list-presets")
            return True
        else:
            print(f"  [FAIL] list-presets failed or output unexpected: {result.stderr}")
            return False
    except FileNotFoundError:
        print("  [FAIL] 'atropos-llm' command not found")
        return False
    except subprocess.TimeoutExpired:
        print("  [FAIL] Command timed out")
        return False


def test_calculation_functions() -> bool:
    """Test core calculation functions."""
    print("\nTesting calculation functions...")
    try:
        from atropos.calculations import estimate_outcome
        from atropos.models import DeploymentScenario, OptimizationStrategy

        # Create a minimal scenario
        scenario = DeploymentScenario(
            name="smoke-test",
            parameters_b=7,
            memory_gb=8,
            throughput_toks_per_sec=20,
            power_watts=200,
            requests_per_day=10000,
            tokens_per_request=800,
            electricity_cost_per_kwh=0.12,
            annual_hardware_cost_usd=12000,
            one_time_project_cost_usd=15000,
        )

        strategy = OptimizationStrategy(
            name="test",
            parameter_reduction_fraction=0.2,
            memory_reduction_fraction=0.2,
            throughput_improvement_fraction=0.1,
            power_reduction_fraction=0.05,
            quality_risk="low",
        )

        outcome = estimate_outcome(scenario, strategy)
        print(f"  [OK] estimate_outcome: ${outcome.annual_total_savings_usd:.2f} savings")
        return True
    except Exception as e:
        print(f"  [FAIL] Calculation test failed: {e}")
        return False


def main() -> int:
    """Run all smoke tests."""
    print("=" * 60)
    print("Atropos Smoke Test")
    print("=" * 60)

    success = True
    success = test_imports() and success
    success = test_version() and success
    success = test_cli_help() and success
    success = test_list_presets() and success
    success = test_calculation_functions() and success

    print("\n" + "=" * 60)
    if success:
        print("[PASS] All smoke tests passed!")
        return 0
    else:
        print("[FAIL] Some smoke tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
