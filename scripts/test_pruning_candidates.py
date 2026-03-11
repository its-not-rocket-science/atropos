#!/usr/bin/env python3
"""Test downloaded candidate models with Atropos validation.

This script tests the 5 candidate models downloaded by download_test_models.py:
- gpt2 (124M) - edge-coder preset
- gpt2-medium (355M) - small-coder preset
- gpt2-xl (1.5B) - medium-coder preset
- facebook/opt-1.3b (1.3B) - medium-coder preset
- EleutherAI/pythia-2.8b (2.8B) - large-coder preset

Usage:
    python scripts/test_pruning_candidates.py [--test-data-dir PATH] [--device cpu|cuda]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# Model to preset mapping (available presets: edge-coder, medium-coder, frontier-assistant)
MODEL_PRESETS = {
    "gpt2": "edge-coder",
    "gpt2-medium": "edge-coder",  # Use edge-coder for small models
    "gpt2-xl": "medium-coder",
    "facebook/opt-1.3b": "medium-coder",
    "EleutherAI/pythia-2.8b": "medium-coder",  # Use medium-coder for 2.8B
}

# Available pruning strategies to test
STRATEGIES = ["mild_pruning", "structured_pruning"]


@dataclass
class ValidationResult:
    """Result of validating a model."""

    model_id: str
    preset: str
    strategy: str
    status: str  # "success", "failed", "skipped"
    validation_time_sec: float = 0.0
    memory_variance_pct: float | None = None
    throughput_variance_pct: float | None = None
    savings_accuracy: float | None = None
    error_message: str = ""
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TestReport:
    """Report of all test operations."""

    total_tests: int = 0
    successful: int = 0
    failed: int = 0
    skipped: int = 0
    results: list[ValidationResult] = field(default_factory=list)
    start_time: str = ""
    end_time: str = ""
    device: str = ""

    @property
    def duration_sec(self) -> float:
        """Calculate total duration."""
        if self.start_time and self.end_time:
            start = datetime.fromisoformat(self.start_time)
            end = datetime.fromisoformat(self.end_time)
            return (end - start).total_seconds()
        return 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_tests": self.total_tests,
            "successful": self.successful,
            "failed": self.failed,
            "skipped": self.skipped,
            "duration_sec": self.duration_sec,
            "device": self.device,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "results": [r.to_dict() for r in self.results],
        }


def check_atropos() -> bool:
    """Check if Atropos CLI is available."""
    import shutil
    if shutil.which("atropos"):
        print("[OK] Atropos CLI found")
        return True

    print("[FAIL] Atropos CLI not found")
    print("\nInstall with: pip install -e .")
    return False


def run_atropos_validation(
    model_id: str,
    preset: str,
    strategy: str,
    device: str = "cpu",
    timeout_sec: float = 300.0,
) -> ValidationResult:
    """Run Atropos validation on a model.

    Args:
        model_id: HuggingFace model ID
        preset: Atropos preset name
        strategy: Pruning strategy name
        device: Device to run on
        timeout_sec: Timeout in seconds

    Returns:
        ValidationResult with status and metrics
    """
    print(f"\n  Validating {model_id}")
    print(f"    Preset: {preset}, Strategy: {strategy}")

    start_time = time.time()
    result = ValidationResult(
        model_id=model_id,
        preset=preset,
        strategy=strategy,
        status="failed",
        timestamp=datetime.now().isoformat(),
    )

    try:
        cmd = [
            "atropos",
            "validate",
            preset,
            "--model",
            model_id,
            "--device",
            device,
            "--strategy",
            strategy,
            "--format",
            "json",
        ]

        proc_result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )

        validation_time = time.time() - start_time
        result.validation_time_sec = validation_time

        if proc_result.returncode == 0:
            # Parse JSON output (skip any text before the JSON)
            try:
                stdout = proc_result.stdout
                # Find the first JSON object
                json_start = stdout.find("{")
                if json_start == -1:
                    raise json.JSONDecodeError("No JSON found in output", stdout, 0)

                data = json.loads(stdout[json_start:])

                # Extract metrics from comparisons
                comparisons = {c["name"]: c for c in data.get("comparisons", [])}

                result.memory_variance_pct = comparisons.get("Memory", {}).get(
                    "variance_pct"
                )
                result.throughput_variance_pct = comparisons.get("Throughput", {}).get(
                    "variance_pct"
                )
                result.savings_accuracy = data.get("savings_accuracy")
                result.status = "success"

                print(f"    [OK] Success ({validation_time:.1f}s)")
                if result.memory_variance_pct is not None:
                    print(
                        f"      Memory variance: {result.memory_variance_pct:+.1f}%"
                    )
                if result.savings_accuracy is not None:
                    print(f"      Accuracy: {result.savings_accuracy:.1f}%")

            except json.JSONDecodeError as e:
                result.error_message = f"Failed to parse JSON: {e}"
                print(f"    [FAIL] Failed: {result.error_message}")
        else:
            result.error_message = proc_result.stderr[:200]
            print(f"    [FAIL] Failed: {result.error_message}")

    except subprocess.TimeoutExpired:
        result.error_message = f"Timeout after {timeout_sec}s"
        result.validation_time_sec = time.time() - start_time
        print(f"    [FAIL] Timeout")
    except Exception as e:
        result.error_message = str(e)
        result.validation_time_sec = time.time() - start_time
        print(f"    [FAIL] Error: {e}")

    return result


def test_all_models(
    device: str = "cpu",
    models: list[str] | None = None,
    strategies: list[str] | None = None,
) -> TestReport:
    """Test all candidate models.

    Args:
        device: Device to run on
        models: Specific models to test (default: all)
        strategies: Specific strategies to test (default: all)

    Returns:
        TestReport with all results
    """
    models = models or list(MODEL_PRESETS.keys())
    strategies = strategies or STRATEGIES

    # Calculate total tests
    total_tests = len(models) * len(strategies)

    report = TestReport(
        total_tests=total_tests,
        start_time=datetime.now().isoformat(),
        device=device,
    )

    print("=" * 60)
    print("Testing Candidate Models with Atropos Validation")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Models: {len(models)}")
    print(f"Strategies: {len(strategies)}")
    print(f"Total tests: {total_tests}")

    test_num = 0
    for model_id in models:
        if model_id not in MODEL_PRESETS:
            print(f"\n[WARNING] Unknown model: {model_id}, skipping")
            continue

        preset = MODEL_PRESETS[model_id]

        for strategy in strategies:
            test_num += 1
            print(f"\n[{test_num}/{total_tests}] Testing {model_id} with {strategy}")

            result = run_atropos_validation(
                model_id=model_id,
                preset=preset,
                strategy=strategy,
                device=device,
            )
            report.results.append(result)

            if result.status == "success":
                report.successful += 1
            elif result.status == "skipped":
                report.skipped += 1
            else:
                report.failed += 1

    report.end_time = datetime.now().isoformat()

    return report


def print_summary(report: TestReport) -> None:
    """Print a summary of the test operation."""
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"Total tests: {report.total_tests}")
    print(f"Successful: {report.successful} [OK]")
    print(f"Failed: {report.failed} [FAIL]")
    print(f"Skipped: {report.skipped} [SKIP]")
    print(f"Duration: {report.duration_sec:.1f}s")
    print(f"Device: {report.device}")

    print("\nResults by Model:")
    print("-" * 80)

    # Group by model
    by_model: dict[str, list[ValidationResult]] = {}
    for r in report.results:
        by_model.setdefault(r.model_id, []).append(r)

    for model_id, results in by_model.items():
        print(f"\n{model_id}:")
        for r in results:
            icon = "[OK]" if r.status == "success" else "[FAIL]" if r.status == "failed" else "[SKIP]"
            print(f"  {icon} {r.strategy:20s} ({r.validation_time_sec:.1f}s)")
            if r.memory_variance_pct is not None:
                print(f"      Memory var: {r.memory_variance_pct:+.1f}%")
            if r.savings_accuracy is not None:
                print(f"      Accuracy: {r.savings_accuracy:.1f}%")
            if r.error_message:
                print(f"      Error: {r.error_message[:80]}")

    # Print strategy comparison
    print("\n\nResults by Strategy:")
    print("-" * 60)

    by_strategy: dict[str, list[ValidationResult]] = {}
    for r in report.results:
        by_strategy.setdefault(r.strategy, []).append(r)

    for strategy, results in by_strategy.items():
        success_count = sum(1 for r in results if r.status == "success")
        avg_accuracy = sum(
            r.savings_accuracy for r in results if r.savings_accuracy is not None
        ) / max(1, sum(1 for r in results if r.savings_accuracy is not None))

        print(f"\n{strategy}:")
        print(f"  Success rate: {success_count}/{len(results)}")
        print(f"  Avg accuracy: {avg_accuracy:.1f}%")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Test candidate models with Atropos validation"
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device for testing (default: cpu)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Specific model IDs to test (default: all candidates)",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=None,
        help=f"Strategies to test (default: {STRATEGIES})",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("test_data/validation_results.json"),
        help="Output JSON file for report (default: test_data/validation_results.json)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Timeout per test in seconds (default: 300)",
    )

    args = parser.parse_args()

    # Check Atropos is available
    if not check_atropos():
        sys.exit(1)

    # Run tests
    report = test_all_models(
        device=args.device,
        models=args.models,
        strategies=args.strategies,
    )

    # Print summary
    print_summary(report)

    # Save report
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report.to_dict(), indent=2))
    print(f"\n\nReport saved to: {args.output}")

    # Exit with error code if any tests failed
    if report.failed > 0:
        print(f"\n[WARNING] {report.failed} test(s) failed")
        sys.exit(1)

    print("\n[OK] All tests passed!")


if __name__ == "__main__":
    main()
