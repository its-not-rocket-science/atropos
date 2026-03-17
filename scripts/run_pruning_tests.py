#!/usr/bin/env python3
"""Run BLOOM and GPT-J pruning tests with timeout."""

import subprocess
import sys
from pathlib import Path


def run_with_timeout(cmd, timeout_sec=1800):  # 30 minutes
    """Run command with timeout, returns (success, output)."""
    print(f"Running: {' '.join(cmd)}")
    print(f"Timeout: {timeout_sec}s")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        output = result.stdout + "\n" + result.stderr
        success = result.returncode == 0
        if success:
            print("[OK] Success")
        else:
            print("[FAIL] Failed with return code", result.returncode)
        return success, output
    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] Timeout after {timeout_sec}s")
        return False, "Timeout"
    except Exception as e:
        print(f"[ERROR] Error: {e}")
        return False, str(e)


def main():
    script_dir = Path(__file__).parent
    tests = [
        ("BLOOM", script_dir / "test_bloom_pruning.py"),
        ("GPT-J", script_dir / "test_gptj_pruning.py"),
    ]

    overall = True
    for name, script in tests:
        print(f"\n{'=' * 60}")
        print(f"Testing {name}")
        print(f"{'=' * 60}")
        if not script.exists():
            print(f"Script not found: {script}")
            overall = False
            continue

        # Use small model for GPT-J to speed up
        if "gptj" in script.name.lower():
            cmd = [
                sys.executable,
                str(script),
                "--model",
                "Milos/slovak-gpt-j-162M",
                "--device",
                "cpu",
            ]
        else:
            cmd = [sys.executable, str(script)]

        success, output = run_with_timeout(cmd, timeout_sec=1800)  # 30 minutes

        # Print last 20 lines of output
        lines = output.strip().split("\n")
        if len(lines) > 20:
            print("\n... (output truncated) ...")
            for line in lines[-20:]:
                print(line)
        else:
            print(output)

        if not success:
            overall = False

    print(f"\n{'=' * 60}")
    if overall:
        print("All tests passed!")
        sys.exit(0)
    else:
        print("Some tests failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
