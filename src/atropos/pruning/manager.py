"""CLI-facing setup and smoke-test helpers for pruning integrations."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass

from .base import PruningResult, get_pruning_framework


@dataclass
class SetupSummary:
    success: bool
    message: str


def setup_pruning_environment(fix: bool = False) -> SetupSummary:
    """Check runtime prerequisites and optionally build framework containers."""
    check_cmd = ["docker", "compose", "version"]
    try:
        subprocess.run(check_cmd, check=True, capture_output=True, text=True)
    except Exception:
        check_cmd = ["podman", "compose", "version"]
        try:
            subprocess.run(check_cmd, check=True, capture_output=True, text=True)
        except Exception:
            return SetupSummary(
                success=False,
                message=(
                    "No compatible rootless container runtime found. Install Docker or Podman "
                    "(rootless mode) and re-run `atropos-llm setup-pruning`."
                ),
            )

    build_cmd = check_cmd[:2] + ["-f", "docker-compose.yml", "build"]
    if fix:
        build_cmd.append("--no-cache")

    completed = subprocess.run(build_cmd, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        return SetupSummary(
            success=False,
            message=f"Container build failed: {completed.stderr[-500:]}",
        )

    return SetupSummary(
        success=True,
        message="Pruning environments are ready (native + container fallback configured).",
    )


def test_pruning_framework(name: str) -> PruningResult:
    """Run a smoke test with fallback behavior for a framework."""
    framework = get_pruning_framework(name)
    return framework.test_integration()
