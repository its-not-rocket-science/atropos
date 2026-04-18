"""pytest configuration and shared fixtures."""

from __future__ import annotations

import sys
from importlib.util import find_spec
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def pytest_addoption(parser: object) -> None:
    """Register no-op coverage flags when pytest-cov is unavailable.

    CI normally installs pytest-cov. Local constrained environments may not have the plugin,
    which would otherwise fail argument parsing due to project-level addopts.
    """

    if find_spec("pytest_cov") is not None:
        return

    # Use dynamic access to avoid importing pytest types at runtime.
    cov_group = parser.getgroup("cov")
    cov_group.addoption("--cov", action="append", default=[])
    cov_group.addoption("--cov-report", action="append", default=[])
