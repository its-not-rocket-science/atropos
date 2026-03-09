"""Web dashboard for interactive Atropos analysis."""

from __future__ import annotations

__all__ = ["create_app", "run_dashboard"]

try:
    from .app import create_app, run_dashboard
except ImportError as e:
    raise ImportError(
        "Dashboard dependencies not installed. Install with: pip install dash plotly pandas"
    ) from e
