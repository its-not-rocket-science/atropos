"""Backward-compatible WorkerManager shim."""

from __future__ import annotations

from dataclasses import dataclass

from .worker_runtime import WorkerRuntime


@dataclass
class WorkerManager(WorkerRuntime):
    """Compatibility alias for WorkerRuntime."""
