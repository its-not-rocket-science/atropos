"""Checkpoint management primitives for environment state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CheckpointManager:
    """Persist environment snapshots.

    This default implementation is intentionally in-memory for deterministic
    tests and no-IO local workflows.
    """

    snapshots: list[dict[str, Any]] = field(default_factory=list)

    def save(self, state: dict[str, Any]) -> None:
        self.snapshots.append(dict(state))

    def latest(self) -> dict[str, Any] | None:
        if not self.snapshots:
            return None
        return dict(self.snapshots[-1])

    def reset(self) -> None:
        self.snapshots.clear()
