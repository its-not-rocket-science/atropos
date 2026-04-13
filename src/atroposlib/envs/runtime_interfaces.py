"""Formal runtime interfaces for environment execution seams."""

from __future__ import annotations

from typing import Any, Protocol


class ItemSource(Protocol):
    """Produces runtime work items from environment payloads."""

    def prepare_item(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Translate an inbound environment payload into a runtime work item."""


class BacklogManager(Protocol):
    """Owns queueing and worker selection during runtime execution."""

    def orchestrate(
        self,
        work_item: dict[str, Any],
        requested_workers: int = 1,
        *,
        env: str = "default",
        trainer_feedback: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Select workers and return a runtime payload for downstream systems."""


class SendToApiPath(Protocol):
    """Boundary for sending runtime payloads to model/API backends."""

    def send(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Send runtime payload and return transport result."""


class RolloutCollector(Protocol):
    """Transforms transport responses into environment-visible rollouts."""

    def collect(self, transport_result: dict[str, Any]) -> dict[str, Any]:
        """Collect and normalize a rollout result."""


class EvalRunner(Protocol):
    """Evaluation entrypoint for runtime-driven environments."""

    def evaluate(self, payload: dict[str, Any], worker_count: int = 1) -> dict[str, Any]:
        """Run evaluation for the given payload."""
