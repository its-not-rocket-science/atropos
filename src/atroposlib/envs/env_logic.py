"""Environment-logic contract separated from runtime infrastructure."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class EnvLogic(Protocol):
    """User-defined business logic for preparing and finalizing steps."""

    def prepare_step(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Prepare user payload before worker orchestration and transport."""

    def finalize_step(self, transport_result: dict[str, Any]) -> dict[str, Any]:
        """Transform transport output to environment-facing result."""


@dataclass
class PassthroughEnvLogic:
    """Default logic that preserves the previous BaseEnv payload semantics."""

    def prepare_step(self, payload: dict[str, Any]) -> dict[str, Any]:
        return dict(payload)

    def finalize_step(self, transport_result: dict[str, Any]) -> dict[str, Any]:
        return dict(transport_result)
