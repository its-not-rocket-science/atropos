"""Composable BaseEnv collaborators used during incremental refactors.

These classes intentionally keep behavior minimal and explicit so `BaseEnv`
can delegate work without breaking existing call-sites.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EnvTransportClient:
    """Transport boundary for API/model calls.

    Override `send` in production adapters (HTTP/gRPC/SDK).
    """

    def send(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {"ok": True, "payload": payload}


@dataclass
class EnvLogger:
    """Structured logging boundary for runtime events."""

    events: list[dict[str, Any]] = field(default_factory=list)

    def info(self, event: str, **metadata: Any) -> None:
        self.events.append({"level": "info", "event": event, "metadata": metadata})

    def error(self, event: str, **metadata: Any) -> None:
        self.events.append({"level": "error", "event": event, "metadata": metadata})


@dataclass
class EnvCheckpointManager:
    """Checkpoint persistence boundary.

    Uses in-memory storage by default to remain side-effect free.
    """

    snapshots: list[dict[str, Any]] = field(default_factory=list)

    def save(self, state: dict[str, Any]) -> None:
        self.snapshots.append(dict(state))


@dataclass
class EnvCliBuilder:
    """CLI argument generation boundary."""

    def build(self, config: dict[str, Any]) -> list[str]:
        args: list[str] = []
        for key, value in sorted(config.items()):
            flag = f"--{key.replace('_', '-')}"
            args.extend([flag, str(value)])
        return args


@dataclass
class EnvConfigMerger:
    """Deterministic YAML/CLI merge boundary.

    CLI flags override YAML defaults.
    """

    def merge(self, yaml_config: dict[str, Any], cli_config: dict[str, Any]) -> dict[str, Any]:
        merged = dict(yaml_config)
        merged.update(cli_config)
        return merged


@dataclass
class EnvRuntime:
    """Worker orchestration boundary.

    This skeleton keeps orchestration explicit and compatible with synchronous
    call-sites. Concrete runtime implementations can override `run`.
    """

    def run(self, work_item: dict[str, Any], worker_count: int = 1) -> dict[str, Any]:
        return {
            "worker_count": worker_count,
            "work_item": work_item,
            "status": "processed",
        }
