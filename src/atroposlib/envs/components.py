"""Backward compatibility aliases for the former components module."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .checkpoint_manager import CheckpointManager
from .logging_manager import LoggingManager
from .transport_client import TransportClient
from .worker_manager import WorkerManager

EnvRuntime = WorkerManager
EnvTransportClient = TransportClient
EnvLogger = LoggingManager
EnvCheckpointManager = CheckpointManager


@dataclass
class EnvCliBuilder:
    """Legacy CLI builder retained for compatibility."""

    def build(self, config: dict[str, Any]) -> list[str]:
        args: list[str] = []
        for key, value in sorted(config.items()):
            args.extend([f"--{key.replace('_', '-')}", str(value)])
        return args


@dataclass
class EnvConfigMerger:
    """Legacy config merge helper retained for compatibility."""

    def merge(self, yaml_config: dict[str, Any], cli_config: dict[str, Any]) -> dict[str, Any]:
        merged = dict(yaml_config)
        merged.update(cli_config)
        return merged
