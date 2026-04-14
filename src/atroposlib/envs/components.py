"""Backward compatibility aliases for extracted environment components."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from atroposlib.cli.adapters import CliAdapter

from .checkpoint_manager import CheckpointManager
from .metrics_logger import MetricsLogger
from .transport_client import TransportClient
from .worker_runtime import WorkerRuntime

EnvRuntime = WorkerRuntime
EnvTransportClient = TransportClient
EnvLogger = MetricsLogger
EnvCheckpointManager = CheckpointManager


@dataclass
class EnvCliBuilder:
    """Legacy CLI builder retained for compatibility."""

    adapter: CliAdapter = field(default_factory=CliAdapter)

    def build(self, config: dict[str, Any]) -> list[str]:
        return self.adapter.build_cli_args(config)


@dataclass
class EnvConfigMerger:
    """Legacy config merge helper retained for compatibility."""

    adapter: CliAdapter = field(default_factory=CliAdapter)

    def merge(self, yaml_config: dict[str, Any], cli_config: dict[str, Any]) -> dict[str, Any]:
        return self.adapter.merge_yaml_and_cli(yaml_config, cli_config)
