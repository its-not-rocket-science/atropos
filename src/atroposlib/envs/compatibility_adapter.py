"""Compatibility adapter for legacy BaseEnv method surface."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from atroposlib.cli.adapters import CliAdapter

from .checkpoint_manager import CheckpointManager
from .metrics_logger import MetricsLogger
from .transport_client import TransportClient
from .worker_runtime import WorkerRuntime


@dataclass
class BaseEnvCompatibilityAdapter:
    """Encapsulates legacy convenience APIs retained by ``BaseEnv``."""

    worker_runtime: WorkerRuntime
    transport_client: TransportClient
    metrics_logger: MetricsLogger
    checkpoint_manager: CheckpointManager
    cli_adapter: CliAdapter

    def serve(
        self,
        step_fn: Callable[..., dict[str, Any]],
        payload_or_stream: dict[str, Any] | Iterable[dict[str, Any]],
        worker_count: int,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        if isinstance(payload_or_stream, dict):
            return step_fn(payload_or_stream, worker_count=worker_count)
        return [step_fn(payload, worker_count=worker_count) for payload in payload_or_stream]

    def reset_runtime_state(self) -> None:
        self.metrics_logger.reset()
        self.checkpoint_manager.reset()

    def build_cli_args(self, config: dict[str, Any]) -> list[str]:
        return self.cli_adapter.build_cli_args(config)

    def merge_yaml_and_cli(
        self,
        yaml_config: dict[str, Any],
        cli_config: dict[str, Any],
    ) -> dict[str, Any]:
        return self.cli_adapter.merge_yaml_and_cli(yaml_config, cli_config)

    def orchestrate_workers(
        self,
        work_item: dict[str, Any],
        worker_count: int,
    ) -> dict[str, Any]:
        return self.worker_runtime.orchestrate(work_item, requested_workers=worker_count)

    def call_api(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.transport_client.send(payload)

    def log_event(self, event: str, **metadata: Any) -> None:
        self.metrics_logger.log_event(event, **metadata)

    def checkpoint(self, state: dict[str, Any]) -> None:
        self.checkpoint_manager.save(state)
