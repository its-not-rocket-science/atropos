"""BaseEnv compatibility facade.

`BaseEnv` previously mixed environment contract, worker orchestration, API
transport, logging, checkpointing, CLI generation, and YAML/CLI merging.

This module preserves the external behavior shape while moving each concern to
explicit collaborators that can be swapped incrementally.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .components import (
    EnvCheckpointManager,
    EnvCliBuilder,
    EnvConfigMerger,
    EnvLogger,
    EnvRuntime,
    EnvTransportClient,
)


@dataclass
class BaseEnv:
    """Backward-compatible environment facade with explicit collaborators."""

    runtime: EnvRuntime = field(default_factory=EnvRuntime)
    transport: EnvTransportClient = field(default_factory=EnvTransportClient)
    logger: EnvLogger = field(default_factory=EnvLogger)
    checkpoint_manager: EnvCheckpointManager = field(default_factory=EnvCheckpointManager)
    cli_builder: EnvCliBuilder = field(default_factory=EnvCliBuilder)
    config_merger: EnvConfigMerger = field(default_factory=EnvConfigMerger)

    # ------------------------
    # Environment contract API
    # ------------------------
    def step(self, payload: dict[str, Any], worker_count: int = 1) -> dict[str, Any]:
        """Execute one environment step.

        Behavior remains: run workers -> call transport -> log -> checkpoint.
        """

        self.logger.info("step_started", worker_count=worker_count)
        runtime_result = self.runtime.run(payload, worker_count=worker_count)
        transport_result = self.transport.send(runtime_result)
        self.checkpoint_manager.save(transport_result)
        self.logger.info("step_finished", status=transport_result.get("ok", False))
        return transport_result

    def reset(self) -> None:
        """Reset volatile state while preserving configured collaborators."""

        self.logger.events.clear()
        self.checkpoint_manager.snapshots.clear()

    # -------------------------------
    # CLI + config compatibility API
    # -------------------------------
    def build_cli_args(self, config: dict[str, Any]) -> list[str]:
        return self.cli_builder.build(config)

    def merge_yaml_and_cli(
        self,
        yaml_config: dict[str, Any],
        cli_config: dict[str, Any],
    ) -> dict[str, Any]:
        return self.config_merger.merge(yaml_config, cli_config)

    # -------------------------------------
    # Legacy method aliases for compatibility
    # -------------------------------------
    def orchestrate_workers(self, work_item: dict[str, Any], worker_count: int = 1) -> dict[str, Any]:
        return self.runtime.run(work_item, worker_count=worker_count)

    def call_api(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.transport.send(payload)

    def log_event(self, event: str, **metadata: Any) -> None:
        self.logger.info(event, **metadata)

    def checkpoint(self, state: dict[str, Any]) -> None:
        self.checkpoint_manager.save(state)
