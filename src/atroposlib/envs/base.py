"""Thin BaseEnv orchestration facade.

BaseEnv now composes specialized collaborators:
- WorkerManager
- TransportClient
- LoggingManager
- CheckpointManager
- EnvLogic
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .checkpoint_manager import CheckpointManager
from .env_logic import EnvLogic, PassthroughEnvLogic
from .logging_manager import LoggingManager
from .transport_client import TransportClient
from .worker_manager import WorkerManager


@dataclass
class BaseEnv:
    """Backward-compatible facade orchestrating composable components."""

    worker_manager: WorkerManager = field(default_factory=WorkerManager)
    transport_client: TransportClient = field(default_factory=TransportClient)
    logging_manager: LoggingManager = field(default_factory=LoggingManager)
    checkpoint_manager: CheckpointManager = field(default_factory=CheckpointManager)
    env_logic: EnvLogic = field(default_factory=PassthroughEnvLogic)

    def step(self, payload: dict[str, Any], worker_count: int = 1) -> dict[str, Any]:
        self.logging_manager.log_event("step_started", worker_count=worker_count)
        prepared = self.env_logic.prepare_step(payload)
        runtime_result = self.worker_manager.orchestrate(prepared, requested_workers=worker_count)
        transport_result = self.transport_client.send(runtime_result)
        finalized = self.env_logic.finalize_step(transport_result)
        self.checkpoint_manager.save(finalized)
        self.logging_manager.log_event("step_finished", status=finalized.get("ok", False))
        return finalized

    def reset(self) -> None:
        self.logging_manager.reset()
        self.checkpoint_manager.reset()

    # -------------------------------
    # CLI + config compatibility API
    # -------------------------------
    def build_cli_args(self, config: dict[str, Any]) -> list[str]:
        args: list[str] = []
        for key, value in sorted(config.items()):
            args.extend([f"--{key.replace('_', '-')}", str(value)])
        return args

    def merge_yaml_and_cli(
        self,
        yaml_config: dict[str, Any],
        cli_config: dict[str, Any],
    ) -> dict[str, Any]:
        merged = dict(yaml_config)
        merged.update(cli_config)
        return merged

    # -------------------------------------
    # Legacy method aliases for compatibility
    # -------------------------------------
    def orchestrate_workers(
        self,
        work_item: dict[str, Any],
        worker_count: int = 1,
    ) -> dict[str, Any]:
        return self.worker_manager.orchestrate(work_item, requested_workers=worker_count)

    def call_api(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.transport_client.send(payload)

    def log_event(self, event: str, **metadata: Any) -> None:
        self.logging_manager.log_event(event, **metadata)

    def checkpoint(self, state: dict[str, Any]) -> None:
        self.checkpoint_manager.save(state)

    # Compatibility properties for code that referenced old collaborator names.
    @property
    def runtime(self) -> WorkerManager:
        return self.worker_manager

    @property
    def transport(self) -> TransportClient:
        return self.transport_client

    @property
    def logger(self) -> LoggingManager:
        return self.logging_manager
