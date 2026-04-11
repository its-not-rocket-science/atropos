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
from time import perf_counter
from typing import Any

from atropos.reproducibility import SeedManager, apply_global_seed

from ..observability import OBSERVABILITY, timed_rollout, tracing_span
from .checkpoint_manager import CheckpointManager
from .env_logic import EnvLogic, PassthroughEnvLogic
from .logging_manager import LoggingManager
from .transport_client import TransportClient
from .worker_manager import WorkerManager


@dataclass(init=False)
class BaseEnv:
    """Backward-compatible facade orchestrating composable components."""

    worker_manager: WorkerManager = field(default_factory=WorkerManager)
    transport_client: TransportClient = field(default_factory=TransportClient)
    logging_manager: LoggingManager = field(default_factory=LoggingManager)
    checkpoint_manager: CheckpointManager = field(default_factory=CheckpointManager)
    env_logic: EnvLogic = field(default_factory=PassthroughEnvLogic)

    def __init__(
        self,
        worker_manager: WorkerManager | None = None,
        transport_client: TransportClient | None = None,
        *,
        transport: TransportClient | None = None,
        logging_manager: LoggingManager | None = None,
        checkpoint_manager: CheckpointManager | None = None,
        env_logic: EnvLogic | None = None,
        seed: int | None = None,
    ) -> None:
        self.worker_manager = worker_manager or WorkerManager()
        self.transport_client = transport or transport_client or TransportClient()
        self.logging_manager = logging_manager or LoggingManager()
        self.checkpoint_manager = checkpoint_manager or CheckpointManager()
        self.env_logic = env_logic or PassthroughEnvLogic()
        self.seed_manager: SeedManager | None = None
        self.seed_metadata: dict[str, Any] = {}
        if seed is not None:
            self.seed_manager = SeedManager(seed)
            self.seed_metadata = apply_global_seed(seed, component="base_env")
            self.worker_manager.seed_manager = self.seed_manager

    def step(self, payload: dict[str, Any], worker_count: int = 1) -> dict[str, Any]:
        env_name = str(payload.get("env", "default"))
        self.logging_manager.log_event("step_started", worker_count=worker_count, env=env_name)
        with tracing_span(
            "baseenv.step",
            attributes={"atropos.env": env_name, "atropos.worker.requested": worker_count},
        ):
            with timed_rollout(env_name):
                started = perf_counter()
                prepared = self.env_logic.prepare_step(payload)
                runtime_result = self.worker_manager.orchestrate(
                    prepared,
                    requested_workers=worker_count,
                )
                if self.seed_manager is not None:
                    inference_seed = self.seed_manager.derive_seed(
                        env_name,
                        stage="inference",
                        worker_id=int(runtime_result.get("worker_count", worker_count)),
                    )
                    runtime_result["inference_seed"] = inference_seed
                transport_result = self.transport_client.send(runtime_result)
                finalized = self.env_logic.finalize_step(transport_result)
                self.checkpoint_manager.save(finalized)
                selected_workers = int(runtime_result.get("worker_count", worker_count))
                max_workers = max(
                    1,
                    int(getattr(self.worker_manager, "max_workers", selected_workers)),
                )
                OBSERVABILITY.set_worker_utilization(
                    env=env_name,
                    utilization_ratio=selected_workers / max_workers,
                )
                self.logging_manager.log_event(
                    "step_finished",
                    status=finalized.get("ok", False),
                    env=env_name,
                    rollout_latency_seconds=perf_counter() - started,
                )
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
