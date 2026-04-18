"""Thin BaseEnv orchestration facade.

BaseEnv exposes a stable environment contract while delegating runtime details
(worker orchestration, transport, metrics, checkpointing, and CLI shims) to
specialized collaborators.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from atroposlib.cli.adapters import CliAdapter

from .checkpoint_manager import CheckpointManager
from .compatibility_adapter import BaseEnvCompatibilityAdapter
from .dependency_factory import BaseEnvDependencyFactory
from .env_logic import EnvLogic, PassthroughEnvLogic
from .metrics_logger import MetricsLogger
from .runtime_controller import RuntimeController
from .transport_client import TransportClient
from .worker_runtime import WorkerRuntime


@dataclass(init=False)
class BaseEnv:
    """Backward-compatible facade orchestrating composable components."""

    runtime_controller: RuntimeController
    worker_runtime: WorkerRuntime = field(default_factory=WorkerRuntime)
    transport_client: TransportClient = field(default_factory=TransportClient)
    metrics_logger: MetricsLogger = field(default_factory=MetricsLogger)
    checkpoint_manager: CheckpointManager = field(default_factory=CheckpointManager)
    cli_adapter: CliAdapter = field(default_factory=CliAdapter)
    env_logic: EnvLogic = field(default_factory=PassthroughEnvLogic)

    def __init__(
        self,
        worker_runtime: WorkerRuntime | None = None,
        transport_client: TransportClient | None = None,
        *,
        worker_manager: WorkerRuntime | None = None,
        transport: TransportClient | None = None,
        metrics_logger: MetricsLogger | None = None,
        logging_manager: MetricsLogger | None = None,
        checkpoint_manager: CheckpointManager | None = None,
        cli_adapter: CliAdapter | None = None,
        env_logic: EnvLogic | None = None,
        seed: int | None = None,
        dependency_factory: BaseEnvDependencyFactory | None = None,
    ) -> None:
        factory = dependency_factory or BaseEnvDependencyFactory()
        wiring = factory.build(
            worker_runtime=worker_runtime,
            worker_manager=worker_manager,
            transport_client=transport_client,
            transport=transport,
            metrics_logger=metrics_logger,
            logging_manager=logging_manager,
            checkpoint_manager=checkpoint_manager,
            cli_adapter=cli_adapter,
            env_logic=env_logic,
            seed=seed,
        )
        self.worker_runtime = wiring.collaborators.worker_runtime
        self.transport_client = wiring.collaborators.transport_client
        self.metrics_logger = wiring.collaborators.metrics_logger
        self.checkpoint_manager = wiring.collaborators.checkpoint_manager
        self.cli_adapter = wiring.collaborators.cli_adapter
        self.env_logic = wiring.collaborators.env_logic
        self.seed_manager = wiring.seed_manager
        self.seed_metadata: dict[str, Any] = wiring.seed_metadata
        self.runtime_controller = wiring.runtime_controller
        self._compatibility = BaseEnvCompatibilityAdapter(
            worker_runtime=self.worker_runtime,
            transport_client=self.transport_client,
            metrics_logger=self.metrics_logger,
            checkpoint_manager=self.checkpoint_manager,
            cli_adapter=self.cli_adapter,
        )

    def step(self, payload: dict[str, Any], worker_count: int = 1) -> dict[str, Any]:
        return self.runtime_controller.run_step(payload=payload, worker_count=worker_count)

    def process(self, payload: dict[str, Any], worker_count: int = 1) -> dict[str, Any]:
        """Compatibility alias for step()."""

        return self.step(payload=payload, worker_count=worker_count)

    def evaluate(self, payload: dict[str, Any], worker_count: int = 1) -> dict[str, Any]:
        """Compatibility alias for step() used by evaluation flows."""

        return self.runtime_controller.evaluate(payload=payload, worker_count=worker_count)

    def serve(
        self,
        payload_or_stream: dict[str, Any] | Iterable[dict[str, Any]],
        worker_count: int = 1,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Compatibility endpoint for serving single requests or streams."""

        return self._compatibility.serve(self.step, payload_or_stream, worker_count)

    def reset(self) -> None:
        self._compatibility.reset_runtime_state()

    # -------------------------------
    # CLI + config compatibility API
    # -------------------------------
    def build_cli_args(self, config: dict[str, Any]) -> list[str]:
        return self._compatibility.build_cli_args(config)

    def merge_yaml_and_cli(
        self,
        yaml_config: dict[str, Any],
        cli_config: dict[str, Any],
    ) -> dict[str, Any]:
        return self._compatibility.merge_yaml_and_cli(yaml_config, cli_config)

    # -------------------------------------
    # Legacy method aliases for compatibility
    # -------------------------------------
    def orchestrate_workers(
        self,
        work_item: dict[str, Any],
        worker_count: int = 1,
    ) -> dict[str, Any]:
        return self._compatibility.orchestrate_workers(work_item, worker_count)

    def call_api(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._compatibility.call_api(payload)

    def log_event(self, event: str, **metadata: Any) -> None:
        self._compatibility.log_event(event, **metadata)

    def checkpoint(self, state: dict[str, Any]) -> None:
        self._compatibility.checkpoint(state)

    # Compatibility properties for code that referenced old collaborator names.
    @property
    def runtime(self) -> WorkerRuntime:
        return self.worker_runtime

    @property
    def worker_manager(self) -> WorkerRuntime:
        return self.worker_runtime

    @property
    def transport(self) -> TransportClient:
        return self.transport_client

    @property
    def logger(self) -> MetricsLogger:
        return self.metrics_logger

    @property
    def logging_manager(self) -> MetricsLogger:
        return self.metrics_logger
