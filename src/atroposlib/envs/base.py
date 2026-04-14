"""Thin BaseEnv orchestration facade.

BaseEnv composes specialized collaborators:
- WorkerRuntime
- TransportClient
- CheckpointManager
- MetricsLogger
- CliAdapter
- EnvLogic
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from atropos.reproducibility import SeedManager, apply_global_seed
from atroposlib.cli.adapters import CliAdapter

from .checkpoint_manager import CheckpointManager
from .env_logic import EnvLogic, EnvLogicItemSource, EnvLogicRolloutCollector, PassthroughEnvLogic
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
    ) -> None:
        runtime = worker_runtime or worker_manager
        logger = metrics_logger or logging_manager
        self.worker_runtime = runtime or WorkerRuntime()
        self.transport_client = transport or transport_client or TransportClient()
        self.metrics_logger = logger or MetricsLogger()
        self.checkpoint_manager = checkpoint_manager or CheckpointManager()
        self.cli_adapter = cli_adapter or CliAdapter()
        self.env_logic = env_logic or PassthroughEnvLogic()
        self.seed_manager: SeedManager | None = None
        self.seed_metadata: dict[str, Any] = {}
        if seed is not None:
            self.seed_manager = SeedManager(seed)
            self.seed_metadata = apply_global_seed(seed, component="base_env")
            self.worker_runtime.seed_manager = self.seed_manager
        self.runtime_controller = RuntimeController(
            item_source=EnvLogicItemSource(self.env_logic),
            backlog_manager=self.worker_runtime,
            send_to_api=self.transport_client,
            rollout_collector=EnvLogicRolloutCollector(self.env_logic),
            metrics_logger=self.metrics_logger,
            checkpoint_manager=self.checkpoint_manager,
            seed_manager=self.seed_manager,
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

        if isinstance(payload_or_stream, dict):
            return self.step(payload_or_stream, worker_count=worker_count)
        return [self.step(payload, worker_count=worker_count) for payload in payload_or_stream]

    def reset(self) -> None:
        self.metrics_logger.reset()
        self.checkpoint_manager.reset()

    # -------------------------------
    # CLI + config compatibility API
    # -------------------------------
    def build_cli_args(self, config: dict[str, Any]) -> list[str]:
        return self.cli_adapter.build_cli_args(config)

    def merge_yaml_and_cli(
        self,
        yaml_config: dict[str, Any],
        cli_config: dict[str, Any],
    ) -> dict[str, Any]:
        return self.cli_adapter.merge_yaml_and_cli(yaml_config, cli_config)

    # -------------------------------------
    # Legacy method aliases for compatibility
    # -------------------------------------
    def orchestrate_workers(
        self,
        work_item: dict[str, Any],
        worker_count: int = 1,
    ) -> dict[str, Any]:
        return self.worker_runtime.orchestrate(work_item, requested_workers=worker_count)

    def call_api(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.transport_client.send(payload)

    def log_event(self, event: str, **metadata: Any) -> None:
        self.metrics_logger.log_event(event, **metadata)

    def checkpoint(self, state: dict[str, Any]) -> None:
        self.checkpoint_manager.save(state)

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
