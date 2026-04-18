"""Composition helpers for building BaseEnv runtime collaborators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from atropos.reproducibility import SeedManager, apply_global_seed
from atroposlib.cli.adapters import CliAdapter

from .checkpoint_manager import CheckpointManager
from .env_logic import EnvLogic, EnvLogicItemSource, EnvLogicRolloutCollector, PassthroughEnvLogic
from .metrics_logger import MetricsLogger
from .runtime_controller import RuntimeController
from .transport_client import TransportClient
from .worker_runtime import WorkerRuntime


@dataclass
class RuntimeCollaborators:
    """Concrete collaborator instances used by ``BaseEnv``."""

    worker_runtime: WorkerRuntime
    transport_client: TransportClient
    metrics_logger: MetricsLogger
    checkpoint_manager: CheckpointManager
    cli_adapter: CliAdapter
    env_logic: EnvLogic


@dataclass
class RuntimeWiring:
    """Composed runtime dependencies, including deterministic seed metadata."""

    collaborators: RuntimeCollaborators
    runtime_controller: RuntimeController
    seed_manager: SeedManager | None
    seed_metadata: dict[str, Any]


@dataclass
class BaseEnvDependencyFactory:
    """Factory responsible for wiring ``BaseEnv`` runtime dependencies."""

    def build(
        self,
        *,
        worker_runtime: WorkerRuntime | None = None,
        worker_manager: WorkerRuntime | None = None,
        transport_client: TransportClient | None = None,
        transport: TransportClient | None = None,
        metrics_logger: MetricsLogger | None = None,
        logging_manager: MetricsLogger | None = None,
        checkpoint_manager: CheckpointManager | None = None,
        cli_adapter: CliAdapter | None = None,
        env_logic: EnvLogic | None = None,
        seed: int | None = None,
    ) -> RuntimeWiring:
        runtime = worker_runtime or worker_manager or WorkerRuntime()
        logger = metrics_logger or logging_manager or MetricsLogger()
        transport_path = transport or transport_client or TransportClient()
        checkpoints = checkpoint_manager or CheckpointManager()
        cli = cli_adapter or CliAdapter()
        logic = env_logic or PassthroughEnvLogic()

        seed_manager: SeedManager | None = None
        seed_metadata: dict[str, Any] = {}
        if seed is not None:
            seed_manager = SeedManager(seed)
            seed_metadata = apply_global_seed(seed, component="base_env")
            runtime.seed_manager = seed_manager

        controller = RuntimeController(
            item_source=EnvLogicItemSource(logic),
            backlog_manager=runtime,
            send_to_api=transport_path,
            rollout_collector=EnvLogicRolloutCollector(logic),
            metrics_logger=logger,
            checkpoint_manager=checkpoints,
            seed_manager=seed_manager,
        )
        return RuntimeWiring(
            collaborators=RuntimeCollaborators(
                worker_runtime=runtime,
                transport_client=transport_path,
                metrics_logger=logger,
                checkpoint_manager=checkpoints,
                cli_adapter=cli,
                env_logic=logic,
            ),
            runtime_controller=controller,
            seed_manager=seed_manager,
            seed_metadata=seed_metadata,
        )
