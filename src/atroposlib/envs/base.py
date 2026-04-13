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
from time import perf_counter
from typing import Any

from atropos.reproducibility import SeedManager, apply_global_seed

from ..observability import OBSERVABILITY, timed_rollout, tracing_span
from .checkpoint_manager import CheckpointManager
from .cli_adapter import CliAdapter
from .env_logic import EnvLogic, PassthroughEnvLogic
from .metrics_logger import MetricsLogger
from .transport_client import TransportClient
from .worker_runtime import WorkerRuntime


@dataclass(init=False)
class BaseEnv:
    """Backward-compatible facade orchestrating composable components."""

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

    def step(self, payload: dict[str, Any], worker_count: int = 1) -> dict[str, Any]:
        env_name = str(payload.get("env", "default"))
        self.metrics_logger.log_event("step_started", worker_count=worker_count, env=env_name)
        with tracing_span(
            "baseenv.step",
            attributes={"atropos.env": env_name, "atropos.worker.requested": worker_count},
        ):
            with timed_rollout(env_name):
                started = perf_counter()
                prepared = self.env_logic.prepare_step(payload)
                trainer_feedback = payload.get("trainer_feedback")
                runtime_result = self.worker_runtime.orchestrate(
                    prepared,
                    requested_workers=worker_count,
                    env=env_name,
                    trainer_feedback=(
                        trainer_feedback if isinstance(trainer_feedback, dict) else None
                    ),
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
                    int(getattr(self.worker_runtime, "max_workers", selected_workers)),
                )
                OBSERVABILITY.set_worker_utilization(
                    env=env_name,
                    utilization_ratio=selected_workers / max_workers,
                )
                trainer_queue_depth = int(runtime_result.get("trainer_queue_depth", 0))
                OBSERVABILITY.set_trainer_queue_depth(
                    env=env_name,
                    queue_depth=trainer_queue_depth,
                )
                rate_limit = float(runtime_result.get("rate_limit", 1.0))
                OBSERVABILITY.set_env_rate_limit(env=env_name, rate_limit=rate_limit)
                self.metrics_logger.log_event(
                    "step_finished",
                    status=finalized.get("ok", False),
                    env=env_name,
                    rollout_latency_seconds=perf_counter() - started,
                )
                return finalized

    def process(self, payload: dict[str, Any], worker_count: int = 1) -> dict[str, Any]:
        """Compatibility alias for step()."""

        return self.step(payload=payload, worker_count=worker_count)

    def evaluate(self, payload: dict[str, Any], worker_count: int = 1) -> dict[str, Any]:
        """Compatibility alias for step() used by evaluation flows."""

        return self.step(payload=payload, worker_count=worker_count)

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
