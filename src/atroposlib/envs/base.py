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

from atroposlib.observability import BaseEnvTracingHooks, RuntimeMetrics

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
    tracing_hooks: BaseEnvTracingHooks = field(default_factory=BaseEnvTracingHooks)
    runtime_metrics: RuntimeMetrics = field(default_factory=RuntimeMetrics)
    env_name: str = "default"

    def __init__(
        self,
        worker_manager: WorkerManager | None = None,
        transport_client: TransportClient | None = None,
        *,
        transport: TransportClient | None = None,
        logging_manager: LoggingManager | None = None,
        checkpoint_manager: CheckpointManager | None = None,
        env_logic: EnvLogic | None = None,
        tracing_hooks: BaseEnvTracingHooks | None = None,
        runtime_metrics: RuntimeMetrics | None = None,
        env_name: str = "default",
    ) -> None:
        self.worker_manager = worker_manager or WorkerManager()
        self.transport_client = transport or transport_client or TransportClient()
        self.logging_manager = logging_manager or LoggingManager()
        self.checkpoint_manager = checkpoint_manager or CheckpointManager()
        self.env_logic = env_logic or PassthroughEnvLogic()
        self.tracing_hooks = tracing_hooks or BaseEnvTracingHooks(env_name=env_name)
        self.runtime_metrics = runtime_metrics or RuntimeMetrics()
        self.env_name = env_name

    def step(self, payload: dict[str, Any], worker_count: int = 1) -> dict[str, Any]:
        self.logging_manager.log_event("step_started", worker_count=worker_count, env=self.env_name)
        with self.tracing_hooks.step_span(worker_count=worker_count, payload=payload) as span:
            prepared = self.env_logic.prepare_step(payload)
            runtime_result = self.worker_manager.orchestrate(
                prepared,
                requested_workers=worker_count,
            )
            transport_result = self.transport_client.send(runtime_result)
            finalized = self.env_logic.finalize_step(transport_result)

            backlog_size = float(runtime_result.get("backlog_size", 0))
            selected_workers = float(runtime_result.get("worker_count", 0))
            utilization = min(
                1.0,
                selected_workers / float(max(1, self.worker_manager.max_workers)),
            )
            self.runtime_metrics.queue_size.labels(env=self.env_name).set(backlog_size)
            self.runtime_metrics.worker_utilization.labels(env=self.env_name).set(utilization)

            if span is not None:
                span.set_attribute("atropos.queue.size", backlog_size)
                span.set_attribute("atropos.worker.selected", selected_workers)
                span.set_attribute("atropos.worker.utilization", utilization)
                span.set_attribute("atropos.step.ok", bool(finalized.get("ok", False)))

            self.checkpoint_manager.save(finalized)
        self.logging_manager.log_event(
            "step_finished",
            status=finalized.get("ok", False),
            env=self.env_name,
            worker_utilization=utilization,
            queue_size=backlog_size,
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
