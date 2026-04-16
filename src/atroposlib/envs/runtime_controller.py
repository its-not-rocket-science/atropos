"""Runtime loop controller extracted from BaseEnv."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from atropos.reproducibility import SeedManager

from ..logging_utils import build_log_context
from ..observability import OBSERVABILITY, timed_rollout, tracing_span
from .checkpoint_manager import CheckpointManager
from .metrics_logger import MetricsLogger
from .runtime_interfaces import (
    BacklogManager,
    EvalRunner,
    ItemSource,
    RolloutCollector,
    SendToApiPath,
)

RUNTIME_LOGGER = logging.getLogger("atroposlib.envs.runtime")


@dataclass
class RuntimeController(EvalRunner):
    """Coordinates runtime collaborators through formal interfaces."""

    item_source: ItemSource
    backlog_manager: BacklogManager
    send_to_api: SendToApiPath
    rollout_collector: RolloutCollector
    metrics_logger: MetricsLogger
    checkpoint_manager: CheckpointManager
    seed_manager: SeedManager | None = None

    def run_step(self, payload: dict[str, Any], worker_count: int = 1) -> dict[str, Any]:
        env_name = str(payload.get("env", "default"))
        request_id = payload.get("request_id")
        batch_id = payload.get("batch_id")
        current_step = payload.get("current_step")
        self.metrics_logger.log_event("step_started", worker_count=worker_count, env=env_name)
        RUNTIME_LOGGER.info(
            "runtime_step_started",
            extra=build_log_context(
                env_id=env_name,
                request_id=request_id,
                batch_id=batch_id,
                worker_id=worker_count,
                current_step=current_step,
            ),
        )
        with tracing_span(
            "baseenv.step",
            attributes={
                "atropos.env": env_name,
                "atropos.worker.requested": worker_count,
                "atropos.request_id": str(request_id) if request_id else "",
                "atropos.batch_id": str(batch_id) if batch_id else "",
            },
        ):
            with timed_rollout(env_name):
                started = perf_counter()
                with tracing_span(
                    "baseenv.environment_item_fetch",
                    attributes={
                        "atropos.env": env_name,
                        "atropos.payload.keys": len(payload),
                    },
                ):
                    prepared = self.item_source.prepare_item(payload)
                trainer_feedback = payload.get("trainer_feedback")
                runtime_result = self.backlog_manager.orchestrate(
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
                try:
                    with tracing_span(
                        "baseenv.send_to_api",
                        attributes={
                            "atropos.env": env_name,
                            "atropos.worker.selected": int(
                                runtime_result.get("worker_count", worker_count)
                            ),
                            "atropos.backlog_size": int(runtime_result.get("backlog_size", 0)),
                        },
                    ):
                        transport_result = self.send_to_api.send(runtime_result)
                except Exception:
                    OBSERVABILITY.observe_failed_send(env=env_name)
                    raise
                with tracing_span(
                    "baseenv.postprocess",
                    attributes={
                        "atropos.env": env_name,
                        "atropos.transport.ok": bool(transport_result.get("ok", False)),
                    },
                ):
                    with tracing_span(
                        "baseenv.trajectory_collection",
                        attributes={
                            "atropos.env": env_name,
                            "atropos.transport.keys": len(transport_result),
                        },
                    ):
                        finalized = self.rollout_collector.collect(transport_result)
                self.checkpoint_manager.save(finalized)
                selected_workers = int(runtime_result.get("worker_count", worker_count))
                OBSERVABILITY.set_worker_count(env=env_name, worker_count=selected_workers)
                max_workers = max(
                    1,
                    int(getattr(self.backlog_manager, "max_workers", selected_workers)),
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
                duration = perf_counter() - started
                self.metrics_logger.log_event(
                    "step_finished",
                    status=finalized.get("ok", False),
                    env=env_name,
                    rollout_latency_seconds=duration,
                )
                RUNTIME_LOGGER.info(
                    "runtime_step_finished",
                    extra=build_log_context(
                        env_id=env_name,
                        request_id=request_id,
                        batch_id=batch_id,
                        worker_id=int(runtime_result.get("worker_count", worker_count)),
                        current_step=current_step,
                        status=bool(finalized.get("ok", False)),
                        rollout_latency_seconds=duration,
                    ),
                )
                return finalized

    def evaluate(self, payload: dict[str, Any], worker_count: int = 1) -> dict[str, Any]:
        env_name = str(payload.get("env", "default"))
        started = perf_counter()
        try:
            return self.run_step(payload, worker_count=worker_count)
        finally:
            OBSERVABILITY.observe_eval_duration(
                env=env_name,
                duration_seconds=perf_counter() - started,
            )
