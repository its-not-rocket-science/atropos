"""Worker orchestration and backlog management for environment execution."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from atropos.reproducibility import SeedManager

from .distributed_execution import (
    AsyncioTaskExecutionBackend,
    RetryPolicy,
    TaskExecutionBackend,
    TaskSpec,
)


@dataclass
class WorkerManager:
    """Manage in-memory backlog and worker scaling policy.

    The default implementation is intentionally deterministic and side-effect free,
    which makes it easy to unit test and safe to run in constrained CI.
    """

    min_workers: int = 1
    max_workers: int = 32
    backlog: list[dict[str, Any]] = field(default_factory=list)
    seed_manager: SeedManager | None = None
    environment_rates: dict[str, float] = field(default_factory=dict)
    trainer_queue_depths: dict[str, int] = field(default_factory=dict)
    target_queue_depth: int = 8
    scale_up_gain: float = 0.25
    scale_down_gain: float = 0.15
    max_rate_limit: float = 1.0
    min_rate_limit: float = 0.2
    execution_backend: TaskExecutionBackend = field(
        default_factory=AsyncioTaskExecutionBackend
    )
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)

    def enqueue(self, work_item: dict[str, Any]) -> int:
        self.backlog.append(dict(work_item))
        return len(self.backlog)

    def _rate_limit_for_env(self, env: str) -> float:
        return self.environment_rates.get(env, self.max_rate_limit)

    def update_trainer_feedback(self, env: str, feedback: dict[str, Any] | None) -> None:
        if not feedback:
            return
        queue_depth = feedback.get("queue_depth")
        if queue_depth is not None:
            queue_depth_int = max(0, int(queue_depth))
            self.trainer_queue_depths[env] = queue_depth_int
            if queue_depth_int > self.target_queue_depth:
                overload = queue_depth_int - self.target_queue_depth
                overload_ratio = overload / max(1, self.target_queue_depth)
                factor = max(0.0, 1.0 - self.scale_up_gain * overload_ratio)
            else:
                available = self.target_queue_depth - queue_depth_int
                available_ratio = available / max(1, self.target_queue_depth)
                factor = 1.0 + self.scale_down_gain * available_ratio
            existing = self._rate_limit_for_env(env)
            updated = existing * factor
            self.environment_rates[env] = max(
                self.min_rate_limit,
                min(updated, self.max_rate_limit),
            )

    def recommended_workers(self, requested_workers: int, env: str) -> int:
        bounded_requested = max(self.min_workers, min(requested_workers, self.max_workers))
        effective_backlog_pressure = max(1, len(self.backlog))
        target_workers = max(bounded_requested, effective_backlog_pressure)
        rate_limited_workers = int(round(target_workers * self._rate_limit_for_env(env)))
        return max(self.min_workers, min(self.max_workers, rate_limited_workers))

    def execute_batch(
        self,
        task_payloads: list[dict[str, Any]],
        task_fn: Callable[[dict[str, Any]], Any],
        *,
        requested_workers: int = 1,
        env: str = "default",
    ) -> dict[str, Any]:
        """Execute a batch of tasks through the configured execution backend."""

        worker_count = self.recommended_workers(requested_workers, env=env)
        tasks = [
            TaskSpec(task_id=f"task-{idx}", payload=payload)
            for idx, payload in enumerate(task_payloads)
        ]
        results = self.execution_backend.run_tasks(
            tasks,
            task_fn,
            retry_policy=self.retry_policy,
            worker_count=worker_count,
        )
        failed = [result for result in results if not result.ok]
        return {
            "worker_count": worker_count,
            "backend": self.execution_backend.name,
            "supports_multi_node": self.execution_backend.supports_multi_node,
            "total_tasks": len(results),
            "failed_tasks": len(failed),
            "results": results,
        }

    def pop_next(self) -> dict[str, Any]:
        if not self.backlog:
            raise RuntimeError("Cannot pop from an empty backlog")
        return self.backlog.pop(0)

    def orchestrate(
        self,
        work_item: dict[str, Any],
        requested_workers: int = 1,
        *,
        env: str = "default",
        trainer_feedback: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.update_trainer_feedback(env, trainer_feedback)
        self.enqueue(work_item)
        selected_workers = self.recommended_workers(requested_workers, env=env)
        next_item = self.pop_next()
        worker_seed = None
        if self.seed_manager is not None:
            worker_seed = self.seed_manager.derive_seed(
                "worker_manager",
                stage="orchestrate",
                worker_id=selected_workers,
            )
        return {
            "worker_count": selected_workers,
            "work_item": next_item,
            "status": "processed",
            "backlog_size": len(self.backlog),
            "worker_seed": worker_seed,
            "env": env,
            "rate_limit": self._rate_limit_for_env(env),
            "trainer_queue_depth": self.trainer_queue_depths.get(env, 0),
        }
