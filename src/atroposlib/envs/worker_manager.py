"""Worker orchestration and backlog management for environment execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from atropos.reproducibility import SeedManager


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

    def enqueue(self, work_item: dict[str, Any]) -> int:
        self.backlog.append(dict(work_item))
        return len(self.backlog)

    def recommended_workers(self, requested_workers: int) -> int:
        bounded_requested = max(self.min_workers, min(requested_workers, self.max_workers))
        backlog_pressure = max(1, len(self.backlog))
        return min(self.max_workers, max(bounded_requested, backlog_pressure))

    def pop_next(self) -> dict[str, Any]:
        if not self.backlog:
            raise RuntimeError("Cannot pop from an empty backlog")
        return self.backlog.pop(0)

    def orchestrate(self, work_item: dict[str, Any], requested_workers: int = 1) -> dict[str, Any]:
        self.enqueue(work_item)
        selected_workers = self.recommended_workers(requested_workers)
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
        }
