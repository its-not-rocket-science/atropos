from __future__ import annotations

import pytest

from atroposlib.envs.worker_manager import WorkerManager


@pytest.mark.stress
def test_high_concurrency_rollout_generation() -> None:
    manager = WorkerManager(max_workers=64)

    payloads = [{"rollout_id": idx} for idx in range(400)]
    output = manager.execute_batch(
        payloads,
        lambda payload: {"rollout_id": payload["rollout_id"], "ok": True},
        requested_workers=64,
        env="stress",
    )

    assert output["worker_count"] == 64
    assert output["total_tasks"] == 400
    assert output["failed_tasks"] == 0
    assert len(output["results"]) == 400
    assert all(result.ok for result in output["results"])
