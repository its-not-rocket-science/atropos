from __future__ import annotations

from atroposlib.envs.checkpoint_manager import CheckpointManager
from atroposlib.envs.env_logic import PassthroughEnvLogic
from atroposlib.envs.logging_manager import LoggingManager
from atroposlib.envs.transport_client import TransportClient
from atroposlib.envs.worker_manager import WorkerManager


class AlwaysFailTransport(TransportClient):
    def _send_once(self, payload: dict[str, object]) -> dict[str, object]:
        raise ValueError(f"cannot send: {payload}")


def test_passthrough_logic_returns_copies() -> None:
    logic = PassthroughEnvLogic()
    payload = {"task": "hello"}
    prepared = logic.prepare_step(payload)
    payload["task"] = "mutated"

    transport_result = {"ok": True, "payload": {"x": 1}}
    finalized = logic.finalize_step(transport_result)
    transport_result["payload"]["x"] = 2

    assert prepared == {"task": "hello"}
    assert finalized == {"ok": True, "payload": {"x": 2}}


def test_logging_manager_records_and_resets_events_and_metrics() -> None:
    logger = LoggingManager()

    logger.log_event("step_started", env="qa")
    logger.record_metrics({"throughput": 12.5}, step=3)

    assert logger.events == [{"level": "info", "event": "step_started", "metadata": {"env": "qa"}}]
    assert logger.metrics == [{"throughput": 12.5, "step": 3.0}]

    logger.reset()
    assert logger.events == []
    assert logger.metrics == []


def test_checkpoint_manager_returns_latest_snapshot_copy() -> None:
    checkpoint_manager = CheckpointManager()
    checkpoint_manager.save({"iteration": 1})
    checkpoint_manager.save({"iteration": 2})

    latest = checkpoint_manager.latest()
    assert latest == {"iteration": 2}

    assert latest is not None
    latest["iteration"] = 100
    assert checkpoint_manager.latest() == {"iteration": 2}


def test_transport_client_raises_after_retries_exhausted() -> None:
    transport = AlwaysFailTransport(max_retries=2, retriable_exceptions=(ValueError,))

    try:
        transport.send({"task": "x"})
    except RuntimeError as exc:
        assert "Transport retries exhausted" in str(exc)
    else:
        raise AssertionError("expected transport send to fail")


def test_worker_manager_recommended_workers_respects_bounds_and_backlog() -> None:
    manager = WorkerManager(min_workers=2, max_workers=5)

    for idx in range(8):
        manager.enqueue({"idx": idx})

    selected = manager.recommended_workers(requested_workers=1, env="default")
    assert selected == 5
