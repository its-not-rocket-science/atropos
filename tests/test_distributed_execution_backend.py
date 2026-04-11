from __future__ import annotations

import sys
from typing import Any

import pytest

from atroposlib.envs.distributed_execution import (
    AsyncioTaskExecutionBackend,
    RayTaskExecutionBackend,
    RetryPolicy,
    TaskSpec,
)
from atroposlib.envs.worker_manager import WorkerManager


def test_asyncio_backend_retries_and_succeeds() -> None:
    backend = AsyncioTaskExecutionBackend()
    calls: dict[str, int] = {}

    def flaky_task(payload: dict[str, Any]) -> dict[str, Any]:
        task_id = str(payload["task_id"])
        calls[task_id] = calls.get(task_id, 0) + 1
        if calls[task_id] == 1:
            raise RuntimeError("transient")
        return {"ok": True, "task_id": task_id}

    tasks = [TaskSpec(task_id=f"id-{idx}", payload={"task_id": f"id-{idx}"}) for idx in range(4)]
    results = backend.run_tasks(
        tasks,
        flaky_task,
        retry_policy=RetryPolicy(max_retries=1),
        worker_count=2,
    )

    assert all(result.ok for result in results)
    assert all(result.attempts == 2 for result in results)


def test_worker_manager_execute_batch_reports_backend_metadata() -> None:
    manager = WorkerManager(max_workers=6)

    output = manager.execute_batch(
        [{"value": 2}, {"value": 4}],
        lambda payload: {"double": payload["value"] * 2},
        requested_workers=3,
        env="eval",
    )

    assert output["backend"] == "asyncio"
    assert output["supports_multi_node"] is False
    assert output["worker_count"] == 3
    assert output["failed_tasks"] == 0


def test_ray_backend_raises_when_ray_is_not_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "ray", raising=False)
    backend = RayTaskExecutionBackend()

    with pytest.raises(RuntimeError, match="ray is not installed"):
        backend.run_tasks(
            [TaskSpec(task_id="id-0", payload={"x": 1})],
            lambda payload: payload["x"],
            retry_policy=RetryPolicy(max_retries=0),
            worker_count=1,
        )


def test_ray_backend_runs_tasks_with_fake_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeRef:
        def __init__(self, value: Any = None, error: Exception | None = None) -> None:
            self.value = value
            self.error = error

    class FakeRemoteFunction:
        def __init__(self, fn: Any) -> None:
            self.fn = fn

        def options(self, **_: Any) -> FakeRemoteFunction:
            return self

        def remote(self, *args: Any, **kwargs: Any) -> FakeRef:
            try:
                return FakeRef(value=self.fn(*args, **kwargs))
            except Exception as exc:  # noqa: BLE001
                return FakeRef(error=exc)

    class FakeRay:
        def __init__(self) -> None:
            self._initialized = False

        def is_initialized(self) -> bool:
            return self._initialized

        def init(self, **_: Any) -> None:
            self._initialized = True

        def remote(self, fn: Any) -> FakeRemoteFunction:
            return FakeRemoteFunction(fn)

        def wait(
            self, refs: list[FakeRef], num_returns: int
        ) -> tuple[list[FakeRef], list[FakeRef]]:
            return refs[:num_returns], refs[num_returns:]

        def get(self, ref: FakeRef) -> Any:
            if ref.error is not None:
                raise ref.error
            return ref.value

    fake_ray = FakeRay()
    monkeypatch.setitem(sys.modules, "ray", fake_ray)

    backend = RayTaskExecutionBackend(address="local")
    tasks = [
        TaskSpec(task_id="a", payload={"value": 3}),
        TaskSpec(task_id="b", payload={"value": 7}),
    ]
    results = backend.run_tasks(
        tasks,
        lambda payload: payload["value"] + 1,
        retry_policy=RetryPolicy(max_retries=0),
        worker_count=2,
    )

    assert all(result.ok for result in results)
    assert [result.result for result in results] == [4, 8]
    assert all(result.backend == "ray" for result in results)
