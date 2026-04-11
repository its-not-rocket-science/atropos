"""Pluggable task execution backends for local and distributed workers."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from typing import Any

TaskCallable = Callable[[dict[str, Any]], Any]


@dataclass(frozen=True)
class RetryPolicy:
    """Retry policy shared across execution backends."""

    max_retries: int = 2
    retryable_exceptions: tuple[type[Exception], ...] = (Exception,)


@dataclass(frozen=True)
class TaskSpec:
    """A unit of work for a task executor."""

    task_id: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class TaskResult:
    """Execution outcome for a task."""

    task_id: str
    ok: bool
    attempts: int
    elapsed_seconds: float
    backend: str
    worker_node: str
    result: Any = None
    error: str | None = None


class TaskExecutionBackend(ABC):
    """Abstraction layer for executing task batches."""

    name: str = "unknown"

    @property
    @abstractmethod
    def supports_multi_node(self) -> bool:
        """Whether the backend can schedule work on multiple nodes."""

    @abstractmethod
    def run_tasks(
        self,
        tasks: list[TaskSpec],
        task_fn: TaskCallable,
        *,
        retry_policy: RetryPolicy,
        worker_count: int,
    ) -> list[TaskResult]:
        """Run a batch of tasks and return ordered results."""


class AsyncioTaskExecutionBackend(TaskExecutionBackend):
    """Baseline local backend using asyncio workers."""

    name = "asyncio"

    @property
    def supports_multi_node(self) -> bool:
        return False

    async def _run_async(
        self,
        tasks: list[TaskSpec],
        task_fn: TaskCallable,
        *,
        retry_policy: RetryPolicy,
        worker_count: int,
    ) -> list[TaskResult]:
        semaphore = asyncio.Semaphore(max(1, worker_count))

        async def _execute(task: TaskSpec) -> TaskResult:
            async with semaphore:
                started = perf_counter()
                for attempt in range(1, retry_policy.max_retries + 2):
                    try:
                        maybe_coro = task_fn(task.payload)
                        if asyncio.iscoroutine(maybe_coro):
                            value = await maybe_coro
                        else:
                            value = maybe_coro
                        return TaskResult(
                            task_id=task.task_id,
                            ok=True,
                            attempts=attempt,
                            elapsed_seconds=perf_counter() - started,
                            backend=self.name,
                            worker_node="local",
                            result=value,
                        )
                    except retry_policy.retryable_exceptions as exc:
                        if attempt > retry_policy.max_retries:
                            return TaskResult(
                                task_id=task.task_id,
                                ok=False,
                                attempts=attempt,
                                elapsed_seconds=perf_counter() - started,
                                backend=self.name,
                                worker_node="local",
                                error=str(exc),
                            )
                return TaskResult(
                    task_id=task.task_id,
                    ok=False,
                    attempts=retry_policy.max_retries + 1,
                    elapsed_seconds=perf_counter() - started,
                    backend=self.name,
                    worker_node="local",
                    error="unknown execution error",
                )

        return list(await asyncio.gather(*(_execute(task) for task in tasks)))

    def run_tasks(
        self,
        tasks: list[TaskSpec],
        task_fn: TaskCallable,
        *,
        retry_policy: RetryPolicy,
        worker_count: int,
    ) -> list[TaskResult]:
        return asyncio.run(
            self._run_async(
                tasks,
                task_fn,
                retry_policy=retry_policy,
                worker_count=worker_count,
            )
        )


class RayTaskExecutionBackend(TaskExecutionBackend):
    """Distributed backend powered by Ray."""

    name = "ray"

    def __init__(
        self,
        *,
        address: str = "auto",
        namespace: str = "atropos",
        ignore_reinit_error: bool = True,
    ) -> None:
        self.address = address
        self.namespace = namespace
        self.ignore_reinit_error = ignore_reinit_error

    @property
    def supports_multi_node(self) -> bool:
        return True

    def _ensure_initialized(self) -> Any:
        try:
            import ray
        except ImportError as exc:  # pragma: no cover - covered by tests via exception path
            raise RuntimeError(
                "Ray backend requested but ray is not installed. Install with `pip install ray`."
            ) from exc

        if not ray.is_initialized():
            init_kwargs: dict[str, Any] = {
                "namespace": self.namespace,
                "ignore_reinit_error": self.ignore_reinit_error,
            }
            if self.address != "auto":
                init_kwargs["address"] = self.address
            try:
                ray.init(**init_kwargs)
            except Exception:
                if self.address == "auto":
                    ray.init(ignore_reinit_error=self.ignore_reinit_error, namespace=self.namespace)
                else:
                    raise
        return ray

    def run_tasks(
        self,
        tasks: list[TaskSpec],
        task_fn: TaskCallable,
        *,
        retry_policy: RetryPolicy,
        worker_count: int,
    ) -> list[TaskResult]:
        ray = self._ensure_initialized()

        def _ray_worker(payload: dict[str, Any], task_id: str) -> tuple[str, str, Any]:
            import socket

            return task_id, socket.gethostname(), task_fn(payload)
        ray_worker = ray.remote(_ray_worker)

        pending: dict[str, tuple[TaskSpec, int]] = {task.task_id: (task, 1) for task in tasks}
        started_by_id = {task.task_id: perf_counter() for task in tasks}
        refs = {
            task_id: ray_worker.options(num_cpus=1).remote(spec.payload, task_id)
            for task_id, (spec, _) in pending.items()
        }
        ordered_results: dict[str, TaskResult] = {}

        max_inflight = max(1, worker_count)
        while refs:
            current_refs = list(refs.values())
            ready_refs, _ = ray.wait(current_refs, num_returns=min(len(current_refs), max_inflight))
            for ready_ref in ready_refs:
                task_id = next(item_id for item_id, ref in refs.items() if ref == ready_ref)
                task, attempt = pending[task_id]
                try:
                    resolved_task_id, host, value = ray.get(ready_ref)
                    ordered_results[resolved_task_id] = TaskResult(
                        task_id=resolved_task_id,
                        ok=True,
                        attempts=attempt,
                        elapsed_seconds=perf_counter() - started_by_id[resolved_task_id],
                        backend=self.name,
                        worker_node=host,
                        result=value,
                    )
                    refs.pop(task_id)
                    pending.pop(task_id)
                except retry_policy.retryable_exceptions as exc:
                    if attempt <= retry_policy.max_retries:
                        pending[task_id] = (task, attempt + 1)
                        refs[task_id] = ray_worker.options(num_cpus=1).remote(
                            task.payload, task_id
                        )
                    else:
                        ordered_results[task_id] = TaskResult(
                            task_id=task_id,
                            ok=False,
                            attempts=attempt,
                            elapsed_seconds=perf_counter() - started_by_id[task_id],
                            backend=self.name,
                            worker_node="unknown",
                            error=str(exc),
                        )
                        refs.pop(task_id)
                        pending.pop(task_id)

        return [ordered_results[task.task_id] for task in tasks]
