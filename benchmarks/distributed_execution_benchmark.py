"""Benchmark asyncio vs distributed execution backends."""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass

from atroposlib.envs.distributed_execution import (
    AsyncioTaskExecutionBackend,
    RayTaskExecutionBackend,
    RetryPolicy,
    TaskResult,
    TaskSpec,
)


@dataclass(frozen=True)
class BenchmarkSummary:
    backend: str
    task_count: int
    worker_count: int
    wall_time_seconds: float
    throughput_tps: float
    failed_tasks: int


def _simulated_task(payload: dict[str, float | int]) -> dict[str, float]:
    # Sleep-based synthetic load to keep benchmark deterministic and lightweight.
    time.sleep(float(payload["sleep_seconds"]))
    value = float(payload["value"])
    return {"score": value * value}


def _run_benchmark(
    *,
    backend_name: str,
    task_count: int,
    worker_count: int,
    sleep_seconds: float,
) -> tuple[BenchmarkSummary, list[TaskResult]]:
    tasks = [
        TaskSpec(
            task_id=f"task-{idx}",
            payload={"sleep_seconds": sleep_seconds, "value": random.uniform(0.5, 1.5)},
        )
        for idx in range(task_count)
    ]
    retry_policy = RetryPolicy(max_retries=1)
    if backend_name == "asyncio":
        backend = AsyncioTaskExecutionBackend()
    elif backend_name == "ray":
        backend = RayTaskExecutionBackend(address="auto")
    else:
        raise ValueError(f"Unsupported backend: {backend_name}")

    started = time.perf_counter()
    results = backend.run_tasks(
        tasks,
        _simulated_task,
        retry_policy=retry_policy,
        worker_count=worker_count,
    )
    elapsed = time.perf_counter() - started
    failed = sum(1 for item in results if not item.ok)
    return (
        BenchmarkSummary(
            backend=backend.name,
            task_count=task_count,
            worker_count=worker_count,
            wall_time_seconds=elapsed,
            throughput_tps=task_count / max(elapsed, 1e-9),
            failed_tasks=failed,
        ),
        results,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-count", type=int, default=64)
    parser.add_argument("--worker-count", type=int, default=8)
    parser.add_argument("--sleep-seconds", type=float, default=0.01)
    args = parser.parse_args()

    baseline_summary, _ = _run_benchmark(
        backend_name="asyncio",
        task_count=args.task_count,
        worker_count=args.worker_count,
        sleep_seconds=args.sleep_seconds,
    )

    payload: dict[str, object] = {"asyncio": asdict(baseline_summary)}

    try:
        ray_summary, _ = _run_benchmark(
            backend_name="ray",
            task_count=args.task_count,
            worker_count=args.worker_count,
            sleep_seconds=args.sleep_seconds,
        )
        speedup = baseline_summary.wall_time_seconds / max(ray_summary.wall_time_seconds, 1e-9)
        payload["ray"] = asdict(ray_summary)
        payload["ray_vs_asyncio_speedup"] = speedup
    except RuntimeError as exc:
        payload["ray"] = {"skipped": True, "reason": str(exc)}

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
